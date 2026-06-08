import asyncio
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import discord
import davey
import numpy as np
from discord.opus import OpusError
from discord.ext import commands
import discord.ext.voice_recv as voice_recv
from discord.ext.voice_recv.opus import PacketDecoder
from dotenv import load_dotenv
from faster_whisper import WhisperModel


load_dotenv()
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

DISCORD_SAMPLE_RATE = 48_000
WHISPER_SAMPLE_RATE = 16_000
CHANNELS = 2
SAMPLE_WIDTH = 2
BYTES_PER_SECOND = DISCORD_SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH

log = logging.getLogger("discord_voice_transcribe")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        log.warning("Invalid integer for %s=%r; using %s", name, value, default)
        return default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        log.warning("Invalid float for %s=%r; using %s", name, value, default)
        return default


def _env_language(name: str, default: Optional[str] = "en") -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default

    value = value.strip()
    if not value or value.lower() == "auto":
        return None
    return value


@dataclass(frozen=True)
class Settings:
    token: Optional[str] = os.getenv("DISCORD_TOKEN")
    voice_channel_name: str = os.getenv("VOICE_CHANNEL_NAME", "Meeting")
    whisper_model: str = os.getenv("WHISPER_MODEL", "base")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    whisper_compute_type: str = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    whisper_language: Optional[str] = _env_language("WHISPER_LANGUAGE", "en")
    whisper_beam_size: int = _env_int("WHISPER_BEAM_SIZE", 1)
    min_chunk_seconds: float = _env_float("MIN_CHUNK_SECONDS", 0.8)
    max_chunk_seconds: float = _env_float("MAX_CHUNK_SECONDS", 12.0)
    silence_flush_seconds: float = _env_float("SILENCE_FLUSH_SECONDS", 0.8)
    transcription_workers: int = _env_int("TRANSCRIPTION_WORKERS", 1)
    post_transcripts_to_discord: bool = _env_bool("POST_TRANSCRIPTS_TO_DISCORD", False)
    verbose_voice_recv_logs: bool = _env_bool("VERBOSE_VOICE_RECV_LOGS", False)


settings = Settings()


def patch_voice_recv_dave_decrypt() -> None:
    """Add DAVE receive decryption until discord-ext-voice-recv handles it natively."""
    if getattr(PacketDecoder, "_discord_voice_transcribe_dave_patch", False):
        return

    original_decode_packet = PacketDecoder._decode_packet

    def _drop_bad_packet(self, reason: str, *, user_id: Optional[int] = None):
        log.debug(
            "Dropping undecodable voice packet: reason=%s user_id=%s ssrc=%s",
            reason,
            user_id,
            self.ssrc,
        )
        assert self._decoder is not None
        return self._decoder.decode(None, fec=False)

    def _decode_packet_with_dave(self, packet):
        if packet:
            voice_client = self.sink.voice_client
            state = getattr(voice_client, "_connection", None) if voice_client else None
            dave_session = getattr(state, "dave_session", None)
            dave_required = bool(getattr(state, "dave_protocol_version", 0))
            can_decrypt = bool(dave_session and getattr(state, "can_encrypt", False))

            if dave_required:
                if not can_decrypt:
                    log.debug("Dropping packet while DAVE session is not ready for SSRC %s", self.ssrc)
                    return packet, _drop_bad_packet(self, "dave_not_ready")

                user_id = self._cached_id
                if user_id is None and voice_client is not None:
                    user_id = voice_client._get_id_from_ssrc(self.ssrc)
                    self._cached_id = user_id

                if user_id is None:
                    log.debug("Dropping DAVE packet for unknown SSRC %s", self.ssrc)
                    return packet, _drop_bad_packet(self, "unknown_ssrc")

                try:
                    packet.decrypted_data = dave_session.decrypt(
                        int(user_id),
                        davey.MediaType.audio,
                        bytes(packet.decrypted_data),
                    )
                except Exception as exc:
                    if "UnencryptedWhenPassthroughDisabled" in str(exc):
                        log.debug(
                            "DAVE marked packet as unencrypted; trying normal Opus decode for user_id=%s ssrc=%s",
                            user_id,
                            self.ssrc,
                        )
                        try:
                            return original_decode_packet(self, packet)
                        except OpusError:
                            return packet, _drop_bad_packet(
                                self,
                                "unencrypted_packet_not_valid_opus",
                                user_id=user_id,
                            )

                    log.warning(
                        "Failed to DAVE-decrypt packet for user_id=%s ssrc=%s",
                        user_id,
                        self.ssrc,
                        exc_info=True,
                    )
                    return packet, _drop_bad_packet(self, "dave_decrypt_failed", user_id=user_id)

        try:
            return original_decode_packet(self, packet)
        except OpusError:
            return packet, _drop_bad_packet(self, "opus_decode_failed")

    PacketDecoder._decode_packet = _decode_packet_with_dave
    PacketDecoder._discord_voice_transcribe_dave_patch = True


patch_voice_recv_dave_decrypt()

intents = discord.Intents.default()
intents.guilds = True
intents.voice_states = True
intents.members = True
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)


def pcm_stereo_48k_to_mono_16k(pcm_bytes: bytes) -> np.ndarray:
    """Convert Discord 48 kHz stereo s16le PCM to Whisper's 16 kHz mono float32."""
    samples = np.frombuffer(pcm_bytes, dtype=np.int16)
    if samples.size < CHANNELS:
        return np.empty(0, dtype=np.float32)

    aligned_size = samples.size - (samples.size % CHANNELS)
    stereo = samples[:aligned_size].reshape(-1, CHANNELS)
    mono_48k = stereo.mean(axis=1, dtype=np.float32) / 32768.0

    if DISCORD_SAMPLE_RATE == WHISPER_SAMPLE_RATE:
        return mono_48k.astype(np.float32, copy=False)

    if DISCORD_SAMPLE_RATE == 48_000 and WHISPER_SAMPLE_RATE == 16_000:
        aligned = mono_48k.size - (mono_48k.size % 3)
        if aligned == 0:
            return np.empty(0, dtype=np.float32)
        return mono_48k[:aligned].reshape(-1, 3).mean(axis=1).astype(np.float32)

    duration = mono_48k.size / DISCORD_SAMPLE_RATE
    target_size = int(duration * WHISPER_SAMPLE_RATE)
    if target_size <= 0:
        return np.empty(0, dtype=np.float32)

    source_x = np.linspace(0.0, duration, num=mono_48k.size, endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_size, endpoint=False)
    return np.interp(target_x, source_x, mono_48k).astype(np.float32)


class WhisperTranscriber:
    def __init__(self, config: Settings):
        self.config = config
        self._model: Optional[WhisperModel] = None
        self._model_lock = threading.Lock()
        self._force_cpu = False
        self._executor = ThreadPoolExecutor(max_workers=max(1, config.transcription_workers))
        self._user_locks: dict[int, asyncio.Lock] = {}
        self._global_slots = asyncio.Semaphore(max(1, config.transcription_workers))

    def _model_device(self) -> str:
        return "cpu" if self._force_cpu else self.config.whisper_device

    def _model_compute_type(self) -> str:
        return "int8" if self._force_cpu else self.config.whisper_compute_type

    def _get_model(self) -> WhisperModel:
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    device = self._model_device()
                    compute_type = self._model_compute_type()
                    log.info(
                        "Loading faster-whisper model=%s device=%s compute_type=%s",
                        self.config.whisper_model,
                        device,
                        compute_type,
                    )
                    self._model = WhisperModel(
                        self.config.whisper_model,
                        device=device,
                        compute_type=compute_type,
                    )
        return self._model

    def _fallback_to_cpu(self) -> None:
        with self._model_lock:
            if not self._force_cpu:
                log.warning(
                    "CUDA runtime is unavailable for faster-whisper; falling back to CPU int8. "
                    "Set WHISPER_DEVICE=cpu to skip this retry on Windows."
                )
            self._force_cpu = True
            self._model = None

    def _is_cuda_runtime_error(self, exc: RuntimeError) -> bool:
        message = str(exc).lower()
        return any(token in message for token in ("cublas", "cudnn", "cuda", "ctranslate2"))

    def _run_transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._get_model().transcribe(
            audio,
            language=self.config.whisper_language,
            vad_filter=True,
            vad_parameters={
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 160,
            },
            beam_size=self.config.whisper_beam_size,
            best_of=1,
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.65,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()

    def _transcribe_blocking(self, pcm_bytes: bytes) -> str:
        audio = pcm_stereo_48k_to_mono_16k(pcm_bytes)
        if audio.size < int(self.config.min_chunk_seconds * WHISPER_SAMPLE_RATE):
            return ""

        try:
            return self._run_transcribe(audio)
        except RuntimeError as exc:
            if self._model_device() in {"auto", "cuda"} and self._is_cuda_runtime_error(exc):
                self._fallback_to_cpu()
                return self._run_transcribe(audio)
            raise

    async def transcribe_member(
        self,
        member: discord.abc.User,
        pcm_bytes: bytes,
        text_channel: Optional[discord.abc.Messageable],
    ) -> None:
        if len(pcm_bytes) < int(self.config.min_chunk_seconds * BYTES_PER_SECOND):
            return

        user_lock = self._user_locks.setdefault(member.id, asyncio.Lock())
        async with user_lock:
            async with self._global_slots:
                loop = asyncio.get_running_loop()
                try:
                    text = await loop.run_in_executor(
                        self._executor,
                        self._transcribe_blocking,
                        pcm_bytes,
                    )
                except Exception:
                    log.exception("Failed to transcribe audio for user_id=%s", member.id)
                    return

        if not text:
            return

        name = getattr(member, "display_name", None) or member.name
        line = f"{name}: {text}"
        print(line, flush=True)

        if self.config.post_transcripts_to_discord and text_channel is not None:
            try:
                await text_channel.send(line)
            except discord.HTTPException:
                log.exception("Failed to post transcript line to Discord")

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)


transcriber = WhisperTranscriber(settings)


class LiveTranscriptionSink(voice_recv.AudioSink):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        transcriber: WhisperTranscriber,
        text_channel: Optional[discord.abc.Messageable],
    ):
        super().__init__()
        self.loop = loop
        self.transcriber = transcriber
        self.text_channel = text_channel
        self.buffers: dict[int, bytearray] = {}
        self.members: dict[int, discord.abc.User] = {}
        self.last_audio_at: dict[int, float] = {}
        self._lock = threading.Lock()
        self._max_chunk_bytes = int(settings.max_chunk_seconds * BYTES_PER_SECOND)

    def wants_opus(self) -> bool:
        return False

    def write(self, user: Optional[discord.abc.User], data: voice_recv.VoiceData) -> None:
        if user is None or not data.pcm:
            return

        now = self.loop.time()
        chunk: Optional[bytes] = None

        with self._lock:
            buffer = self.buffers.setdefault(user.id, bytearray())
            buffer.extend(data.pcm)
            self.members[user.id] = user
            self.last_audio_at[user.id] = now

            if len(buffer) >= self._max_chunk_bytes:
                chunk = bytes(buffer)
                buffer.clear()

        if chunk:
            self._submit(user, chunk)

    @voice_recv.AudioSink.listener()
    def on_voice_member_speaking_stop(self, member: discord.Member) -> None:
        observed_at = self.last_audio_at.get(member.id)
        if observed_at is None:
            return

        asyncio.run_coroutine_threadsafe(
            self._flush_after_silence(member, observed_at),
            self.loop,
        )

    async def _flush_after_silence(self, member: discord.Member, observed_at: float) -> None:
        await asyncio.sleep(settings.silence_flush_seconds)

        with self._lock:
            if self.last_audio_at.get(member.id) != observed_at:
                return

        self.flush_user(member.id)

    def flush_user(self, user_id: int) -> None:
        with self._lock:
            buffer = self.buffers.get(user_id)
            member = self.members.get(user_id)
            if not buffer or member is None:
                return
            chunk = bytes(buffer)
            buffer.clear()

        self._submit(member, chunk)

    def _submit(self, member: discord.abc.User, pcm_chunk: bytes) -> None:
        asyncio.run_coroutine_threadsafe(
            self.transcriber.transcribe_member(member, pcm_chunk, self.text_channel),
            self.loop,
        )

    def cleanup(self) -> None:
        with self._lock:
            user_ids = list(self.buffers)

        for user_id in user_ids:
            self.flush_user(user_id)


def _listen_after(error: Optional[Exception]) -> None:
    if error:
        log.error(
            "Voice receive stopped with an error",
            exc_info=(type(error), error, error.__traceback__),
        )


async def _connect_and_listen(
    ctx: commands.Context,
    channel: discord.VoiceChannel,
) -> None:
    existing = ctx.voice_client

    try:
        if existing and existing.is_connected():
            if existing.channel != channel:
                await existing.move_to(channel)
            if isinstance(existing, voice_recv.VoiceRecvClient) and existing.is_listening():
                existing.stop_listening()
            vc = existing
        else:
            vc = await channel.connect(
                cls=voice_recv.VoiceRecvClient,
                self_deaf=False,
                self_mute=True,
            )
    except RuntimeError as exc:
        await ctx.reply(f"Could not connect to voice: `{exc}`")
        return
    except discord.ClientException as exc:
        await ctx.reply(f"Could not connect to voice: `{exc}`")
        return

    if not isinstance(vc, voice_recv.VoiceRecvClient):
        await ctx.reply("Connected voice client does not support receiving audio.")
        return

    sink = LiveTranscriptionSink(asyncio.get_running_loop(), transcriber, ctx.channel)
    vc.listen(sink, after=_listen_after)

    consent_note = "Make sure everyone in the voice channel knows transcription is active."
    await ctx.reply(f"Joined **{channel.name}** and started live transcription. {consent_note}")


@bot.event
async def on_ready() -> None:
    log.info("Logged in as %s (id=%s)", bot.user, bot.user.id if bot.user else "unknown")


@bot.command()
async def join(ctx: commands.Context) -> None:
    if not ctx.author.voice or not ctx.author.voice.channel:
        await ctx.reply("You must be inside a voice channel first.")
        return

    await _connect_and_listen(ctx, ctx.author.voice.channel)


@bot.command(name="join_meeting", aliases=["meeting"])
async def join_meeting(ctx: commands.Context) -> None:
    if ctx.guild is None:
        await ctx.reply("This command must be used in a server.")
        return

    meeting_channel = discord.utils.get(ctx.guild.voice_channels, name=settings.voice_channel_name)
    if meeting_channel is None:
        await ctx.reply(f"I could not find a voice channel named `{settings.voice_channel_name}`.")
        return

    await _connect_and_listen(ctx, meeting_channel)


@bot.command()
async def status(ctx: commands.Context) -> None:
    vc = ctx.voice_client
    if not vc or not vc.is_connected():
        await ctx.reply("I am not connected to a voice channel.")
        return

    listening = isinstance(vc, voice_recv.VoiceRecvClient) and vc.is_listening()
    privacy_code = getattr(vc, "voice_privacy_code", None)
    message = f"Connected to **{vc.channel.name}**. Listening: `{listening}`."
    if privacy_code:
        message += f" DAVE privacy code: `{privacy_code}`."
    await ctx.reply(message)


@bot.command()
async def leave(ctx: commands.Context) -> None:
    vc = ctx.voice_client
    if not vc:
        await ctx.reply("I am not connected to a voice channel.")
        return

    if isinstance(vc, voice_recv.VoiceRecvClient) and vc.is_listening():
        vc.stop_listening()

    await vc.disconnect(force=False)
    await ctx.reply("Stopped transcription and left the voice channel.")


@bot.event
async def on_command_error(ctx: commands.Context, error: commands.CommandError) -> None:
    if isinstance(error, commands.CommandNotFound):
        return
    log.exception("Command failed", exc_info=error)
    await ctx.reply(f"Command failed: `{error}`")


async def main() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy_logger in ("filelock", "httpcore", "httpx", "huggingface_hub"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    if not settings.verbose_voice_recv_logs:
        logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.WARNING)
        logging.getLogger("discord.ext.voice_recv.reader").setLevel(logging.WARNING)
        logging.getLogger("discord.ext.voice_recv.router").setLevel(logging.WARNING)

    if not settings.token:
        raise RuntimeError("DISCORD_TOKEN is missing from .env or the environment")

    async with bot:
        await bot.start(settings.token)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        transcriber.shutdown()
