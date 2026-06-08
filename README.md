# discord-voice-transcribe

Python Discord bot that joins a voice channel, receives per-user audio, and transcribes each speaker with `faster-whisper`.

## Reality check

Discord voice receive is not an official high-level bot feature in `discord.py`. This project uses `discord-ext-voice-recv`, which is experimental. Discord voice calls now require DAVE/E2EE support for non-stage voice channels, so this project requires a recent `discord.py[voice]` plus `davey`.

Always tell users in the voice channel that transcription is active and get consent where required.

## Setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Create `.env`:

```env
DISCORD_TOKEN=your_bot_token_here
VOICE_CHANNEL_NAME=Meeting
WHISPER_MODEL=small
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
WHISPER_LANGUAGE=en
```

For NVIDIA GPU, try:

```env
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
```

Use `cuda` only if the CUDA 12 runtime libraries are installed and visible to Python. If you see `cublas64_12.dll is not found`, switch back to `WHISPER_DEVICE=cpu`.

Leave `WHISPER_LANGUAGE` empty for auto-detect.

## Discord configuration

In the Discord Developer Portal, enable:

- Server Members Intent
- Message Content Intent

Invite the bot with:

- View Channels
- Connect
- Send Messages
- Read Message History

The bot uses prefix commands, so run them from a server text channel.

## Run

```powershell
.\venv\Scripts\python.exe bot.py
```

Commands:

- `!join` joins your current voice channel.
- `!join_meeting` joins the channel named by `VOICE_CHANNEL_NAME`.
- `!meeting` is an alias for `!join_meeting`.
- `!status` shows connection/listening state.
- `!leave` stops transcription and disconnects.

Transcripts are printed to the terminal by default:

```text
Jimmy: Hello Amir
Amir: Hello Jimmy how are you
```

Set `POST_TRANSCRIPTS_TO_DISCORD=true` to also post transcript lines into the text channel where the join command was used.

The first transcription can be slower because `faster-whisper` loads the model and may download it if it is not already cached.

## Useful tuning

```env
MIN_CHUNK_SECONDS=0.8
MAX_CHUNK_SECONDS=12
SILENCE_FLUSH_SECONDS=0.8
TRANSCRIPTION_WORKERS=1
WHISPER_BEAM_SIZE=1
LOG_LEVEL=INFO
VERBOSE_VOICE_RECV_LOGS=false
```

Lower `MAX_CHUNK_SECONDS` reduces latency during long speech. Larger Whisper models improve accuracy but increase CPU/GPU load.

## Troubleshooting

If you see `discord.opus.OpusError: corrupted stream`, Discord is probably sending DAVE-encrypted Opus frames and the receive extension is trying to decode them before DAVE decryption. `bot.py` applies a small compatibility patch for this at startup. Some calls can also produce undecodable transition packets while a DAVE session exists; the patch drops those frames instead of stopping the receiver.

If transcription still stops immediately after joining, run with:

```env
LOG_LEVEL=DEBUG
```

Then check whether DAVE decryption is failing, the SSRC-to-user mapping is missing, or the voice session never becomes ready.

Set `VERBOSE_VOICE_RECV_LOGS=true` only when debugging the receive extension itself. It produces a lot of RTCP packet noise.
