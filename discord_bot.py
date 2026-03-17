"""
discord_bot.py  ─  SynchronVoice Employee Transcription Bot
============================================================
• Auto-joins a voice channel the moment the first employee arrives.
• Records each speaker's audio separately (per Discord user ID).
• Every FLUSH_INTERVAL_SEC seconds it transcribes each buffer with
  Groq Whisper Large v3 Turbo and posts an embed to #transcriptions.
• Handles 8-hour sessions via chunked processing (never loads a full
  session into memory).
• When the last employee leaves, the session is finalised and the
  transcript .txt is uploaded to #transcriptions.

Requirements:  pip install -r requirements.txt
Environment variables (put in .env):
    DISCORD_BOT_TOKEN  – your bot token
    GROQ_API_KEY       – your Groq API key

NOTE: Uses discord-ext-voice-recv (already installed) for per-user audio.
      No FFmpeg required for receiving audio.
"""
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import threading
import wave
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import discord
from discord.ext import commands, voice_recv
from discord.opus import Decoder as OpusDecoder, OpusError
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("synchronvoice")

# Silence the chatty "WS payload has extra keys" INFO messages from voice_recv
logging.getLogger("discord.ext.voice_recv.gateway").setLevel(logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Config / constants
# ─────────────────────────────────────────────────────────────────────────────
DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GROQ_API_KEY  = os.getenv("GROQ_API_KEY")

TRANSCRIPTIONS_DIR = Path("transcriptions")
TRANSCRIPTIONS_DIR.mkdir(exist_ok=True)

# Discord voice delivers raw PCM: 48 kHz, stereo, 16-bit signed little-endian
SAMPLE_RATE    = 48_000
CHANNELS       = 2
SAMPLE_WIDTH   = 2                                          # bytes per sample
BYTES_PER_SEC  = SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH     # 192 000 B/s

FLUSH_INTERVAL_SEC = 10        # drain buffers this often
MIN_AUDIO_BYTES    = BYTES_PER_SEC * 1       # skip < 1 s (silence / noise)
MAX_CHUNK_BYTES    = BYTES_PER_SEC * 55      # ≤ 55 s keeps Groq under 25 MB

groq_client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers  (no Discord dependency – easily unit-testable)
# ─────────────────────────────────────────────────────────────────────────────

def pcm_to_wav(pcm: bytes) -> bytes:
    """Wrap raw Discord PCM bytes in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


def call_groq(wav_bytes: bytes, user_id: int) -> str:
    """Send a WAV buffer to Groq Whisper and return the transcript text."""
    audio_file = io.BytesIO(wav_bytes)
    audio_file.name = f"audio_{user_id}.wav"
    resp = groq_client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-large-v3-turbo",
        response_format="text",
        temperature=0.0,
    )
    # Groq SDK may return a plain str or an object with .text
    return resp if isinstance(resp, str) else resp.text


def resolve_member(bot: commands.Bot, user_id: int) -> Optional[discord.Member]:
    """Find a guild member by user ID across all guilds the bot is in."""
    for guild in bot.guilds:
        m = guild.get_member(user_id)
        if m:
            return m
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TranscriptionSession  ─  manages on-disk logs for one recording session
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptionSession:
    """
    Creates two files at session start:
        transcriptions/<YYYY-MM-DD_HH-MM-SS>_<channel>.txt   ← human-readable
        transcriptions/<YYYY-MM-DD_HH-MM-SS>_<channel>.json  ← structured data
    Both are written incrementally (append on every segment), so a crash
    never loses already-captured data.
    """

    def __init__(self, guild_id: int, channel_name: str) -> None:
        self.guild_id     = guild_id
        self.channel_name = channel_name
        self.started_at   = datetime.now()
        self.session_id   = self.started_at.strftime("%Y-%m-%d_%H-%M-%S")
        self.entries: list[dict] = []

        # Sanitise channel name for use in a filename
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in channel_name)
        self.txt_path  = TRANSCRIPTIONS_DIR / f"{self.session_id}_{safe}.txt"
        self.json_path = TRANSCRIPTIONS_DIR / f"{self.session_id}_{safe}.json"

        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write("=== SynchronVoice Transcription Session ===\n")
            f.write(f"Channel : {channel_name}\n")
            f.write(f"Started : {self.started_at:%Y-%m-%d %H:%M:%S}\n")
            f.write("=" * 44 + "\n\n")

        logger.info("Session started → %s", self.txt_path)

    def append(self, username: str, text: str) -> None:
        """Append one transcription segment (thread-safe: only file I/O)."""
        ts = datetime.now()
        self.entries.append({"timestamp": ts.isoformat(), "username": username, "text": text.strip()})
        line = f"[{ts:%H:%M:%S}] {username}: {text.strip()}\n"
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(line)

    def finalize(self) -> Path:
        """Write session summary footer + JSON dump.  Returns the .txt path."""
        ended_at = datetime.now()
        duration = ended_at - self.started_at
        h, rem   = divmod(int(duration.total_seconds()), 3600)
        m, s     = divmod(rem, 60)

        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'=' * 44}\n")
            f.write(f"Ended    : {ended_at:%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Duration : {h:02d}h {m:02d}m {s:02d}s\n")
            f.write(f"Speakers : {len({e['username'] for e in self.entries})}\n")
            f.write(f"Segments : {len(self.entries)}\n")

        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": self.session_id,
                    "channel": self.channel_name,
                    "started_at": self.started_at.isoformat(),
                    "ended_at": ended_at.isoformat(),
                    "duration_seconds": int(duration.total_seconds()),
                    "entries": self.entries,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        logger.info("Session finalised → %s", self.txt_path)
        return self.txt_path


# ─────────────────────────────────────────────────────────────────────────────
# EmployeeSink  ─  per-user audio buffer + periodic transcription
# ─────────────────────────────────────────────────────────────────────────────

class EmployeeSink(voice_recv.AudioSink):
    """
    voice_recv calls write(user, data) on a background thread for every
    received audio packet.  We buffer raw PCM per user (thread-safe via a
    threading.Lock), then every FLUSH_INTERVAL_SEC seconds an asyncio task:
      1. Slices up to MAX_CHUNK_BYTES from each user's buffer.
      2. Wraps it in a WAV container.
      3. Sends it to Groq Whisper (in a thread-pool executor).
      4. Appends the transcript to the session log and posts to Discord.
    """

    def __init__(
        self,
        bot: commands.Bot,
        session: TranscriptionSession,
        text_channel: Optional[discord.TextChannel],
    ) -> None:
        super().__init__()
        self.bot          = bot
        self.session      = session
        self.text_channel = text_channel
        # Keyed by Discord user ID → accumulated raw PCM bytes
        self.buffers: dict[int, bytearray] = defaultdict(bytearray)
        # Cache user display names captured in the write() thread
        self._user_names: dict[int, str] = {}
        # Per-user Opus decoders (one stateful decoder per speaker)
        self._decoders: dict[int, OpusDecoder] = {}
        # threading.Lock because write() is called from voice_recv's audio thread
        self._buf_lock    = threading.Lock()
        self._task: Optional[asyncio.Task] = None

    # ── voice_recv.AudioSink hooks ───────────────────────────────────────────

    def wants_opus(self) -> bool:
        """Return True → voice_recv skips its internal Opus decode step.
        We decode manually so we can catch OpusError on corrupted first packets
        instead of crashing the PacketRouter thread.
        """
        return True

    def _get_decoder(self, user_id: int) -> OpusDecoder:
        if user_id not in self._decoders:
            self._decoders[user_id] = OpusDecoder()
        return self._decoders[user_id]

    def write(self, user: Optional[discord.User], data: voice_recv.VoiceData) -> None:
        """Called from the voice_recv audio thread on every ~20 ms Opus packet."""
        if user is None:
            return
        opus_bytes = data.opus   # raw decrypted Opus (populated when wants_opus=True)
        if not opus_bytes:
            return
        try:
            pcm = self._get_decoder(user.id).decode(opus_bytes, fec=False)
        except OpusError:
            # Skip corrupted packets (common right after initial connection)
            return
        if not pcm:
            return
        with self._buf_lock:
            self.buffers[user.id] += pcm
            # Cache name while we have the User object
            if user.id not in self._user_names:
                self._user_names[user.id] = user.display_name

    def cleanup(self) -> None:
        """Cancel our flush task when voice_recv stops."""
        if self._task and not self._task.done():
            self._task.cancel()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start_processing(self) -> None:
        """Start the periodic flush-and-transcribe loop on the event loop."""
        self._task = asyncio.create_task(self._flush_loop())

    # ── internal processing ──────────────────────────────────────────────────

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(FLUSH_INTERVAL_SEC)
            await self._drain_buffers()

    async def _drain_buffers(self) -> None:
        """Snapshot buffers (under lock) then transcribe each chunk in parallel."""
        to_process: dict[int, bytes] = {}
        with self._buf_lock:
            for user_id in list(self.buffers):
                buf = self.buffers[user_id]
                if len(buf) < MIN_AUDIO_BYTES:
                    continue
                chunk = bytes(buf[:MAX_CHUNK_BYTES])
                self.buffers[user_id] = bytearray(buf[MAX_CHUNK_BYTES:])
                to_process[user_id] = chunk

        if to_process:
            await asyncio.gather(
                *(self._transcribe(uid, chunk) for uid, chunk in to_process.items()),
                return_exceptions=True,
            )

    async def drain_final(self) -> None:
        """Drain everything remaining in buffers at session end."""
        to_process: dict[int, bytes] = {}
        with self._buf_lock:
            for user_id, buf in list(self.buffers.items()):
                if len(buf) >= MIN_AUDIO_BYTES:
                    to_process[user_id] = bytes(buf)
                self.buffers[user_id] = bytearray()

        if to_process:
            await asyncio.gather(
                *(self._transcribe(uid, chunk) for uid, chunk in to_process.items()),
                return_exceptions=True,
            )

    async def _transcribe(self, user_id: int, pcm: bytes) -> None:
        try:
            wav_bytes = pcm_to_wav(pcm)
            loop = asyncio.get_event_loop()
            # Groq SDK is synchronous – run it in a thread so we don't block
            text = await loop.run_in_executor(None, call_groq, wav_bytes, user_id)
            if not text or not text.strip():
                return

            # Prefer guild Member (has server nickname) over plain User
            member   = resolve_member(self.bot, user_id)
            username = (
                member.display_name
                if member
                else self._user_names.get(user_id, f"User_{user_id}")
            )

            self.session.append(username, text)
            logger.info("[%s] %s", username, text[:120])

            if self.text_channel:
                embed = discord.Embed(description=text.strip(), color=0x57F287)
                embed.set_author(
                    name=username,
                    icon_url=member.display_avatar.url if member else None,
                )
                embed.timestamp = discord.utils.utcnow()
                await self.text_channel.send(embed=embed)

        except Exception:
            logger.exception("Transcription failed for user %d", user_id)


# ─────────────────────────────────────────────────────────────────────────────
# Bot setup
# ─────────────────────────────────────────────────────────────────────────────

intents = discord.Intents.default()
intents.voice_states    = True
intents.message_content = True
intents.members         = True   # required: enable "Server Members Intent" in dev portal

bot = commands.Bot(command_prefix="!", intents=intents)

# guild_id → { vc, sink, session, text_channel }
active_sessions: dict[int, dict] = {}
_joining_guilds: set[int] = set()   # guard against duplicate join races


# ─────────────────────────────────────────────────────────────────────────────
# Events
# ─────────────────────────────────────────────────────────────────────────────

@bot.event
async def on_ready() -> None:
    logger.info("Logged in as %s (ID: %d)", bot.user, bot.user.id)
    for g in bot.guilds:
        logger.info("  Guild: %s", g.name)
    logger.info("Listening for employees to join voice channels…")


@bot.event
async def on_voice_state_update(
    member: discord.Member,
    before: discord.VoiceState,
    after: discord.VoiceState,
) -> None:
    # Ignore the bot's own state changes
    if member.bot:
        return

    guild = member.guild

    # ── Employee joined a voice channel ──────────────────────────────────────
    if before.channel is None and after.channel is not None:
        if guild.id not in active_sessions and guild.id not in _joining_guilds:
            await _start_session(after.channel)

    # ── Employee left or moved away from the channel the bot is in ───────────
    elif before.channel is not None and after.channel != before.channel:
        if guild.id in active_sessions:
            bot_channel = active_sessions[guild.id]["vc"].channel
            if before.channel == bot_channel:
                remaining = [m for m in bot_channel.members if not m.bot]
                if not remaining:
                    await _end_session(guild)


# ─────────────────────────────────────────────────────────────────────────────
# Session helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _start_session(channel: discord.VoiceChannel) -> None:
    guild = channel.guild
    _joining_guilds.add(guild.id)
    try:
        text_channel = discord.utils.get(guild.text_channels, name="transcriptions")
        if text_channel is None:
            logger.warning(
                "No #transcriptions channel found in %s — transcripts will only be saved to disk.",
                guild.name,
            )

        try:
            vc = await channel.connect(cls=voice_recv.VoiceRecvClient)
        except discord.ClientException:
            logger.warning("Already connected in %s", guild.name)
            return
        except Exception:
            logger.exception("Failed to join %s in %s", channel.name, guild.name)
            return

        session = TranscriptionSession(guild.id, channel.name)
        sink    = EmployeeSink(bot, session, text_channel)

        # voice_recv: listen() starts the receive loop; no after-callback needed
        vc.listen(sink)
        sink.start_processing()

        active_sessions[guild.id] = {
            "vc": vc,
            "sink": sink,
            "session": session,
            "text_channel": text_channel,
        }
        logger.info("▶ Session started – #%s (%s)", channel.name, guild.name)

        if text_channel:
            await text_channel.send(
                f"🔴 **Recording started** in **#{channel.name}**\n"
                f"Session `{session.session_id}` — speaker transcriptions appear below in real-time."
            )
    finally:
        _joining_guilds.discard(guild.id)


async def _end_session(guild: discord.Guild) -> None:
    info = active_sessions.pop(guild.id, None)
    if not info:
        return

    vc:      discord.VoiceClient  = info["vc"]
    sink:    EmployeeSink          = info["sink"]
    session: TranscriptionSession  = info["session"]
    text_channel                   = info["text_channel"]

    # 1. Cancel our flush loop
    sink.cleanup()

    # 2. Tell voice_recv to stop the receive pipeline
    try:
        if vc.is_listening():
            vc.stop_listening()
    except Exception:
        pass

    # 3. Transcribe whatever audio is still in the buffers
    await sink.drain_final()

    # 4. Finalise files
    log_path = session.finalize()

    # 5. Disconnect
    try:
        await vc.disconnect()
    except Exception:
        pass

    logger.info("⏹ Session ended for %s → %s", guild.name, log_path)

    if not text_channel:
        return

    dur    = datetime.now() - session.started_at
    h, rem = divmod(int(dur.total_seconds()), 3600)
    m, s   = divmod(rem, 60)
    speakers = sorted({e["username"] for e in session.entries})

    embed = discord.Embed(title="✅ Session Complete", color=0x3498DB)
    embed.add_field(name="Channel",  value=session.channel_name,            inline=True)
    embed.add_field(name="Duration", value=f"{h:02d}h {m:02d}m {s:02d}s",  inline=True)
    embed.add_field(name="Speakers", value=", ".join(speakers) or "–",      inline=False)
    embed.add_field(name="Segments", value=str(len(session.entries)),        inline=True)
    embed.add_field(name="Saved to", value=str(log_path),                   inline=False)
    embed.timestamp = discord.utils.utcnow()
    await text_channel.send(embed=embed)

    # Upload the transcript if it is within Discord's file size limit
    if log_path.exists() and log_path.stat().st_size < 8_000_000:
        await text_channel.send(
            "📄 Full transcript:",
            file=discord.File(str(log_path)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Commands  (all require Manage Channels except !status / !transcript)
# ─────────────────────────────────────────────────────────────────────────────

@bot.command(name="join")
@commands.has_permissions(manage_channels=True)
async def cmd_join(ctx: commands.Context) -> None:
    """Manually start recording in your current voice channel."""
    if not ctx.author.voice:
        await ctx.send("❌ You must be in a voice channel first.")
        return
    if ctx.guild.id in active_sessions:
        await ctx.send("⚠️ Already recording in this server.")
        return
    await _start_session(ctx.author.voice.channel)
    await ctx.send(f"✅ Joined **{ctx.author.voice.channel.name}** – recording started.")


@bot.command(name="leave")
@commands.has_permissions(manage_channels=True)
async def cmd_leave(ctx: commands.Context) -> None:
    """Stop recording and leave the voice channel."""
    if ctx.guild.id not in active_sessions:
        await ctx.send("❌ Bot is not recording in this server.")
        return
    await ctx.send("⏹ Ending session…")
    await _end_session(ctx.guild)


@bot.command(name="status")
async def cmd_status(ctx: commands.Context) -> None:
    """Show the current session status and who is present."""
    if ctx.guild.id not in active_sessions:
        await ctx.send("😴 No active recording session.")
        return

    info    = active_sessions[ctx.guild.id]
    session = info["session"]
    vc      = info["vc"]
    dur     = datetime.now() - session.started_at
    h, rem  = divmod(int(dur.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    present = [mem.display_name for mem in vc.channel.members if not mem.bot]

    embed = discord.Embed(title="📊 Session Status", color=0xF39C12)
    embed.add_field(name="Channel",      value=vc.channel.name,                        inline=True)
    embed.add_field(name="Duration",     value=f"{h:02d}h {m:02d}m {s:02d}s",         inline=True)
    embed.add_field(name="Present now",  value=", ".join(present) or "–",               inline=False)
    embed.add_field(name="Segments",     value=str(len(session.entries)),                inline=True)
    embed.add_field(name="Session file", value=str(session.txt_path),                   inline=False)
    await ctx.send(embed=embed)


@bot.command(name="transcript")
async def cmd_transcript(ctx: commands.Context) -> None:
    """Upload the current (in-progress) transcript file."""
    if ctx.guild.id not in active_sessions:
        await ctx.send("❌ No active session.")
        return
    path = active_sessions[ctx.guild.id]["session"].txt_path
    if path.stat().st_size < 8_000_000:
        await ctx.send("📄 Current transcript:", file=discord.File(str(path)))
    else:
        await ctx.send(f"📄 Transcript is large — find it at `{path}`.")


@bot.command(name="help_sv")
async def cmd_help(ctx: commands.Context) -> None:
    """Show all SynchronVoice commands."""
    embed = discord.Embed(
        title="🎙 SynchronVoice – Command Reference",
        color=0x5865F2,
    )
    embed.add_field(
        name="Auto behaviour",
        value="Bot auto-joins when the first employee enters a voice channel and leaves when the last one does.",
        inline=False,
    )
    embed.add_field(name="!join",       value="Manually start recording (needs Manage Channels)", inline=False)
    embed.add_field(name="!leave",      value="Manually stop recording (needs Manage Channels)",  inline=False)
    embed.add_field(name="!status",     value="Show current session info",                         inline=False)
    embed.add_field(name="!transcript", value="Download the in-progress transcript right now",     inline=False)
    embed.add_field(name="!help_sv",    value="Show this message",                                 inline=False)
    await ctx.send(embed=embed)


@bot.event
async def on_command_error(ctx: commands.Context, error: Exception) -> None:
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("❌ You need **Manage Channels** permission for that command.")
    elif isinstance(error, commands.CommandNotFound):
        pass  # silently ignore unknown commands
    else:
        logger.error("Command error in %s: %s", ctx.command, error)
        await ctx.send(f"❌ {error}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise SystemExit("DISCORD_BOT_TOKEN is not set in .env")
    if not GROQ_API_KEY:
        raise SystemExit("GROQ_API_KEY is not set in .env")
    bot.run(DISCORD_TOKEN)
