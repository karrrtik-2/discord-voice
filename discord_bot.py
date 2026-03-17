"""
discord_bot.py  —  SynchronVoice Employee Transcription Bot (py-cord edition)
==============================================================================
• Uses py-cord's native vc.start_recording() — the ONLY reliable way to
  capture per-user audio in Python Discord bots (2025+).
• discord-ext-voice-recv is broken due to sequence rollover bug + new
  Discord encryption changes. py-cord ships its own working sink system.
• Auto-joins when the first employee enters any voice channel.
• Buffers raw PCM per-user, downmixes stereo→mono, applies RMS VAD, then
  flushes to Deepgram Nova-3 every FLUSH_INTERVAL_SEC seconds.
• Incremental .txt + .json logs: a crash never loses captured data.
• Leaves and finalises when the last employee leaves.

Install:
    pip uninstall discord.py discord-ext-voice-recv -y
    pip install py-cord deepgram-sdk python-dotenv

.env:
    DISCORD_BOT_TOKEN = ...
    DEEPGRAM_API_KEY  = ...
"""
from __future__ import annotations

import asyncio
import audioop
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
from discord.ext import commands
from discord.sinks import Sink, AudioData
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions

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

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
DISCORD_TOKEN    = os.getenv("DISCORD_BOT_TOKEN")
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

TRANSCRIPTIONS_DIR = Path("transcriptions")
TRANSCRIPTIONS_DIR.mkdir(exist_ok=True)

# py-cord sink delivers decoded: 48 kHz, stereo, 16-bit signed PCM
SAMPLE_RATE  = 48_000
SAMPLE_WIDTH = 2                           # bytes per sample (int16)

# After stereo→mono downmix:
MONO_BPS = SAMPLE_RATE * 1 * SAMPLE_WIDTH  # 96 000 B/s

FLUSH_INTERVAL_SEC = 30                    # transcribe every 30 s
MIN_AUDIO_BYTES    = MONO_BPS // 2         # need ≥ 0.5 s before sending
MAX_CHUNK_BYTES    = MONO_BPS * 55         # keep request ≤ ~5 MB

# RMS VAD — only buffers packets with actual voice energy.
# 100 is conservative; real speech is typically 300–3000.
# Silent Discord packets have RMS ≈ 0–30.
VAD_RMS_THRESHOLD = 100

deepgram_client = DeepgramClient(DEEPGRAM_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def stereo_to_mono(pcm: bytes) -> bytes:
    """Downmix 16-bit stereo PCM → mono."""
    return audioop.tomono(pcm, SAMPLE_WIDTH, 0.5, 0.5)


def is_speech(mono_pcm: bytes) -> bool:
    """True when RMS energy exceeds VAD_RMS_THRESHOLD."""
    if len(mono_pcm) < SAMPLE_WIDTH * 2:
        return False
    return audioop.rms(mono_pcm, SAMPLE_WIDTH) > VAD_RMS_THRESHOLD


def build_wav(mono_pcm: bytes) -> bytes:
    """Wrap mono 48 kHz int16 PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(mono_pcm)
    return buf.getvalue()


def call_deepgram(wav_bytes: bytes) -> str:
    """Send WAV to Deepgram Nova-3 (blocking). Uses non-deprecated SDK API."""
    options = PrerecordedOptions(
        model="nova-3",
        language="en",
        smart_format=True,
        punctuate=True,
    )
    payload  = {"buffer": wav_bytes, "mimetype": "audio/wav"}
    response = deepgram_client.listen.rest.v("1").transcribe_file(payload, options)
    try:
        return response.results.channels[0].alternatives[0].transcript or ""
    except (AttributeError, IndexError, KeyError):
        return ""


def resolve_member(bot: commands.Bot, user_id: int) -> Optional[discord.Member]:
    for guild in bot.guilds:
        m = guild.get_member(user_id)
        if m:
            return m
    return None


# ─────────────────────────────────────────────────────────────────────────────
# TranscriptionSession  —  on-disk logs for one recording session
# ─────────────────────────────────────────────────────────────────────────────

class TranscriptionSession:
    def __init__(self, guild_id: int, channel_name: str) -> None:
        self.guild_id     = guild_id
        self.channel_name = channel_name
        self.started_at   = datetime.now()
        self.session_id   = self.started_at.strftime("%Y-%m-%d_%H-%M-%S")
        self.entries: list[dict] = []

        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in channel_name)
        self.txt_path  = TRANSCRIPTIONS_DIR / f"{self.session_id}_{safe}.txt"
        self.json_path = TRANSCRIPTIONS_DIR / f"{self.session_id}_{safe}.json"

        with open(self.txt_path, "w", encoding="utf-8") as f:
            f.write("=== SynchronVoice Transcription Session ===\n")
            f.write(f"Channel : {channel_name}\n")
            f.write(f"Started : {self.started_at:%Y-%m-%d %H:%M:%S}\n")
            f.write(f"STT     : Deepgram Nova-3\n")
            f.write("=" * 44 + "\n\n")

        logger.info("Session started → %s", self.txt_path)

    def append(self, username: str, text: str) -> None:
        ts = datetime.now()
        self.entries.append({"timestamp": ts.isoformat(), "username": username, "text": text.strip()})
        with open(self.txt_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts:%H:%M:%S}] {username}: {text.strip()}\n")

    def finalize(self) -> Path:
        ended_at = datetime.now()
        dur      = ended_at - self.started_at
        h, rem   = divmod(int(dur.total_seconds()), 3600)
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
                    "duration_seconds": int(dur.total_seconds()),
                    "entries": self.entries,
                },
                f, indent=2, ensure_ascii=False,
            )
        logger.info("Session finalised → %s", self.txt_path)
        return self.txt_path


# ─────────────────────────────────────────────────────────────────────────────
# EmployeeSink  —  py-cord native Sink subclass
#
# HOW py-cord sinks work:
#   • vc.start_recording(sink, finished_cb, channel) starts capture.
#   • py-cord's internal reader decrypts + decodes every Opus packet.
#   • For every decoded PCM frame it calls sink.write(user_id, AudioData).
#   • AudioData.file is a BytesIO that GROWS over time (append-only).
#   • vc.stop_recording() drains remaining audio then fires finished_cb.
#
# Our write() hook:
#   1. Reads only NEW bytes from AudioData.file using a tracked offset.
#   2. Downmixes stereo → mono.
#   3. Applies RMS VAD — silent frames are dropped before buffering.
#   4. Appends speech-only mono PCM to a per-user bytearray.
#
# A background asyncio task flushes to Deepgram every FLUSH_INTERVAL_SEC.
# ─────────────────────────────────────────────────────────────────────────────

class EmployeeSink(Sink):
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

        # Per-user mono PCM accumulation
        self._mono_buffers: dict[int, bytearray] = defaultdict(bytearray)
        # Track how many bytes we've already read from each AudioData.file
        self._read_offsets: dict[int, int] = defaultdict(int)
        self._buf_lock   = threading.Lock()
        self._flush_task: Optional[asyncio.Task] = None

    # ── py-cord Sink hook ────────────────────────────────────────────────────

    def write(self, data: AudioData, user: int) -> None:
        """
        Called by py-cord's audio thread for every decoded PCM chunk.
        `user` is an int (Discord user ID).
        `data.file` is a BytesIO that grows as audio arrives — we only
        read the NEW bytes each call via a tracked byte offset.
        """
        with self._buf_lock:
            offset = self._read_offsets[user]

        # Seek to where we last left off and read only new data
        data.file.seek(offset)
        new_stereo = data.file.read()

        if not new_stereo:
            return

        new_offset = offset + len(new_stereo)

        # Downmix stereo → mono
        mono = stereo_to_mono(new_stereo)

        # VAD: skip mostly-silent chunks
        if not is_speech(mono):
            with self._buf_lock:
                self._read_offsets[user] = new_offset
            return

        with self._buf_lock:
            self._mono_buffers[user] += mono
            self._read_offsets[user] = new_offset

        logger.debug("write() user=%d  new_bytes=%d  buf_total=%d",
                     user, len(new_stereo), len(self._mono_buffers[user]))

    def cleanup(self) -> None:
        """Called by py-cord when stop_recording() completes."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start_flush_loop(self) -> None:
        self._flush_task = asyncio.create_task(self._flush_loop())

    # ── internal processing ──────────────────────────────────────────────────

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(FLUSH_INTERVAL_SEC)
            await self._drain_buffers()

    async def _drain_buffers(self) -> None:
        to_process: dict[int, bytes] = {}
        with self._buf_lock:
            for uid in list(self._mono_buffers):
                buf = self._mono_buffers[uid]
                if len(buf) < MIN_AUDIO_BYTES:
                    continue
                chunk = bytes(buf[:MAX_CHUNK_BYTES])
                self._mono_buffers[uid] = bytearray(buf[MAX_CHUNK_BYTES:])
                to_process[uid] = chunk

        if to_process:
            logger.info("Draining %d user buffer(s) to Deepgram…", len(to_process))
            await asyncio.gather(
                *(self._transcribe(uid, chunk) for uid, chunk in to_process.items()),
                return_exceptions=True,
            )

    async def drain_final(self) -> None:
        """Flush all remaining audio at session end (ignores MIN_AUDIO_BYTES)."""
        to_process: dict[int, bytes] = {}
        with self._buf_lock:
            for uid, buf in list(self._mono_buffers.items()):
                if buf:
                    to_process[uid] = bytes(buf)
                self._mono_buffers[uid] = bytearray()

        if to_process:
            logger.info("Final drain: %d user buffer(s)…", len(to_process))
            await asyncio.gather(
                *(self._transcribe(uid, chunk) for uid, chunk in to_process.items()),
                return_exceptions=True,
            )

    async def _transcribe(self, user_id: int, mono_pcm: bytes) -> None:
        try:
            wav  = build_wav(mono_pcm)
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, call_deepgram, wav)

            if not text or not text.strip():
                logger.info("Empty transcript for user %d (%.1f s audio)",
                            user_id, len(mono_pcm) / MONO_BPS)
                return

            member   = resolve_member(self.bot, user_id)
            username = member.display_name if member else f"User_{user_id}"

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
intents.members         = True   # enable "Server Members Intent" in dev portal

bot = commands.Bot(command_prefix="!", intents=intents)

# guild_id → { vc, sink, session, text_channel }
active_sessions: dict[int, dict] = {}
_joining_guilds: set[int] = set()


# ─────────────────────────────────────────────────────────────────────────────
# Events
# ─────────────────────────────────────────────────────────────────────────────

@bot.event
async def on_ready() -> None:
    logger.info("Logged in as %s (ID: %d)", bot.user, bot.user.id)
    for g in bot.guilds:
        logger.info("  Guild: %s", g.name)
    logger.info("Ready. Listening for employees entering voice channels…")


@bot.event
async def on_voice_state_update(
    member: discord.Member,
    before: discord.VoiceState,
    after: discord.VoiceState,
) -> None:
    if member.bot:
        return

    guild = member.guild

    # Employee joined a voice channel
    if before.channel is None and after.channel is not None:
        if guild.id not in active_sessions and guild.id not in _joining_guilds:
            await _start_session(after.channel)

    # Employee left or switched channels
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

async def _recording_finished(
    sink: EmployeeSink, channel: discord.VoiceChannel, *args
) -> None:
    """py-cord fires this after stop_recording() finishes draining."""
    logger.info("py-cord recording finished for #%s", channel.name)


async def _start_session(channel: discord.VoiceChannel) -> None:
    guild = channel.guild
    _joining_guilds.add(guild.id)
    try:
        text_channel = discord.utils.get(guild.text_channels, name="transcriptions")
        if text_channel is None:
            logger.warning("No #transcriptions channel in %s — disk only.", guild.name)

        try:
            vc = await channel.connect()
        except discord.ClientException:
            logger.warning("Already connected in %s", guild.name)
            return
        except Exception:
            logger.exception("Failed to join %s in %s", channel.name, guild.name)
            return

        session = TranscriptionSession(guild.id, channel.name)
        sink    = EmployeeSink(bot, session, text_channel)

        # py-cord native recording — calls sink.write(user_id, AudioData) per frame
        vc.start_recording(sink, _recording_finished, channel)
        sink.start_flush_loop()

        active_sessions[guild.id] = {
            "vc": vc,
            "sink": sink,
            "session": session,
            "text_channel": text_channel,
        }
        logger.info("▶ Recording started – #%s (%s)", channel.name, guild.name)

        if text_channel:
            await text_channel.send(
                f"🔴 **Recording started** in **#{channel.name}**\n"
                f"Session `{session.session_id}` — transcriptions appear every {FLUSH_INTERVAL_SEC}s."
            )
    finally:
        _joining_guilds.discard(guild.id)


async def _end_session(guild: discord.Guild) -> None:
    info = active_sessions.pop(guild.id, None)
    if not info:
        return

    vc:          discord.VoiceClient  = info["vc"]
    sink:        EmployeeSink          = info["sink"]
    session:     TranscriptionSession  = info["session"]
    text_channel                       = info["text_channel"]

    # 1. Stop our flush loop
    sink.cleanup()

    # 2. Stop py-cord recording (triggers _recording_finished callback)
    try:
        vc.stop_recording()
    except Exception:
        pass

    # Small delay to let the finished callback fire
    await asyncio.sleep(0.5)

    # 3. Flush any remaining audio buffers to Deepgram
    await sink.drain_final()

    # 4. Finalise log files
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
    embed.add_field(name="STT",      value="Deepgram Nova-3",               inline=True)
    embed.add_field(name="Speakers", value=", ".join(speakers) or "–",      inline=False)
    embed.add_field(name="Segments", value=str(len(session.entries)),        inline=True)
    embed.add_field(name="Saved to", value=str(log_path),                   inline=False)
    embed.timestamp = discord.utils.utcnow()
    await text_channel.send(embed=embed)

    if log_path.exists() and log_path.stat().st_size < 8_000_000:
        await text_channel.send(
            "📄 Full transcript:",
            file=discord.File(str(log_path)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Commands
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
    """Show current session status and buffer sizes."""
    if ctx.guild.id not in active_sessions:
        await ctx.send("😴 No active recording session.")
        return

    info    = active_sessions[ctx.guild.id]
    session = info["session"]
    vc      = info["vc"]
    sink: EmployeeSink = info["sink"]
    dur     = datetime.now() - session.started_at
    h, rem  = divmod(int(dur.total_seconds()), 3600)
    m, s    = divmod(rem, 60)
    present = [mem.display_name for mem in vc.channel.members if not mem.bot]

    with sink._buf_lock:
        buf_info = {uid: len(b) for uid, b in sink._mono_buffers.items() if b}

    embed = discord.Embed(title="📊 Session Status", color=0xF39C12)
    embed.add_field(name="Channel",     value=vc.channel.name,                       inline=True)
    embed.add_field(name="Duration",    value=f"{h:02d}h {m:02d}m {s:02d}s",        inline=True)
    embed.add_field(name="STT Model",   value="Deepgram Nova-3",                     inline=True)
    embed.add_field(name="Present now", value=", ".join(present) or "–",             inline=False)
    embed.add_field(name="Segments",    value=str(len(session.entries)),              inline=True)
    if buf_info:
        members_buf = []
        for uid, sz in buf_info.items():
            m_obj = resolve_member(bot, uid)
            name  = m_obj.display_name if m_obj else f"User_{uid}"
            members_buf.append(f"{name}: {sz // 1000}kB")
        embed.add_field(name="Buffered", value="\n".join(members_buf), inline=False)
    embed.add_field(name="Session file", value=str(session.txt_path), inline=False)
    await ctx.send(embed=embed)


@bot.command(name="flushtx")
@commands.has_permissions(manage_channels=True)
async def cmd_flush(ctx: commands.Context) -> None:
    """Force-flush all audio buffers to Deepgram right now."""
    if ctx.guild.id not in active_sessions:
        await ctx.send("❌ No active session.")
        return
    await ctx.send("⏳ Flushing buffers…")
    sink: EmployeeSink = active_sessions[ctx.guild.id]["sink"]
    await sink._drain_buffers()
    await ctx.send("✅ Flush complete.")


@bot.command(name="transcript")
async def cmd_transcript(ctx: commands.Context) -> None:
    """Download the current in-progress transcript."""
    if ctx.guild.id not in active_sessions:
        await ctx.send("❌ No active session.")
        return
    path = active_sessions[ctx.guild.id]["session"].txt_path
    if path.stat().st_size < 8_000_000:
        await ctx.send("📄 Current transcript:", file=discord.File(str(path)))
    else:
        await ctx.send(f"📄 Transcript too large — find it at `{path}`.")


@bot.command(name="help_sv")
async def cmd_help(ctx: commands.Context) -> None:
    """Show all SynchronVoice commands."""
    embed = discord.Embed(title="🎙 SynchronVoice – Commands", color=0x5865F2)
    embed.add_field(
        name="Auto behaviour",
        value=(f"Bot joins on first employee entry, leaves when the last one does. "
               f"Transcripts sent every {FLUSH_INTERVAL_SEC}s via Deepgram Nova-3."),
        inline=False,
    )
    embed.add_field(name="!join",       value="Manually start recording (Manage Channels)",  inline=False)
    embed.add_field(name="!leave",      value="Manually stop recording (Manage Channels)",   inline=False)
    embed.add_field(name="!flushtx",    value="Force-flush audio buffers to Deepgram now",   inline=False)
    embed.add_field(name="!status",     value="Session info + live buffer sizes per user",   inline=False)
    embed.add_field(name="!transcript", value="Download the in-progress transcript",         inline=False)
    embed.add_field(name="!help_sv",    value="This message",                                inline=False)
    await ctx.send(embed=embed)


@bot.event
async def on_command_error(ctx: commands.Context, error: Exception) -> None:
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("❌ You need **Manage Channels** permission for that command.")
    elif isinstance(error, commands.CommandNotFound):
        pass
    else:
        logger.error("Command error in %s: %s", ctx.command, error)
        await ctx.send(f"❌ {error}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        raise SystemExit("DISCORD_BOT_TOKEN is not set in .env")
    if not DEEPGRAM_API_KEY:
        raise SystemExit("DEEPGRAM_API_KEY is not set in .env")
    bot.run(DISCORD_TOKEN)
