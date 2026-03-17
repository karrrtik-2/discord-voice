"""
test_bot.py  ─  Unit tests for discord_bot.py
==============================================
Tests cover the pure (non-Discord) helpers and the TranscriptionSession
class.  No Discord connection is required.

Run:
    pytest test_bot.py -v
"""
from __future__ import annotations

import io
import json
import math
import struct
import wave
from pathlib import Path

import pytest

# ── import the pieces we want to test ────────────────────────────────────────
from discord_bot import (
    BYTES_PER_SEC,
    CHANNELS,
    FLUSH_INTERVAL_SEC,
    MAX_CHUNK_BYTES,
    MIN_AUDIO_BYTES,
    SAMPLE_RATE,
    SAMPLE_WIDTH,
    TranscriptionSession,
    pcm_to_wav,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers used in tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_pcm(duration_sec: float = 1.0, amplitude: int = 8000) -> bytes:
    """Generate a 440 Hz sine-wave PCM block (stereo, 16-bit, 48 kHz)."""
    n_samples = int(SAMPLE_RATE * duration_sec)
    frames = []
    for i in range(n_samples):
        sample = int(amplitude * math.sin(2 * math.pi * 440 * i / SAMPLE_RATE))
        frames.append(struct.pack("<hh", sample, sample))  # L + R
    return b"".join(frames)


def _wav_params(wav_bytes: bytes) -> tuple[int, int, int, int]:
    """Return (nchannels, sampwidth, framerate, nframes) from WAV bytes."""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        return wf.getnchannels(), wf.getsampwidth(), wf.getframerate(), wf.getnframes()


# ─────────────────────────────────────────────────────────────────────────────
# 1. Constants
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_bytes_per_sec_formula(self):
        assert BYTES_PER_SEC == SAMPLE_RATE * CHANNELS * SAMPLE_WIDTH

    def test_min_audio_is_at_least_one_second(self):
        assert MIN_AUDIO_BYTES >= BYTES_PER_SEC * 1

    def test_max_chunk_under_25_mb_groq_limit(self):
        # Groq Whisper enforces a 25 MB file limit; WAV header overhead is ~44 B
        assert MAX_CHUNK_BYTES < 25 * 1_000_000

    def test_flush_interval_is_positive(self):
        assert FLUSH_INTERVAL_SEC > 0

    def test_sample_rate_is_48khz(self):
        assert SAMPLE_RATE == 48_000

    def test_channels_is_stereo(self):
        assert CHANNELS == 2


# ─────────────────────────────────────────────────────────────────────────────
# 2. pcm_to_wav
# ─────────────────────────────────────────────────────────────────────────────

class TestPcmToWav:
    def test_produces_valid_wav_header(self):
        wav = pcm_to_wav(_make_pcm(0.5))
        assert wav[:4] == b"RIFF"
        assert wav[8:12] == b"WAVE"

    def test_channel_count_matches_discord(self):
        ch, *_ = _wav_params(pcm_to_wav(_make_pcm(1.0)))
        assert ch == CHANNELS

    def test_sample_width_matches_discord(self):
        _, sw, *_ = _wav_params(pcm_to_wav(_make_pcm(1.0)))
        assert sw == SAMPLE_WIDTH

    def test_frame_rate_matches_discord(self):
        _, _, fr, _ = _wav_params(pcm_to_wav(_make_pcm(1.0)))
        assert fr == SAMPLE_RATE

    def test_frame_count_matches_input(self):
        duration = 2.0
        _, _, _, frames = _wav_params(pcm_to_wav(_make_pcm(duration)))
        assert frames == int(SAMPLE_RATE * duration)

    def test_empty_pcm_gives_zero_frames(self):
        _, _, _, frames = _wav_params(pcm_to_wav(b""))
        assert frames == 0

    def test_output_is_larger_than_input_due_to_header(self):
        pcm = _make_pcm(1.0)
        wav = pcm_to_wav(pcm)
        assert len(wav) > len(pcm)

    def test_large_chunk_stays_within_groq_limit(self):
        # MAX_CHUNK_BYTES of PCM should produce a WAV still under 25 MB
        wav = pcm_to_wav(bytes(MAX_CHUNK_BYTES))
        assert len(wav) < 25 * 1_000_000


# ─────────────────────────────────────────────────────────────────────────────
# 3. TranscriptionSession
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TranscriptionSession:
    """Creates a TranscriptionSession whose files land in pytest's tmp_path."""
    import discord_bot
    monkeypatch.setattr(discord_bot, "TRANSCRIPTIONS_DIR", tmp_path)
    return TranscriptionSession(guild_id=111, channel_name="general")


class TestTranscriptionSessionCreation:
    def test_txt_file_is_created(self, session: TranscriptionSession):
        assert session.txt_path.exists()

    def test_json_file_does_not_exist_before_finalize(self, session: TranscriptionSession):
        assert not session.json_path.exists()

    def test_txt_header_contains_channel(self, session: TranscriptionSession):
        content = session.txt_path.read_text(encoding="utf-8")
        assert "general" in content

    def test_txt_header_contains_started(self, session: TranscriptionSession):
        content = session.txt_path.read_text(encoding="utf-8")
        assert "Started" in content

    def test_session_id_is_timestamp_format(self, session: TranscriptionSession):
        # format: YYYY-MM-DD_HH-MM-SS
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}", session.session_id)

    def test_special_chars_in_channel_name_are_sanitised(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        import discord_bot
        monkeypatch.setattr(discord_bot, "TRANSCRIPTIONS_DIR", tmp_path)
        s = TranscriptionSession(guild_id=999, channel_name="my channel #1!")
        assert s.txt_path.exists()
        assert " " not in s.txt_path.name
        assert "#" not in s.txt_path.name
        assert "!" not in s.txt_path.name


class TestTranscriptionSessionAppend:
    def test_append_writes_username_to_file(self, session: TranscriptionSession):
        session.append("Alice", "Hello everyone")
        assert "Alice" in session.txt_path.read_text(encoding="utf-8")

    def test_append_writes_text_to_file(self, session: TranscriptionSession):
        session.append("Alice", "Hello everyone")
        assert "Hello everyone" in session.txt_path.read_text(encoding="utf-8")

    def test_append_stores_entry_in_memory(self, session: TranscriptionSession):
        session.append("Bob", "Good morning")
        assert len(session.entries) == 1

    def test_append_entry_has_correct_fields(self, session: TranscriptionSession):
        session.append("Bob", "Good morning")
        entry = session.entries[0]
        assert entry["username"] == "Bob"
        assert entry["text"] == "Good morning"
        assert "timestamp" in entry

    def test_append_trims_whitespace(self, session: TranscriptionSession):
        session.append("Alice", "  hi there  ")
        assert session.entries[0]["text"] == "hi there"

    def test_multiple_speakers_stored(self, session: TranscriptionSession):
        session.append("Alice", "Hi")
        session.append("Bob", "Hey")
        session.append("Alice", "How are you?")
        assert len(session.entries) == 3
        speakers = {e["username"] for e in session.entries}
        assert speakers == {"Alice", "Bob"}

    def test_entries_preserve_order(self, session: TranscriptionSession):
        for i in range(5):
            session.append(f"User{i}", f"Message {i}")
        for i, entry in enumerate(session.entries):
            assert entry["text"] == f"Message {i}"


class TestTranscriptionSessionFinalize:
    def test_finalize_returns_txt_path(self, session: TranscriptionSession):
        path = session.finalize()
        assert path == session.txt_path
        assert path.suffix == ".txt"

    def test_finalize_creates_json(self, session: TranscriptionSession):
        session.append("Alice", "test")
        session.finalize()
        assert session.json_path.exists()

    def test_finalize_json_has_required_keys(self, session: TranscriptionSession):
        session.append("Alice", "Hello")
        session.finalize()
        data = json.loads(session.json_path.read_text(encoding="utf-8"))
        for key in ("session_id", "channel", "started_at", "ended_at", "duration_seconds", "entries"):
            assert key in data, f"Missing key: {key}"

    def test_finalize_json_entries_match_appended(self, session: TranscriptionSession):
        session.append("Alice", "Hello")
        session.append("Bob", "World")
        session.finalize()
        data = json.loads(session.json_path.read_text(encoding="utf-8"))
        assert len(data["entries"]) == 2

    def test_finalize_json_duration_non_negative(self, session: TranscriptionSession):
        session.finalize()
        data = json.loads(session.json_path.read_text(encoding="utf-8"))
        assert data["duration_seconds"] >= 0

    def test_finalize_json_channel_matches(self, session: TranscriptionSession):
        session.finalize()
        data = json.loads(session.json_path.read_text(encoding="utf-8"))
        assert data["channel"] == "general"

    def test_finalize_txt_footer_has_ended(self, session: TranscriptionSession):
        session.finalize()
        content = session.txt_path.read_text(encoding="utf-8")
        assert "Ended" in content

    def test_finalize_txt_footer_has_duration(self, session: TranscriptionSession):
        session.finalize()
        content = session.txt_path.read_text(encoding="utf-8")
        assert "Duration" in content

    def test_finalize_txt_footer_has_speakers_count(self, session: TranscriptionSession):
        session.append("Alice", "Hi")
        session.finalize()
        content = session.txt_path.read_text(encoding="utf-8")
        assert "Speakers" in content

    def test_finalize_txt_footer_has_segments_count(self, session: TranscriptionSession):
        session.append("Alice", "Hi")
        session.append("Bob", "Hey")
        session.finalize()
        content = session.txt_path.read_text(encoding="utf-8")
        assert "Segments" in content

    def test_finalize_is_idempotent_for_files(self, session: TranscriptionSession):
        """Calling finalize twice should not raise, just append a second footer."""
        session.finalize()
        session.finalize()  # should not raise
        assert session.txt_path.exists()

    def test_empty_session_finalizes_cleanly(self, session: TranscriptionSession):
        path = session.finalize()
        assert path.exists()
        data = json.loads(session.json_path.read_text(encoding="utf-8"))
        assert data["entries"] == []
