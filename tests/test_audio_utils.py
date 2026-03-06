"""src/audio/utils.py 단위 테스트."""

from __future__ import annotations

import struct

import numpy as np
import pytest
from src.audio.utils import bytes_to_frames, compute_db_rms, has_sufficient_energy, resample

# ---------------------------------------------------------------------------
# bytes_to_frames
# ---------------------------------------------------------------------------


class TestBytesToFrames:
    def test_16bit_mono_roundtrip(self):
        samples = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        raw = samples.tobytes()
        out = bytes_to_frames(raw, sample_width=2, channels=1)
        max_val = float(np.iinfo(np.int16).max)
        expected = samples.astype(np.float32) / max_val
        np.testing.assert_array_almost_equal(out, expected, decimal=4)

    def test_16bit_stereo_averages_channels(self):
        # L=32767, R=-32767 -> mono avg ≈ 0
        raw = struct.pack("<hh", 32767, -32767)
        out = bytes_to_frames(raw, sample_width=2, channels=2)
        assert len(out) == 1
        assert abs(out[0]) < 0.01

    def test_empty_input(self):
        out = bytes_to_frames(b"", sample_width=2, channels=1)
        assert len(out) == 0

    def test_unsupported_sample_width_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            bytes_to_frames(b"\x00\x00", sample_width=1, channels=1)


# ---------------------------------------------------------------------------
# resample
# ---------------------------------------------------------------------------


class TestResample:
    def test_same_rate_returns_copy(self):
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        out = resample(audio, 16000, 16000)
        np.testing.assert_array_almost_equal(out, audio)
        assert out is not audio

    def test_double_rate_doubles_length(self):
        audio = np.array([0.0, 1.0], dtype=np.float32)
        out = resample(audio, 1000, 2000)
        assert len(out) == 4

    def test_half_rate_halves_length(self):
        audio = np.ones(100, dtype=np.float32)
        out = resample(audio, 16000, 8000)
        assert len(out) == 50

    def test_empty_audio(self):
        audio = np.array([], dtype=np.float32)
        out = resample(audio, 16000, 8000)
        assert len(out) == 0


# ---------------------------------------------------------------------------
# compute_db_rms
# ---------------------------------------------------------------------------


class TestComputeDbRms:
    def test_silence_returns_negative(self):
        audio = np.zeros(100, dtype=np.float32)
        db = compute_db_rms(audio)
        assert db < 0

    def test_full_scale_sine_approx_0db(self):
        t = np.linspace(0, 1, 1000)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        db = compute_db_rms(audio)
        # RMS of sin = 1/sqrt(2) ≈ -3dB
        assert -5 < db < 0

    def test_half_amplitude_approx_minus_6db(self):
        audio = np.ones(100, dtype=np.float32) * 0.5
        db = compute_db_rms(audio)
        assert -7 < db < -5

    def test_empty_returns_low_value(self):
        audio = np.array([], dtype=np.float32)
        db = compute_db_rms(audio)
        assert db < -50 or db == float("-inf")


# ---------------------------------------------------------------------------
# has_sufficient_energy
# ---------------------------------------------------------------------------


class TestHasSufficientEnergy:
    def test_silence_returns_false(self):
        raw = np.zeros(1600, dtype=np.int16).tobytes()
        assert has_sufficient_energy(raw, threshold_db=-40.0) is False

    def test_loud_speech_returns_true(self):
        raw = (np.ones(1600, dtype=np.int16) * 10000).tobytes()
        assert has_sufficient_energy(raw, threshold_db=-40.0) is True

    def test_short_buffer_returns_false(self):
        assert has_sufficient_energy(b"", threshold_db=-40.0) is False
