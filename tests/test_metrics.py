"""src/utils/metrics.py 단위 테스트."""

from __future__ import annotations

import time

import pytest
from src.utils.metrics import (
    RTFTimer,
    calculate_cer,
    calculate_wer,
    get_memory_usage_mb,
)

# ---------------------------------------------------------------------------
# CER (Character Error Rate)
# ---------------------------------------------------------------------------


class TestCalculateCER:
    def test_identical_strings_return_zero(self):
        assert calculate_cer("안녕하세요", "안녕하세요") == 0.0

    def test_completely_different_strings(self):
        cer = calculate_cer("abc", "xyz")
        assert cer > 0.0

    def test_partial_error(self):
        cer = calculate_cer("abcd", "abxd")
        assert 0.0 < cer < 1.0

    def test_empty_reference_empty_hypothesis(self):
        assert calculate_cer("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis(self):
        assert calculate_cer("", "abc") == 1.0

    def test_nonempty_reference_empty_hypothesis(self):
        cer = calculate_cer("abc", "")
        assert cer > 0.0

    def test_single_character(self):
        assert calculate_cer("a", "a") == 0.0
        assert calculate_cer("a", "b") > 0.0


# ---------------------------------------------------------------------------
# WER (Word Error Rate)
# ---------------------------------------------------------------------------


class TestCalculateWER:
    def test_identical_sentences_return_zero(self):
        assert calculate_wer("hello world", "hello world") == 0.0

    def test_word_substitution(self):
        wer = calculate_wer("the cat sat", "the dog sat")
        assert wer == pytest.approx(1 / 3)

    def test_word_insertion(self):
        wer = calculate_wer("the cat", "the big cat")
        assert wer > 0.0

    def test_word_deletion(self):
        wer = calculate_wer("the big cat", "the cat")
        assert wer > 0.0

    def test_empty_reference_empty_hypothesis(self):
        assert calculate_wer("", "") == 0.0

    def test_empty_reference_nonempty_hypothesis(self):
        assert calculate_wer("", "hello") == 1.0

    def test_nonempty_reference_empty_hypothesis(self):
        wer = calculate_wer("hello world", "")
        assert wer > 0.0

    def test_completely_different_sentences(self):
        wer = calculate_wer("the cat sat", "big dog ran")
        assert wer == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# RTFTimer
# ---------------------------------------------------------------------------


class TestRTFTimer:
    def test_elapsed_and_rtf_positive(self):
        with RTFTimer(audio_duration=10.0) as timer:
            time.sleep(0.01)
        assert timer.elapsed > 0.0
        assert timer.rtf > 0.0

    def test_rtf_equals_elapsed_over_duration(self):
        with RTFTimer(audio_duration=5.0) as timer:
            time.sleep(0.01)
        assert timer.rtf == pytest.approx(timer.elapsed / 5.0)

    def test_negative_audio_duration_raises(self):
        with pytest.raises(ValueError, match="positive"):
            RTFTimer(audio_duration=-1.0)

    def test_zero_audio_duration_raises(self):
        with pytest.raises(ValueError, match="positive"):
            RTFTimer(audio_duration=0.0)

    def test_context_manager_returns_self(self):
        timer = RTFTimer(audio_duration=1.0)
        with timer as t:
            assert t is timer

    def test_initial_values_are_zero(self):
        timer = RTFTimer(audio_duration=1.0)
        assert timer.elapsed == 0.0
        assert timer.rtf == 0.0


# ---------------------------------------------------------------------------
# 메모리 사용량
# ---------------------------------------------------------------------------


class TestGetMemoryUsageMB:
    def test_returns_positive_value(self):
        usage = get_memory_usage_mb()
        assert usage > 0.0

    def test_returns_float(self):
        usage = get_memory_usage_mb()
        assert isinstance(usage, float)
