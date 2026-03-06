"""성능 측정 유틸리티 — CER, WER, RTF, 메모리 사용량."""

from __future__ import annotations

import time
from types import TracebackType

import jiwer
import psutil


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Character Error Rate를 계산한다 (한국어 등 문자 단위 평가용)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return float(jiwer.cer(reference, hypothesis))


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Word Error Rate를 계산한다 (영어·일본어 등 단어 단위 평가용)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return float(jiwer.wer(reference, hypothesis))


class RTFTimer:
    """컨텍스트 매니저로 Real-Time Factor를 측정한다.

    RTF = 처리 시간 / 오디오 길이  (< 1.0 이면 실시간보다 빠름)

    사용법::

        with RTFTimer(audio_duration=10.0) as timer:
            process(audio)
        print(timer.rtf)
    """

    def __init__(self, audio_duration: float) -> None:
        if audio_duration <= 0:
            raise ValueError(f"audio_duration must be positive, got {audio_duration}")
        self.audio_duration = audio_duration
        self.elapsed: float = 0.0
        self.rtf: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> RTFTimer:
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.elapsed = time.perf_counter() - self._start
        self.rtf = self.elapsed / self.audio_duration


def get_memory_usage_mb() -> float:
    """현재 프로세스의 RSS(Resident Set Size)를 MB 단위로 반환한다."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)
