"""ASR 엔진 추상 인터페이스."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscriptionResult:
    """전사 결과."""

    text: str
    language: str = ""
    is_final: bool = True


class BaseASREngine(ABC):
    """모든 ASR 엔진이 구현할 공통 인터페이스."""

    @abstractmethod
    def transcribe(
        self,
        audio: bytes | Path | str,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """오디오를 전사한다.

        Args:
            audio: PCM bytes, WAV 파일 경로, 또는 numpy 배열(미구현 시 Path/str만)
            sample_rate: 샘플레이트 (bytes 입력 시 필수)

        Returns:
            TranscriptionResult
        """
        ...

    @abstractmethod
    def initialize(self) -> None:
        """엔진을 초기화하고 모델을 로드한다."""
        ...

    @abstractmethod
    def release(self) -> None:
        """리소스를 해제한다."""
        ...
