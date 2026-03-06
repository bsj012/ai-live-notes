"""src/asr/base.py 단위 테스트."""

from __future__ import annotations

from pathlib import Path

from src.asr.base import BaseASREngine, TranscriptionResult


class MockASREngine(BaseASREngine):
    """테스트용 추상 클래스 구현."""

    def transcribe(
        self,
        audio: bytes | Path | str,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        if isinstance(audio, bytes):
            return TranscriptionResult(text=f"bytes_{len(audio)}", is_final=True)
        return TranscriptionResult(text="file_result", is_final=True)

    def initialize(self) -> None:
        pass

    def release(self) -> None:
        pass


class TestTranscriptionResult:
    def test_default_values(self):
        r = TranscriptionResult(text="hello")
        assert r.text == "hello"
        assert r.language == ""
        assert r.is_final is True

    def test_full_constructor(self):
        r = TranscriptionResult(text="안녕", language="ko", is_final=False)
        assert r.text == "안녕"
        assert r.language == "ko"
        assert r.is_final is False


class TestBaseASREngine:
    def test_mock_engine_transcribe_bytes(self):
        engine = MockASREngine()
        result = engine.transcribe(b"\x00" * 1024, sample_rate=16000)
        assert result.text == "bytes_1024"

    def test_mock_engine_transcribe_path(self):
        engine = MockASREngine()
        result = engine.transcribe(Path("/tmp/test.wav"))
        assert result.text == "file_result"

    def test_mock_engine_transcribe_str(self):
        engine = MockASREngine()
        result = engine.transcribe("/tmp/test.wav")
        assert result.text == "file_result"

    def test_initialize_release_no_error(self):
        engine = MockASREngine()
        engine.initialize()
        engine.release()
