"""src/asr/streaming/funasr_engine.py 단위 테스트 (mock)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("funasr")

from src.asr.streaming.funasr_engine import FunASREngine
from src.config import ASRConfig, load_config, reset_config


@pytest.fixture(autouse=True)
def _isolate_config():
    reset_config()
    yield
    reset_config()


@pytest.fixture()
def _mock_funasr():
    """funasr AutoModel을 mock하여 모델 로드 없이 테스트한다."""
    mock_model = MagicMock()
    mock_model.generate.return_value = [{"text": "mock transcription"}]
    mock_auto_model = MagicMock(return_value=mock_model)
    with patch("funasr.AutoModel", mock_auto_model):
        yield mock_model


class TestFunASREngine:
    def test_initialize_loads_model(self, _mock_funasr):
        load_config()
        from funasr import AutoModel

        engine = FunASREngine()
        engine.initialize()
        engine.release()
        AutoModel.assert_called_once()

    def test_transcribe_path_returns_result(self, _mock_funasr, tmp_path: Path):
        load_config()
        wav = tmp_path / "test.wav"
        wav.write_bytes(b"dummy")
        engine = FunASREngine()
        engine._model = _mock_funasr

        result = engine.transcribe(wav)

        assert result.text == "mock transcription"
        assert result.language == "ko"

    def test_transcribe_bytes_creates_temp_wav(self, _mock_funasr):
        load_config()
        pcm = b"\x00\x00" * 1000
        engine = FunASREngine()
        engine._model = _mock_funasr

        result = engine.transcribe(pcm, sample_rate=16000)

        assert result.text == "mock transcription"
        _mock_funasr.generate.assert_called_once()

    def test_custom_config_language(self, _mock_funasr, tmp_path: Path):
        load_config()
        cfg = ASRConfig(language="en")
        engine = FunASREngine(config=cfg)
        engine._model = _mock_funasr
        wav = tmp_path / "test.wav"
        wav.write_bytes(b"dummy")

        engine.transcribe(wav)

        call_kw = _mock_funasr.generate.call_args[1]
        assert call_kw["language"] == "英文"
