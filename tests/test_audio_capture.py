"""src/audio/capture.py 단위 테스트 (PyAudio mock)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from src.audio.capture import AudioCapture
from src.config import (
    AppConfig,
    AudioConfig,
    OutputConfig,
    load_config,
    reset_config,
)


@pytest.fixture(autouse=True)
def _isolate_config():
    reset_config()
    yield
    reset_config()


@pytest.fixture()
def _mock_pyaudio():
    """PyAudio를 mock하여 하드웨어 접근 없이 테스트한다."""
    mock_pa = MagicMock()
    mock_stream = MagicMock()
    mock_pa.open.return_value = mock_stream
    with patch("src.audio.capture.pyaudio.PyAudio", return_value=mock_pa):
        yield mock_pa


@pytest.fixture()
def _patch_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """recordings_dir를 tmp_path로 우회한다."""
    load_config()
    fake_config = AppConfig(
        output=OutputConfig(recordings_dir=str(tmp_path / "recordings"))
    )
    monkeypatch.setattr("src.audio.capture.get_config", lambda: fake_config)
    return tmp_path / "recordings"


class TestAudioCapture:
    def test_start_stop_no_error(self, _mock_pyaudio, _patch_output_dir):
        capture = AudioCapture()
        capture.start()
        assert capture._stream is not None
        capture.stop()
        assert capture._stream is None
        _mock_pyaudio.open.assert_called_once()
        _mock_pyaudio.terminate.assert_called_once()

    def test_chunk_queue_available(self, _mock_pyaudio, _patch_output_dir):
        capture = AudioCapture()
        assert capture.chunk_queue is capture._chunk_queue
        capture.start()
        capture.stop()

    def test_wav_file_created_on_start(self, _mock_pyaudio, _patch_output_dir: Path):
        capture = AudioCapture()
        capture.start()
        wav_files = list(_patch_output_dir.glob("*.wav"))
        capture.stop()
        assert len(wav_files) >= 1

    def test_custom_config_used(self, _mock_pyaudio, _patch_output_dir):
        cfg = AudioConfig(sample_rate=8000, chunk_size=512)
        capture = AudioCapture(config=cfg)
        capture.start()
        call_kw = _mock_pyaudio.open.call_args[1]
        assert call_kw["rate"] == 8000
        assert call_kw["frames_per_buffer"] == 512
        capture.stop()
