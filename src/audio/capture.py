"""마이크 캡처 모듈 — PyAudio 실시간 스트림, 큐, WAV 저장."""

from __future__ import annotations

import queue
import sys
import threading
import wave
from datetime import datetime
from pathlib import Path

import pyaudio

from src.config import AudioConfig, get_config
from src.utils.logger import get_logger

_log = get_logger("src.audio.capture")

# PyAudio format: 16-bit = paInt16
_PA_FORMAT = pyaudio.paInt16

_MIC_PERMISSION_HINT = """
마이크 접근 권한이 필요합니다.

macOS: 시스템 설정 → 개인 정보 보호 및 보안 → 마이크
  - "터미널" 또는 "Cursor" 앱이 목록에 있고 체크되어 있는지 확인하세요.
  - 권한이 거부된 경우 앱을 다시 실행하면 권한 요청 창이 뜹니다.
"""


class AudioCapture:
    """PyAudio 기반 실시간 마이크 캡처.

    - 오디오 청크를 queue.Queue에 put (ASR 엔진에서 consume)
    - output/recordings/에 .wav 파일 동시 저장
    """

    def __init__(self, config: AudioConfig | None = None) -> None:
        self._config = config or get_config().audio
        self._output_dir = Path(get_config().output.recordings_dir)
        self._chunk_queue: queue.Queue[bytes] = queue.Queue()
        self._pa: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None
        self._wav_file: wave.Wave_write | None = None
        self._wav_path: Path | None = None
        self._running = threading.Event()

    @property
    def chunk_queue(self) -> queue.Queue[bytes]:
        """ASR 엔진에서 consume할 청크 큐."""
        return self._chunk_queue

    @staticmethod
    def check_mic_available(sample_rate: int = 16000) -> bool:
        """마이크 접근 가능 여부를 확인한다. 권한 거부 시 False, 안내 메시지 출력."""
        pa = None
        try:
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=_PA_FORMAT,
                channels=1,
                rate=sample_rate,
                input=True,
                frames_per_buffer=256,
            )
            stream.close()
            return True
        except OSError as e:
            _log.warning("Microphone access failed: %s", e)
            print(_MIC_PERMISSION_HINT, file=sys.stderr)
            return False
        except Exception as e:
            _log.warning("Microphone check failed: %s", e)
            print(_MIC_PERMISSION_HINT, file=sys.stderr)
            return False
        finally:
            if pa is not None:
                try:
                    pa.terminate()
                except Exception:
                    pass

    def start(self) -> None:
        """스트림을 시작하고 WAV 기록을 시작한다."""
        if self._stream is not None:
            _log.warning("Capture already started")
            return

        self._output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._wav_path = self._output_dir / f"recording_{ts}.wav"

        self._pa = pyaudio.PyAudio()
        self._wav_file = wave.open(str(self._wav_path), "wb")  # noqa: SIM115
        self._wav_file.setnchannels(self._config.channels)
        self._wav_file.setsampwidth(self._config.bit_depth // 8)
        self._wav_file.setframerate(self._config.sample_rate)

        def _callback(
            in_data: bytes,
            frame_count: int,
            time_info: dict,
            status: int,
        ) -> tuple[bytes, int]:
            if in_data:
                self._chunk_queue.put(in_data)
                if self._wav_file is not None:
                    self._wav_file.writeframes(in_data)
            return (b"", pyaudio.paContinue)

        self._stream = self._pa.open(
            format=_PA_FORMAT,
            channels=self._config.channels,
            rate=self._config.sample_rate,
            input=True,
            frames_per_buffer=self._config.chunk_size,
            stream_callback=_callback,
        )
        self._running.set()
        self._stream.start_stream()
        _log.info("Capture started, recording to %s", self._wav_path)

    def stop(self) -> None:
        """스트림을 종료하고 리소스를 해제한다."""
        self._running.clear()
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception as e:
                _log.warning("Error closing stream: %s", e)
            self._stream = None
        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception as e:
                _log.warning("Error terminating PyAudio: %s", e)
            self._pa = None
        if self._wav_file is not None:
            try:
                self._wav_file.close()
            except Exception as e:
                _log.warning("Error closing WAV file: %s", e)
            self._wav_file = None
        if self._wav_path is not None:
            _log.info("Recording saved to %s", self._wav_path)
            self._wav_path = None
