"""M2 검증: 마이크 → FunASR 실시간 전사 → 터미널 출력.

실행: conda run -n aln python scripts/poc_funasr_test.py [wav_path]
  - 인자 없음: 마이크 입력 → 실시간 전사
  - wav_path: WAV 파일 단독 전사 테스트

종료: Ctrl+C (마이크 모드)

필요: funasr, torch 설치 (pip install funasr torch)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.utils import has_sufficient_energy
from src.asr.streaming.funasr_engine import FunASREngine
from src.config import load_config


def run_file_mode(wav_path: Path) -> None:
    """WAV 파일 단독 전사 테스트."""
    load_config()
    engine = FunASREngine()
    try:
        engine.initialize()
        result = engine.transcribe(wav_path)
        print(result.text)
    finally:
        engine.release()


def run_mic_mode() -> None:
    from src.audio.capture import AudioCapture

    cfg = load_config()

    if not AudioCapture.check_mic_available(cfg.audio.sample_rate):
        print("마이크 권한을 확인한 후 다시 실행해 주세요.", file=sys.stderr)
        sys.exit(1)

    # 16kHz mono 16-bit: 2초 = 32000 samples = 64000 bytes
    buffer_duration_sec = 2.0
    bytes_per_sec = cfg.audio.sample_rate * 2  # 16-bit mono
    target_bytes = int(buffer_duration_sec * bytes_per_sec)

    capture = AudioCapture()
    engine = FunASREngine()
    buffer = bytearray()

    try:
        engine.initialize()
        capture.start()

        print("Listening... (Ctrl+C to stop)")
        print("-" * 40)

        while True:
            try:
                chunk = capture.chunk_queue.get(timeout=0.5)
            except Exception:
                continue

            buffer.extend(chunk)
            if len(buffer) >= target_bytes:
                buf = bytes(buffer)
                buffer.clear()
                if not has_sufficient_energy(buf, threshold_db=-40.0):
                    continue
                result = engine.transcribe(buf, cfg.audio.sample_rate)
                if result.text.strip():
                    print(f"\n>>> {result.text}")

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        if buffer:
            buf = bytes(buffer)
            if has_sufficient_energy(buf, threshold_db=-40.0):
                result = engine.transcribe(buf, cfg.audio.sample_rate)
                if result.text.strip():
                    print(f"\n>>> {result.text}")
        capture.stop()
        engine.release()
        print("Done.")


def main() -> None:
    if len(sys.argv) > 1:
        wav_path = Path(sys.argv[1])
        if wav_path.exists():
            run_file_mode(wav_path)
        else:
            print(f"File not found: {wav_path}")
            sys.exit(1)
    else:
        run_mic_mode()


if __name__ == "__main__":
    main()
