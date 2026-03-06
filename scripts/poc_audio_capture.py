"""M1 검증: 마이크 캡처 + dB 터미널 출력 + WAV 저장.

실행: conda run -n aln python scripts/poc_audio_capture.py
종료: Ctrl+C
"""

from __future__ import annotations

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.capture import AudioCapture
from src.audio.utils import bytes_to_frames, compute_db_rms
from src.config import get_config, load_config


def main() -> None:
    load_config()
    capture = AudioCapture()

    try:
        capture.start()
        cfg = get_config().audio
        sample_width = cfg.bit_depth // 8

        print("Capturing... (Ctrl+C to stop)")
        print("-" * 40)

        while True:
            try:
                chunk = capture.chunk_queue.get(timeout=0.1)
            except Exception:
                continue

            frames = bytes_to_frames(chunk, sample_width, cfg.channels)
            db = compute_db_rms(frames)
            level = min(50, max(0, int(db + 60)))
            bar = "|" * level + " " * (50 - level)
            print(f"\r dB: {db:6.1f}  {bar}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        capture.stop()
        print("Done.")


if __name__ == "__main__":
    main()
