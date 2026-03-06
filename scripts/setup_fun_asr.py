"""Fun-ASR 저장소 초기화 스크립트.

FunASR-MLT-Nano 실행에 필요한 model.py(FunASRNano 클래스)는
Fun-ASR GitHub 저장소에만 있으므로, 이 스크립트로 저장소를 클론한다.

실행: conda run -n aln python scripts/setup_fun_asr.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FUN_ASR_REPO = _PROJECT_ROOT / "models" / "Fun-ASR"
_MODEL_PY = _FUN_ASR_REPO / "model.py"
_FUN_ASR_URL = "https://github.com/FunAudioLLM/Fun-ASR.git"


def main() -> int:
    if _MODEL_PY.exists():
        print(f"Fun-ASR already present: {_FUN_ASR_REPO}")
        return 0

    _FUN_ASR_REPO.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning Fun-ASR into {_FUN_ASR_REPO}...")

    try:
        subprocess.run(
            ["git", "clone", _FUN_ASR_URL, str(_FUN_ASR_REPO)],
            check=True,
            capture_output=False,
        )
    except subprocess.CalledProcessError as e:
        print(f"Clone failed: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError:
        print("git not found. Install git and retry.", file=sys.stderr)
        return 1

    if _MODEL_PY.exists():
        print("Fun-ASR setup complete.")
        return 0

    print("Clone succeeded but model.py not found.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
