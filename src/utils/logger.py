"""구조화 로깅 모듈 — 콘솔 + 파일 듀얼 출력."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_LOG_DIR = _PROJECT_ROOT / "logs"
_LOG_FILE = _LOG_DIR / "app.log"
_FMT = "[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

_initialized = False


def _ensure_log_dir() -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)


def _init_root_logger(level: str = "INFO") -> None:
    """루트 로거에 콘솔·파일 핸들러를 한 번만 부착한다."""
    global _initialized  # noqa: PLW0603
    if _initialized:
        return

    _ensure_log_dir()

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    formatter = logging.Formatter(_FMT, datefmt=_DATE_FMT)

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """모듈별 로거를 생성·반환한다.

    최초 호출 시 루트 로거를 초기화하며, 로그 레벨은
    config.yaml의 ``logging.level`` 값을 따른다.
    """
    if not _initialized:
        try:
            from src.config import get_config

            level = get_config().logging_level
        except Exception:
            level = "INFO"
        _init_root_logger(level)

    return logging.getLogger(name)
