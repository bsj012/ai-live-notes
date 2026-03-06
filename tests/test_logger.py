"""src/utils/logger.py 단위 테스트."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import src.utils.logger as logger_mod
from src.utils.logger import get_logger


@pytest.fixture(autouse=True)
def _isolate_logger():
    """매 테스트 전후로 로거 전역 상태를 초기화한다."""
    logger_mod._initialized = False

    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level

    yield

    logger_mod._initialized = False

    root = logging.getLogger()
    for handler in root.handlers[:]:
        if handler not in original_handlers:
            handler.close()
            root.removeHandler(handler)
    root.setLevel(original_level)


@pytest.fixture()
def _patch_log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """로그 파일 경로를 tmp_path로 우회하여 실제 logs/ 에 쓰지 않도록 한다."""
    log_dir = tmp_path / "logs"
    log_file = log_dir / "app.log"
    monkeypatch.setattr(logger_mod, "_LOG_DIR", log_dir)
    monkeypatch.setattr(logger_mod, "_LOG_FILE", log_file)
    return log_file


# ---------------------------------------------------------------------------
# 로거 생성 및 반환 타입
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_returns_logging_logger(self, _patch_log_file: Path):
        logger = get_logger("test.module")
        assert isinstance(logger, logging.Logger)

    def test_logger_has_correct_name(self, _patch_log_file: Path):
        logger = get_logger("myapp.core")
        assert logger.name == "myapp.core"

    def test_different_names_return_different_loggers(self, _patch_log_file: Path):
        a = get_logger("module.a")
        b = get_logger("module.b")
        assert a is not b
        assert a.name != b.name


# ---------------------------------------------------------------------------
# 듀얼 출력 (콘솔 + 파일)
# ---------------------------------------------------------------------------


class TestDualOutput:
    def test_root_logger_has_console_and_file_handlers(self, _patch_log_file: Path):
        get_logger("dual.test")

        root = logging.getLogger()
        handler_types = {type(h) for h in root.handlers}
        assert logging.StreamHandler in handler_types
        assert logging.FileHandler in handler_types

    def test_log_message_written_to_file(self, _patch_log_file: Path):
        logger = get_logger("file.writer")
        logger.info("file-write-check")

        content = _patch_log_file.read_text(encoding="utf-8")
        assert "file-write-check" in content
        assert "file.writer" in content

    def test_log_message_written_to_stderr(
        self, _patch_log_file: Path, capsys: pytest.CaptureFixture[str]
    ):
        logger = get_logger("stderr.writer")
        logger.warning("stderr-check")

        captured = capsys.readouterr()
        assert "stderr-check" in captured.err

    def test_log_dir_created_automatically(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        log_dir = tmp_path / "nested" / "logs"
        monkeypatch.setattr(logger_mod, "_LOG_DIR", log_dir)
        monkeypatch.setattr(logger_mod, "_LOG_FILE", log_dir / "app.log")

        get_logger("dir.create")

        assert log_dir.exists()


# ---------------------------------------------------------------------------
# 레벨 필터링
# ---------------------------------------------------------------------------


class TestLevelFiltering:
    def test_debug_filtered_at_info_level(self, _patch_log_file: Path):
        logger_mod._init_root_logger("INFO")
        logger = get_logger("level.filter")

        logger.debug("should-not-appear")
        logger.info("should-appear")

        for handler in logging.getLogger().handlers:
            handler.flush()

        content = _patch_log_file.read_text(encoding="utf-8")
        assert "should-not-appear" not in content
        assert "should-appear" in content

    def test_debug_visible_at_debug_level(self, _patch_log_file: Path):
        logger_mod._init_root_logger("DEBUG")
        logger = get_logger("level.debug")

        logger.debug("debug-visible")

        for handler in logging.getLogger().handlers:
            handler.flush()

        content = _patch_log_file.read_text(encoding="utf-8")
        assert "debug-visible" in content

    def test_warning_visible_at_info_level(self, _patch_log_file: Path):
        logger_mod._init_root_logger("INFO")
        logger = get_logger("level.warn")

        logger.warning("warn-msg")

        for handler in logging.getLogger().handlers:
            handler.flush()

        content = _patch_log_file.read_text(encoding="utf-8")
        assert "warn-msg" in content


# ---------------------------------------------------------------------------
# 초기화 동작
# ---------------------------------------------------------------------------


class TestInitialization:
    def test_init_called_only_once(self, _patch_log_file: Path):
        get_logger("init.first")
        handler_count = len(logging.getLogger().handlers)

        get_logger("init.second")
        assert len(logging.getLogger().handlers) == handler_count

    def test_falls_back_to_info_when_config_unavailable(
        self,
        _patch_log_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        def _raise(*_args, **_kwargs):
            raise RuntimeError("config unavailable")

        monkeypatch.setattr("src.utils.logger.get_config", _raise, raising=False)

        logger = get_logger("fallback.test")
        assert logging.getLogger().level == logging.INFO
        assert isinstance(logger, logging.Logger)
