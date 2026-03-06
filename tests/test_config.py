"""src/config.py 단위 테스트."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from src.config import (
    AppConfig,
    ASRConfig,
    AudioConfig,
    LLMConfig,
    OutputConfig,
    VADConfig,
    get_config,
    load_config,
    reset_config,
)


@pytest.fixture(autouse=True)
def _isolate_singleton():
    """매 테스트 전후로 싱글톤 인스턴스를 초기화한다."""
    reset_config()
    yield
    reset_config()


# ---------------------------------------------------------------------------
# 기본값 생성 (config.yaml 없는 경우)
# ---------------------------------------------------------------------------


class TestDefaultConfig:
    def test_load_without_yaml_returns_defaults(self, tmp_path: Path):
        cfg = load_config(tmp_path / "nonexistent.yaml")

        assert isinstance(cfg, AppConfig)
        assert cfg.audio == AudioConfig()
        assert cfg.asr.engine == "funasr"
        assert cfg.asr.language == "ko"
        assert cfg.asr.device == "mps"
        assert cfg.asr.vad == VADConfig()
        assert cfg.llm == LLMConfig()
        assert cfg.output == OutputConfig()
        assert cfg.logging_level == "INFO"

    def test_default_sub_configs(self):
        audio = AudioConfig()
        assert audio.sample_rate == 16000
        assert audio.channels == 1
        assert audio.bit_depth == 16
        assert audio.chunk_size == 1024

        vad = VADConfig()
        assert vad.model is None
        assert vad.max_segment_time == 30000
        assert vad.speech_to_sil_time_thres == 150
        assert vad.max_end_silence_time == 800

        asr = ASRConfig()
        assert asr.engine == "funasr"
        assert asr.language == "ko"
        assert asr.device == "mps"

        llm = LLMConfig()
        assert llm.provider == "openai"
        assert llm.model == "gpt-4o"
        assert llm.temperature == 0.3

        output = OutputConfig()
        assert output.transcripts_dir == "output/transcripts"
        assert output.notes_dir == "output/notes"
        assert output.recordings_dir == "output/recordings"


# ---------------------------------------------------------------------------
# config.yaml 정상 로드
# ---------------------------------------------------------------------------


class TestLoadFromYAML:
    def test_loads_values_from_yaml(self, tmp_path: Path):
        yaml_content = textwrap.dedent("""\
            audio:
              sample_rate: 44100
              channels: 2
              bit_depth: 24
              chunk_size: 2048
            asr:
              engine: sensevoice
              language: en
              device: cpu
              vad:
                model: custom-vad
                max_segment_time: 15000
            llm:
              provider: gemini
              model: gemini-pro
              temperature: 0.7
            output:
              transcripts_dir: out/t
              notes_dir: out/n
              recordings_dir: out/r
            logging:
              level: DEBUG
        """)
        config_file = tmp_path / "config.yaml"
        config_file.write_text(yaml_content, encoding="utf-8")

        cfg = load_config(config_file)

        assert cfg.audio.sample_rate == 44100
        assert cfg.audio.channels == 2
        assert cfg.audio.bit_depth == 24
        assert cfg.audio.chunk_size == 2048

        assert cfg.asr.engine == "sensevoice"
        assert cfg.asr.language == "en"
        assert cfg.asr.device == "cpu"
        assert cfg.asr.vad.model == "custom-vad"
        assert cfg.asr.vad.max_segment_time == 15000

        assert cfg.llm.provider == "gemini"
        assert cfg.llm.model == "gemini-pro"
        assert cfg.llm.temperature == 0.7

        assert cfg.output.transcripts_dir == "out/t"
        assert cfg.output.notes_dir == "out/n"
        assert cfg.output.recordings_dir == "out/r"

        assert cfg.logging_level == "DEBUG"

    def test_partial_yaml_uses_defaults_for_missing(self, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("llm:\n  provider: claude\n", encoding="utf-8")

        cfg = load_config(config_file)

        assert cfg.llm.provider == "claude"
        assert cfg.llm.temperature == 0.3
        assert cfg.audio == AudioConfig()
        assert cfg.asr.engine == "funasr"

    def test_empty_yaml_uses_defaults(self, tmp_path: Path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("", encoding="utf-8")

        cfg = load_config(config_file)

        assert cfg.audio == AudioConfig()
        assert cfg.logging_level == "INFO"


# ---------------------------------------------------------------------------
# 잘못된 설정값 ValueError 검증
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.fixture()
    def _yaml_file(self, tmp_path: Path):
        self._config_path = tmp_path / "config.yaml"
        return self._config_path

    def _write_and_load(self, path: Path, content: str) -> AppConfig:
        path.write_text(textwrap.dedent(content), encoding="utf-8")
        return load_config(path)

    def test_invalid_engine(self, _yaml_file: Path):
        with pytest.raises(ValueError, match="asr.engine"):
            self._write_and_load(_yaml_file, "asr:\n  engine: whisper\n")

    def test_invalid_language(self, _yaml_file: Path):
        with pytest.raises(ValueError, match="asr.language"):
            self._write_and_load(_yaml_file, "asr:\n  language: zh\n")

    def test_invalid_device(self, _yaml_file: Path):
        with pytest.raises(ValueError, match="asr.device"):
            self._write_and_load(_yaml_file, "asr:\n  device: cuda\n")

    def test_invalid_provider(self, _yaml_file: Path):
        with pytest.raises(ValueError, match="llm.provider"):
            self._write_and_load(_yaml_file, "llm:\n  provider: huggingface\n")

    def test_invalid_temperature_too_high(self, _yaml_file: Path):
        with pytest.raises(ValueError, match="llm.temperature"):
            self._write_and_load(_yaml_file, "llm:\n  temperature: 3.0\n")

    def test_invalid_temperature_negative(self, _yaml_file: Path):
        with pytest.raises(ValueError, match="llm.temperature"):
            self._write_and_load(_yaml_file, "llm:\n  temperature: -0.1\n")

    def test_invalid_sample_rate(self, _yaml_file: Path):
        with pytest.raises(ValueError, match="audio.sample_rate"):
            self._write_and_load(_yaml_file, "audio:\n  sample_rate: 0\n")

    def test_invalid_log_level(self, _yaml_file: Path):
        with pytest.raises(ValueError, match="logging.level"):
            self._write_and_load(_yaml_file, "logging:\n  level: TRACE\n")


# ---------------------------------------------------------------------------
# reset_config / 싱글톤
# ---------------------------------------------------------------------------


class TestSingleton:
    def test_get_config_returns_same_instance(self, tmp_path: Path):
        load_config(tmp_path / "none.yaml")
        first = get_config()
        second = get_config()
        assert first is second

    def test_reset_then_reload_gives_new_instance(self, tmp_path: Path):
        load_config(tmp_path / "none.yaml")
        first = get_config()

        reset_config()

        load_config(tmp_path / "none.yaml")
        second = get_config()
        assert first is not second

    def test_get_config_auto_loads_if_not_initialized(self):
        cfg = get_config()
        assert isinstance(cfg, AppConfig)
