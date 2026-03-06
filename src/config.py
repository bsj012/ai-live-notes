"""설정 관리 모듈 — .env API 키 + config.yaml 전역 설정 로드."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Dataclass 정의
# ---------------------------------------------------------------------------


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    chunk_size: int = 1024


@dataclass
class VADConfig:
    """VAD 설정. model이 None이면 VAD 비활성화 (ModelScope 다운로드 불필요)."""

    model: str | None = None  # "fsmn-vad" 또는 null (비활성화)
    max_segment_time: int = 30000
    speech_to_sil_time_thres: int = 150  # ms, 음성→침묵 전환 임계값
    max_end_silence_time: int = 800      # ms, 발화 끝 침묵 허용


@dataclass
class ASRConfig:
    engine: str = "funasr"
    language: str = "ko"
    device: str = "mps"
    vad: VADConfig = field(default_factory=VADConfig)
    fun_asr_repo_path: str | None = None  # null이면 models/Fun-ASR 사용


@dataclass
class LLMConfig:
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.3


@dataclass
class OutputConfig:
    transcripts_dir: str = "output/transcripts"
    notes_dir: str = "output/notes"
    recordings_dir: str = "output/recordings"


@dataclass
class APIKeys:
    """환경변수(.env)에서 로드된 API 키."""

    openai_api_key: str = ""
    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3"


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    api_keys: APIKeys = field(default_factory=APIKeys)
    logging_level: str = "INFO"


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------

_VALID_ENGINES = {"funasr", "zipformer", "sensevoice"}
_VALID_LANGUAGES = {"ko", "en", "ja", "zh"}
_VALID_DEVICES = {"mps", "cpu"}
_VALID_LLM_PROVIDERS = {"openai", "gemini", "claude", "ollama"}
_VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def _validate(config: AppConfig) -> None:
    """설정값 범위를 검증하고, 잘못된 값이면 ValueError를 발생시킨다."""
    if config.asr.engine not in _VALID_ENGINES:
        raise ValueError(
            f"asr.engine must be one of {_VALID_ENGINES}, got '{config.asr.engine}'"
        )
    if config.asr.language not in _VALID_LANGUAGES:
        raise ValueError(
            f"asr.language must be one of {_VALID_LANGUAGES}, "
            f"got '{config.asr.language}'"
        )
    if config.asr.device not in _VALID_DEVICES:
        raise ValueError(
            f"asr.device must be one of {_VALID_DEVICES}, got '{config.asr.device}'"
        )
    if config.llm.provider not in _VALID_LLM_PROVIDERS:
        raise ValueError(
            f"llm.provider must be one of {_VALID_LLM_PROVIDERS}, "
            f"got '{config.llm.provider}'"
        )
    if config.logging_level not in _VALID_LOG_LEVELS:
        raise ValueError(
            f"logging.level must be one of {_VALID_LOG_LEVELS}, "
            f"got '{config.logging_level}'"
        )
    if not (0.0 <= config.llm.temperature <= 2.0):
        raise ValueError(
            f"llm.temperature must be between 0.0 and 2.0, got {config.llm.temperature}"
        )
    if config.audio.sample_rate <= 0:
        raise ValueError(
            f"audio.sample_rate must be positive, got {config.audio.sample_rate}"
        )


# ---------------------------------------------------------------------------
# YAML → dataclass 매핑
# ---------------------------------------------------------------------------


def _build_config(raw: dict[str, Any]) -> AppConfig:
    """YAML에서 파싱된 dict를 AppConfig dataclass 트리로 변환한다."""
    audio_raw = raw.get("audio", {})
    asr_raw = raw.get("asr", {})
    llm_raw = raw.get("llm", {})
    output_raw = raw.get("output", {})
    logging_raw = raw.get("logging", {})

    vad_raw = asr_raw.pop("vad", {}) if isinstance(asr_raw, dict) else {}
    vad = VADConfig(**vad_raw) if vad_raw else VADConfig()

    audio = AudioConfig(**audio_raw) if audio_raw else AudioConfig()
    asr = ASRConfig(**asr_raw, vad=vad) if asr_raw else ASRConfig(vad=vad)
    llm = LLMConfig(**llm_raw) if llm_raw else LLMConfig()
    output = OutputConfig(**output_raw) if output_raw else OutputConfig()

    return AppConfig(
        audio=audio,
        asr=asr,
        llm=llm,
        output=output,
        logging_level=logging_raw.get("level", "INFO"),
    )


def _load_api_keys() -> APIKeys:
    """환경변수에서 API 키를 읽어 APIKeys 객체로 반환한다."""
    return APIKeys(
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3"),
    )


# ---------------------------------------------------------------------------
# 싱글톤 퍼블릭 API
# ---------------------------------------------------------------------------

_config_instance: AppConfig | None = None


def load_config(config_path: str | Path | None = None) -> AppConfig:
    """config.yaml + .env를 로드하여 AppConfig를 생성·검증·반환한다.

    config_path가 None이면 프로젝트 루트의 config.yaml을 사용한다.
    파일이 존재하지 않으면 기본값으로 동작한다.
    """
    global _config_instance  # noqa: PLW0603

    load_dotenv(_PROJECT_ROOT / ".env")

    # HuggingFace 모델 캐시를 프로젝트 models/ 아래로 (HF_HOME 미설정 시)
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(_PROJECT_ROOT / "models" / "huggingface")

    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if path.exists():
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        raw = {}

    config = _build_config(raw)
    config.api_keys = _load_api_keys()
    _validate(config)

    _config_instance = config
    return config


def get_config() -> AppConfig:
    """이미 로드된 AppConfig를 반환한다.

    아직 로드 전이면 자동으로 load_config()를 호출한다.
    """
    if _config_instance is None:
        return load_config()
    return _config_instance


def reset_config() -> None:
    """싱글톤 인스턴스를 초기화한다(테스트용)."""
    global _config_instance  # noqa: PLW0603
    _config_instance = None
