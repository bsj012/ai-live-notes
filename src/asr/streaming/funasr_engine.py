"""FunASR-MLT-Nano 스트리밍 엔진 — VAD + pseudo-streaming."""

from __future__ import annotations

import tempfile
from pathlib import Path

from src.asr.base import BaseASREngine, TranscriptionResult
from src.config import ASRConfig, get_config
from src.utils.logger import get_logger

_log = get_logger("src.asr.streaming.funasr_engine")

# FunASR inference_with_vad는 FunASR-MLT-Nano 타임스탬프 형식과 맞지 않아 KeyError: 0 발생.
# VAD 세그먼트별 개별 전사 후 병합으로 우회.

_MODEL_ID = "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
_LANG_MAP = {"ko": "韩文", "en": "英文", "ja": "日文", "zh": "中文"}

# Fun-ASR model.py 경로 (FunASRNano 클래스 등록용)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_FUN_ASR_REPO = _PROJECT_ROOT / "models" / "Fun-ASR"


class FunASREngine(BaseASREngine):
    """FunASR-MLT-Nano 기반 ASR 엔진.

    VAD + pseudo-streaming: 오디오 청크를 모아 세그먼트 단위로 전사.
    """

    def __init__(self, config: ASRConfig | None = None) -> None:
        self._config = config or get_config().asr
        self._model = None

    def _get_remote_code_path(self) -> Path | None:
        """Fun-ASR model.py 경로를 반환한다. 없으면 None."""
        if getattr(self._config, "fun_asr_repo_path", None):
            repo = Path(self._config.fun_asr_repo_path)
        else:
            repo = _DEFAULT_FUN_ASR_REPO
        path = repo / "model.py"
        return path if path.exists() else None

    def initialize(self) -> None:
        """모델을 로드한다."""
        if self._model is not None:
            _log.warning("Model already initialized")
            return

        try:
            from funasr import AutoModel
            from funasr.utils.dynamic_import import import_module_from_path
        except ImportError as e:
            raise RuntimeError(
                "funasr not installed. Run: pip install funasr torch"
            ) from e

        lang = _LANG_MAP.get(self._config.language, "韩文")
        vad_model = getattr(self._config.vad, "model", None) or None
        vad_kwargs: dict = {}
        if vad_model:
            # model_conf를 여기서 넣으면 build_model이 download를 건너뛰어 등록 실패.
            # fsmn-vad 기본값(speech_to_sil=150ms, max_end_silence=800ms) 사용.
            vad_kwargs = {
                "max_single_segment_time": self._config.vad.max_segment_time,
                "hub": "hf",
            }

        # FunASRNano 등록: hub=hf일 때 download_from_hf는 remote_code를 처리하지 않음.
        # model.py를 수동 로드하여 tables.model_classes에 FunASRNano 등록.
        remote_code_path = self._get_remote_code_path()
        if remote_code_path is not None:
            import_module_from_path(str(remote_code_path))

        auto_model_kwargs: dict = {
            "model": _MODEL_ID,
            "trust_remote_code": True,
            "vad_model": vad_model,
            "vad_kwargs": vad_kwargs,
            "device": self._config.device,
            "hub": "hf",
            "disable_update": True,
        }

        _log.info(
            "Loading FunASR-MLT-Nano (device=%s, lang=%s)",
            self._config.device,
            lang,
        )
        self._model = AutoModel(**auto_model_kwargs)

        if self._model.vad_model is not None:
            vad_cfg = self._config.vad
            # vad_opts는 FsmnVADStreaming에 있음 (encoder가 아님)
            self._model.vad_model.vad_opts.speech_to_sil_time_thres = (
                vad_cfg.speech_to_sil_time_thres
            )
            self._model.vad_model.vad_opts.max_end_silence_time = (
                vad_cfg.max_end_silence_time
            )
            _log.info(
                "VAD overrides applied: speech_to_sil=%dms, max_end_silence=%dms",
                vad_cfg.speech_to_sil_time_thres,
                vad_cfg.max_end_silence_time,
            )

        _log.info("FunASR-MLT-Nano loaded")

    def release(self) -> None:
        """리소스를 해제한다."""
        self._model = None
        _log.info("FunASR engine released")

    def transcribe(
        self,
        audio: bytes | Path | str,
        sample_rate: int = 16000,
    ) -> TranscriptionResult:
        """오디오를 전사한다.

        Args:
            audio: WAV 파일 경로, 또는 PCM bytes (16-bit mono)
            sample_rate: bytes 입력 시 샘플레이트

        Returns:
            TranscriptionResult
        """
        if self._model is None:
            self.initialize()

        if isinstance(audio, bytes):
            path = self._bytes_to_wav(audio, sample_rate)
            try:
                return self._transcribe_path(path)
            finally:
                Path(path).unlink(missing_ok=True)
        else:
            return self._transcribe_path(str(Path(audio).resolve()))

    def _transcribe_path(self, path: str) -> TranscriptionResult:
        """WAV 경로를 전사한다. VAD 사용 시 세그먼트별 전사 후 병합(KeyError: 0 우회)."""
        lang = _LANG_MAP.get(self._config.language, "韩文")
        cfg = {"cache": {}, "batch_size": 1, "language": lang}

        try:
            if self._model.vad_model is not None:
                text = self._transcribe_with_vad_segments(path, lang, cfg)
            else:
                res = self._model.generate(input=[path], **cfg)
                text = res[0]["text"] if res and len(res) > 0 else ""
        except ValueError as e:
            if "disallowed special token" in str(e).lower() or "<|" in str(e):
                _log.warning("CTC/tiktoken 특수 토큰 에러, 빈 결과 반환: %s", e)
                text = ""
            else:
                raise

        return TranscriptionResult(text=text, language=self._config.language)

    def _transcribe_with_vad_segments(self, path: str, lang: str, cfg: dict) -> str:
        """VAD로 세그먼트 추출 후 각각 전사하여 병합. inference_with_vad KeyError: 0 우회."""
        from funasr.utils.load_utils import load_audio_text_image_video
        from funasr.utils.vad_utils import slice_padding_audio_samples

        # 1) VAD로 세그먼트 추출
        vad_res = self._model.inference(
            [path],
            model=self._model.vad_model,
            kwargs=self._model.vad_kwargs,
            **cfg,
        )
        if not vad_res or not vad_res[0].get("value"):
            # 음성 없음 → ASR 호출 생략 (침묵/노이즈에서 할루시네이션 방지)
            return ""

        vadsegments = vad_res[0]["value"]
        if not vadsegments:
            return ""

        # [[start_ms, end_ms], ...] → slice_padding_audio_samples 형식 [(seg, idx), ...]
        segments_for_slice = [([s[0], s[1]], i) for i, s in enumerate(vadsegments)]

        # 2) 오디오 로드
        fs = self._model.kwargs.get("frontend", None)
        fs = fs.fs if fs and hasattr(fs, "fs") else 16000
        speech = load_audio_text_image_video(path, fs=fs, audio_fs=cfg.get("fs", 16000))
        speech_lengths = len(speech)

        # 3) 세그먼트별 전사 (slice 형식: segment = ( [start,end], idx ) → segment[0][0], segment[0][1])
        speech_list, _ = slice_padding_audio_samples(
            speech, speech_lengths, segments_for_slice
        )
        texts: list[str] = []
        for speech_chunk in speech_list:
            n_samples = (
                speech_chunk.shape[-1]
                if hasattr(speech_chunk, "shape")
                else len(speech_chunk)
            )
            if n_samples < 800:  # 50ms 미만 스킵 (노이즈 버스트 필터)
                continue
            try:
                res = self._model.inference(
                    [speech_chunk],
                    model=self._model.model,
                    kwargs=self._model.kwargs,
                    **cfg,
                )
            except ValueError as e:
                if "disallowed special token" in str(e).lower() or "<|" in str(e):
                    _log.debug("CTC 특수 토큰 에러, 세그먼트 스킵: %s", e)
                    continue
                raise
            if res and res[0].get("text", "").strip():
                texts.append(res[0]["text"].strip())

        return " ".join(texts)

    def _bytes_to_wav(self, pcm: bytes, sample_rate: int) -> str:
        """PCM bytes를 임시 WAV 파일로 저장한다."""
        import wave

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm)
        return path
