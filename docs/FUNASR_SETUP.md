# FunASR-MLT-Nano 설치 및 트러블슈팅

FunASR-MLT-Nano 실행 시 필요한 설정과 시행착오를 정리한 문서입니다.

## 1. 사전 준비

### 1.1 Fun-ASR GitHub 저장소 클론

FunASRNano 클래스는 HuggingFace 모델 저장소에 없고, [FunAudioLLM/Fun-ASR](https://github.com/FunAudioLLM/Fun-ASR) GitHub에만 `model.py`로 정의되어 있습니다.

```bash
python scripts/setup_fun_asr.py
```

또는 수동:

```bash
mkdir -p models
git clone https://github.com/FunAudioLLM/Fun-ASR.git models/Fun-ASR
```

필요 파일: `model.py`, `ctc.py`, `tools/` (저장소 클론 시 모두 포함)

### 1.2 의존성 설치

```bash
conda run -n aln pip install -r requirements.txt
```

필수 패키지: `funasr`, `torch`, `torchaudio`, `transformers`, `huggingface_hub`, `openai-whisper`

### 1.3 HuggingFace 모델 캐시

`config.py`에서 `HF_HOME` 미설정 시 `models/huggingface`로 설정됩니다. HuggingFace 모델 다운로드는 이 경로에 저장됩니다.

## 2. 알려진 오류 및 해결

### 2.1 `AssertionError: FunASRNano is not registered`

**원인**: FunASR의 `download_from_hf`는 `remote_code`를 처리하지 않음. ModelScope(`download_from_ms`)만 처리.

**해결**: `funasr_engine.py`에서 `import_module_from_path(model.py)`를 AutoModel 호출 전에 수동 실행하여 `tables.model_classes`에 FunASRNano 등록. (이미 구현됨)

### 2.2 `No module named 'huggingface_hub'`

**해결**: `pip install huggingface_hub`

### 2.3 `UnboundLocalError: local variable 'AutoTokenizer' referenced before assignment`

**원인**: `transformers` 관련 import 문제 (FunASR 내부).

**해결**: `pip install transformers` 확인

### 2.4 `UnboundLocalError: local variable 'get_tokenizer' referenced before assignment`

**원인**: `openai-whisper` 미설치.

**해결**: `pip install openai-whisper`

### 2.5 `AssertionError: fsmn-vad is not registered`

**원인**: VAD 모델은 ModelScope에서 다운로드. `~/.cache/modelscope` 사용. 네트워크/권한 문제 시 실패.

**해결**: 네트워크 연결 확인, `~/.cache/modelscope` 쓰기 권한 확인

**우회**: `config.yaml`에서 `asr.vad.model: null`로 설정하면 VAD 비활성화. ModelScope 다운로드 없이 실행 가능.

### 2.6 `KeyError: 0` (inference_with_vad)

**원인**: FunASR 라이브러리 내부 버그 (FunASR-MLT-Nano 타임스탬프 형식과 inference_with_vad 불일치).

**해결**: `funasr_engine.py`에서 VAD 세그먼트별 개별 전사 후 병합으로 우회. `generate()` 대신 VAD → slice → inference 순으로 직접 처리.

## 3. 실행 검증

```bash
# 마이크 모드 (Ctrl+C로 종료) — macOS 권한 요청 시 허용 필요
conda run -n aln python scripts/poc_funasr_test.py

# WAV 파일 모드
conda run -n aln python scripts/poc_funasr_test.py output/recordings/recording_xxx.wav
```

### 마이크 권한 (macOS)

마이크 모드 실행 시 시스템 권한이 필요합니다:

- **시스템 설정** → **개인 정보 보호 및 보안** → **마이크**
- "터미널" 또는 "Cursor" 앱이 목록에 있고 체크되어 있는지 확인
- 권한 없이 실행 시 `check_mic_available()`에서 안내 메시지 출력 후 종료

## 4. 참고

- [Fun-ASR-MLT-Nano HuggingFace](https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512)
- [Fun-ASR GitHub](https://github.com/FunAudioLLM/Fun-ASR)
- [FunASR Issue #49](https://github.com/FunAudioLLM/Fun-ASR/issues/49) — FunASRNano 등록 관련
