# AI Live Notes

맥북 환경에서 실시간 음성 전사(ASR) + AI 강의 노트 자동 생성 애플리케이션.

FunASR-MLT-Nano로 실시간 전사(KO/EN/JA/ZH 31개 언어)하고, VibeVoice-ASR로 고품질 후처리(화자 분리 + 타임스탬프)한 뒤, LLM으로 구조화된 노트를 생성합니다.

## 환경 설정

### 1. Python 환경 (conda)

```bash
conda create -n aln python=3.10 -y
conda activate aln
pip install -r requirements.txt
```

> Python 3.10 필수 (conda 환경 `aln` 사용)

### 2. 시스템 의존성

```bash
# macOS (PortAudio — PyAudio 빌드에 필요)
brew install portaudio
```

### 3. 설정 파일

```bash
cp .env.example .env
# .env 파일을 편집하여 사용할 서비스의 API 키 입력
```

`config.yaml`에서 ASR 엔진, 언어, 오디오 파라미터, LLM 프로바이더 등 전역 설정을 관리합니다.
자세한 설정 항목은 [`plan.md`](plan.md) 섹션 2.16을 참조하세요.

지원하는 LLM 프로바이더 (API 키를 입력한 프로바이더만 활성화):

| 프로바이더 | 환경 변수 |
|---|---|
| OpenAI | `OPENAI_API_KEY` |
| Google Gemini | `GEMINI_API_KEY` |
| Anthropic Claude | `ANTHROPIC_API_KEY` |
| Ollama (로컬) | 토큰 불필요 |

### 4. HuggingFace 로그인 (벤치마크용)

CER 벤치마크에 사용하는 데이터셋이 gated dataset이므로, 접근 전 로그인이 필요합니다.

```bash
pip install huggingface_hub
huggingface-cli login
# HuggingFace 토큰 입력 (https://huggingface.co/settings/tokens)
```

### 5. ASR 모델

#### FunASR-MLT-Nano-2512 (실시간 전사 — 메인)

FunASR 프레임워크에서 자동으로 모델을 다운로드합니다.

```python
from funasr import AutoModel

model = AutoModel(
    model="FunAudioLLM/Fun-ASR-MLT-Nano-2512",
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="mps",  # Apple Silicon (또는 "cpu")
)
```

- 31개 언어 지원 (한국어, 영어, 일본어, 중국어 포함)
- 800M 파라미터, `device="mps"` 로 맥북에서 CUDA 없이 구동

#### Streaming Zipformer (실시간 전사 — 대안)

```bash
# 추후 models/download_models.sh 스크립트 제공 예정
```

모델 목록:
- `sherpa-onnx-streaming-zipformer-korean-2024-06-16` (한국어)
- `sherpa-onnx-streaming-zipformer-en-2023-06-26` (영어)
- 일본어 스트리밍 Zipformer 모델은 존재하지 않음

#### VibeVoice-ASR (고품질 후처리)

```bash
# 9B 모델, BF16 기준 ~18GB — Apple Silicon 권장
# 추후 다운로드 가이드 제공 예정
```

## 사용법

> Phase 1 (CLI 프로토타입) 구현 후 업데이트 예정

```bash
conda activate aln
# python main.py  (추후 구현)
```

## 출력 결과

| 디렉토리 | 내용 |
|---|---|
| `output/recordings/` | 녹음 .wav 파일 |
| `output/transcripts/` | 전사 텍스트 |
| `output/notes/` | LLM 생성 구조화 노트 |

## 문서

- [`spec.md`](spec.md) — 프로젝트 요구사항 정의서
- [`plan.md`](plan.md) — 구현 계획

## 라이선스

[MIT License](LICENSE)
