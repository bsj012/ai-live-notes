# 구현 계획 (Plan.md)

## 1. 디렉토리 구조

```
ai-live-notes/
├── main.py                     # CLI 엔트리포인트 (루트 레벨)
├── config.yaml                 # 전역 설정 (오디오, ASR, LLM 파라미터)
├── src/
│   ├── __init__.py
│   ├── asr/                    # ASR 엔진
│   │   ├── __init__.py
│   │   ├── base.py             # ASR 엔진 추상 인터페이스
│   │   ├── streaming/          # 실시간 ASR
│   │   │   ├── __init__.py
│   │   │   ├── funasr_engine.py    # FunASR-MLT-Nano (메인)
│   │   │   ├── zipformer_engine.py # Streaming Zipformer (대안)
│   │   │   └── sensevoice_engine.py # SenseVoice (대안)
│   │   └── batch/              # 후처리 ASR
│   │       ├── __init__.py
│   │       ├── vibevoice_engine.py  # VibeVoice-ASR 배치 전사
│   │       └── diarization.py       # 화자 분리
│   ├── llm/                    # LLM 요약
│   │   ├── __init__.py
│   │   ├── summarizer.py       # 전사→노트 변환 로직
│   │   └── providers/          # LLM 프로바이더
│   │       ├── __init__.py
│   │       ├── base.py         # 프로바이더 추상 인터페이스
│   │       ├── openai_provider.py
│   │       ├── gemini_provider.py
│   │       ├── claude_provider.py
│   │       └── ollama_provider.py
│   ├── pipeline/               # 파이프라인 오케스트레이션
│   │   ├── __init__.py
│   │   └── runner.py           # ASR→후처리→LLM 파이프라인 실행기
│   ├── audio/                  # 오디오 처리
│   │   ├── __init__.py
│   │   ├── capture.py          # PyAudio 마이크 캡처
│   │   └── utils.py            # 오디오 포맷 변환, 리샘플링
│   ├── utils/                  # 공통 유틸리티
│   │   ├── __init__.py
│   │   ├── logger.py           # 구조화 로깅
│   │   └── metrics.py          # WER, RTF 등 성능 측정
│   └── config.py               # 설정 관리 (.env + config.yaml 로드)
├── scripts/                    # 유틸리티 스크립트
│   ├── poc_funasr_test.py      # FunASR PoC 테스트
│   ├── benchmark_cer.py        # 한국어 CER 벤치마크 (HuggingFace 데이터셋)
│   └── compare_asr_engines.py  # 4개 ASR 엔진 결과 비교
├── models/                     # ASR 모델 파일 (git-ignored)
│   ├── funasr/                 # FunASR 모델 캐시
│   ├── sherpa_onnx/            # Zipformer, SenseVoice 모델
│   ├── vibevoice/              # VibeVoice-ASR 모델
│   └── download_models.sh
├── output/                     # 출력 결과 (git-ignored)
│   ├── transcripts/            # 전사 텍스트
│   ├── notes/                  # LLM 생성 노트
│   └── recordings/             # 녹음 .wav 파일
├── tests/                      # 테스트 오디오 및 유닛테스트
│   ├── audio_samples/          # 테스트 오디오 파일
│   └── test_asr.py
├── Requirement.md
├── Plan.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
└── LICENSE
```

## 2. 모듈별 구현 명세

### 2.1 `src/audio/capture.py` — 마이크 캡처

- PyAudio를 이용한 실시간 오디오 스트림 캡처
- 16-bit, mono, 16kHz 샘플링
- 오디오 청크를 콜백/큐 방식으로 ASR 엔진에 전달
- 동시에 `output/recordings/`에 .wav 파일 저장 (후처리용)

### 2.2 `src/audio/utils.py` — 오디오 유틸리티

- 오디오 포맷 변환 (sample rate, bit depth)
- 리샘플링 (다양한 입력 소스 대응)
- 오디오 레벨(dB) 계산

### 2.3 `src/asr/base.py` — ASR 엔진 추상 인터페이스

- 모든 ASR 엔진이 구현할 공통 인터페이스 정의
- `transcribe(audio_chunk) -> TranscriptionResult`
- 엔진 초기화, 언어 설정, 리소스 해제 메서드

### 2.4 `src/asr/streaming/funasr_engine.py` — FunASR-MLT-Nano (메인)

- `funasr.AutoModel`로 `FunAudioLLM/Fun-ASR-MLT-Nano-2512` 로드
- `device="mps"` (Apple Silicon) 또는 `device="cpu"` 설정
- VAD(`fsmn-vad`) + ASR 조합으로 pseudo-streaming 구현
- 언어 파라미터 설정 (한국어/영어/일본어)
- 핫워드 지원

### 2.5 `src/asr/streaming/zipformer_engine.py` — Streaming Zipformer (대안)

- `sherpa-onnx` 기반 진정한 스트리밍 ASR
- 언어별 모델 로드 (KO: `sherpa-onnx-streaming-zipformer-korean-2024-06-16`,
  EN: `sherpa-onnx-streaming-zipformer-en-2023-06-26`)
- 실시간 오디오 청크를 받아 즉시 전사 반환

### 2.6 `src/asr/streaming/sensevoice_engine.py` — SenseVoice (대안)

- `sherpa-onnx` 기반, VAD + offline 조합
- 단일 모델로 KO/EN/JA/ZH/Cantonese 지원
- FunASR-MLT-Nano와 성능 비교용

### 2.7 `src/asr/batch/vibevoice_engine.py` — VibeVoice-ASR (후처리)

- 녹음된 .wav 파일을 VibeVoice-ASR 9B로 배치 전사
- Rich Transcription 출력: ASR + 화자 분리 + 타임스탬프를 단일 패스로 수행
- 커스텀 핫워드(프롬프트 기반 context injection) 설정
- 고정밀 전사 결과 반환
- **한계**: SFT가 영어·중국어에 집중되어 한국어 성능은 M4.5에서 검증 필요

### 2.8 `src/asr/batch/diarization.py` — 화자 분리 결과 파싱/포맷팅

- VibeVoice-ASR의 Rich Transcription 출력에서 화자·타임스탬프 정보를 파싱
- 구조화된 데이터 모델로 변환 (화자 라벨, 시작/종료 시간, 발화 텍스트)
- LLM 요약에 적합한 형태로 포맷팅 (화자별 발화 묶기, 시간순 정렬)
- 향후 독립 화자 분리 모델(pyannote 등)로 교체 가능하도록 인터페이스 분리

### 2.9 `src/llm/providers/` — LLM 프로바이더

- `base.py`: 추상 인터페이스 (`generate(prompt) -> str`)
- 각 프로바이더: API 키 존재 시에만 활성화
- 프로바이더 자동 감지: `.env`에 키가 있는 프로바이더만 로드
- Ollama: 로컬 구동, 토큰 불필요

### 2.10 `src/llm/summarizer.py` — 전사→노트 변환

- 전사 텍스트를 구조화된 노트로 변환하는 프롬프트 설계 및 실행
- LLM 프로바이더 선택 로직 (`config.yaml` 기반)
- 핵심 포인트, 주제별 분류, 액션 아이템 추출
- 결과를 `output/notes/`에 저장

### 2.11 `src/pipeline/runner.py` — 파이프라인 오케스트레이션

- ASR → 후처리 → LLM 요약의 전체 파이프라인 실행 관리
- 각 단계 간 데이터 흐름 제어
- 에러 핸들링 및 재시도 로직
- 실시간 전사 모드 / 배치 모드 분기

### 2.12 `src/utils/logger.py` — 구조화 로깅

- 프로젝트 전역 로거 설정
- 파일 + 콘솔 듀얼 출력
- 로그 레벨 `config.yaml`에서 관리

### 2.13 `src/utils/metrics.py` — 성능 측정

- WER (Word Error Rate) 계산
- CER (Character Error Rate) 계산
- RTF (Real Time Factor) 측정
- 메모리 사용량 모니터링

### 2.14 `src/config.py` — 설정 관리

- `.env` 파일에서 API 키 로드 (`python-dotenv`)
- `config.yaml`에서 전역 설정 로드 (ASR 엔진 선택, 언어, 오디오 파라미터 등)
- 설정 값 검증 및 기본값 제공

### 2.15 `main.py` — CLI 엔트리포인트 (루트)

- `argparse` 기반 CLI 인터페이스
- `pipeline/runner.py`를 호출하여 파이프라인 실행
- 명령줄 옵션: 언어 선택, 엔진 선택, 모드 선택 (실시간/배치)
- 실시간 전사 → 터미널 출력
- 녹음 종료 시 후처리 ASR + LLM 요약 실행

### 2.16 `config.yaml` — 전역 설정 파일

```yaml
audio:
  sample_rate: 16000
  channels: 1
  bit_depth: 16
  chunk_size: 1024

asr:
  engine: funasr          # funasr / zipformer / sensevoice
  language: ko            # ko / en / ja
  device: mps             # mps / cpu
  vad:
    model: fsmn-vad
    max_segment_time: 30000

llm:
  provider: openai        # openai / gemini / claude / ollama
  model: gpt-4o
  temperature: 0.3

output:
  transcripts_dir: output/transcripts
  notes_dir: output/notes
  recordings_dir: output/recordings

logging:
  level: INFO
```

## 3. 의존성

```
# requirements.txt
funasr              # FunASR-MLT-Nano (메인 ASR)
sherpa-onnx         # Streaming Zipformer, SenseVoice (대안 ASR)
pyaudio             # 마이크 캡처
openai              # OpenAI LLM
anthropic           # Claude LLM
google-generativeai # Gemini LLM
python-dotenv       # .env 설정 로드
pyyaml              # config.yaml 파싱
datasets            # HuggingFace 데이터셋 로드
huggingface_hub     # HuggingFace 모델/데이터 접근
jiwer               # WER/CER 계산 라이브러리
torch               # VibeVoice-ASR 추론 (M4에서 필요)
transformers        # VibeVoice-ASR 모델 로드 (M4에서 필요)
openai-whisper      # Whisper large-v3 로컬 베이스라인 (M4.5에서 필요)
```

## 4. 구현 순서 (마일스톤별)

### M0: 프로젝트 기반 설정

1. `config.yaml` 작성 (전역 설정 파일)
2. `src/config.py` 구현 (`.env` + `config.yaml` 로드)
3. `src/utils/logger.py` 구현 (구조화 로깅)
4. `src/utils/metrics.py` 구현 (WER, RTF 등 성능 측정 유틸)
5. 디렉토리 구조 생성 (`output/`, `models/`, `tests/`)

### M1: 마이크 입력 캡처

1. `src/audio/capture.py` 구현
2. `src/audio/utils.py` 구현 (오디오 포맷 변환, 리샘플링)
3. PyAudio로 16kHz mono 오디오 스트림 캡처
4. 터미널에 오디오 레벨(dB) 표시하여 정상 동작 확인
5. `output/recordings/`에 .wav 파일 동시 저장

### M2: 실시간 ASR PoC

1. `scripts/poc_funasr_test.py` 작성 — FunASR-MLT-Nano 단독 테스트
2. `src/asr/base.py` ASR 엔진 추상 인터페이스 구현
3. `src/asr/streaming/funasr_engine.py` 구현
4. VAD + FunASR-MLT-Nano 파이프라인 연결
5. 마이크 입력 → 실시간 전사 → 터미널 출력

### M3: 다국어 전사 검증

1. 한국어 전사 테스트 — **CER** 측정 (`src/utils/metrics.py` 활용)
2. 영어 전사 테스트 — WER 측정
3. 일본어 전사 테스트 — WER 측정
4. (옵션) Streaming Zipformer, SenseVoice와 비교 테스트

### M4: 고품질 후처리 ASR

1. `src/asr/batch/vibevoice_engine.py` 구현
2. `src/asr/batch/diarization.py` 구현
3. VibeVoice-ASR 9B 모델 로드 및 배치 전사
4. 화자 분리 + 타임스탬프 출력 확인
5. Apple Silicon 로컬 추론 가능 여부 확인

### M4.5: 한국어 CER 정량 벤치마크 (Go/No-Go 게이트)

1. HuggingFace 데이터셋 준비
   - `jp1924/KsponSpeech` — 한국어 자발 발화 (eval set 500~1000 샘플)
   - `jp1924/KoreanUniversityLectureData` — 한국어 대학 강의 (도메인 적합)
2. 베이스라인 ASR 준비
   - **Whisper large-v3 (로컬)** — 오픈소스 베이스라인, FunASR 공식 벤치마크에 비교 데이터 존재
   - **OpenAI Whisper API** (`gpt-4o-audio-preview`) — 상용 베이스라인, 절대 품질 기준점
3. `scripts/benchmark_cer.py` 구현
   - `huggingface_hub` 로그인 + `datasets` 라이브러리로 데이터 로드
   - 4개 엔진(FunASR, VibeVoice, Whisper-local, OpenAI API) 동일 데이터로 전사
   - `jiwer` 라이브러리로 CER/WER 계산
4. `scripts/compare_asr_engines.py` 구현
   - 4개 엔진 결과를 테이블로 비교 출력
5. 비교 대상 및 측정 항목

| 항목 | FunASR-MLT-Nano | VibeVoice-ASR 9B | Whisper large-v3 (로컬) | OpenAI Whisper API |
|---|---|---|---|---|
| 역할 | 실시간 ASR 후보 | 후처리 ASR 후보 | 오픈소스 베이스라인 | 상용 베이스라인 |
| 모드 | pseudo-streaming | 배치 전사 | 배치 (로컬) | 배치 (API) |
| CER (KsponSpeech) | 측정 | 측정 | 측정 | 측정 |
| CER (강의 데이터) | 측정 | 측정 | 측정 | 측정 |
| RTF | 측정 | 측정 | 측정 | N/A (API) |
| 메모리 사용량 | MPS / CPU | MPS / CPU | MPS / CPU | N/A |
| 비용 | 무료 | 무료 | 무료 | ~$0.006/분 |

> **비용 추정**: KsponSpeech 500 샘플 (평균 10초, 총 ~83분) 기준 OpenAI API 비용 약 $0.50

6. Go/No-Go 판단 기준

| 조건 | 판단 | 조치 |
|---|---|---|
| FunASR CER ≤ 10% | Go | 실시간 ASR 채택 확정, LLM 요약 품질 영향 최소화 |
| FunASR CER 10~15% | 조건부 Go | 채택하되 핫워드 튜닝으로 보완, LLM 프롬프트에 교정 지시 추가 |
| FunASR CER > 15% | Hold | 모델 교체 또는 파인튜닝 검토 (공식 벤치마크에서 한국어 미검증) |
| FunASR CER ≤ OpenAI CER + 3%p | Go | 상용 수준에 근접, 오픈소스 이점(비용, 프라이버시) 확보 |
| FunASR CER > OpenAI CER + 10%p | Hold | 실시간 경로에 OpenAI API 직접 사용 검토 |
| VibeVoice CER < FunASR CER - 5%p | Go | 2단계 파이프라인 가치 확인, 후처리 채택 |
| VibeVoice CER ≥ FunASR CER - 5%p | 재검토 | 후처리 단계 필요성 재평가 |
| 두 엔진 모두 CER > 20% | No-Go | 한국어 특화 모델 또는 파인튜닝 필수 |

> **근거**: ASR CER 10% 이상에서 LLM 요약 품질(사실 정합성)이 저하되기 시작함.
> 특히 고유명사·전문용어 오류는 WER/CER 수치 대비 요약 품질에 불균형적으로 큰 영향을 미침
> (Shapira et al. 2025, Pulikodan et al. 2025).
> 단, LLM의 언어 이해 능력으로 일부 ASR 오류를 보정할 수 있으므로, 10~15% 구간은 조건부 허용.
> 상용 STT 대비 격차가 3%p 이내면 오픈소스 이점(비용, 프라이버시, 오프라인)이 충분히 가치 있음.

7. 참고: 공식 벤치마크 현황

| 모델 | 한국어 벤치마크 | 비고 |
|---|---|---|
| FunASR-MLT-Nano | **미공개** | 공식 벤치마크에 한국어 테스트셋 미포함 (ZH/EN/ID/TH/VI만 공개) |
| FunASR-Nano (중국어 버전) | Industry Avg WER 16.72% | 0.8B, 중국어/영어 중심 벤치마크 |
| VibeVoice-ASR | **미공개** | SFT가 영어·중국어·코드스위칭에 집중, 한국어 성능 저하 가능 |
| Whisper large-v3 | Fleurs-zh CER 4.71% | FunASR-ML-Nano(3.51%)와 공식 비교 존재, 한국어는 미포함 |
| OpenAI Whisper API | 미공개 | 상용 서비스, 한국어 지원, 업계 표준 품질 기준 |

> **핵심 리스크**: 4개 모델 모두 한국어 공식 벤치마크가 없으므로, M4.5에서의 자체 측정이 필수적임

### M5: LLM 요약

1. `src/llm/providers/base.py` 추상 인터페이스 구현
2. OpenAI, Gemini, Claude, Ollama 프로바이더 각각 구현
3. `src/llm/summarizer.py` 구현 — 전사→노트 변환 프롬프트 설계
4. 프로바이더 자동 감지 로직
5. 결과를 `output/notes/`에 저장

### M6: 통합 파이프라인

1. `src/pipeline/runner.py` 구현 — ASR→후처리→LLM 전체 흐름 관리
2. `main.py`(루트)에서 `pipeline/runner.py` 호출
3. CLI 인터페이스: 시작/중지, 언어 선택, 엔진 선택, 모드 선택
4. End-to-End 동작 테스트
5. 에러 처리 및 안정성 개선

## 5. 데이터 흐름

```
[마이크] ──▶ [audio/capture.py] ──▶ [오디오 청크 큐]
                │                           │
                │                           ▼
                │                   [VAD (fsmn-vad)]
                │                           │
                │                     음성 구간 감지
                │                           │
                │                           ▼
                │              [asr/streaming/funasr_engine.py]
                │                           │
                │                      전사 텍스트
                │                           │
                │                           ▼
                │                  [터미널 실시간 출력]
                │
                ▼
    [output/recordings/*.wav]
                │
           녹음 종료 후
                │
        ┌───────┴───────────────────────────────┐
        │              pipeline/runner.py        │
        │                                        │
        │    ▼                                   │
        │  [asr/batch/vibevoice_engine.py]       │
        │    │                                   │
        │    ▼                                   │
        │  [asr/batch/diarization.py]            │
        │    │  고품질 전사 + 화자 분리            │
        │    ▼                                   │
        │  [output/transcripts/]                 │
        │    │                                   │
        │    ▼                                   │
        │  [llm/summarizer.py]                   │
        │    │  구조화된 노트 생성                 │
        │    ▼                                   │
        │  [output/notes/]                       │
        └────────────────────────────────────────┘
```

## 6. PoC 테스트 계획 (M2/M3)

### FunASR-MLT-Nano-2512 맥북 테스트

```python
from funasr import AutoModel

model = AutoModel(
    model="FunAudioLLM/Fun-ASR-MLT-Nano-2512",
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="mps",  # Apple Silicon
)

# 한국어 테스트
res = model.generate(input=["test_ko.wav"], cache={}, language="韩文")
# 영어 테스트
res = model.generate(input=["test_en.wav"], cache={}, language="英文")
# 일본어 테스트
res = model.generate(input=["test_ja.wav"], cache={}, language="日文")
```

### 측정 항목

| 항목 | 측정 방법 |
|---|---|
| 전사 정확도 (한국어 CER / 영어·일본어 WER) | 레퍼런스 전사와 비교 |
| 처리 속도 (RTF) | 처리 시간 / 오디오 길이 |
| 메모리 사용량 | `mps` vs `cpu` 비교 |
| 지연시간 | 마이크 입력 → 전사 출력 시간 |

## 7. 한국어 CER 벤치마크 계획 (M4.5)

### 평가 데이터셋

| 데이터셋 | 용도 | 특징 |
|---|---|---|
| [KsponSpeech](https://huggingface.co/datasets/jp1924/KsponSpeech) | 범용 한국어 ASR 벤치마크 | 자발 발화, 100K~1M 샘플, Parquet |
| [KoreanUniversityLectureData](https://huggingface.co/datasets/jp1924/KoreanUniversityLectureData) | 도메인 적합성 검증 | 대학 강의 음성, 프로젝트 목적과 직결 |

> **참고**: 두 데이터셋 모두 gated dataset으로 HuggingFace 로그인 + 연락처 정보 동의 후 접근 가능

### 한국어에 CER을 사용하는 이유

한국어는 어절 단위 WER 사용 시 조사·어미 변화로 인해 오류율이 과도하게 높게 측정됨.
음절/자소 단위 CER이 실제 전사 품질을 더 정확하게 반영.

### CER과 LLM 요약 품질의 관계

| CER 범위 | 요약 품질 영향 | 근거 |
|---|---|---|
| ≤ 5% | 영향 미미 | LLM의 언어 이해 능력으로 대부분의 오류 보정 가능 |
| 5~10% | 경미한 영향 | 일반적 오류는 LLM이 보정, 고유명사 오류 시 사실 정합성 저하 시작 |
| 10~15% | 유의미한 영향 | 사실 정합성(factual consistency) 저하, 핵심 용어 누락/왜곡 발생 |
| > 15% | 심각한 영향 | 요약 신뢰도 대폭 하락, 전사 교정 단계 필수 |

> - ASR 오류가 요약 품질에 미치는 영향은 오류 유형에 따라 불균형적임 (Shapira et al. 2025)
> - 고유명사·전문용어 오류는 WER/CER 수치 대비 요약에 과도한 영향을 미침 (Kroll & Kraus 2024)
> - 표준 WER/CER만으로는 다운스트림 태스크 성능을 정확히 예측하기 어려움 (Pulikodan et al. 2025)
> - LLM 기반 교정으로 10~20%의 WER 상대 개선이 가능하나 한계 존재 (arxiv:2310.11532)

### 공식 벤치마크 현황 및 한국어 리스크

**FunASR-MLT-Nano-2512**:
- 공식 벤치마크에 한국어 테스트셋 **미포함** (중국어, 영어, 인도네시아어, 태국어, 베트남어만 공개)
- Fun-ASR-Nano 산업 데이터셋 평균 WER: 16.72% (주로 중국어 기준)
- Fleurs 영어 WER: 5.49%, Fleurs 중국어 CER: 3.51%

**VibeVoice-ASR 9B**:
- 한국어 공식 벤치마크 **미공개**
- SFT(Supervised Fine-Tuning)가 **영어·중국어·코드스위칭에 집중** — 비주력 언어 성능 저하 가능 (논문 Limitations에 명시)
- tcpWER (화자 분리 포함): AMI_IHM 20.82%, AISHELL4 25.35% (영어/중국어 기준)
- 순수 WER은 16개 세팅 중 8개에서 최저 (Gemini-2.5/3-Pro 대비)

**Whisper large-v3** (오픈소스 베이스라인):
- FunASR 공식 벤치마크에서 직접 비교됨 (Fleurs-zh CER 4.71% vs FunASR-ML-Nano 3.51%)
- 한국어 공식 수치 미공개이나, 다국어 지원 범위가 넓어 합리적 베이스라인
- 로컬 구동 가능 (`device="mps"`)

**OpenAI Whisper API** (상용 베이스라인):
- 업계 표준 상용 STT, 한국어 지원
- 절대 품질 기준점으로 활용 — "상용 대비 몇 %p 차이인가"가 핵심 판단 근거
- 비용: $0.006/분 (벤치마크 규모에서 무시 가능)

### 벤치마크 코드 예시

```python
from datasets import load_dataset
from jiwer import cer
from openai import OpenAI
import whisper
import time

ds = load_dataset("jp1924/KsponSpeech", split="test[:500]")
client = OpenAI()
whisper_model = whisper.load_model("large-v3", device="mps")

ENGINES = ["funasr", "vibevoice", "whisper_local", "openai_api"]
results = {e: [] for e in ENGINES}

for sample in ds:
    audio_path = sample["audio"]["path"]
    reference = sample["text"]

    # 1. FunASR-MLT-Nano (실시간 후보)
    t0 = time.time()
    hyp = funasr_model.generate(input=[audio_path], cache={}, language="韩文")
    results["funasr"].append({
        "ref": reference, "hyp": hyp,
        "rtf": (time.time() - t0) / sample["audio"]["duration"]
    })

    # 2. VibeVoice-ASR (후처리 후보)
    t0 = time.time()
    hyp = vibevoice_model.transcribe(audio_path)
    results["vibevoice"].append({
        "ref": reference, "hyp": hyp,
        "rtf": (time.time() - t0) / sample["audio"]["duration"]
    })

    # 3. Whisper large-v3 (오픈소스 베이스라인)
    t0 = time.time()
    hyp = whisper_model.transcribe(audio_path, language="ko")["text"]
    results["whisper_local"].append({
        "ref": reference, "hyp": hyp,
        "rtf": (time.time() - t0) / sample["audio"]["duration"]
    })

    # 4. OpenAI Whisper API (상용 베이스라인)
    with open(audio_path, "rb") as f:
        hyp = client.audio.transcriptions.create(
            model="whisper-1", file=f, language="ko"
        ).text
    results["openai_api"].append({"ref": reference, "hyp": hyp})

# CER 비교 출력
for engine in ENGINES:
    engine_cer = cer(
        [r["ref"] for r in results[engine]],
        [r["hyp"] for r in results[engine]]
    )
    print(f"{engine:20s} CER: {engine_cer:.2%}")
```
