# 프로젝트 요구사항 정의서 (Requirement.md)

## 1. 프로젝트 목적

맥북 환경에서 실시간으로 음성을 전사(ASR)하고, 전사된 텍스트를 바탕으로
강의 노트(요약)를 생성하는 AI 네이티브 애플리케이션 개발.

## 2. 핵심 기능

### 2.1 실시간 음성 인식 (Real-time ASR)

- 마이크 입력을 통한 저지연 스트리밍 전사
- 목표 지연시간: 500ms 이하
- 전사 결과를 터미널(CLI)에 실시간 출력

### 2.2 다국어 지원

- 필수: 한국어(Korean), 영어(English), 일본어(Japanese)
- Phase 1에서 3개 언어 모두 PoC 검증 수행
- FunASR-MLT-Nano는 단일 모델로 31개 언어를 지원하므로 언어 전환이 간편

### 2.3 고품질 후처리 (Post-processing)

- 녹음 종료 후 전체 오디오에 대한 고정밀 전사
- 화자 분리(Speaker Diarization) 및 타임스탬프 부여
- 커스텀 핫워드(전문 용어, 고유명사) 반영

### 2.4 강의 요약 생성

- 전사된 전체 텍스트를 LLM을 통해 구조화된 노트로 요약
- 핵심 포인트, 주제별 분류, 액션 아이템 추출

### 2.5 단계적 개발

1. **Phase 1 — CLI 프로토타입**: 터미널 기반 실시간 전사 + 요약
2. **Phase 2 — macOS Native**: SwiftUI 앱으로 고도화

## 3. 기술 스택 (Technical Stack)

### 3.1 ASR 엔진 및 모델 (2단계 파이프라인)

| 용도 | 엔진 | 모델 | 비고 |
|---|---|---|---|
| **실시간 ASR (메인)** | `funasr` | FunASR-MLT-Nano-2512 (800M) | 31개 언어, KO/EN/JA 지원, VAD+pseudo-streaming |
| **실시간 ASR (대안)** | `sherpa-onnx` | Streaming Zipformer | KO/EN만 스트리밍 가능 (JA 모델 없음) |
| **실시간 ASR (대안)** | `sherpa-onnx` | SenseVoice | KO/EN/JA/ZH/Cantonese 단일 모델, offline |
| **고품질 후처리 ASR** | Transformers / vLLM | VibeVoice-ASR (9B) | 배치 전용, 화자 분리 + 타임스탬프, 51개 언어 |

#### 모델 선정 근거

- **FunASR-MLT-Nano-2512 (메인 후보)**: Tongyi Lab의 end-to-end ASR 대형 모델.
  800M 파라미터, 수십만 시간 학습 데이터, 31개 언어 지원(KO/EN/JA 포함).
  `device="mps"` 로 Apple Silicon에서 CUDA 없이 구동 가능.
  VAD(`fsmn-vad`) 조합으로 pseudo-streaming 구현.
  핫워드 지원으로 도메인 특화 정확도 향상 가능.
  Open-source 벤치마크에서 WER 기준 상위 성능 (Fun-ASR-Nano 평균 WER 16.72%).
- **Streaming Zipformer (대안)**: sherpa-onnx 기반 진정한 스트리밍 처리 가능.
  한국어(`sherpa-onnx-streaming-zipformer-korean-2024-06-16`),
  영어(`sherpa-onnx-streaming-zipformer-en-2023-06-26`) 모델 존재.
  일본어 스트리밍 모델은 존재하지 않음.
  Python/Swift 바인딩 지원으로 Phase 2 전환에 유리.
- **SenseVoice (대안)**: sherpa-onnx에서 KO/EN/JA를 단일 모델로 지원.
  VAD와 조합하여 pseudo-streaming 가능. Zipformer 대비 높은 정확도.
- **VibeVoice-ASR (후처리)**: 실시간 스트리밍 불가
  (60분 오디오를 단일 패스로 배치 처리하는 구조).
  9B 파라미터, 51개 언어 지원, 코드 스위칭 가능.
  ASR + 화자 분리 + 타임스탬프를 동시 수행하여 고품질 후처리 단계에 적합.
  **한계**: SFT가 영어·중국어·코드스위칭에 집중 — 한국어 등 비주력 언어는 성능 저하 가능
  (논문 Limitations 섹션에 multilingual forgetting 리스크 명시).
  한국어 CER은 M4.5에서 자체 측정 필요.

### 3.2 LLM 프로바이더 (요약 엔진)

| 프로바이더 | 활성화 조건 | 비고 |
|---|---|---|
| **OpenAI** (GPT-4o 등) | API 토큰 입력 시 | 클라우드 |
| **Google Gemini** (2.5 Pro 등) | API 토큰 입력 시 | 클라우드 |
| **Anthropic Claude** (Sonnet 등) | API 토큰 입력 시 | 클라우드 |
| **Ollama** (llama3, mistral 등) | 토큰 불필요, 로컬 구동 | 오프라인 대응 |

- 프로바이더 추상화 레이어를 두어 동적으로 활성화/전환
- 사용자가 API 키를 입력한 프로바이더만 활성화

### 3.3 오디오 처리

- **Phase 1 (Python CLI)**: `PyAudio` (마이크 캡처)
- **Phase 2 (Native)**: `AVFoundation` (macOS 오디오 프레임워크)

### 3.4 개발 환경

- **Phase 1**: Python 3.10+
- **Phase 2**: Swift 5.9+ / SwiftUI
- **IDE**: Cursor
- **타겟 OS**: macOS 14+ (Sonoma)
- **conda 환경**: `aln`

## 4. 시스템 아키텍처 (개요)

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────┐
│  Microphone  │───▶│  VAD (fsmn-vad)   │───▶│  실시간 전사  │
│  (PyAudio)   │    │  + FunASR-MLT     │    │  (CLI 출력)   │
│              │    │  -Nano-2512       │    │              │
└─────────────┘    └──────────────────┘    └──────┬───────┘
                                                   │
                          ┌────────────────────────┘
                          ▼
┌─────────────┐    ┌──────────────────┐    ┌──────────────┐
│  저장된 오디오 │───▶│  VibeVoice-ASR    │───▶│  고품질 전사   │
│  (.wav)      │    │  (9B, 배치처리)    │    │  + 화자 분리   │
└─────────────┘    └──────────────────┘    └──────┬───────┘
                                                   │
                                                   ▼
                                           ┌──────────────┐
                                           │  LLM 요약     │
                                           │  (노트 생성)   │
                                           └──────────────┘
```

## 5. 제약 및 가정

- **타겟 디바이스**: Apple Silicon (M-series) 맥북
  - Intel Mac 호환성은 우선순위 낮음 (VibeVoice-ASR 9B 모델 구동에 Apple Silicon 권장)
- **실시간 지연시간**: ASR 500ms 이하 목표
- **FunASR-MLT-Nano 구동**: `device="mps"` (Apple Silicon) 또는 `device="cpu"`
  - 800M 모델이므로 Apple Silicon 통합 메모리에서 충분히 구동 가능
- **VibeVoice-ASR 구동 요건**: 9B 모델, BF16 기준 ~18GB 메모리
  - Apple Silicon 통합 메모리 활용 또는 외부 GPU 서버 추론 고려
- **네트워크**: LLM 요약 단계에서 API 호출 필요 (오프라인 시 Ollama 대안)
- **오디오 포맷**: 16-bit, 단일 채널 (mono), 16kHz 샘플링 레이트 기본

## 6. Phase 1 마일스톤 (CLI 프로토타입)

| 단계 | 목표 | 완료 기준 |
|---|---|---|
| M0 | 프로젝트 기반 설정 | `config.yaml`, 로거, 메트릭스 유틸, 디렉토리 구조 생성 |
| M1 | 마이크 입력 캡처 | PyAudio로 실시간 오디오 스트림 확보 |
| M2 | 실시간 ASR PoC | FunASR-MLT-Nano + VAD로 터미널에 전사 출력 |
| M3 | 다국어 전사 검증 | 한국어(CER), 영어(WER), 일본어(WER) 전사 정확도 및 RTF 측정 |
| M4 | 고품질 후처리 ASR | VibeVoice-ASR로 녹음 파일 배치 전사 + 화자 분리 |
| M4.5 | 한국어 CER 벤치마크 | KsponSpeech/강의 데이터로 4개 엔진(FunASR, VibeVoice, Whisper, OpenAI) CER 비교 (Go/No-Go) |
| M5 | LLM 요약 | 전사 결과를 LLM에 전달하여 구조화된 노트 생성 |
| M6 | 통합 파이프라인 | M0~M5 연결, CLI에서 End-to-End 동작 |

## 7. 검증 항목 (PoC에서 확인)

### FunASR-MLT-Nano-2512 (메인)

- [ ] 맥북 Apple Silicon에서 `device="mps"` 구동 확인
- [ ] 한국어 전사 정확도 (CER) — 공식 벤치마크에 한국어 미포함, 자체 측정 필수
- [ ] 영어 전사 정확도 (WER)
- [ ] 일본어 전사 정확도 (WER)
- [ ] VAD + FunASR-MLT-Nano 파이프라인 지연시간 측정 (RTF)
- [ ] 핫워드 기능 검증

### 대안 모델 비교 (옵션)

- [ ] Streaming Zipformer 한국어/영어 모델 vs FunASR-MLT-Nano 품질/속도 비교
- [ ] SenseVoice vs FunASR-MLT-Nano 품질/속도 비교

### 한국어 CER 정량 벤치마크 (M4.5 — Go/No-Go)

- [ ] HuggingFace 데이터셋 접근: [KsponSpeech](https://huggingface.co/datasets/jp1924/KsponSpeech), [KoreanUniversityLectureData](https://huggingface.co/datasets/jp1924/KoreanUniversityLectureData)
- [ ] FunASR-MLT-Nano 한국어 CER 측정 (목표: ≤ 10%, 조건부 허용: ≤ 15%)
- [ ] VibeVoice-ASR 한국어 CER 측정 (FunASR 대비 5%p 이상 낮아야 2단계 파이프라인 가치 확인)
- [ ] Whisper large-v3 (로컬) 한국어 CER 측정 — 오픈소스 베이스라인
- [ ] OpenAI Whisper API 한국어 CER 측정 — 상용 베이스라인 (절대 품질 기준점)
- [ ] FunASR CER ≤ OpenAI CER + 3%p 이면 오픈소스 채택 확정
- [ ] CER > 10% 시 LLM 요약 품질 영향 평가 (사실 정합성, 고유명사 정확도)
- [ ] 두 엔진 모두 CER > 20% 시 모델 교체/파인튜닝 검토 (No-Go)

### 후처리 및 요약

- [ ] VibeVoice-ASR 9B 모델의 Apple Silicon 로컬 추론 가능 여부
- [ ] VibeVoice-ASR 한국어 성능 확인 (SFT가 영어·중국어 집중, 한국어 성능 저하 리스크)
- [ ] VibeVoice-ASR 화자 분리 정확도 (DER)
- [ ] 실시간 전사 ↔ VibeVoice 후처리 전사 간 품질 차이
- [ ] LLM 프로바이더별 요약 품질 비교
