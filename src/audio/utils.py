"""오디오 유틸리티 — 포맷 변환, 리샘플링, dB 계산."""

from __future__ import annotations

import numpy as np


def bytes_to_frames(raw: bytes, sample_width: int, channels: int) -> np.ndarray:
    """PyAudio raw bytes를 float32 numpy 배열로 변환한다.

    Args:
        raw: PCM raw bytes (little-endian)
        sample_width: 샘플당 바이트 수 (예: 16-bit → 2)
        channels: 채널 수

    Returns:
        float32 배열, 범위 [-1.0, 1.0], shape (num_samples,)
    """
    if sample_width == 2:  # 16-bit
        dtype = np.int16
    elif sample_width == 4:  # 32-bit
        dtype = np.int32
    else:
        raise ValueError(f"Unsupported sample_width: {sample_width}")

    samples = np.frombuffer(raw, dtype=dtype)
    samples = samples.reshape(-1, channels)
    samples = samples.mean(axis=1) if channels > 1 else samples.squeeze(axis=1)

    max_val = float(np.iinfo(dtype).max)
    return samples.astype(np.float32) / max_val


def resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """오디오를 target_sr으로 리샘플링한다.

    Args:
        audio: float32 오디오 배열 (mono)
        orig_sr: 원본 샘플레이트
        target_sr: 목표 샘플레이트

    Returns:
        리샘플된 float32 배열
    """
    if orig_sr == target_sr:
        return audio.copy()

    orig_len = len(audio)
    target_len = int(orig_len * target_sr / orig_sr)
    if target_len <= 0:
        return np.array([], dtype=np.float32)

    indices = np.linspace(0, orig_len - 1, target_len, dtype=np.float32)
    return np.interp(indices, np.arange(orig_len), audio).astype(np.float32)


def has_sufficient_energy(
    pcm_bytes: bytes,
    sample_width: int = 2,
    channels: int = 1,
    threshold_db: float = -40.0,
) -> bool:
    """PCM 버퍼에 말소리 수준의 에너지가 있는지 확인한다.

    침묵/노이즈만 있을 때 ASR 할루시네이션을 줄이기 위한 게이트.
    0dB = full scale 기준, 말소리 -20~-10dB, 조용한 방 -50dB 이하.

    Args:
        pcm_bytes: 16-bit PCM raw bytes
        sample_width: 샘플당 바이트 (기본 2)
        channels: 채널 수 (기본 1)
        threshold_db: 이 dB 이상이면 음성 있다고 판단 (기본 -40)

    Returns:
        에너지가 threshold 이상이면 True
    """
    if len(pcm_bytes) < sample_width * channels:
        return False
    audio = bytes_to_frames(pcm_bytes, sample_width, channels)
    db = compute_db_rms(audio)
    return db >= threshold_db


def compute_db_rms(audio: np.ndarray, eps: float = 1e-10) -> float:
    """RMS 기반 dB를 계산한다. 0dB = full scale (1.0).

    Args:
        audio: float32 오디오 배열 (mono)
        eps: 무음 시 -inf 방지용 최소값

    Returns:
        dB 값 (무음에 가까우면 -inf에 가까운 음수)
    """
    if audio.size == 0:
        return float("-inf" if eps == 0 else -100.0)

    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    rms = max(rms, eps)
    return float(20.0 * np.log10(rms))
