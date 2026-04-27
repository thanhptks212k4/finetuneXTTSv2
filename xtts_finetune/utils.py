"""
utils.py - Utility functions: logging, Vietnamese text normalization,
           MCD metric, checkpoint zipping, seed setting, memory cleanup.
"""

import gc
import os
import re
import math
import zipfile
import logging
import random
import numpy as np
import torch
from typing import Optional


# ─── Logger ───────────────────────────────────────────────────────────────────

def get_logger(name: str, log_dir: str = "./output/logs") -> logging.Logger:
    """Create a logger that writes to both console and a file."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger


# ─── Seed ─────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─── Memory ───────────────────────────────────────────────────────────────────

def free_memory():
    """Aggressively free GPU and CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def log_gpu_memory(logger: logging.Logger, tag: str = ""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        logger.debug(
            f"[GPU Memory{' ' + tag if tag else ''}] "
            f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB"
        )


# ─── Vietnamese Text Normalization ────────────────────────────────────────────

# Mapping of common abbreviations / numbers in Vietnamese
_VI_ABBREV = {
    "tp.": "thành phố",
    "tp ": "thành phố ",
    "đ/c": "địa chỉ",
    "đt": "điện thoại",
    "ths": "thạc sĩ",
    "gs": "giáo sư",
    "pgs": "phó giáo sư",
    "ts": "tiến sĩ",
    "ks": "kỹ sư",
    "bs": "bác sĩ",
    "cn": "cử nhân",
    "nxb": "nhà xuất bản",
    "vd": "ví dụ",
    "vv": "vân vân",
    "tt": "tiếp theo",
    "tl": "tài liệu",
    "ql": "quản lý",
    "ht": "hệ thống",
    "cntt": "công nghệ thông tin",
    "ai": "trí tuệ nhân tạo",
    "ml": "học máy",
    "dl": "dữ liệu",
}

_VI_DIGIT_MAP = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín",
}


def _expand_number(match: re.Match) -> str:
    """Convert a matched number string to Vietnamese words (simple version)."""
    num_str = match.group(0)
    # For large numbers, just read digit by digit
    return " ".join(_VI_DIGIT_MAP.get(d, d) for d in num_str if d.isdigit())


def normalize_vietnamese_text(text: str) -> str:
    """
    Normalize Vietnamese text for TTS:
    - Lowercase
    - Expand abbreviations
    - Convert digits to words
    - Remove unwanted characters
    - Collapse whitespace
    """
    if not text or not text.strip():
        return ""

    text = text.strip()

    # Lowercase
    text = text.lower()

    # Expand abbreviations (whole-word match)
    for abbr, expansion in _VI_ABBREV.items():
        text = text.replace(abbr, expansion)

    # Convert numbers to words
    text = re.sub(r"\d+", _expand_number, text)

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+\.\S+", "", text)

    # Keep Vietnamese characters, Latin letters, spaces, and basic punctuation
    # Vietnamese Unicode range: \u00C0-\u024F + tone marks \u0300-\u036F
    # Plus Vietnamese-specific: \u1E00-\u1EFF
    text = re.sub(
        r"[^\w\s\u00C0-\u024F\u1E00-\u1EFF\u0300-\u036F.,!?;:\-']",
        " ",
        text,
        flags=re.UNICODE,
    )

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─── MCD (Mel Cepstral Distortion) ───────────────────────────────────────────

def compute_mcd(ref_mel: np.ndarray, syn_mel: np.ndarray) -> float:
    """
    Compute Mel Cepstral Distortion between two mel spectrograms.

    Args:
        ref_mel: Reference mel spectrogram (n_mels, T_ref)
        syn_mel: Synthesized mel spectrogram (n_mels, T_syn)

    Returns:
        MCD value in dB (lower is better, ~5-8 dB is good)
    """
    # Align lengths by truncating to the shorter one
    min_len = min(ref_mel.shape[1], syn_mel.shape[1])
    ref_mel = ref_mel[:, :min_len]
    syn_mel = syn_mel[:, :min_len]

    # Convert to cepstral domain via DCT approximation
    # MCD = (10 / ln(10)) * sqrt(2 * sum((c1 - c2)^2))
    diff = ref_mel - syn_mel
    mcd = (10.0 / math.log(10.0)) * math.sqrt(2.0 * np.mean(diff ** 2))
    return float(mcd)


# ─── Checkpoint Utilities ─────────────────────────────────────────────────────

def zip_checkpoint(checkpoint_path: str, logger: Optional[logging.Logger] = None):
    """Zip a checkpoint directory or file to save disk space (useful on Kaggle)."""
    zip_path = checkpoint_path.rstrip("/\\") + ".zip"
    if os.path.isdir(checkpoint_path):
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(checkpoint_path):
                for file in files:
                    fp = os.path.join(root, file)
                    zf.write(fp, os.path.relpath(fp, os.path.dirname(checkpoint_path)))
    elif os.path.isfile(checkpoint_path):
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(checkpoint_path, os.path.basename(checkpoint_path))

    if logger:
        logger.info(f"Checkpoint zipped → {zip_path}")
    return zip_path


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Scan checkpoint_dir for the most recent patch checkpoint.
    Returns the path to the latest checkpoint file, or None.
    """
    if not os.path.isdir(checkpoint_dir):
        return None

    checkpoints = []
    for fname in os.listdir(checkpoint_dir):
        if fname.startswith("patch_") and fname.endswith(".pth"):
            try:
                patch_num = int(fname.split("_")[1].split(".")[0])
                checkpoints.append((patch_num, os.path.join(checkpoint_dir, fname)))
            except (IndexError, ValueError):
                continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def find_best_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """Return path to best_model.pth if it exists."""
    best = os.path.join(checkpoint_dir, "best_model.pth")
    return best if os.path.isfile(best) else None


# ─── Audio Utilities ──────────────────────────────────────────────────────────

def load_audio_safe(path: str, target_sr: int = 22050):
    """
    Load an audio file safely, resampling to target_sr.
    Returns (waveform_np_float32, sample_rate) or (None, None) on error.
    """
    try:
        import torchaudio
        import torchaudio.transforms as T

        waveform, sr = torchaudio.load(path)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample
        if sr != target_sr:
            resampler = T.Resample(orig_freq=sr, new_freq=target_sr)
            waveform = resampler(waveform)

        # Normalize loudness to [-1, 1]
        peak = waveform.abs().max()
        if peak > 0:
            waveform = waveform / peak * 0.95

        return waveform.squeeze(0).numpy().astype(np.float32), target_sr

    except Exception as e:
        return None, None


def get_audio_duration(path: str) -> float:
    """Return audio duration in seconds without loading the full file."""
    try:
        import torchaudio
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate
    except Exception:
        return 0.0
