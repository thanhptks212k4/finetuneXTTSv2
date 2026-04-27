"""
dataset.py - Dataset loading, validation, filtering, and patch management
             for XTTS v2 Vietnamese fine-tuning.

Manifest format (JSONL):
    {"audio": "path/to/audio.wav", "text": "transcription"}
"""

import os
import json
import math
import random
import logging
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .utils import (
    normalize_vietnamese_text,
    get_audio_duration,
    load_audio_safe,
    get_logger,
    free_memory,
)
from .config import TrainingConfig


# ─── Manifest Loading & Validation ────────────────────────────────────────────

def resolve_manifest_item(item: dict, audio_root_remap: dict = None) -> dict:
    """
    Normalize a raw manifest item into a standard dict with keys:
        'audio'      — absolute path to wav file
        'text'       — transcript string
        'speaker_id' — optional speaker id (default: 'default')

    Handles multiple field name conventions:
        audio : 'audio', 'audio_filepath', 'audio_path', 'path'
        text  : 'text', 'transcript', 'sentence', 'transcription'

    audio_root_remap: dict mapping old path prefix → new path prefix.
        Example: {"/kaggle/input/data-speech-to-text":
                  "/kaggle/input/datasets/tinthnhphm21022004/data-speech-to-text"}
    """
    # Unwrap nested 'root' key (some NeMo-style manifests)
    if "root" in item and isinstance(item["root"], dict):
        item = item["root"]

    # Resolve audio path
    audio = (
        item.get("audio")
        or item.get("audio_filepath")
        or item.get("audio_path")
        or item.get("path")
    )

    # Resolve text
    text = (
        item.get("text")
        or item.get("transcript")
        or item.get("sentence")
        or item.get("transcription")
    )

    # Apply path remapping (fix broken absolute paths in manifests)
    if audio and audio_root_remap:
        for old_prefix, new_prefix in audio_root_remap.items():
            if audio.startswith(old_prefix):
                audio = new_prefix + audio[len(old_prefix):]
                break

    return {
        "audio":      audio or "",
        "text":       (text or "").strip(),
        "speaker_id": item.get("speaker_id", item.get("speaker", "default")),
    }


def load_manifest(
    path: str,
    logger: logging.Logger,
    audio_root_remap: dict = None,
) -> List[Dict]:
    """
    Load a JSONL manifest file.

    Supports multiple field name conventions (see resolve_manifest_item).
    audio_root_remap: optional dict to fix broken audio paths in the manifest.
        Example:
            {"/kaggle/input/data-speech-to-text":
             "/kaggle/input/datasets/tinthnhphm21022004/data-speech-to-text"}
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Manifest not found: {path}")

    samples = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_no}: JSON parse error — {e}")
                skipped += 1
                continue

            item = resolve_manifest_item(raw, audio_root_remap)

            if not item["audio"] or not item["text"]:
                skipped += 1
                continue

            samples.append(item)

    logger.info(f"Loaded {len(samples)} samples from {path} ({skipped} skipped)")
    return samples


def validate_and_filter(
    samples: List[Dict],
    config: TrainingConfig,
    logger: logging.Logger,
) -> List[Dict]:
    """
    Validate each sample:
    - Audio file must exist
    - Duration must be within [min_audio_length, max_audio_length]
    - Text must be non-empty after normalization
    Returns filtered + normalized list.
    """
    valid = []
    stats = {"missing_file": 0, "too_short": 0, "too_long": 0, "empty_text": 0}

    for item in samples:
        audio_path = item.get("audio", "")

        # Check file existence
        if not audio_path or not os.path.isfile(audio_path):
            stats["missing_file"] += 1
            continue

        # Check duration (fast — no full load)
        duration = get_audio_duration(audio_path)
        if duration < config.min_audio_length:
            stats["too_short"] += 1
            continue
        if duration > config.max_audio_length:
            stats["too_long"] += 1
            continue

        # Normalize text
        text = normalize_vietnamese_text(item.get("text", ""))
        if not text:
            stats["empty_text"] += 1
            continue

        valid.append({
            "audio":      audio_path,
            "text":       text,
            "duration":   duration,
            "speaker_id": item.get("speaker_id", "default"),
        })

    logger.info(
        f"Validation complete: {len(valid)} valid | "
        f"missing={stats['missing_file']} | "
        f"too_short={stats['too_short']} | "
        f"too_long={stats['too_long']} | "
        f"empty_text={stats['empty_text']}"
    )
    return valid


# ─── Patch Management ─────────────────────────────────────────────────────────

def split_into_patches(
    samples: List[Dict],
    patch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> List[List[Dict]]:
    """
    Split the full sample list into patches of `patch_size`.
    Optionally shuffle before splitting.
    """
    if shuffle:
        rng = random.Random(seed)
        samples = samples.copy()
        rng.shuffle(samples)

    n_patches = math.ceil(len(samples) / patch_size)
    patches = []
    for i in range(n_patches):
        start = i * patch_size
        end = min(start + patch_size, len(samples))
        patches.append(samples[start:end])

    return patches


# ─── PyTorch Dataset ──────────────────────────────────────────────────────────

class XTTSDataset(Dataset):
    """
    PyTorch Dataset for a single patch of XTTS training data.

    Each item returns:
        {
            "audio":      np.ndarray (float32, shape [T])
            "text":       str
            "speaker_id": str
            "duration":   float
        }
    Audio is loaded lazily on __getitem__ to keep RAM usage low.
    """

    def __init__(
        self,
        samples: List[Dict],
        config: TrainingConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.samples = samples
        self.config = config
        self.logger = logger or get_logger("XTTSDataset")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Optional[Dict]:
        item = self.samples[idx]
        audio, sr = load_audio_safe(item["audio"], target_sr=self.config.sample_rate)

        if audio is None:
            # Return None; collate_fn will skip this sample
            self.logger.warning(f"Failed to load audio: {item['audio']}")
            return None

        return {
            "audio": audio,          # np.ndarray float32
            "text": item["text"],
            "speaker_id": item["speaker_id"],
            "duration": item["duration"],
            "audio_path": item["audio"],
        }


def collate_fn(batch: List[Optional[Dict]]) -> Optional[Dict]:
    """
    Custom collate: skip None items, pad audio to same length.
    Returns None if the entire batch is invalid.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # Pad audio sequences to max length in batch
    max_len = max(b["audio"].shape[0] for b in batch)
    audio_padded = np.zeros((len(batch), max_len), dtype=np.float32)
    audio_lengths = []

    for i, b in enumerate(batch):
        length = b["audio"].shape[0]
        audio_padded[i, :length] = b["audio"]
        audio_lengths.append(length)

    return {
        "audio": torch.from_numpy(audio_padded),          # [B, T]
        "audio_lengths": torch.tensor(audio_lengths, dtype=torch.long),
        "text": [b["text"] for b in batch],
        "speaker_id": [b["speaker_id"] for b in batch],
        "audio_path": [b["audio_path"] for b in batch],
    }


def build_dataloader(
    samples: List[Dict],
    config: TrainingConfig,
    shuffle: bool = True,
    logger: Optional[logging.Logger] = None,
) -> DataLoader:
    """Build a DataLoader for a given list of samples (one patch)."""
    dataset = XTTSDataset(samples, config, logger)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        persistent_workers=config.num_workers > 0,
    )


# ─── Validation Dataset ───────────────────────────────────────────────────────

class ValidationDataset(Dataset):
    """
    Small validation dataset — loads everything into memory upfront
    since val sets are typically small.
    """

    def __init__(
        self,
        samples: List[Dict],
        config: TrainingConfig,
        logger: Optional[logging.Logger] = None,
    ):
        self.config = config
        self.logger = logger or get_logger("ValidationDataset")
        self.items = self._load_all(samples)

    def _load_all(self, samples: List[Dict]) -> List[Dict]:
        loaded = []
        for item in samples:
            audio, sr = load_audio_safe(item["audio"], target_sr=self.config.sample_rate)
            if audio is None:
                continue
            loaded.append({
                "audio": audio,
                "text": item["text"],
                "speaker_id": item["speaker_id"],
                "audio_path": item["audio"],
            })
        self.logger.info(f"Validation dataset: {len(loaded)} samples loaded into memory")
        return loaded

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def build_val_dataloader(
    samples: List[Dict],
    config: TrainingConfig,
    logger: Optional[logging.Logger] = None,
) -> DataLoader:
    """Build a DataLoader for the validation set."""
    dataset = ValidationDataset(samples, config, logger)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,          # keep val simple
        collate_fn=collate_fn,
        pin_memory=False,
    )
