"""
prepare_data.py - Helper script to build JSONL manifests from a raw dataset directory.

Supports two common layouts:

Layout A (flat):
    dataset/
        audio1.wav
        audio1.txt
        audio2.wav
        audio2.txt

Layout B (LJSpeech-style):
    dataset/
        wavs/
            audio1.wav
            audio2.wav
        metadata.csv   (format: filename|text  OR  filename|normalized_text|text)

Usage:
    # Layout A
    python -m xtts_finetune.prepare_data \
        --input_dir ./raw_data \
        --layout flat \
        --output_dir ./data \
        --val_ratio 0.05

    # Layout B (LJSpeech)
    python -m xtts_finetune.prepare_data \
        --input_dir ./raw_data \
        --layout ljspeech \
        --output_dir ./data \
        --val_ratio 0.05
"""

import os
import json
import random
import argparse
import logging
from typing import List, Dict, Tuple, Optional

from .utils import get_logger, normalize_vietnamese_text, get_audio_duration
from .config import TrainingConfig


def scan_flat_layout(input_dir: str, logger: logging.Logger) -> List[Dict]:
    """
    Scan a flat directory where each .wav has a matching .txt file.
    """
    samples = []
    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith(".wav"):
            continue
        base = os.path.splitext(fname)[0]
        txt_path = os.path.join(input_dir, base + ".txt")
        wav_path = os.path.join(input_dir, fname)

        if not os.path.isfile(txt_path):
            logger.warning(f"No transcript for: {fname}")
            continue

        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        samples.append({"audio": wav_path, "text": text})

    logger.info(f"Flat layout: found {len(samples)} pairs in {input_dir}")
    return samples


def scan_ljspeech_layout(input_dir: str, logger: logging.Logger) -> List[Dict]:
    """
    Scan an LJSpeech-style directory with wavs/ folder and metadata.csv.
    """
    wav_dir = os.path.join(input_dir, "wavs")
    metadata_path = os.path.join(input_dir, "metadata.csv")

    if not os.path.isdir(wav_dir):
        raise FileNotFoundError(f"wavs/ directory not found in: {input_dir}")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found in: {input_dir}")

    samples = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) < 2:
                continue

            fname = parts[0].strip()
            # LJSpeech format: id|normalized|raw  OR  id|text
            text = parts[-1].strip()  # use last column as text

            # Try with and without .wav extension
            wav_path = os.path.join(wav_dir, fname)
            if not os.path.isfile(wav_path):
                wav_path = os.path.join(wav_dir, fname + ".wav")
            if not os.path.isfile(wav_path):
                logger.warning(f"Audio not found: {fname}")
                continue

            samples.append({"audio": wav_path, "text": text})

    logger.info(f"LJSpeech layout: found {len(samples)} samples in {input_dir}")
    return samples


def split_train_val(
    samples: List[Dict],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Split samples into train/val sets."""
    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(len(shuffled) * val_ratio))
    val = shuffled[:n_val]
    train = shuffled[n_val:]
    return train, val


def write_manifest(samples: List[Dict], path: str, logger: logging.Logger):
    """Write samples to a JSONL manifest file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Manifest written: {path} ({len(samples)} samples)")


def prepare_data(
    input_dir: str,
    output_dir: str,
    layout: str = "flat",
    val_ratio: float = 0.05,
    normalize_text: bool = True,
    min_duration: float = 1.0,
    max_duration: float = 20.0,
    seed: int = 42,
    logger: Optional[logging.Logger] = None,
):
    """
    Full data preparation pipeline:
    1. Scan input directory
    2. Normalize text
    3. Filter by duration
    4. Split train/val
    5. Write JSONL manifests
    """
    if logger is None:
        logger = get_logger("prepare_data")

    os.makedirs(output_dir, exist_ok=True)

    # ── Scan ──────────────────────────────────────────────────────────────────
    if layout == "flat":
        samples = scan_flat_layout(input_dir, logger)
    elif layout == "ljspeech":
        samples = scan_ljspeech_layout(input_dir, logger)
    else:
        raise ValueError(f"Unknown layout: {layout}. Use 'flat' or 'ljspeech'")

    # ── Normalize text ────────────────────────────────────────────────────────
    if normalize_text:
        before = len(samples)
        normalized = []
        for item in samples:
            text = normalize_vietnamese_text(item["text"])
            if text:
                item["text"] = text
                normalized.append(item)
        samples = normalized
        logger.info(f"Text normalization: {before} → {len(samples)} samples")

    # ── Filter by duration ────────────────────────────────────────────────────
    logger.info("Filtering by audio duration (this may take a while)...")
    filtered = []
    for item in samples:
        dur = get_audio_duration(item["audio"])
        if min_duration <= dur <= max_duration:
            item["duration"] = dur
            filtered.append(item)
    logger.info(f"Duration filter: {len(samples)} → {len(filtered)} samples")
    samples = filtered

    # ── Split ─────────────────────────────────────────────────────────────────
    train_samples, val_samples = split_train_val(samples, val_ratio, seed)
    logger.info(f"Split: {len(train_samples)} train | {len(val_samples)} val")

    # ── Write manifests ───────────────────────────────────────────────────────
    write_manifest(train_samples, os.path.join(output_dir, "train.jsonl"), logger)
    write_manifest(val_samples, os.path.join(output_dir, "val.jsonl"), logger)

    # ── Stats ─────────────────────────────────────────────────────────────────
    if samples:
        durations = [s.get("duration", 0) for s in samples]
        total_hours = sum(durations) / 3600
        avg_dur = sum(durations) / len(durations)
        logger.info(
            f"\nDataset statistics:\n"
            f"  Total samples:  {len(samples)}\n"
            f"  Total duration: {total_hours:.2f} hours\n"
            f"  Avg duration:   {avg_dur:.2f}s\n"
            f"  Min duration:   {min(durations):.2f}s\n"
            f"  Max duration:   {max(durations):.2f}s\n"
        )

    return train_samples, val_samples


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare XTTS training data manifests")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--layout", type=str, default="flat",
                        choices=["flat", "ljspeech"])
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--min_duration", type=float, default=1.0)
    parser.add_argument("--max_duration", type=float, default=20.0)
    parser.add_argument("--no_normalize", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger = get_logger("prepare_data")
    prepare_data(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        layout=args.layout,
        val_ratio=args.val_ratio,
        normalize_text=not args.no_normalize,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        seed=args.seed,
        logger=logger,
    )


if __name__ == "__main__":
    main()
