"""
inference.py - Post-training inference: load best checkpoint and generate audio.

Usage:
    python -m xtts_finetune.inference \
        --text "Xin chào, đây là giọng nói tổng hợp." \
        --reference_audio ./data/reference.wav \
        --output ./output/inference_test.wav
"""

import os
import argparse
import logging
from typing import Optional
import numpy as np
import torch
import torchaudio

from .config import TrainingConfig
from .model_loader import (
    download_base_model,
    load_xtts_model,
    configure_trainable_params,
    extract_speaker_embedding,
    load_checkpoint,
)
from .utils import get_logger, free_memory, normalize_vietnamese_text


def run_inference(
    text: str,
    reference_audio: str,
    output_path: str,
    checkpoint_path: Optional[str] = None,
    config: Optional[TrainingConfig] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Load the best checkpoint and synthesize speech from text.

    Args:
        text:             Vietnamese text to synthesize
        reference_audio:  Path to reference audio for speaker conditioning
        output_path:      Where to save the output .wav file
        checkpoint_path:  Path to .pth checkpoint (uses best_model.pth if None)
        config:           TrainingConfig (uses defaults if None)
        logger:           Logger instance
    """
    if config is None:
        config = TrainingConfig()
    if logger is None:
        logger = get_logger("inference", config.log_dir)

    # ── Resolve checkpoint ────────────────────────────────────────────────────
    if checkpoint_path is None:
        from .utils import find_best_checkpoint, find_latest_checkpoint
        checkpoint_path = find_best_checkpoint(config.checkpoint_dir)
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
        if checkpoint_path is None:
            logger.warning("No fine-tuned checkpoint found. Using base model.")

    # ── Download base model if needed ─────────────────────────────────────────
    download_base_model(config, logger)

    # ── Load model ────────────────────────────────────────────────────────────
    model, xtts_config = load_xtts_model(config, logger, checkpoint_path=checkpoint_path)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")

    # ── Extract speaker embedding ─────────────────────────────────────────────
    logger.info(f"Extracting speaker embedding from: {reference_audio}")
    speaker_embedding = extract_speaker_embedding(
        model, xtts_config, reference_audio, device, logger
    )

    if speaker_embedding is None:
        logger.warning("Speaker embedding extraction failed. Using zero embedding.")
        speaker_embedding = torch.zeros(1, 512, device=device)

    # ── Normalize text ────────────────────────────────────────────────────────
    text_normalized = normalize_vietnamese_text(text)
    logger.info(f"Input text: '{text}'")
    logger.info(f"Normalized: '{text_normalized}'")

    # ── Inference ─────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        if hasattr(model, "inference"):
            out = model.inference(
                text=text_normalized,
                language="vi",
                gpt_cond_latent=speaker_embedding,
                speaker_embedding=speaker_embedding,
                temperature=0.7,
                length_penalty=1.0,
                repetition_penalty=2.0,
                top_k=50,
                top_p=0.85,
                enable_text_splitting=True,
            )
            wav = out.get("wav", None)
        else:
            logger.error("Model does not have inference() method")
            return

    if wav is None:
        logger.error("Inference returned no audio")
        return

    # ── Save output ───────────────────────────────────────────────────────────
    if isinstance(wav, torch.Tensor):
        wav_np = wav.cpu().numpy()
    else:
        wav_np = np.array(wav, dtype=np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    wav_tensor = torch.from_numpy(wav_np).unsqueeze(0)
    torchaudio.save(output_path, wav_tensor, config.sample_rate)
    logger.info(f"Audio saved: {output_path} ({wav_np.shape[0] / config.sample_rate:.2f}s)")

    free_memory()
    return output_path


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="XTTS v2 Vietnamese TTS Inference"
    )
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument(
        "--reference_audio", type=str, required=True,
        help="Reference audio for speaker conditioning"
    )
    parser.add_argument(
        "--output", type=str, default="./output/inference_output.wav",
        help="Output .wav file path"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to .pth checkpoint (default: best_model.pth)"
    )
    parser.add_argument(
        "--base_model_dir", type=str, default="./base_model",
        help="Directory containing XTTS base model files"
    )
    args = parser.parse_args()

    config = TrainingConfig(
        base_model_dir=args.base_model_dir,
        reference_audio=args.reference_audio,
    )
    logger = get_logger("inference", config.log_dir)

    run_inference(
        text=args.text,
        reference_audio=args.reference_audio,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        config=config,
        logger=logger,
    )


if __name__ == "__main__":
    main()
