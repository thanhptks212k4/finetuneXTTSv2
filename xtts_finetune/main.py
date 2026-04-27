"""
main.py - Entry point for XTTS v2 Vietnamese fine-tuning pipeline.

Usage:
    # Full training from scratch
    python -m xtts_finetune.main

    # Resume from patch 5
    python -m xtts_finetune.main --resume_patch 5

    # Custom config
    python -m xtts_finetune.main \
        --train_manifest ./data/train.jsonl \
        --val_manifest ./data/val.jsonl \
        --reference_audio ./data/reference.wav \
        --batch_size 4 \
        --epochs_per_patch 2

    # Inference only
    python -m xtts_finetune.main --inference_only \
        --text "Xin chào thế giới" \
        --reference_audio ./data/reference.wav
"""

import os
import sys
import argparse
import logging

import torch

# Add parent directory to path if running as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xtts_finetune.config import TrainingConfig
from xtts_finetune.utils import (
    get_logger,
    set_seed,
    free_memory,
    log_gpu_memory,
    find_latest_checkpoint,
    find_best_checkpoint,
)
from xtts_finetune.dataset import (
    load_manifest,
    validate_and_filter,
)
from xtts_finetune.model_loader import (
    download_base_model,
    load_xtts_model,
    configure_trainable_params,
    enable_gradient_checkpointing,
    apply_lora,
    extract_speaker_embedding,
    load_checkpoint,
)
from xtts_finetune.trainer import XTTSTrainer
from xtts_finetune.inference import run_inference


# ─── Argument Parsing ─────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="XTTS v2 Vietnamese Fine-Tuning Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Paths
    parser.add_argument("--hf_repo_id", type=str, default="anhnh2002/vnTTS")
    parser.add_argument("--base_model_dir", type=str, default="./base_model")
    parser.add_argument("--train_manifest", type=str, default="./data/train.jsonl")
    parser.add_argument("--val_manifest", type=str, default="./data/val.jsonl")
    parser.add_argument("--reference_audio", type=str, default="./data/reference.wav")
    parser.add_argument("--output_dir", type=str, default="./output")

    # Training
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs_per_patch", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--patch_size", type=int, default=5000)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--resume_patch", type=int, default=0)
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to specific checkpoint to resume from")

    # Flags
    parser.add_argument("--freeze_encoder", action="store_true", default=True)
    parser.add_argument("--no_freeze_encoder", dest="freeze_encoder", action="store_false")
    parser.add_argument("--use_fp16", action="store_true", default=True)
    parser.add_argument("--no_fp16", dest="use_fp16", action="store_false")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--use_lora", action="store_true", default=False)
    parser.add_argument("--speaker_mode", type=str, default="single",
                        choices=["single", "multi"])
    parser.add_argument("--zip_checkpoints", action="store_true", default=False)

    # Inference mode
    parser.add_argument("--inference_only", action="store_true", default=False)
    parser.add_argument("--text", type=str, default=None,
                        help="Text for inference mode")
    parser.add_argument("--inference_output", type=str,
                        default="./output/inference_output.wav")

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)

    return parser.parse_args()


# ─── Build Config from Args ───────────────────────────────────────────────────

def build_config(args: argparse.Namespace) -> TrainingConfig:
    """Map CLI arguments to TrainingConfig."""
    return TrainingConfig(
        hf_repo_id=args.hf_repo_id,
        base_model_dir=args.base_model_dir,
        train_manifest=args.train_manifest,
        val_manifest=args.val_manifest,
        reference_audio=args.reference_audio,
        output_dir=args.output_dir,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        sample_dir=os.path.join(args.output_dir, "samples"),
        log_dir=os.path.join(args.output_dir, "logs"),
        batch_size=args.batch_size,
        epochs_per_patch=args.epochs_per_patch,
        learning_rate=args.learning_rate,
        patch_size=args.patch_size,
        grad_accum_steps=args.grad_accum_steps,
        resume_patch=args.resume_patch,
        freeze_encoder=args.freeze_encoder,
        use_fp16=args.use_fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        use_lora=args.use_lora,
        speaker_mode=args.speaker_mode,
        zip_checkpoints=args.zip_checkpoints,
        seed=args.seed,
        num_workers=args.num_workers,
    )


# ─── System Info ──────────────────────────────────────────────────────────────

def print_system_info(logger: logging.Logger):
    """Print GPU/CPU info for debugging."""
    logger.info("=" * 60)
    logger.info("XTTS v2 Vietnamese Fine-Tuning Pipeline")
    logger.info("=" * 60)
    logger.info(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(
                f"GPU {i}: {props.name} | "
                f"VRAM: {props.total_memory / 1024**3:.1f} GB"
            )
    else:
        logger.warning("No GPU detected — training will be slow on CPU")

    logger.info("=" * 60)


# ─── Main Training Pipeline ───────────────────────────────────────────────────

def run_training(config: TrainingConfig, logger: logging.Logger):
    """Execute the full training pipeline."""

    # ── Step 1: Download base model ───────────────────────────────────────────
    logger.info("Step 1/7: Downloading base model...")
    download_base_model(config, logger)

    # ── Step 2: Load and validate datasets ───────────────────────────────────
    logger.info("Step 2/7: Loading and validating datasets...")
    train_raw = load_manifest(config.train_manifest, logger,
                              audio_root_remap=config.audio_root_remap)
    val_raw   = load_manifest(config.val_manifest, logger,
                              audio_root_remap=config.audio_root_remap)

    train_samples = validate_and_filter(train_raw, config, logger)
    val_samples = validate_and_filter(val_raw, config, logger)

    if len(train_samples) == 0:
        logger.error("No valid training samples found. Aborting.")
        return

    if len(val_samples) == 0:
        logger.warning("No valid validation samples. Using first 100 training samples.")
        val_samples = train_samples[:100]
        train_samples = train_samples[100:]

    logger.info(
        f"Dataset ready: {len(train_samples)} train | {len(val_samples)} val"
    )

    # ── Step 3: Load XTTS model ───────────────────────────────────────────────
    logger.info("Step 3/7: Loading XTTS model...")

    # Determine if we're resuming from a checkpoint
    resume_ckpt = None
    if config.resume_patch > 0:
        resume_ckpt = find_latest_checkpoint(config.checkpoint_dir)
        if resume_ckpt:
            logger.info(f"Will resume from checkpoint: {resume_ckpt}")
        else:
            logger.warning("Resume requested but no checkpoint found. Starting fresh.")

    model, xtts_config = load_xtts_model(config, logger)

    # ── Step 4: Configure trainable parameters ────────────────────────────────
    logger.info("Step 4/7: Configuring trainable parameters...")
    model = configure_trainable_params(model, config, logger)

    # Optional: gradient checkpointing
    if config.gradient_checkpointing:
        enable_gradient_checkpointing(model, logger)

    # Optional: LoRA
    if config.use_lora:
        logger.info("Applying LoRA...")
        model = apply_lora(model, config, logger)

    # ── Step 5: Extract speaker embedding ────────────────────────────────────
    logger.info("Step 5/7: Extracting speaker embedding...")
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    speaker_embedding = None

    if config.speaker_mode == "single":
        if os.path.isfile(config.reference_audio):
            speaker_embedding = extract_speaker_embedding(
                model, xtts_config, config.reference_audio, device, logger
            )
        else:
            logger.warning(
                f"Reference audio not found: {config.reference_audio}. "
                "Speaker embedding will be None (model may use default)."
            )
    else:
        logger.info("Multi-speaker mode: embeddings will be extracted per batch")

    # ── Step 6: Resume checkpoint if needed ──────────────────────────────────
    if resume_ckpt:
        logger.info("Step 6/7: Loading resume checkpoint...")
        # Build a temporary optimizer to load state into
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        temp_optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)
        meta = load_checkpoint(resume_ckpt, model, temp_optimizer, device=str(device), logger=logger)
        del temp_optimizer
        free_memory()
    else:
        logger.info("Step 6/7: Starting from base model (no resume checkpoint)")

    log_gpu_memory(logger, "after model load")

    # ── Step 7: Train ─────────────────────────────────────────────────────────
    logger.info("Step 7/7: Starting training...")
    trainer = XTTSTrainer(
        model=model,
        xtts_config=xtts_config,
        config=config,
        speaker_embedding=speaker_embedding,
    )

    # If resuming, restore global step
    if resume_ckpt:
        trainer.global_step = meta.get("step", 0)
        logger.info(f"Resuming from global step: {trainer.global_step}")

    final_metrics = trainer.train(train_samples, val_samples)

    logger.info(
        f"\nTraining complete!\n"
        f"  Best val loss: {trainer.best_val_loss:.4f}\n"
        f"  Final val MCD: {final_metrics.get('val_mcd', 'N/A'):.2f} dB\n"
        f"  Checkpoints:   {config.checkpoint_dir}\n"
        f"  Samples:       {config.sample_dir}\n"
    )

    # Cleanup
    del model, trainer
    free_memory()


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    config = build_config(args)
    logger = get_logger("main", config.log_dir)

    set_seed(config.seed)
    print_system_info(logger)

    if args.inference_only:
        # ── Inference mode ────────────────────────────────────────────────────
        if not args.text:
            logger.error("--text is required for inference mode")
            sys.exit(1)

        run_inference(
            text=args.text,
            reference_audio=args.reference_audio,
            output_path=args.inference_output,
            config=config,
            logger=logger,
        )
    else:
        # ── Training mode ─────────────────────────────────────────────────────
        run_training(config, logger)


if __name__ == "__main__":
    main()
