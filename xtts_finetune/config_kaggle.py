"""
config_kaggle.py - Optimized configuration for Kaggle environment
Reduces disk usage and memory consumption
"""

from .config import TrainingConfig


def get_kaggle_config() -> TrainingConfig:
    """
    Get optimized config for Kaggle with:
    - Smaller batch size
    - Less frequent checkpointing
    - Reduced patch size
    - Disabled checkpoint zipping
    """
    config = TrainingConfig()
    
    # Reduce batch size to save memory
    config.batch_size = 2
    config.grad_accum_steps = 8  # Keep effective batch size = 16
    
    # Reduce patch size to save disk space
    config.patch_size = 3000
    
    # Less frequent evaluation and checkpointing
    config.eval_every_n_steps = 1000
    config.save_every_n_steps = 1000
    
    # Disable checkpoint zipping (saves time and disk I/O)
    config.zip_checkpoints = False
    
    # Reduce workers to save memory
    config.num_workers = 1
    
    # Enable all memory optimizations
    config.use_fp16 = True
    config.gradient_checkpointing = True
    config.freeze_encoder = True
    
    return config
