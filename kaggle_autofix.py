#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kaggle_autofix.py - Comprehensive auto-fix for Kaggle XTTS training

This module automatically:
1. Detects the real audio directory by walking the dataset
2. Analyzes manifest format to detect path prefixes
3. Builds audio_root_remap dictionary
4. Rewrites fixed manifests with correct paths
5. Validates that fixed manifests have > 0 valid samples
6. Selects optimal reference audio automatically
7. Returns a ready-to-use TrainingConfig

Usage in Kaggle notebook:
    from kaggle_autofix import auto_fix_kaggle_paths
    config = auto_fix_kaggle_paths()
    # Now use config for training
"""

import os
import json
import glob
from typing import Dict, List, Tuple, Optional
import soundfile as sf


def find_audio_directory(base_path: str, verbose: bool = True) -> Optional[str]:
    """
    Walk the dataset directory tree to find where .wav files actually live.
    
    Args:
        base_path: Root path to search (e.g., /kaggle/input/datasets/...)
        verbose: Print progress messages
        
    Returns:
        Absolute path to directory containing .wav files, or None if not found
    """
    if verbose:
        print(f"🔍 Searching for .wav files in: {base_path}")
    
    if not os.path.exists(base_path):
        if verbose:
            print(f"❌ Base path does not exist: {base_path}")
        return None
    
    wav_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.wav'):
                wav_files.append(os.path.join(root, file))
        
        # Stop after finding first directory with .wav files
        if wav_files:
            break
    
    if not wav_files:
        if verbose:
            print(f"❌ No .wav files found in {base_path}")
        return None
    
    audio_dir = os.path.dirname(wav_files[0])
    
    if verbose:
        print(f"✅ Found {len(wav_files)} .wav files")
        print(f"✅ Audio directory: {audio_dir}")
        print(f"   Example file: {os.path.basename(wav_files[0])}")
    
    return audio_dir


def analyze_manifest_paths(manifest_path: str, sample_size: int = 100, verbose: bool = True) -> List[str]:
    """
    Read sample lines from manifest to detect what path prefixes are used.
    
    Args:
        manifest_path: Path to JSONL manifest file
        sample_size: Number of lines to sample
        verbose: Print progress messages
        
    Returns:
        List of unique directory prefixes found in manifest audio paths
    """
    if verbose:
        print(f"\n🔍 Analyzing manifest: {os.path.basename(manifest_path)}")
    
    if not os.path.exists(manifest_path):
        if verbose:
            print(f"❌ Manifest not found: {manifest_path}")
        return []
    
    path_prefixes = set()
    total_lines = 0
    
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            
            total_lines += 1
            try:
                obj = json.loads(line.strip())
                audio_path = obj.get('audio', '')
                
                if audio_path:
                    # Extract directory part
                    if '/' in audio_path:
                        dir_part = os.path.dirname(audio_path)
                        if dir_part:
                            path_prefixes.add(dir_part)
                    else:
                        # Relative path (just filename)
                        path_prefixes.add('')
            except json.JSONDecodeError:
                continue
    
    if verbose:
        print(f"✅ Analyzed {total_lines} samples")
        if path_prefixes:
            print(f"✅ Found {len(path_prefixes)} unique path pattern(s):")
            for prefix in sorted(path_prefixes)[:5]:
                print(f"   - {prefix if prefix else '(relative paths)'}")
        else:
            print(f"⚠️  No audio paths found in manifest")
    
    return sorted(path_prefixes)


def build_path_remap(manifest_prefixes: List[str], actual_audio_dir: str, verbose: bool = True) -> Dict[str, str]:
    """
    Build a dictionary mapping old manifest paths to new actual paths.
    
    Args:
        manifest_prefixes: List of path prefixes found in manifest
        actual_audio_dir: Actual directory where .wav files exist
        verbose: Print progress messages
        
    Returns:
        Dictionary mapping old prefix -> new prefix
    """
    if verbose:
        print(f"\n🔧 Building path remap...")
    
    remap = {}
    
    for old_prefix in manifest_prefixes:
        if old_prefix and old_prefix != actual_audio_dir:
            remap[old_prefix] = actual_audio_dir
    
    if verbose:
        if remap:
            print(f"✅ Remap dictionary ({len(remap)} mapping(s)):")
            for old, new in remap.items():
                print(f"   {old}")
                print(f"   → {new}")
        else:
            print(f"✅ No remap needed (paths already correct)")
    
    return remap


def fix_and_validate_manifest(
    src_manifest: str,
    dst_manifest: str,
    audio_dir: str,
    path_remap: Optional[Dict[str, str]] = None,
    verbose: bool = True
) -> Tuple[int, int, List[Dict]]:
    """
    Fix manifest paths and validate that audio files exist.
    
    Args:
        src_manifest: Source manifest path
        dst_manifest: Destination manifest path (will be created)
        audio_dir: Directory containing actual .wav files
        path_remap: Optional dict mapping old paths to new paths
        verbose: Print progress messages
        
    Returns:
        Tuple of (valid_count, missing_count, valid_samples_list)
    """
    if verbose:
        print(f"\n🔧 Fixing manifest: {os.path.basename(src_manifest)}")
    
    if not os.path.exists(src_manifest):
        if verbose:
            print(f"❌ Source manifest not found: {src_manifest}")
        return 0, 0, []
    
    valid_samples = []
    stats = {'total': 0, 'valid': 0, 'missing': 0, 'invalid_json': 0, 'empty_text': 0}
    
    with open(src_manifest, 'r', encoding='utf-8') as fin:
        for line_no, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            
            stats['total'] += 1
            
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                stats['invalid_json'] += 1
                if verbose and stats['invalid_json'] <= 3:
                    print(f"⚠️  Line {line_no}: Invalid JSON - {e}")
                continue
            
            # Get audio path and text
            audio_path = obj.get('audio', '')
            text = obj.get('text', '').strip()
            
            if not text:
                stats['empty_text'] += 1
                continue
            
            if not audio_path:
                stats['missing'] += 1
                continue
            
            # Apply path remap
            if path_remap:
                for old_prefix, new_prefix in path_remap.items():
                    if audio_path.startswith(old_prefix):
                        # Replace old prefix with new prefix
                        audio_path = audio_path.replace(old_prefix, new_prefix, 1)
                        break
            
            # If path is not absolute or still doesn't exist, try with audio_dir + basename
            if not os.path.isabs(audio_path) or not os.path.exists(audio_path):
                basename = os.path.basename(audio_path)
                audio_path = os.path.join(audio_dir, basename)
            
            # Validate file exists
            if os.path.exists(audio_path):
                obj['audio'] = audio_path
                valid_samples.append(obj)
                stats['valid'] += 1
            else:
                stats['missing'] += 1
                if verbose and stats['missing'] <= 3:
                    print(f"⚠️  File not found: {os.path.basename(audio_path)}")
    
    # Write fixed manifest
    os.makedirs(os.path.dirname(dst_manifest), exist_ok=True)
    with open(dst_manifest, 'w', encoding='utf-8') as fout:
        for obj in valid_samples:
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
    
    if verbose:
        print(f"✅ Manifest fixed:")
        print(f"   Total samples: {stats['total']:,}")
        print(f"   Valid samples: {stats['valid']:,}")
        print(f"   Missing files: {stats['missing']:,}")
        if stats['invalid_json'] > 0:
            print(f"   Invalid JSON: {stats['invalid_json']:,}")
        if stats['empty_text'] > 0:
            print(f"   Empty text: {stats['empty_text']:,}")
        print(f"   Saved to: {dst_manifest}")
    
    return stats['valid'], stats['missing'], valid_samples


def select_reference_audio(
    audio_dir: str,
    valid_samples: Optional[List[Dict]] = None,
    target_duration_min: float = 5.0,
    target_duration_max: float = 10.0,
    verbose: bool = True
) -> Optional[str]:
    """
    Automatically select the best reference audio for speaker embedding.
    
    Criteria:
    - Duration between 5-10 seconds (preferred)
    - Highest sample rate
    - Clean audio (no clipping)
    
    Args:
        audio_dir: Directory containing .wav files
        valid_samples: Optional list of valid samples from manifest
        target_duration_min: Minimum preferred duration (seconds)
        target_duration_max: Maximum preferred duration (seconds)
        verbose: Print progress messages
        
    Returns:
        Path to selected reference audio, or None if no suitable file found
    """
    if verbose:
        print(f"\n🎤 Selecting reference audio...")
    
    # Get candidate files
    candidates = []
    
    if valid_samples:
        # Use files from manifest
        candidate_paths = [s['audio'] for s in valid_samples[:100]]
    else:
        # Use all .wav files in directory
        candidate_paths = glob.glob(os.path.join(audio_dir, '*.wav'))[:100]
    
    if not candidate_paths:
        if verbose:
            print(f"❌ No audio files found")
        return None
    
    # Evaluate each candidate
    for audio_path in candidate_paths:
        if not os.path.exists(audio_path):
            continue
        
        try:
            info = sf.info(audio_path)
            duration = info.frames / info.samplerate
            
            # Score based on duration and sample rate
            duration_score = 0
            if target_duration_min <= duration <= target_duration_max:
                duration_score = 100  # Perfect duration
            elif duration < target_duration_min:
                duration_score = 50 * (duration / target_duration_min)  # Too short
            else:
                duration_score = 50 * (target_duration_max / duration)  # Too long
            
            sr_score = info.samplerate / 1000  # Higher sample rate is better
            
            total_score = duration_score + sr_score
            
            candidates.append({
                'path': audio_path,
                'duration': duration,
                'sample_rate': info.samplerate,
                'score': total_score
            })
        except Exception as e:
            if verbose:
                print(f"⚠️  Could not read {os.path.basename(audio_path)}: {e}")
            continue
    
    if not candidates:
        if verbose:
            print(f"❌ No valid audio files found")
        return None
    
    # Sort by score and pick the best
    candidates.sort(key=lambda x: x['score'], reverse=True)
    best = candidates[0]
    
    if verbose:
        print(f"✅ Selected reference audio:")
        print(f"   File: {os.path.basename(best['path'])}")
        print(f"   Duration: {best['duration']:.2f}s")
        print(f"   Sample rate: {best['sample_rate']} Hz")
        print(f"   Score: {best['score']:.1f}")
    
    return best['path']


def auto_fix_kaggle_paths(
    audio_base_path: str = '/kaggle/input/datasets/tinthnhphm21022004/data-speech-to-text',
    manifest_base_path: str = '/kaggle/input/datasets/thanhphamtien2102224/weight-phowhisper',
    train_manifest_name: str = 'train_full_manifest.jsonl',
    test_manifest_name: str = 'test_manifest.jsonl',
    output_dir: str = '/kaggle/working/output',
    working_dir: str = '/kaggle/working',
    verbose: bool = True
) -> Optional[object]:
    """
    Comprehensive auto-fix for Kaggle XTTS training paths.
    
    This function:
    1. Finds the real audio directory
    2. Analyzes manifest format
    3. Builds path remap dictionary
    4. Fixes and validates manifests
    5. Selects reference audio
    6. Returns a ready-to-use TrainingConfig
    
    Args:
        audio_base_path: Base path to audio dataset
        manifest_base_path: Base path to manifest files
        train_manifest_name: Name of train manifest file
        test_manifest_name: Name of test manifest file
        output_dir: Output directory for checkpoints/samples/logs
        working_dir: Working directory for fixed manifests
        verbose: Print detailed progress messages
        
    Returns:
        TrainingConfig object ready for training, or None on failure
    """
    if verbose:
        print("=" * 80)
        print("🚀 KAGGLE AUTO-FIX: Detecting and fixing all path issues")
        print("=" * 80)
    
    # Step 1: Find actual audio directory
    audio_dir = find_audio_directory(audio_base_path, verbose=verbose)
    if not audio_dir:
        print("\n❌ FATAL: Could not find audio directory")
        print("   Please check that the dataset is mounted correctly")
        return None
    
    # Step 2: Locate manifest files
    train_manifest_src = os.path.join(manifest_base_path, train_manifest_name)
    test_manifest_src = os.path.join(manifest_base_path, test_manifest_name)
    
    if not os.path.exists(train_manifest_src):
        print(f"\n❌ FATAL: Train manifest not found: {train_manifest_src}")
        return None
    
    if not os.path.exists(test_manifest_src):
        print(f"\n⚠️  WARNING: Test manifest not found: {test_manifest_src}")
        print("   Will use a subset of training data for validation")
        test_manifest_src = None
    
    # Step 3: Analyze manifest paths
    manifest_prefixes = analyze_manifest_paths(train_manifest_src, sample_size=100, verbose=verbose)
    
    # Step 4: Build path remap
    path_remap = build_path_remap(manifest_prefixes, audio_dir, verbose=verbose)
    
    # Step 5: Fix and validate train manifest
    train_manifest_dst = os.path.join(working_dir, 'train_manifest_fixed.jsonl')
    train_valid, train_missing, train_samples = fix_and_validate_manifest(
        train_manifest_src,
        train_manifest_dst,
        audio_dir,
        path_remap,
        verbose=verbose
    )
    
    if train_valid == 0:
        print("\n❌ FATAL: No valid training samples after fixing paths")
        print("   Possible issues:")
        print("   1. Audio files are in a different location than detected")
        print("   2. Manifest audio paths don't match actual filenames")
        print("   3. Audio files are missing from the dataset")
        print(f"\n   Detected audio directory: {audio_dir}")
        print(f"   Please verify this is correct and contains .wav files")
        return None
    
    # Step 6: Fix and validate test manifest
    if test_manifest_src:
        test_manifest_dst = os.path.join(working_dir, 'test_manifest_fixed.jsonl')
        test_valid, test_missing, test_samples = fix_and_validate_manifest(
            test_manifest_src,
            test_manifest_dst,
            audio_dir,
            path_remap,
            verbose=verbose
        )
    else:
        # Use subset of training data
        test_manifest_dst = os.path.join(working_dir, 'test_manifest_fixed.jsonl')
        test_samples = train_samples[:min(100, len(train_samples) // 10)]
        train_samples = train_samples[len(test_samples):]
        
        # Write test manifest
        with open(test_manifest_dst, 'w', encoding='utf-8') as f:
            for obj in test_samples:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        
        # Update train manifest
        with open(train_manifest_dst, 'w', encoding='utf-8') as f:
            for obj in train_samples:
                f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        
        test_valid = len(test_samples)
        train_valid = len(train_samples)
        
        if verbose:
            print(f"\n✅ Created validation set from training data:")
            print(f"   Train samples: {train_valid:,}")
            print(f"   Val samples: {test_valid:,}")
    
    # Step 7: Select reference audio
    reference_audio = select_reference_audio(
        audio_dir,
        valid_samples=train_samples,
        verbose=verbose
    )
    
    if not reference_audio:
        print("\n⚠️  WARNING: Could not auto-select reference audio")
        print("   Using first available .wav file")
        wav_files = glob.glob(os.path.join(audio_dir, '*.wav'))
        reference_audio = wav_files[0] if wav_files else None
    
    if not reference_audio:
        print("\n❌ FATAL: No reference audio available")
        return None
    
    # Step 8: Create TrainingConfig
    if verbose:
        print("\n" + "=" * 80)
        print("✅ AUTO-FIX COMPLETE - Summary:")
        print("=" * 80)
        print(f"📂 Audio directory: {audio_dir}")
        print(f"📄 Train manifest: {train_manifest_dst}")
        print(f"   Valid samples: {train_valid:,}")
        print(f"📄 Test manifest: {test_manifest_dst}")
        print(f"   Valid samples: {test_valid:,}")
        print(f"🎤 Reference audio: {os.path.basename(reference_audio)}")
        print(f"📁 Output directory: {output_dir}")
        print("=" * 80)
    
    # Import TrainingConfig
    try:
        from xtts_finetune.config import TrainingConfig
    except ImportError:
        print("\n⚠️  Could not import TrainingConfig")
        print("   Returning paths as dict instead")
        return {
            'audio_dir': audio_dir,
            'train_manifest': train_manifest_dst,
            'test_manifest': test_manifest_dst,
            'reference_audio': reference_audio,
            'path_remap': path_remap,
            'train_valid': train_valid,
            'test_valid': test_valid,
        }
    
    # Create config with Kaggle T4 optimizations
    config = TrainingConfig(
        # Model
        hf_repo_id='coqui/XTTS-v2',
        base_model_dir=os.path.join(working_dir, 'base_model'),
        
        # Data
        train_manifest=train_manifest_dst,
        val_manifest=test_manifest_dst,
        reference_audio=reference_audio,
        audio_root_remap=path_remap if path_remap else None,
        
        # Training - OPTIMIZED FOR KAGGLE T4 (16GB VRAM)
        patch_size=2000,              # Smaller patches to avoid OOM
        batch_size=2,                 # Small batch for T4
        grad_accum_steps=8,           # Effective batch = 16
        epochs_per_patch=1,
        learning_rate=2e-5,
        
        # Memory optimization
        use_fp16=True,
        gradient_checkpointing=True,
        freeze_encoder=True,
        num_workers=1,
        
        # Evaluation
        eval_every_n_steps=500,
        save_every_n_steps=500,
        
        # Output
        output_dir=output_dir,
        checkpoint_dir=os.path.join(output_dir, 'checkpoints'),
        sample_dir=os.path.join(output_dir, 'samples'),
        log_dir=os.path.join(output_dir, 'logs'),
        
        # Misc
        seed=42,
        speaker_mode='single',
        zip_checkpoints=False,
    )
    
    if verbose:
        print("\n✅ TrainingConfig created with Kaggle T4 optimizations")
        print(f"   Batch size: {config.batch_size}")
        print(f"   Gradient accumulation: {config.grad_accum_steps}")
        print(f"   Effective batch: {config.batch_size * config.grad_accum_steps}")
        print(f"   FP16: {config.use_fp16}")
        print(f"   Gradient checkpointing: {config.gradient_checkpointing}")
        print(f"   Freeze encoder: {config.freeze_encoder}")
    
    return config


if __name__ == '__main__':
    # Test the auto-fix
    config = auto_fix_kaggle_paths(verbose=True)
    if config:
        print("\n✅ Auto-fix successful!")
    else:
        print("\n❌ Auto-fix failed!")
