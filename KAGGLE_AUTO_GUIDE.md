# Hướng dẫn chạy Kaggle Notebook tự động

## File mới: `kaggle_notebook_auto.ipynb`

Notebook này chạy **hoàn toàn tự động** từ đầu đến cuối, không cần tương tác. Bạn có thể bật notebook lên và để nó chạy ngầm.

## Cách sử dụng

### Bước 1: Upload notebook lên Kaggle

1. Vào https://www.kaggle.com/code
2. Click **New Notebook**
3. Click **File** → **Upload Notebook**
4. Chọn file `kaggle_notebook_auto.ipynb` từ repo này
5. Hoặc copy toàn bộ nội dung file và paste vào notebook mới

### Bước 2: Cấu hình Kaggle

#### Settings → Add Data:
- `tinthnhphm21022004/data-speech-to-text` (audio files)
- `thanhphamtien2102224/weight-phowhisper` (manifest JSONL)

#### Settings → Accelerator:
- **GPU T4 x1** (bắt buộc)

#### Settings → Internet:
- **ON** (để download model từ HuggingFace)

#### Settings → Persistence:
- **Files only** (để save checkpoints khi session timeout)

### Bước 3: Chạy notebook

**Option 1: Run All**
- Click **Run All** ở menu trên
- Notebook sẽ chạy tất cả 11 steps tự động

**Option 2: Run cell duy nhất**
- Click vào cell code duy nhất
- Nhấn **Shift + Enter**

### Bước 4: Đóng tab và để chạy ngầm

- Sau khi bắt đầu chạy, bạn có thể **đóng tab browser**
- Kaggle sẽ tiếp tục chạy notebook trong background
- Session timeout: **9 giờ** (đủ để train xong)

### Bước 5: Quay lại kiểm tra

- Mở lại notebook sau vài giờ
- Xem output để biết tiến độ
- Nếu đã xong, download file `xtts_output.zip` từ Output tab

## Output

Sau khi chạy xong, bạn sẽ có:

### Trong notebook output:
```
✅ ALL STEPS COMPLETED SUCCESSFULLY!
================================================================================

📂 Output locations:
   Checkpoints: /kaggle/working/output/checkpoints
   Samples: /kaggle/working/output/samples
   Logs: /kaggle/working/output/logs
   Archive: /kaggle/working/xtts_output.zip

📥 Download:
   Kaggle → Output tab → xtts_output.zip

🎉 Training pipeline completed!
```

### Files trong Output tab:
- **`xtts_output.zip`** - Toàn bộ output (checkpoints + samples + logs)
- Checkpoints: `output/checkpoints/patch_XXXX.pth`, `best_model.pth`
- Audio samples: `output/samples/step_XXXXXX_sample_XX.wav`
- Test audio: `output/test_01.wav`, `test_02.wav`, `test_03.wav`
- Logs: `output/logs/kaggle_auto.log`

## 11 Steps tự động

Notebook sẽ chạy qua 11 bước:

1. ✅ **Fix numpy** - Downgrade numpy 2.x → 1.26.4
2. ✅ **Check GPU** - Verify T4 GPU available
3. ✅ **Clone repo** - Pull latest code from GitHub
4. ✅ **Install deps** - Install Coqui TTS + dependencies
5. ✅ **Setup paths** - Fix manifest paths, find reference audio
6. ✅ **Configure** - Setup training config
7. ✅ **Load dataset** - Load and validate train/val data
8. ✅ **Download model** - Download XTTS v2 base model from HF
9. ✅ **Load model** - Load model, configure trainable params
10. ✅ **Train** - Run full training loop
11. ✅ **Inference** - Generate test audio samples
12. ✅ **Zip output** - Create downloadable archive

## Monitoring

### Xem tiến độ real-time:
- Mở notebook → scroll xuống output
- Sẽ thấy logs từng step

### Kiểm tra GPU usage:
- Kaggle → Session → GPU Usage
- Nên thấy ~80-90% GPU utilization khi training

### Kiểm tra logs chi tiết:
- Sau khi xong, download `xtts_output.zip`
- Extract → xem file `logs/kaggle_auto.log`

## Troubleshooting

### Lỗi "Dataset not found"
- Kiểm tra lại Settings → Add Data
- Đảm bảo 2 datasets đã được add đúng tên

### Lỗi "No GPU detected"
- Settings → Accelerator → chọn **GPU T4 x1**
- Restart session

### Session timeout trước khi xong
- Kaggle free tier: 9 giờ/session
- Nếu dataset quá lớn, có thể cần:
  - Giảm `patch_size` xuống 3000
  - Hoặc chạy trên Kaggle Pro (30 giờ/session)

### Notebook bị stuck
- Refresh page
- Nếu vẫn chạy → đợi thêm
- Nếu đã stop → xem logs để biết lỗi ở đâu

## So sánh với notebook cũ

| Feature | `kaggle_notebook.ipynb` (cũ) | `kaggle_notebook_auto.ipynb` (mới) |
|---|---|---|
| Số cells | 13 cells | 1 cell duy nhất |
| Tương tác | Cần chạy từng cell | Chạy tự động |
| Chạy ngầm | Không | Có ✅ |
| Restart kernel | Cần (sau Cell 0) | Không cần |
| Phù hợp cho | Debug, học tập | Production, chạy ngầm |

## Tips

### Để chạy nhanh hơn:
```python
# Trong notebook, edit config:
config = TrainingConfig(
    batch_size       = 4,        # Tăng từ 2 → 4 (nếu đủ VRAM)
    grad_accum_steps = 4,        # Giảm từ 8 → 4
    patch_size       = 3000,     # Giảm từ 5000 → 3000
    epochs_per_patch = 1,        # Giữ nguyên
)
```

### Để train kỹ hơn:
```python
config = TrainingConfig(
    epochs_per_patch = 2,        # Tăng từ 1 → 2
    learning_rate    = 1e-5,     # Giảm LR
    eval_every_n_steps = 250,    # Eval thường xuyên hơn
)
```

### Để save VRAM:
```python
config = TrainingConfig(
    batch_size       = 1,        # Giảm xuống 1
    grad_accum_steps = 16,       # Tăng lên 16
    use_fp16         = True,     # Bắt buộc
    gradient_checkpointing = True,  # Bắt buộc
)
```

## Khi nào dùng notebook nào?

### Dùng `kaggle_notebook.ipynb` (cũ) khi:
- Đang debug code
- Muốn xem từng bước một
- Cần modify config giữa chừng
- Đang học cách pipeline hoạt động

### Dùng `kaggle_notebook_auto.ipynb` (mới) khi:
- Chỉ cần train và lấy kết quả
- Muốn chạy ngầm, không cần giám sát
- Đã biết config phù hợp
- Cần chạy nhiều experiments song song

## Next Steps

Sau khi training xong:

1. **Download output**
   - Kaggle → Output → `xtts_output.zip`

2. **Test model locally**
   ```bash
   python -m xtts_finetune.inference \
       --checkpoint ./output/checkpoints/best_model.pth \
       --text "Test tiếng Việt" \
       --reference_audio ./reference.wav \
       --output ./test.wav
   ```

3. **Deploy model**
   - Upload checkpoint lên HuggingFace
   - Hoặc serve qua API (FastAPI, Flask)

4. **Continue training**
   - Modify config để train thêm epochs
   - Resume từ checkpoint

---

**File**: `kaggle_notebook_auto.ipynb`  
**Commit**: `e9e6a3f`  
**Status**: ✅ Ready to use
