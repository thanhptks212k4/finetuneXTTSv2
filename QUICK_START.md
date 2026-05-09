# Quick Start - Chạy Training trên Kaggle

## Bước 1: Upload Notebook

1. Vào https://www.kaggle.com/code
2. Click **New Notebook**
3. Click **File** → **Upload Notebook**
4. Chọn file **`kaggle_notebook_auto.ipynb`** từ repo này

## Bước 2: Add Datasets

Settings → Add Data → Add:
- `tinthnhphm21022004/data-speech-to-text`
- `thanhphamtien2102224/weight-phowhisper`

## Bước 3: Chọn GPU

Settings → Accelerator → **GPU T4 x1**

## Bước 4: Bật Internet

Settings → Internet → **ON**

## Bước 5: Run

Click **Run All** hoặc nhấn **Shift + Enter** trên cell duy nhất

## Bước 6: Đóng Tab

Notebook sẽ chạy ngầm trong ~2-4 giờ. Bạn có thể đóng tab.

## Bước 7: Download Kết Quả

Quay lại sau → Output tab → Download **`xtts_output.zip`**

---

## Output Bao Gồm:

- ✅ **Checkpoints**: `best_model.pth`, `patch_XXXX.pth`
- ✅ **Audio samples**: Test audio files
- ✅ **Logs**: Training logs chi tiết

## Thời Gian:

- Setup: ~5 phút
- Training: ~2-4 giờ (tùy dataset size)
- Total: ~2-4 giờ

## Lưu Ý:

- ⚠️ Kaggle free tier: 9 giờ/session (đủ để train xong)
- ⚠️ Nếu session timeout: Chạy lại, code sẽ resume từ checkpoint
- ⚠️ Numpy warnings là bình thường, không ảnh hưởng

## Troubleshooting:

### "Dataset not found"
→ Kiểm tra lại Settings → Add Data

### "No GPU"
→ Settings → Accelerator → GPU T4 x1

### Notebook bị stuck
→ Refresh page, nếu vẫn chạy thì đợi thêm

---

**That's it!** 🎉

Chỉ cần 5 bước setup, click Run, và đợi kết quả!
