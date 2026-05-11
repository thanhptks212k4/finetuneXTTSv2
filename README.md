# XTTS v2 Vietnamese Fine-Tuning

Fine-tune XTTS v2 (Coqui TTS) cho tiếng Việt trên Kaggle.

## 🚀 Cách sử dụng trên Kaggle

### Bước 1: Upload notebook
1. Vào [Kaggle](https://www.kaggle.com/)
2. Tạo notebook mới
3. Upload file `kaggle_notebook_auto.ipynb`

### Bước 2: Thêm datasets
Trong notebook, thêm 2 datasets:
- `tinthnhphm21022004/data-speech-to-text`
- `thanhphamtien2102224/weight-phowhisper`

### Bước 3: Cấu hình
- **Accelerator**: GPU T4 x1
- **Internet**: ON
- **Persistence**: Files only

### Bước 4: Chạy
Click **"Run All"** và đợi ~3 giờ.

## ✅ Các bug đã được fix

1. ✅ W&B hanging - Đã disable trước khi import
2. ✅ Audio path mismatch - Tự động detect và fix paths
3. ✅ audio_root_remap - Đã cấu hình đúng
4. ✅ T4 OOM - Tối ưu cho 16GB VRAM
5. ✅ Reference audio - Tự động chọn file tốt nhất

## 📂 Cấu trúc project

```
finetuneXTTSv2/
├── kaggle_notebook_auto.ipynb  ← File chính để chạy trên Kaggle
├── kaggle_autofix.py           ← Functions tự động fix paths
├── requirements.txt            ← Dependencies
├── README.md                   ← File này
└── xtts_finetune/             ← Python package
    ├── config.py              ← Cấu hình training
    ├── dataset.py             ← Load và validate data
    ├── model_loader.py        ← Load XTTS model
    ├── trainer.py             ← Training loop
    ├── inference.py           ← Inference sau khi train
    ├── utils.py               ← Utility functions
    └── ...
```

## 📊 Kết quả mong đợi

Sau khi training xong:
- **Checkpoints**: `/kaggle/working/output/checkpoints/`
- **Audio samples**: `/kaggle/working/output/samples/`
- **Logs**: `/kaggle/working/output/logs/`

## 🐛 Troubleshooting

### "No valid training samples"
- Kiểm tra datasets đã được add chưa
- Verify đường dẫn trong manifest

### "CUDA out of memory"
- Đã được fix với batch_size=2
- Nếu vẫn lỗi, giảm xuống batch_size=1

### "No GPU detected"
- Kiểm tra Settings → Accelerator → GPU T4 x1

## 📝 License

Dự án sử dụng Coqui TTS (Mozilla Public License 2.0).
