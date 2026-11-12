# Troubleshooting

Daftar masalah nyata yang ditemui selama pengerjaan, beserta penyebab dan solusinya.

| # | Gejala | Penyebab | Solusi |
|---|---|---|---|
| 1 | `zsh: command not found: ollama` | Ollama belum terpasang | Instal dari ollama.com (aplikasi atau `curl -fsSL https://ollama.com/install.sh \| sh`), lalu tutup dan buka lagi Terminal |
| 2 | `zsh: command not found: pip` | macOS hanya menyediakan `python3` | Gunakan `python3 -m pip install ...` |
| 3 | `ImportError: cannot import name 'HfFolder' from 'huggingface_hub'` saat menjalankan Gradio 4.44 | `huggingface_hub` terlalu baru untuk Gradio 4.x | `python3 -m pip install "huggingface_hub<0.26"` |
| 4 | `can't open file '.../app.py': No such file or directory` | Perintah dijalankan dari folder yang salah | `cd` ke folder tempat `app.py` berada |
| 5 | `[Error] 404 Client Error: Not Found for url: .../api/chat` di UI | Nama model di `app.py` tidak ada di Ollama | Cek `ollama list`, samakan `MODEL_DASAR` dan `MODEL_FT` dengan nama di sana |
| 6 | Tidak ada file `.gguf` di Google Drive | Unsloth versi baru menyimpan GGUF di folder `gguf_model_gguf`, sedangkan yang disalin hanya `gguf_model` (bobot 16-bit) | Cari dengan `glob("**/*.gguf")`, salin hanya file `.gguf` |
| 7 | Unduhan dari Google Drive gagal/berat | Folder dikompres jadi zip besar | Unggah satu file GGUF ke Hugging Face, unduh dengan `curl -L -C -` (bisa dilanjutkan) |
| 8 | `ValueError: mount failed` pada `drive.mount` | Jendela izin Google tidak disetujui/diblokir | Izinkan pop-up, pakai akun yang sama, `force_remount=True`; atau lewati Drive |
| 9 | Model `slang-baku` di Ollama ternyata model dasar | Pernah dibuat dengan `FROM qwen2.5:1.5b` (tanpa file GGUF hasil fine-tuning) | Cek `ollama show slang-baku --modelfile`; `FROM` harus menunjuk ke blob berbeda dari model dasar; buat ulang dengan `FROM ./file.gguf` |
| 10 | `ollama create` menampilkan hash layer yang sama dengan sebelumnya | Masih memakai file GGUF lama | Unduh model baru dengan nama file berbeda dan buat model dengan nama baru |
| 11 | Hasil fine-tuning kurang baik, validation loss naik setelah sekitar 2 epoch | Overfit (data kecil, 5 epoch) | Kurangi epoch menjadi 2, tambah dan variasikan data |
| 12 | Peringatan `Both max_new_tokens and max_length seem to have been set` | Konfigurasi bawaan model | Aman diabaikan; `max_new_tokens` yang berlaku |
| 13 | Peringatan `xformers ... cannot load with torch ...` | Versi torch di Colab lebih baru dari xformers | Aman diabaikan; Unsloth memakai attention bawaan PyTorch |
| 14 | `CUDA out of memory` | Batch terlalu besar | Turunkan `per_device_train_batch_size` menjadi 2 dan naikkan `gradient_accumulation_steps` |
| 15 | `TypeError ... max_seq_length` atau `unexpected keyword argument 'tokenizer'` pada SFTTrainer | Versi TRL baru | Ganti `max_seq_length` -> `max_length` dan `tokenizer=` -> `processing_class=` |
| 16 | Token Hugging Face tidak sengaja terlihat | Token ditempel di tempat yang bisa dibaca orang lain | Segera hapus di *Settings -> Access Tokens*, buat token baru, dan masukkan hanya lewat `notebook_login()` |
