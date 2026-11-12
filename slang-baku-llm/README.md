# Penerjemah Slang -> Bahasa Indonesia Baku (Fine-Tuning Local LLM)

Proyek tugas mata kuliah **Generative AI** - Project 2: Fine-Tuning Local LLM.
Penulis: **Faisal Surya Saputra - 103012330152** (dikerjakan individu dengan bantuan asisten AI).

Sistem ini mengubah kalimat bahasa gaul/slang Indonesia menjadi bahasa Indonesia baku. Model dasar **Qwen 2.5 1.5B Instruct** di-fine-tuning dengan **QLoRA (Unsloth)** di Google Colab, dikonversi ke **GGUF (q4_k_m)**, lalu dijalankan **lokal lewat Ollama** dengan antarmuka **Gradio** yang menampilkan perbandingan model dasar vs model hasil fine-tuning.

## Daftar isi
1. [Arsitektur](#1-arsitektur)
2. [Struktur repositori](#2-struktur-repositori)
3. [Prasyarat](#3-prasyarat)
4. [Langkah 1 - Dataset](#4-langkah-1---dataset)
5. [Langkah 2 - Fine-tuning di Colab](#5-langkah-2---fine-tuning-di-colab)
6. [Langkah 3 - Pindahkan model ke komputer lokal](#6-langkah-3---pindahkan-model-ke-komputer-lokal)
7. [Langkah 4 - Jalankan di Ollama](#7-langkah-4---jalankan-di-ollama)
8. [Langkah 5 - UI Gradio dan demo](#8-langkah-5---ui-gradio-dan-demo)
9. [Hasil](#9-hasil)
10. [Troubleshooting](#10-troubleshooting)
11. [Keterbatasan](#11-keterbatasan)
12. [Pembagian peran](#12-pembagian-peran)

## 1. Arsitektur

```
 data/dataset_slang.jsonl
          |
          v
 [Google Colab, GPU T4]
  Unsloth + QLoRA (4-bit) pada Qwen2.5-1.5B-Instruct
  -> kurva loss, log, skor chrF, perbandingan sebelum/sesudah
  -> merge LoRA ke 16-bit -> konversi GGUF q4_k_m (llama.cpp)
          |
          v   (unggah ke Hugging Face, unduh dengan curl)
 [Komputer lokal (macOS)]
  Ollama: ollama create (Modelfile + file .gguf)
          |
          v   REST API http://localhost:11434/api/chat
  app/app.py (Gradio): model dasar (qwen2.5:1.5b) vs model fine-tuned
```

## 2. Struktur repositori

| Path | Isi |
|---|---|
| `data/dataset_slang.jsonl` | Dataset slang -> baku (lihat `data/README.md`) |
| `notebooks/fine_tuning_slang_baku.ipynb` | Notebook Colab: analisis data, training, evaluasi, ekspor GGUF |
| `ollama/Modelfile` | Konfigurasi Ollama (template ChatML, system prompt, stop token) |
| `app/app.py` | UI Gradio yang memanggil Ollama |
| `results/` | Kurva loss, log training, tabel perbandingan |
| `docs/screenshots/` | Screenshot tiap tahap |
| `docs/troubleshooting.md` | Daftar error yang ditemui dan solusinya |

File model `.gguf` (sekitar 940 MB) **tidak disimpan di GitHub** karena melebihi batas ukuran file. Model tersedia di Hugging Face: <https://huggingface.co/faisalwp/slang-baku-gguf>.

## 3. Prasyarat
- Akun Google (Colab, runtime GPU T4 gratis) dan akun Hugging Face (token tipe *Write*).
- Komputer lokal dengan [Ollama](https://ollama.com/download), Python 3.9+, dan RAM minimal 8 GB.
- Pustaka UI: `pip3 install -r requirements.txt`

## 4. Langkah 1 - Dataset
Dataset berisi pasangan `input` (slang) dan `output` (baku) dalam format JSONL, berkategori (kosakata slang, singkatan chat, campuran Inggris, kalimat panjang, dan lain-lain). Detail format, distribusi kategori, dan cara pembuatan ada di [`data/README.md`](data/README.md).

Preprocessing yang dilakukan notebook: merapikan spasi, menghapus duplikat berdasarkan `input`, menampilkan statistik per kategori, lalu membagi data 90% latih / 10% validasi (stratified, `random_state=42`).

## 5. Langkah 2 - Fine-tuning di Colab
1. Buka `notebooks/fine_tuning_slang_baku.ipynb` di <https://colab.research.google.com> (*File -> Upload notebook*).
2. *Runtime -> Change runtime type -> T4 GPU*.
3. Jalankan sel dari atas ke bawah. Saat diminta, unggah `data/dataset_slang.jsonl`.
4. Sel **Konfigurasi eksperimen** mengatur `LORA_R`, `LORA_ALPHA`, `EPOCHS`, dan `TAG` (label untuk nama file hasil).
5. Sel *Uji Model Dasar* menyimpan keluaran sebelum fine-tuning; sel *Uji Model Fine-Tuned* membandingkannya dengan sesudah fine-tuning dan menghitung skor chrF.
6. Sel *Ekspor* membuat file GGUF di `gguf_model_gguf/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf`.

**Hyperparameter (konfigurasi akhir):**

| Parameter | Nilai | Alasan singkat |
|---|---|---|
| Model dasar | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` | Ringan, muat di T4 dan di laptop |
| LoRA rank / alpha | 16 / 16 | Kapasitas cukup untuk tugas gaya bahasa |
| Target modul | q, k, v, o, gate, up, down | Seluruh proyeksi atensi dan MLP |
| Batch efektif | 4 x 2 (gradient accumulation) = 8 | Hemat memori GPU |
| Learning rate | 2e-4, scheduler linear, warmup 5 | Nilai umum untuk LoRA |
| Epoch | 2 | Validation loss mendatar setelah sekitar 2 epoch; 5 epoch terbukti overfit (lihat bagian Hasil) |
| Optimizer | `adamw_8bit`, weight decay 0.01 | Hemat memori |
| Panjang maksimum | 512 token | Kalimat data pendek |
| Loss | Hanya pada bagian jawaban (`train_on_responses_only`) | Model belajar menghasilkan terjemahan, bukan menghafal prompt |

## 6. Langkah 3 - Pindahkan model ke komputer lokal
Mengunduh file besar dari Google Drive sering gagal, jadi jalur yang dipakai adalah Hugging Face.

Di Colab (login dengan token *Write*; **jangan membagikan token**):
```python
from huggingface_hub import notebook_login, HfApi
notebook_login()
api = HfApi()
repo = "faisalwp/slang-baku-gguf"
api.create_repo(repo, exist_ok=True)
api.upload_file(
    path_or_fileobj="/content/gguf_model_gguf/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf",
    path_in_repo="slang-baku-v2.Q4_K_M.gguf",
    repo_id=repo,
)
```

Di terminal lokal (`-C -` membuat unduhan bisa dilanjutkan jika terputus):
```bash
mkdir -p ~/proyek-llm && cd ~/proyek-llm
curl -L -C - -o slang-baku-v2.Q4_K_M.gguf https://huggingface.co/faisalwp/slang-baku-gguf/resolve/main/slang-baku-v2.Q4_K_M.gguf
ls -lh   # sekitar 940 MB
```

## 7. Langkah 4 - Jalankan di Ollama
```bash
cp /path/ke/repo/ollama/Modelfile .
sed 's#^FROM .*#FROM ./slang-baku-v2.Q4_K_M.gguf#' Modelfile > Modelfile.v2
ollama create slang-baku-v2 -f Modelfile.v2
ollama list
ollama run slang-baku-v2
```
Contoh masukan (format harus sama dengan saat training: instruksi, baris kosong, lalu kalimat):
```
Ubah kalimat slang berikut menjadi bahasa Indonesia baku.

gue lagi mager banget hari ini
```
Pembanding untuk demo: `ollama pull qwen2.5:1.5b`.

Catatan: `Modelfile` memakai template ChatML dan prompt sistem yang **identik** dengan prompt saat training. Jika berbeda, kualitas keluaran bisa turun.

## 8. Langkah 5 - UI Gradio dan demo
```bash
pip3 install -r requirements.txt
cd app
sed -i '' 's/MODEL_FT = "slang-baku"/MODEL_FT = "slang-baku-v2"/' app.py   # macOS; sesuaikan nama model
python3 app.py
```
Buka <http://127.0.0.1:7860>. UI menampilkan dua kotak: keluaran model dasar (`qwen2.5:1.5b`) dan keluaran model fine-tuned, untuk kalimat yang sama.

Pastikan aplikasi Ollama sedang berjalan (ikon di menu bar) sebelum menjalankan UI.

## 9. Hasil

**Percobaan 1** - 370 pasang data, 5 epoch (210 langkah):
- chrF pada data validasi: model dasar **47,69** -> fine-tuned **70,70**.
- Validation loss terendah **0,559** pada langkah 60, lalu naik ke 0,659 sementara train loss turun ke 0,01 -> **overfit** setelah sekitar 2 epoch.
- Kesimpulan: konfigurasi diubah menjadi 2 epoch.

**Percobaan 2** - 623 pasang data, 2 epoch:
- Validation loss turun dari sekitar 0,8 ke sekitar 0,49 lalu mendatar, tanpa tanda overfit.
- chrF pada data validasi: dasar **[isi]** -> fine-tuned **[isi]**.

Letakkan di folder `results/`: `kurva_loss_*.png`, `log_training_*.json`, `hasil_perbandingan_*.csv`.

**Contoh perbandingan** (isi dari `hasil_perbandingan_*.csv` atau dari UI):

| No. | Input (Slang / Bahasa Gaul) | Model Dasar (`qwen2.5:1.5b`) | Model *Fine-Tuned* (`slang-baku-v2`) |
| :--- | :--- | :--- | :--- |
| 1 | `anjir, tugas gue numpuk banget minggu ini` | Gue harus belajar banyak materi di sekolah minggu ini. | Anjir, tugas saya sangat banyak bulan ini. |
| 2 | `woi, jangan lupa bawain charger gue ya` | Woii, jangan lupa bawa charger-nya juga ya. | Hai, jangan lupa membawa peralatan pengisi daya saya ya. |
| 3 | `doi udh balikan sm mantannya, bkn gue yg salah` | Saya sudah kembali ke tempat asalnya, tapi tidak ada saya yang salah. | Dia sudah kembali kepada mantan, bukan saya yang salah. |
| 4 | `gila, nilai gue naik drastis semester ini` | Gua mendapat kenaikan nilai yang sangat drastis di semester ini. | Saya sangat tidak jelas, nilai saya meningkat drastis semestri ini. |
| 5 | `sorry bgt gue telat, tadi ban motor gue bocor` | Sorry karena saya lupa, mobil saya pecah hari ini. | Saya sangat minta maaf karena terlambat, tadi motor saya bocor. |
| 6 | `lo mau titip apa? gue mau ke indomaret` | Gue mau ke Indomaret, udh di sana ya. | Apakah kamu ingin membeli sesuatu? Saya ingin ke Indomaret. |
| 7 | `gue males banget deh ngerjain laporan praktikum` | Saya tidak tertarik untuk membuat laporan praktikum. | Saya sangat malas menulis laporan praktikum. |
| 8 | `btw besok kelasnya jadi online apa offline sih?` | besok, apakah kita akan belajar secara online atau offline? | Atau besok kelasnya jadi daring atau tatap muka? |
| 9 | `ntar malem gue nginep di rumah temen` | ntar malam gue ngecek email di rumah temen. | Nanti malam, saya menginap di rumah teman. |
| 10 | `dosennya galak banget, gue jadi gugup pas presentasi` | Dalam suasana rapat dan pentingnya sesi presentasi, saya merasa sangat agen dan malu-malu. | Dosennya sangat serius, saya menjadi gelisah saat presentasi. |
| 11 | `capek banget gue abis nugas semaleman` | Saya sangat lelah setelah bekerja seharian. | Saya sangat lelah setelah mengerjakan tugas semalaman. |
| 12 | `eh lo udah makan belum? gue laper nih` | eh lo udah makan belum? gue lapar deh. | Aduh, kamu sudah makan belum? Saya lapar sekali. |
| 13 | `bokap gue marah gara-gara gue pulang kemaleman` | Gue marah karena gue pulang ke tempat asal. | Ayah saya marah karena saya pulang kemalaman. |
| 14 | `tugas kelompok kita bubar jalan, nggak ada yang mau ngerjain` | Tugas kelas kita susah-susah gampang, tidak ada siapannya untuk membantu. | Tugas kelompok kita tidak berjalan, tidak ada yang mau mengerjakannya. |
| 15 | `gue sih oke aja, terserah lo deh` | *[Model menolak/menjawab dengan peringatan etika]* | Saya saja baik-baik saja, terserah kamu. |

---

Screenshot demo: letakkan di `docs/screenshots/` lalu tautkan di sini.

## 10. Troubleshooting
Ringkasan; daftar lengkap ada di [`docs/troubleshooting.md`](docs/troubleshooting.md).

| Masalah | Solusi singkat |
|---|---|
| `zsh: command not found: ollama` | Pasang Ollama, lalu tutup dan buka lagi Terminal |
| Tidak ada file `.gguf` di folder `gguf_model` | Unsloth versi baru menyimpannya di `gguf_model_gguf` |
| `ImportError: cannot import name 'HfFolder'` (Gradio) | `python3 -m pip install "huggingface_hub<0.26"` |
| `can't open file app.py` | Jalankan dari folder tempat `app.py` berada (`cd`) |
| Google Drive `mount failed` | `force_remount=True`, atau lewati Drive dan unggah ke Hugging Face |
| Hasil `ollama create` memakai hash lama | File `.gguf` yang dipakai masih yang lama; unduh ulang dengan nama berbeda |

## 11. Keterbatasan
- Model 1,5B terbatas pada kata slang yang jarang atau tidak ada di data; keluaran kadang berubah makna.
- Dataset (623 pasang) masih kecil dan dibuat dengan bantuan AI, lalu dikurasi manual; padanan baku bersifat subjektif.
- Evaluasi otomatis memakai chrF pada validasi kecil; belum ada penilaian manusia skala besar.
- Pengembangan lanjutan: data lebih banyak, model 3B, dan penilaian manual terstruktur.

## 12. Pembagian peran
Proyek dikerjakan **individu**. Seluruh peran dipegang penulis:

Asisten AI (Claude) digunakan untuk membantu menyusun draf data, kode notebook, dan dokumentasi; seluruh keluaran dijalankan, diuji, dan dikurasi oleh penulis.

## Lisensi dan atribusi
- Model dasar: Qwen2.5-1.5B-Instruct (Alibaba Cloud), lisensi Apache 2.0.
- Pelatihan: [Unsloth](https://github.com/unslothai/unsloth), TRL, PEFT, llama.cpp; inferensi: [Ollama](https://ollama.com); UI: [Gradio](https://gradio.app).
