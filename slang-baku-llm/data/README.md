# Dataset

`dataset_slang.jsonl` berisi **623 pasang** kalimat slang/bahasa gaul Indonesia -> bahasa Indonesia baku.

## Format
Satu objek JSON per baris (JSONL):

```json
{"instruction": "Ubah kalimat slang berikut menjadi bahasa Indonesia baku.", "input": "gue lagi mager banget hari ini", "output": "Saya sedang malas bergerak hari ini.", "kategori": "slang_medsos"}
```

| Field | Keterangan |
|---|---|
| `instruction` | Instruksi tugas (sama untuk semua baris) |
| `input` | Kalimat slang |
| `output` | Padanan bahasa Indonesia baku |
| `kategori` | Kelompok jenis slang (untuk analisis data; tidak dipakai saat training) |

## Distribusi per kategori
| Kategori | Jumlah |
|---|---|
| `kosakata_slang` | 107 |
| `kata_umum` | 69 |
| `sehari_hari` | 68 |
| `kalimat_panjang` | 65 |
| `singkatan` | 60 |
| `slang_medsos` | 50 |
| `campuran_inggris` | 50 |
| `ekspresif` | 28 |
| `pertanyaan` | 20 |
| `perintah_ajakan` | 20 |
| `konteks_kampus` | 20 |
| `emosi_opini` | 20 |
| `gaul_terbaru` | 20 |
| `imbuhan_verba` | 18 |
| `waktu_jumlah` | 8 |

## Cara pembuatan
Dataset dibuat secara bertahap: draf dihasilkan dengan bantuan asisten AI (Claude), kemudian dikurasi manual (pengecekan makna, penyeragaman padanan baku, penghapusan duplikat berdasarkan kolom `input`). Tidak ada data dari sumber luar.

Aturan padanan yang dipakai agar konsisten, antara lain: *banget* -> "sangat", *mager* -> "malas bergerak", *baper* -> "terbawa perasaan", *nongkrong* -> "berkumpul", *gebetan* -> "orang yang ditaksir", *gue/lo* -> "saya/kamu".

## Pembagian data
Saat training, data dibagi 90% latih dan 10% validasi (stratified berdasarkan `kategori`, `random_state=42`).

## Keterbatasan
- Ukuran data masih kecil untuk keragaman slang Indonesia yang luas.
- Padanan baku bersifat subjektif; sebagian kata slang punya lebih dari satu padanan yang benar.
