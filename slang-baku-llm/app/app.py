"""UI Gradio: Penerjemah Slang -> Bahasa Indonesia Baku (model dasar vs fine-tuned, via Ollama)."""
import requests
import gradio as gr

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_DASAR = "qwen2.5:1.5b"     # model sebelum fine-tuning
MODEL_FT = "slang-baku"          # model hasil fine-tuning (nama saat ollama create)

# HARUS sama persis dengan yang dipakai saat training di notebook
SYSTEM_PROMPT = ("Kamu adalah penerjemah yang mengubah bahasa gaul (slang) "
                 "menjadi bahasa Indonesia baku. Jawab hanya dengan hasil terjemahan.")
INSTRUKSI = "Ubah kalimat slang berikut menjadi bahasa Indonesia baku."

CONTOH = [
    "gue lagi mager banget hari ini",
    "anjir, tugas gue numpuk banget minggu ini",
    "sorry bgt gue telat, tadi ban motor gue bocor",
    "btw besok kelasnya jadi online apa offline sih?",
    "gue udah nge-submit tugasnya semalem, tinggal nunggu approval dosen",
]


def tanya(model: str, kalimat: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{INSTRUKSI}\n\n{kalimat}"},
        ],
        "stream": False,
        "options": {"temperature": 0.2},
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "[Error] Ollama belum berjalan. Jalankan aplikasi Ollama / 'ollama serve' dulu."
    except Exception as e:
        return f"[Error] {e}"


def terjemahkan(kalimat: str):
    kalimat = (kalimat or "").strip()
    if not kalimat:
        return "", ""
    return tanya(MODEL_DASAR, kalimat), tanya(MODEL_FT, kalimat)


with gr.Blocks(title="Penerjemah Slang ke Bahasa Baku") as demo:
    gr.Markdown("# Penerjemah Slang → Bahasa Indonesia Baku\n"
                "Perbandingan **model dasar** (Qwen 2.5 1.5B) vs **model hasil fine-tuning** (QLoRA). "
                "Semua berjalan lokal lewat Ollama.")
    inp = gr.Textbox(label="Kalimat slang", placeholder="contoh: gue lagi mager banget hari ini", lines=2)
    btn = gr.Button("Terjemahkan", variant="primary")
    with gr.Row():
        out_dasar = gr.Textbox(label=f"Sebelum fine-tuning ({MODEL_DASAR})", lines=4)
        out_ft = gr.Textbox(label=f"Sesudah fine-tuning ({MODEL_FT})", lines=4)
    gr.Examples(examples=CONTOH, inputs=inp)
    btn.click(terjemahkan, inputs=inp, outputs=[out_dasar, out_ft])
    inp.submit(terjemahkan, inputs=inp, outputs=[out_dasar, out_ft])

if __name__ == "__main__":
    demo.launch()
