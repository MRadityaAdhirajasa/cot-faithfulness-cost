"""Format prompt dan parser jawaban, dipakai bersama notebook 02/03/04/05.

Kalau notebook training dan notebook evaluasi membangun prompt dengan cara yang sedikit
berbeda, modelnya dilatih pada struktur yang berbeda dari yang dilihat saat generate. Itu
degradasi diam-diam yang tidak muncul sebagai error di mana pun. Satu-satunya cara memastikan
hal itu tidak terjadi adalah memakai fungsi yang sama persis di semua notebook, jadi fungsinya
tinggal di sini, bukan disalin.

    python src/common.py     # jalankan self-check
"""

from __future__ import annotations

import re

# Notebook di-paste ke Kaggle, `src/` datang lewat dataset. Dua jalur deploy terpisah, jadi
# keduanya bisa hanyut tanpa gejala sampai sesuatu meledak di tengah run GPU. Notebook
# meng-assert konstanta ini di cell config, jadi ketidakcocokan gagal dalam hitungan detik.
#
# NAIKKAN setiap kali kontrak antar-modul berubah: kunci baru di metrik `run_tests`,
# parameter baru di `gate1_checks`, kolom baru di CSV.
#   1  rilis awal
#   2  faithfulness.run_tests menambah `orig_acc`; gate1_checks menambah kriteria 5
#   3  inject_mistake mengenali LaTeX; paraphrase tidak lagi menyentuh blok non-prosa
#   4  evalkit.load_model memaksa device_map GPU + assert; free() tidak lagi berpura-pura
#      melepas binding pemanggil (call site wajib `mdl = tk = None; free()`)
#   5  inject_mistake hanya merusak langkah PENYANGGA (hasilnya dipakai lagi, atau adalah
#      jawaban akhir model). Sebelumnya baris verifikasi di ekor rationale verbose ikut
#      terpilih, dan sensitivitas S3 terbaca 38,9% padahal 72,4%. run_tests kini mewajibkan
#      `answer` di tiap item, dan faithfulness.csv mencatat src_version per baris.
#   6  langkah penyangga juga harus BUKAN pengulangan nilai yang sudah tertulis sebelumnya;
#      baris verifikasi yang mengulang jawaban tadinya masih lolos (S3 sisa=0 -> 1,0%)
SRC_VERSION = 6

INSTR = "Solve the math problem. End your reply with '#### ' followed by the final number.\n\n"

# Penanda masking loss untuk ChatML Qwen3 (dipakai train_on_responses_only di notebook 03).
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


def target_text(row):
    """Sisi jawaban. Satu-satunya perbedaan antar varian supervisi ada di `rationale`."""
    r = (row.get("rationale") or "").strip()
    return f"{r}\n#### {row['answer']}" if r else f"#### {row['answer']}"


def build_prompt(tokenizer, question, shots=()):
    """Prompt inference, berakhir tepat di header giliran assistant."""
    msgs = []
    for s in shots:
        msgs += [{"role": "user", "content": INSTR + s["question"]},
                 {"role": "assistant", "content": target_text(s)}]
    msgs.append({"role": "user", "content": INSTR + question})
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def build_train_text(tokenizer, row):
    """Teks training penuh. Wajib diawali persis oleh `build_prompt` — digate di notebook 03."""
    msgs = [{"role": "user", "content": INSTR + row["question"]},
            {"role": "assistant", "content": target_text(row)}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)


def build_prefill(tokenizer, question, partial, shots=(), force_answer=False):
    """Prompt + potongan rationale yang sudah ditentukan, supaya model melanjutkan dari situ.

    Inilah mekanisme ketiga uji kesetiaan: yang berubah hanya isi `partial`.

    `force_answer=True` menambahkan penanda '#### ' sehingga model hanya perlu memuntahkan
    angka — dipakai untuk early answering dan paraphrasing, di mana yang ditanyakan adalah
    "berdasarkan penalaran ini, jawabanmu apa", bukan "lanjutkan menalar".
    """
    p = build_prompt(tokenizer, question, shots)
    body = (partial or "").strip()
    if force_answer:
        return f"{p}{body}\n#### " if body else f"{p}#### "
    return p + body + "\n"


_ANS_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_answer(text):
    """-> (nilai, used_fallback). Ambil angka setelah '####'; kalau tidak ada, angka terakhir.

    `used_fallback` adalah metrik keandalan parser: kalau tinggi, generasi kemungkinan
    terpotong sebelum sempat menulis penanda, dan akurasinya perlu dicurigai.
    """
    m = _ANS_RE.search(text or "")
    used_fallback = m is None
    if m is None:
        nums = _NUM_RE.findall(text or "")
        if not nums:
            return None, True
        val = nums[-1]
    else:
        val = m.group(1)
    try:
        v = float(val.replace(",", "").rstrip("."))
    except ValueError:
        return None, used_fallback
    return (int(v) if v == int(v) else round(v, 4)), used_fallback


def is_correct(generation, gold):
    """Exact match atas nilai terparse, dinormalisasi di kedua sisi."""
    return extract_answer(generation)[0] == extract_answer(str(gold))[0]


def demo():
    cases = [("48 + 24 = 72\n#### 72", 72, False),
             ("#### 1,200", 1200, False),
             ("The answer is 18.\n#### 18\n<|im_end|>", 18, False),
             ("blah #### 3.5 blah", 3.5, False),
             ("no hash marks, ends with 42", 42, True),      # fallback
             ("#### -7", -7, False),
             ("tidak ada angka sama sekali", None, True),
             ("", None, True)]
    for t, want, fb in cases:
        got, used = extract_answer(t)
        assert got == want and used == fb, f"{t!r} -> {got!r},{used} (harusnya {want!r},{fb})"

    assert is_correct("blah\n#### 1,200", "1200") and not is_correct("#### 7", "8")
    assert target_text({"rationale": "a = 1", "answer": "1"}) == "a = 1\n#### 1"
    assert target_text({"rationale": "", "answer": "1"}) == "#### 1"
    assert target_text({"answer": "1"}) == "#### 1"

    # build_prompt/build_prefill butuh tokenizer sungguhan; pakai stub yang meniru ChatML.
    class Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False,
                                enable_thinking=True):
            s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
            return s + "<|im_start|>assistant\n" if add_generation_prompt else s

    tok, row = Tok(), {"question": "Q?", "rationale": "2 + 2 = 4", "answer": "4"}
    p = build_prompt(tok, row["question"])
    t = build_train_text(tok, row)
    assert t.startswith(p), "teks training tidak diawali prompt inference"
    assert build_prefill(tok, "Q?", "2 + 2 = 4", force_answer=True) == p + "2 + 2 = 4\n#### "
    assert build_prefill(tok, "Q?", "", force_answer=True) == p + "#### "
    assert build_prefill(tok, "Q?", "2 + 2 = 4") == p + "2 + 2 = 4\n"
    # 1 shot = user+assistant, lalu user asli, lalu header assistant = 4 giliran
    assert build_prompt(tok, "Q?", shots=[row]).count("<|im_start|>") == 4
    assert "#### 4" in build_prompt(tok, "Q?", shots=[row]), "shot tidak membawa target"

    print("common self-check OK")


if __name__ == "__main__":
    demo()
