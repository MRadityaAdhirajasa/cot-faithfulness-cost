"""Pemuatan model, generasi, dan skoring akurasi — dipakai notebook 02, 03, dan 04.

Satu-satunya modul di `src/` yang butuh GPU. Alasannya ada di sini dan bukan di notebook:
evaluasi in-domain berjalan di notebook 03 sementara OOD dan faithfulness berjalan di
notebook 04, dan keduanya harus memakai decoding, batch size, serta plafon token yang persis
sama. Di proyek sebelumnya dua pass memakai batch size berbeda dan itu berakhir sebagai
limitation yang tidak bisa diperbaiki lagi tanpa mengulang semuanya.

Konstanta di bawah adalah variabel kontrol. Jangan di-override per pemanggilan.
"""

from __future__ import annotations

import gc
import time

import numpy as np
import torch
from unsloth import FastLanguageModel

from common import build_prompt, extract_answer, is_correct

MAX_SEQ = 1024
MAX_NEW = 1024    # 512 memotong generasi long-CoT diam-diam; itu temuan mahal proyek B
BATCH = 16


def load_model(path, max_seq=MAX_SEQ, for_inference=True):
    """Muat model dasar atau adapter LoRA (path direktori adapter juga diterima).

    `device_map={"": 0}` memaksa seluruh modul ke GPU. Tanpa itu, saat VRAM sudah terisi
    accelerate diam-diam menaruh modul yang tidak terkuantisasi (`embed_tokens`, `lm_head`)
    di CPU, dan kegagalannya baru muncul jauh di dalam `generate()` sebagai
    "Expected all tensors to be on the same device". Lebih baik OOM yang jelas.
    """
    kw = dict(max_seq_length=max_seq, load_in_4bit=True, full_finetuning=False)
    try:
        model, tok = FastLanguageModel.from_pretrained(str(path), device_map={"": 0}, **kw)
    except TypeError:                      # versi Unsloth yang tidak meneruskan device_map
        model, tok = FastLanguageModel.from_pretrained(str(path), **kw)

    # Model bnb 4-bit tidak bisa dipindah dengan .to(), jadi ini diperiksa, bukan diperbaiki.
    cpu = sorted({n for n, p in model.named_parameters() if p.device.type != "cuda"})
    assert not cpu, (
        f"{len(cpu)} parameter tertinggal di CPU (mis. {cpu[:3]}). VRAM sudah terisi model "
        f"sebelumnya.\n  Lepas dulu: mdl = tk = None; free()  -- lalu muat ulang.")

    if for_inference:
        try:
            FastLanguageModel.for_inference(model)
        except Exception:
            model.eval()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def make_gen(model, tok, bs=BATCH, max_seq=MAX_SEQ):
    """-> callable(prompts, max_new) -> list[(teks, n_token_baru)], sesuai kontrak run_tests.

    Greedy tanpa kecuali: `do_sample=False`, tanpa temperature dan tanpa top_p. Ketiga uji
    kesetiaan membandingkan jawaban antar kondisi, jadi satu sumber keacakan saja sudah cukup
    membuat "jawaban berubah" tidak berarti apa-apa.
    """
    @torch.inference_mode()
    def gen(prompts, max_new):
        out = []
        for i in range(0, len(prompts), bs):
            enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=max_seq).to("cuda")
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                               pad_token_id=tok.pad_token_id)
            for seq in g[:, enc["input_ids"].shape[1]:]:
                out.append((tok.decode(seq, skip_special_tokens=True),
                            int((seq != tok.pad_token_id).sum())))
        return out
    return gen


def evaluate(model, tok, rows, shots=(), max_new=MAX_NEW, bs=BATCH):
    """Akurasi exact-match plus metrik biaya dan kesehatan pengukuran.

    `truncated_frac` dan `used_fallback_frac` bukan hiasan: keduanya adalah alarm bahwa
    akurasi yang dilaporkan sedang diukur pada generasi yang terpotong atau tidak terparse.
    """
    gen = make_gen(model, tok, bs=bs)
    t0 = time.time()
    outs = gen([build_prompt(tok, r["question"], shots) for r in rows], max_new)
    dt = time.time() - t0

    ok = trunc = fb = 0
    toks = []
    for r, (txt, ntok) in zip(rows, outs):
        toks.append(ntok)
        trunc += ntok >= max_new                  # kena plafon = kemungkinan terpotong
        fb += extract_answer(txt)[1]
        ok += is_correct(txt, r["answer"])

    return {"accuracy": ok / len(rows), "avg_output_tokens": float(np.mean(toks)),
            "latency_s_per_item": dt / len(rows), "truncated_frac": trunc / len(rows),
            "max_tokens_seen": int(max(toks)), "used_fallback_frac": fb / len(rows),
            "eval_max_new": max_new, "n_shot": len(shots)}


def free(*objs):
    """Kumpulkan sampah dan kosongkan cache CUDA.

    **`free(mdl, tk)` saja TIDAK melepas VRAM.** `del` di dalam fungsi hanya menghapus nama
    lokalnya; binding milik pemanggil tetap hidup dan modelnya tetap menghuni GPU. Pemanggil
    yang harus melepas:

        mdl = tk = None
        free()

    Di dalam loop hal ini tersamarkan karena `mdl` di-rebind tiap iterasi, lalu meledak di
    cell berikutnya saat model terakhir masih tertahan dan model kedua ikut dimuat.
    """
    if objs:
        print("PERINGATAN free(): argumen diabaikan. Lepas sendiri lebih dulu -> "
              "mdl = tk = None; free()")
    gc.collect()
    torch.cuda.empty_cache()
