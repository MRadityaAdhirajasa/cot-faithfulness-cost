from __future__ import annotations

import gc
import time

import numpy as np
import torch
from unsloth import FastLanguageModel

from common import build_prompt, extract_answer, is_correct

MAX_SEQ = 1024
MAX_NEW = 1024   
BATCH = 16


def load_model(path, max_seq=MAX_SEQ, for_inference=True):
    kw = dict(max_seq_length=max_seq, load_in_4bit=True, full_finetuning=False)
    try:
        model, tok = FastLanguageModel.from_pretrained(str(path), device_map={"": 0}, **kw)
    except TypeError:                      
        model, tok = FastLanguageModel.from_pretrained(str(path), **kw)

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
    if objs:
        print("PERINGATAN free(): argumen diabaikan. Lepas sendiri lebih dulu -> "
              "mdl = tk = None; free()")
    gc.collect()
    torch.cuda.empty_cache()
