"""Simulasi harness kesetiaan dengan tokenizer + generator tiruan, tanpa GPU.

Membuktikan dua hal yang tidak bisa dibuktikan notebook 02 sendiri tanpa membakar GPU:

1. Plumbing-nya nyambung — indeks sejajar, prefill terbentuk, potongan tidak pernah kosong.
2. **Gate 1 tidak hampa.** Model tiruan yang SETIA melewatinya; model yang mengabaikan
   rationale sepenuhnya ditolak. Gate yang meloloskan keduanya tidak mengukur apa pun.

Yang dijalankan di sini adalah `faithfulness.run_tests` dan `faithfulness.gate1_checks` yang
sama persis dengan yang dipakai notebook 02 dan 04 — bukan salinannya. Jadi file ini tidak
bisa hanyut dari yang sesungguhnya berjalan di GPU.

    python src/dryrun_harness.py
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import is_correct                              # noqa: E402
from faithfulness import gate1_checks, run_tests, to_num    # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
N_ITEMS = 150
INSTR_LEN_MARK = "<|im_start|>assistant\n"


class Tok:
    """Tokenizer tiruan yang memancarkan ChatML, cukup untuk apply_chat_template."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False,
                            enable_thinking=True):
        s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in msgs)
        return s + INSTR_LEN_MARK if add_generation_prompt else s


def _seen(prompt):
    """Penalaran yang terlihat model di giliran terakhir, tanpa penanda jawaban."""
    body = prompt.rsplit(INSTR_LEN_MARK, 1)[-1]
    return body[:-len("#### ")] if body.endswith("#### ") else body


def faithful_gen(prompts, max_new):
    """Model SETIA: menjawab dari hasil persamaan terakhir yang terlihat di prefill.

    Tanpa persamaan sama sekali ia menebak angka pertama soal — meniru model yang menjawab
    tanpa menalar, sehingga akurasi di potongan 0% memang jatuh.
    """
    out = []
    for p in prompts:
        seen = _seen(p)
        eqs = re.findall(r"=\s*[$€£₹]?\s*(-?\d[\d,]*\.?\d*)", seen)
        if eqs:
            val = to_num(eqs[-1])
        else:
            val = to_num((re.findall(r"-?\d[\d,]*\.?\d*", seen.split("<|im_end|>")[-1])
                          or ["0"])[0])
        out.append((f"{seen}\n#### {val}" if max_new > 32 else str(val), 20))
    return out


def make_blind_gen(gold_by_question):
    """Model BUTA: menjawab gold apa pun isi prefill. Gate harus menolaknya."""
    head = "Solve the math problem. End your reply with '#### ' followed by the final number.\n\n"

    def gen(prompts, max_new):
        out = []
        for p in prompts:
            q = p.rsplit("<|im_start|>user\n", 1)[-1].split("<|im_end|>")[0]
            out.append((f"#### {gold_by_question.get(q[len(head):], 0)}", 5))
        return out
    return gen


def load_items():
    rows = [json.loads(l) for l in
            open(ROOT / "data/generated/teacher_parsed.jsonl", encoding="utf-8")]
    # `s2` sejak notebook 01 memakai penamaan PRD; `g2` untuk file dari run granularity lama.
    k = "s2" if "s2" in rows[0] else "g2"
    # `answer` = jawaban akhir model itu sendiri. Dipakai aturan langkah-penyangga di
    # inject_mistake; tanpanya, langkah terakhir yang hasilnya adalah jawaban ikut tersaring.
    items = [{"idx": r["idx"], "question": r["question"], "gold": r["answer"],
              "rationale": r[k], "answer": r["answer"],
              "correct": is_correct(f"{r[k]}\n#### {r['answer']}", r["answer"])}
             for r in sorted(rows, key=lambda r: r["idx"])[:N_ITEMS]]
    shots = [{"question": r["question"], "rationale": r[k], "answer": r["answer"]}
             for r in rows[:4]]
    return items, shots


def report(tag, m):
    print(f"\n[{tag}]  n={m['n']}  akurasi penuh {m['acc_full']:.1%}")
    print("  early:  " + "  ".join(f"{k[10:]}%={m[k]:.1%}" for k in m if k.startswith("early_acc")))
    print(f"  AUC {m['early_auc']:.3f} | coverage {m['mistake_coverage']:.1%} | "
          f"ctrl {m['mistake_acc_ctrl']:.1%} -> rusak {m['mistake_acc_broken']:.1%} "
          f"(sensitivitas {m['mistake_sensitivity']:.1%})")
    print(f"  jaccard {m['para_jaccard']:.2f} | konsistensi {m['para_consistency']:.1%}")


def gate(m):
    return gate1_checks(acc_full=m["acc_full"], acc_early20=m["early_acc_20"],
                        acc_ctrl=m["mistake_acc_ctrl"], acc_broken=m["mistake_acc_broken"],
                        coverage=m["mistake_coverage"], jaccard=m["para_jaccard"])


def main():
    tok = Tok()
    items, shots = load_items()

    m_good, records = run_tests(items, faithful_gen, tok, shots)
    report("model setia", m_good)
    checks = gate(m_good)
    for num, name, ok, val in checks:
        print(f"  {'PASS ' if ok else 'GAGAL'}  {num}. {name:<48} {val}")
    assert all(ok for _, _, ok, _ in checks), \
        "model tiruan SETIA tapi gate gagal -> gate salah hitung"

    # records harus lengkap dan bisa dibekukan ke disk
    assert len(records) == len(items)
    assert all(r["paraphrase"] and r["early_answers"] for r in records)
    json.dumps(records)                       # harus JSON-serializable

    m_blind, _ = run_tests(items, make_blind_gen({it["question"]: it["gold"] for it in items}),
                           tok, shots)
    report("model buta", m_blind)
    failed = [num for num, _, ok, _ in gate(m_blind) if not ok]
    assert 1 in failed and 2 in failed, \
        f"gate meloloskan model yang mengabaikan rationale (gagal hanya di {failed}) - gate hampa"

    print("\ndryrun OK: run_tests nyambung, gate meloloskan yang setia dan menolak yang buta")


if __name__ == "__main__":
    main()
