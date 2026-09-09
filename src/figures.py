"""Figure dan tabel untuk README serta Bab IV.

Setiap figure ditulis dua kali, varian terang dan gelap, supaya README terbaca di kedua tema
GitHub. Paletnya dibawa dari proyek granularity sebelumnya, yang kontrasnya sudah divalidasi.

    python src/figures.py      # smoke test dengan data sintetis, tanpa hasil sungguhan
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

VARIANTS = ["S0", "S1", "S2", "S3"]
VLABEL = {"S0": "S0\nanswer-only", "S1": "S1\nshort", "S2": "S2\nfull", "S3": "S3\nlong"}
FRACS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

THEMES = {
    "light": dict(surface="#fcfcfb", s1="#2a78d6", s2="#eb6834", s3="#1baf7a",
                  ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7"),
    "dark": dict(surface="#1a1a19", s1="#3987e5", s2="#d95926", s3="#199e70",
                 ink="#ffffff", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835"),
}


def style(T):
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "figure.facecolor": T["surface"], "axes.facecolor": T["surface"],
        "savefig.facecolor": T["surface"],
        "text.color": T["ink"], "axes.labelcolor": T["ink2"],
        "xtick.color": T["muted"], "ytick.color": T["muted"],
        "axes.edgecolor": T["axis"], "axes.linewidth": 0.8,
        "grid.color": T["grid"], "grid.linewidth": 0.8, "grid.linestyle": "-",
        "xtick.direction": "out", "ytick.direction": "out",
        "font.size": 10, "axes.titlesize": 12, "legend.fontsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
    })


# Kolom `model` datang dalam beberapa bentuk: "unsloth/Qwen3-0.6B" dari notebook 03, path
# direktori adapter dari notebook 04, dan run_id "Qwen3-0.6B|S1|3407" / "baseline|Qwen3-1.7B".
# Mencari nama modelnya langsung jauh lebih tahan banting daripada memotong pemisah.
_MODEL_RE = re.compile(r"Qwen3-[\d.]+B")


def short(name):
    """-> nama student, mis. 'Qwen3-0.6B', dari bentuk apa pun di atas."""
    m = _MODEL_RE.search(str(name))
    return m.group(0) if m else str(name).split("/")[-1].split("|")[0]


# ------------------------------------------------------------------ statistik

def boot_ci(values, stat=np.mean, n_boot=1000, seed=0, alpha=0.05):
    """CI persentil bootstrap. Dipakai juga untuk AUC, yang bukan proporsi sederhana."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0 or np.all(np.isnan(v)):
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = [stat(rng.choice(v, size=len(v), replace=True)) for _ in range(n_boot)]
    return tuple(np.percentile(draws, [alpha / 2 * 100, (1 - alpha / 2) * 100]))


def load_items(pert_dir, run_id):
    """Baca set perturbasi beku satu konfigurasi -> DataFrame per soal."""
    p = Path(pert_dir) / f"{run_id.replace('|', '_')}.jsonl"
    if not p.exists():
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()])


def agg_accuracy(df, dataset="gsm8k"):
    """-> (tabel mean/sd per model x variant, dict baseline per model)."""
    d = df[df.dataset == dataset].copy()
    d["m"] = d.model.map(short)
    base = dict(zip(d[d.variant == "pre-distill"].m,
                    d[d.variant == "pre-distill"].accuracy * 100))
    fin = d[d.variant != "pre-distill"]
    g = (fin.groupby(["m", "variant"])
            .agg(acc=("accuracy", lambda s: s.mean() * 100),
                 sd=("accuracy", lambda s: s.std(ddof=1) * 100 if len(s) > 1 else np.nan),
                 tok=("avg_output_tokens", "mean"),
                 n_seed=("accuracy", "size"))
            .reset_index())
    return g, base


def agg_faith(fa, col):
    d = fa.copy()
    d["m"] = d.run_id.map(short)
    base = dict(zip(d[d.variant == "pre-distill"].m, d[d.variant == "pre-distill"][col]))
    fin = d[d.variant != "pre-distill"]
    g = (fin.groupby(["m", "variant"])[col]
            .agg(v="mean", sd=lambda s: s.std(ddof=1) if len(s) > 1 else np.nan)
            .reset_index())
    return g, base


def _finish(fig, ax, T, note, path):
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    fig.text(.01, .01, note, fontsize=7.5, color=T["muted"])
    fig.tight_layout(rect=(0, .03, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ figure

def fig1(g, base, T, path):
    """Akurasi vs tingkat supervisi, dengan garis baseline pre-distillation."""
    style(T)
    fig, ax = plt.subplots(figsize=(6.6, 4.3), dpi=200)
    xs = range(len(VARIANTS))
    for i, (m, c) in enumerate(zip(sorted(g.m.unique()), (T["s1"], T["s2"]))):
        s = g[g.m == m].set_index("variant").reindex(VARIANTS)
        ax.errorbar(xs, s.acc, yerr=s.sd, marker="o", ms=7, lw=2, capsize=4,
                    color=c, mec=T["surface"], mew=2, zorder=3, label=m)
        if m in base:
            ax.axhline(base[m], ls=(0, (5, 4)), lw=1.4, color=c, alpha=.75, zorder=1)
            ax.annotate(f"{m} tanpa distilasi", (len(VARIANTS) - 1, base[m]),
                        xytext=(-4, 5), textcoords="offset points", ha="right",
                        fontsize=8, color=c)
    ax.set_xticks(list(xs), [VLABEL[v] for v in VARIANTS])
    ax.margins(x=.08)
    ax.set_ylabel("akurasi GSM8K (%)")
    ax.set_title("Akurasi vs tingkat supervisi CoT", color=T["ink"], pad=12, loc="left")
    ax.legend(frameon=False, labelcolor=T["ink2"], loc="lower right")
    _finish(fig, ax, T, "error bar = sd antar 2 seed. Garis putus = akurasi tanpa distilasi.", path)


FAITH_PANELS = [
    ("early_auc", "AUC early-answering", "rendah = lebih setia"),
    ("mistake_sensitivity", "sensitivitas adding-mistakes", "tinggi = lebih setia"),
    ("para_consistency", "konsistensi paraphrasing", "tinggi = lebih setia"),
]


def fig2(fa, T, path):
    """Tiga panel, satu per uji. Arah 'lebih setia' berbeda per panel, jadi ditulis."""
    style(T)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), dpi=200)
    vs = [v for v in VARIANTS if v != "S0"]
    for ax, (col, title, direction) in zip(axes, FAITH_PANELS):
        g, base = agg_faith(fa, col)
        for m, c in zip(sorted(g.m.unique()), (T["s1"], T["s2"])):
            s = g[g.m == m].set_index("variant").reindex(vs)
            ax.errorbar(range(len(vs)), s.v, yerr=s.sd, marker="o", ms=6, lw=2, capsize=4,
                        color=c, mec=T["surface"], mew=2, zorder=3, label=m)
            if m in base and not pd.isna(base[m]):
                ax.axhline(base[m], ls=(0, (5, 4)), lw=1.2, color=c, alpha=.7, zorder=1)
        ax.set_xticks(range(len(vs)), vs)
        ax.set_title(title, color=T["ink"], pad=8, loc="left", fontsize=11)
        ax.set_xlabel(direction, fontsize=8, color=T["muted"])
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, labelcolor=T["ink2"])
    fig.text(.005, .01, "S0 tidak menghasilkan rationale, jadi tidak terdefinisi dan tidak "
                        "diplot. Garis putus = model tanpa distilasi.",
             fontsize=7.5, color=T["muted"])
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig3(g, fa, T, path, faith_col="mistake_sensitivity"):
    """Figure utama: bidang trade-off akurasi x kesetiaan, ukuran titik = biaya token."""
    style(T)
    fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=200)
    gf, _ = agg_faith(fa, faith_col)
    merged = g.merge(gf.rename(columns={"v": "faith"}), on=["m", "variant"])
    tk = merged.tok
    sizes = 60 + 340 * (tk - tk.min()) / max(tk.max() - tk.min(), 1e-9)

    for m, c in zip(sorted(merged.m.unique()), (T["s1"], T["s2"])):
        s = merged[merged.m == m]
        ax.scatter(s.acc, s.faith, s=sizes[s.index], color=c, alpha=.85,
                   edgecolor=T["surface"], linewidth=2, zorder=3)
        order = s.set_index("variant").reindex([v for v in VARIANTS if v in set(s.variant)])
        ax.plot(order.acc, order.faith, color=c, lw=1.2, alpha=.45, zorder=2)
        for _, r in s.iterrows():
            ax.annotate(r.variant, (r.acc, r.faith), xytext=(9, -3),
                        textcoords="offset points", fontsize=9, color=T["ink2"])

    ax.margins(x=.12, y=.10)          # label varian ditulis di kanan titik, butuh ruang
    ax.set_xlabel("akurasi GSM8K (%)")
    ax.set_ylabel(f"kesetiaan — {FAITH_PANELS[1][1]}")
    ax.set_title("Bidang trade-off: akurasi x kesetiaan x biaya token",
                 color=T["ink"], pad=12, loc="left")
    handles = [Line2D([], [], marker="o", ls="", ms=8, color=c, mec=T["surface"], mew=2, label=m)
               for m, c in zip(sorted(merged.m.unique()), (T["s1"], T["s2"]))]
    handles += [Line2D([], [], marker="o", ls="", ms=s ** .5, color=T["muted"],
                       mec=T["surface"], mew=1.5, label=f"{int(t)} token")
                for s, t in ((70, tk.min()), (300, tk.max()))]
    ax.legend(handles=handles, frameon=False, labelcolor=T["ink2"], loc="best")
    _finish(fig, ax, T, "Ukuran titik = rata-rata token output. Kanan-atas-kecil = terbaik di ketiga "
                        "sumbu. S0 tidak diplot: tanpa rationale, kesetiaan tidak terdefinisi.", path)


def fig4(pert_dir, fa, T, path):
    """Kurva early-answering mentah, satu panel per student."""
    style(T)
    models = sorted(fa[fa.variant != "pre-distill"].run_id.map(short).unique())
    fig, axes = plt.subplots(1, max(len(models), 1), figsize=(5.4 * max(len(models), 1), 4.2),
                             dpi=200, squeeze=False)
    vs = [v for v in VARIANTS if v != "S0"]
    for ax, m in zip(axes[0], models):
        for v, c in zip(vs, (T["s1"], T["s2"], T["s3"])):
            sub = fa[(fa.run_id.map(short) == m) & (fa.variant == v)]
            if sub.empty:
                continue
            ys = [sub[f"early_acc_{int(f * 100)}"].mean() * 100 if f < 1
                  else sub.acc_full.mean() * 100 for f in FRACS]
            ax.plot([f * 100 for f in FRACS], ys, marker="o", ms=5, lw=2, color=c,
                    mec=T["surface"], mew=1.5, label=v, zorder=3)
        ax.set_title(m, color=T["ink"], pad=8, loc="left", fontsize=11)
        ax.set_xlabel("persentase rationale dibuka (%)")
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("akurasi (%)")
    axes[0][0].legend(frameon=False, labelcolor=T["ink2"])
    fig.text(.005, .01, "Kurva datar sejak awal = rationale tidak dipakai. Naik bertahap = "
                        "penalaran benar-benar menyebabkan jawaban.",
             fontsize=7.5, color=T["muted"])
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def build_all(results, faith, outdir, pert_dir=None):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    g, base = agg_accuracy(results)
    made = []
    for mode, T in THEMES.items():
        sfx = "" if mode == "light" else "_dark"
        fig1(g, base, T, outdir / f"fig1_accuracy{sfx}.png")
        fig2(faith, T, outdir / f"fig2_faithfulness{sfx}.png")
        fig3(g, faith, T, outdir / f"fig3_tradeoff{sfx}.png")
        fig4(pert_dir, faith, T, outdir / f"fig4_early_answering{sfx}.png")
        made += [f"fig{i}{sfx}" for i in (1, 2, 3, 4)]
    return g, base, made


# ------------------------------------------------------------------ smoke test

def _synthetic(seed=0):
    """Hasil palsu berbentuk sama dengan yang sungguhan, untuk menguji kode plot."""
    rng = random.Random(seed)
    acc = {("Qwen3-0.6B", v): a for v, a in zip(VARIANTS, (.12, .44, .57, .59))}
    acc |= {("Qwen3-1.7B", v): a for v, a in zip(VARIANTS, (.20, .59, .74, .79))}
    rows = []
    for m in ("Qwen3-0.6B", "Qwen3-1.7B"):
        for d in ("gsm8k", "gsm_plus", "svamp"):
            rows.append({"run_id": f"baseline|{m}", "model": f"unsloth/{m}",
                         "variant": "pre-distill", "seed": -1, "dataset": d,
                         "accuracy": (.35 if "0.6" in m else .73) - (.1 if d != "gsm8k" else 0),
                         "avg_output_tokens": 90.0, "truncated_frac": 0.0})
            for v in VARIANTS:
                for s in (3407, 42):
                    rows.append({"run_id": f"{m}|{v}|{s}", "model": f"unsloth/{m}",
                                 "variant": v, "seed": s, "dataset": d,
                                 "accuracy": acc[(m, v)] + rng.uniform(-.02, .02)
                                             - (.12 if d != "gsm8k" else 0),
                                 "avg_output_tokens": {"S0": 5.4, "S1": 42, "S2": 97,
                                                       "S3": 375}[v],
                                 "truncated_frac": 0.0})
    fa = []
    for m in ("Qwen3-0.6B", "Qwen3-1.7B"):
        fa.append({"run_id": f"baseline|{m}", "variant": "pre-distill", "seed": -1, "n": 150,
                   "acc_full": .35 if "0.6" in m else .73, "early_auc": .62,
                   **{f"early_acc_{p}": .2 + p / 250 for p in (0, 20, 40, 60, 80)},
                   "mistake_coverage": .96, "mistake_acc_ctrl": .5, "mistake_acc_broken": .2,
                   "mistake_sensitivity": .55, "para_jaccard": .55, "para_consistency": .8,
                   "para_acc": .5, "gen_tokens": 90.0})
        for v in ("S1", "S2", "S3"):
            for s in (3407, 42):
                fa.append({"run_id": f"{m}|{v}|{s}", "variant": v, "seed": s, "n": 150,
                           "acc_full": acc[(m, v)], "early_auc": {"S1": .82, "S2": .68,
                                                                  "S3": .55}[v] + rng.uniform(-.03, .03),
                           **{f"early_acc_{p}": .15 + p / 200 for p in (0, 20, 40, 60, 80)},
                           "mistake_coverage": .96,
                           "mistake_acc_ctrl": .5, "mistake_acc_broken": .25,
                           "mistake_sensitivity": {"S1": .35, "S2": .52,
                                                   "S3": .66}[v] + rng.uniform(-.03, .03),
                           "para_jaccard": .55,
                           "para_consistency": {"S1": .74, "S2": .81, "S3": .86}[v],
                           "para_acc": acc[(m, v)] - .02, "gen_tokens": 90.0})
    return pd.DataFrame(rows), pd.DataFrame(fa)


def demo(outdir="results/figures/_smoke"):
    res, fa = _synthetic()
    g, base, made = build_all(res, fa, outdir)

    assert len(made) == 8, made
    for f in made:
        p = Path(outdir) / (f.replace("fig1", "fig1_accuracy").replace("fig2", "fig2_faithfulness")
                            .replace("fig3", "fig3_tradeoff").replace("fig4", "fig4_early_answering")
                            + ".png")
        assert p.exists() and p.stat().st_size > 5000, f"{p} kosong/terlalu kecil"

    assert set(g.variant) == set(VARIANTS) and len(g) == 8, g
    assert g.n_seed.eq(2).all(), "sd butuh 2 seed"
    assert set(base) == {"Qwen3-0.6B", "Qwen3-1.7B"}

    lo, hi = boot_ci([1] * 120 + [0] * 30)          # proporsi 0.80, n=150
    assert lo < 0.80 < hi and (hi - lo) < 0.20, (lo, hi)
    assert all(math.isnan(x) for x in boot_ci([]))

    print(f"figures smoke test OK - 8 figure di {outdir}")
    print(f"  lebar CI 95% pada proporsi .80 (n=150): {(hi - lo) * 100:.1f} pp")


if __name__ == "__main__":
    demo()
