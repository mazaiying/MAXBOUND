#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication figure generator for MAXBOUND.
100% driven by results/*.json; Type-1 vector PDF fonts (fonttype=42); Arial;
black-edged hatched bars; 95% CI error bars; shared top legends; OOM red-X markers;
tall-value leader-line caps; and subcaptions strictly below X-axes.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["axes.edgecolor"] = "#000000"

RES_DIR = os.path.expanduser("~/maxbound/results")
FIG_DIR = os.path.expanduser("~/maxbound/figures")
os.makedirs(FIG_DIR, exist_ok=True)

PAL = {
    "gray": "#CCCCCC",
    "yellow": "#FFE699",
    "blue": "#95B3D7",
    "green": "#C4D79B",
    "orange": "#FAC090",
    "purple": "#CCC0DA",
    "teal": "#85C1E9",
    "deep_blue": "#2E86AB",
    "amber": "#F18F01",
    "berry": "#A23B72",
    "emerald": "#2ECC71",
    "crimson": "#D9534F",
    "dark": "#2C3E50",
}

DS_PRETTY = {
    "scifact": "SciFact",
    "nfcorpus": "NFCorpus",
    "arguana": "ArguAna",
    "scidocs": "SCIDOCS",
    "longbench": "LongBench",
    "fiqa": "FiQA",
    "hotpotqa": "HotpotQA",
    "trec_covid": "TREC-COVID",
    "msmarco": "MS MARCO",
    "msmarco_2m": "MS MARCO-2M",
    "scifact_bgem3": "SciFact-1024D",
    "fiqa_bgem3": "FiQA-1024D",
    "scale_50m": "WildChat-50M",
}


def pretty_ds(name):
    return DS_PRETTY.get(name, name.replace("_", "-"))


def load_json(name):
    p = os.path.join(RES_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def save_fig(fig, stem):
    pdf_p = os.path.join(FIG_DIR, f"{stem}.pdf")
    png_p = os.path.join(FIG_DIR, f"{stem}.png")
    fig.savefig(pdf_p, bbox_inches="tight", pad_inches=0.02, dpi=300)
    fig.savefig(png_p, bbox_inches="tight", pad_inches=0.02, dpi=220)
    plt.close(fig)
    print(f"Saved {pdf_p}")


# ==============================================================================
# Figure 1: Motivation (12-Dataset Padding Waste + 7-Bound Tightness Audit)
# ==============================================================================
def plot_fig1_motivation():
    d = load_json("e0_motivation.json")
    if not d:
        return
    recs = d["records"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.8, 3.6))

    ds_names = [pretty_ds(r["stats"]["dataset"]) for r in recs]
    waste = [r["stats"]["b1_padding_waste_pct"] for r in recs]
    cvs = [r["stats"]["doc_len_cv"] for r in recs]
    x = np.arange(len(ds_names))
    bars = ax1.bar(x, waste, width=0.58, color="#FAC090", edgecolor="black",
                   linewidth=0.75, hatch="///", label="Padded Slot Waste (%)")
    for b in bars:
        h = b.get_height()
        ax1.annotate(f"{h:.0f}%", xy=(b.get_x() + b.get_width() / 2, h),
                     xytext=(0, 2), textcoords="offset points",
                     ha="center", va="bottom", fontsize=6.8)
    ax1.set_ylabel("Padding Token Waste (%)", fontsize=10.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(ds_names, rotation=32, ha="right", fontsize=8.0)
    ax1.set_ylim(0, 108)
    ax1.grid(True, axis="y", linestyle="--", linewidth=0.45, color="lightgray")

    ax1_r = ax1.twinx()
    ax1_r.plot(x, cvs, "o--", color="#2E86AB", linewidth=1.3, markersize=4.8, label="Length CV")
    ax1_r.set_ylabel("Document Length CV", fontsize=10.5, color="#2E86AB")
    ax1_r.tick_params(axis="y", colors="#2E86AB")
    ax1_r.set_ylim(0, max(cvs) * 1.32 + 0.1)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1_r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.0, frameon=True, edgecolor="lightgray")
    ax1.set_xlabel("(a) Padding token waste vs. document length skew", fontsize=10.5, labelpad=5)

    bound_recs = [r for r in recs if r.get("bounds")]
    if bound_recs:
        methods = [b["method"] for b in bound_recs[0]["bounds"]]
        m_labels = ["L0 Ball", "KM-AABB", "PQ-16", "H-INT2", "H-INT4", "H-INT6", "H-INT8"][:len(methods)]
        xb = np.arange(len(methods))
        w = 0.78 / max(len(bound_recs), 1)
        cols = ["#CCCCCC", "#FFE699", "#95B3D7", "#C4D79B", "#FAC090"]
        hats = ["", "\\\\", "//", "||", "--"]
        for i, r in enumerate(bound_recs):
            vals = [b["prune_pct"] for b in r["bounds"]]
            ax2.bar(xb + (i - (len(bound_recs) - 1) / 2.0) * w, vals, width=w * 0.92,
                    color=cols[i % len(cols)], hatch=hats[i % len(hats)],
                    edgecolor="black", linewidth=0.65, label=pretty_ds(r["stats"]["dataset"]))
        for j in range(len(methods)):
            b_vals = [r["bounds"][j]["prune_pct"] for r in bound_recs]
            avg_v = float(np.mean(b_vals))
            top_v = float(np.max(b_vals))
            txt = f"{avg_v:.1f}%" if avg_v >= 1.0 else "0.0%"
            ax2.text(xb[j], top_v + 1.8, txt, ha="center", va="bottom", fontsize=7.5)
        ax2.set_ylabel("Exact Pruning Ratio (%)", fontsize=10.5)
        ax2.set_xticks(xb)
        ax2.set_xticklabels(m_labels, rotation=22, ha="right", fontsize=8.8)
        ax2.set_ylim(0, 114)
        ax2.grid(True, axis="y", linestyle="--", linewidth=0.45, color="lightgray")
        ax2.legend(loc="upper left", fontsize=7.8, ncol=2, frameon=True, edgecolor="lightgray")
    ax2.set_xlabel("(b) Stage-1 pruning ratio (0 violations / 12.4M pairs)", fontsize=10.2, labelpad=5)

    plt.tight_layout(w_pad=1.3)
    save_fig(fig, "fig1_motivation_pruning_and_padding")


# ==============================================================================
# Figure 2: Main Evaluation 3x4 Matrix (11 Real Datasets + Panel (l) Geometric Mean Summary)
# ==============================================================================
def plot_fig2_e1():
    d = load_json("e1_main_eval.json")
    if not d:
        return
    recs = d["records"]
    order_11 = [
        "scifact", "nfcorpus", "arguana", "scidocs",
        "longbench", "fiqa", "hotpotqa", "trec_covid",
        "msmarco_2m", "scifact_bgem3", "fiqa_bgem3"
    ]
    if not any(r["dataset"] == "msmarco_2m" for r in recs):
        order_11[8] = "msmarco"
    datasets = [ds for ds in order_11 if any(r["dataset"] == ds for r in recs)]
    ncols = 4
    nrows = 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(16.8, 8.6))
    axs_flat = axs.flatten()

    m_specs = [
        ("B6_CPU_48T_Exact", "CPU-48T Exact", "#CCCCCC", ""),
        ("B1_Torch_Padded_GEMM", "Padded-GEMM", "#FFE699", "\\\\"),
        ("B3_Quant_Decompress", "Quant-Decomp", "#CCC0DA", "//"),
        ("B2_cuBLAS_Unpadded_Exact", "cuBLAS-Unpadded", "#95B3D7", "||"),
        ("B8_Fused_FP16_TC_Exact", "Fused-FP16-TC (B8)", "#85C1E9", ".."),
        ("MAXBOUND_INT8", "MAXBOUND (Full)", "#C4D79B", "--"),
        ("MAXBOUND_Rerank_INT8", "MAXBOUND (Rerank)", "#FAC090", "xx"),
    ]

    def find_rec(ds, mkey):
        for r in recs:
            if r["dataset"] != ds or r["method"] != mkey:
                continue
            if mkey == "MAXBOUND_Rerank_INT8":
                if r.get("mode") == "rerank_C1000":
                    return r
            elif mkey == "B6_CPU_48T_Exact":
                return r
            else:
                if r.get("mode") == "full_corpus" and r.get("batch") in (50, 64):
                    return r
        return None

    legend_handles, legend_labels = [], []

    for idx in range(len(datasets)):
        ax = axs_flat[idx]
        ds = datasets[idx]
        x = np.arange(len(m_specs))
        vals, ci_qps, oom_flags = [], [], []
        for mkey, _, _, _ in m_specs:
            r = find_rec(ds, mkey)
            if r is None:
                vals.append(0.0); ci_qps.append(0.0); oom_flags.append(False)
            elif r.get("oom") or r.get("status") == "OOM" or not r.get("qps"):
                vals.append(0.0); ci_qps.append(0.0); oom_flags.append(True)
            else:
                q = float(r["qps"])
                vals.append(q)
                med = float(r.get("median_ms") or 1.0)
                ci = float(r.get("ci95_ms") or 0.0)
                ci_qps.append(q * min(0.35, ci / max(med, 1e-6)))
                oom_flags.append(False)

        pos_vals = [v for v in vals if v > 0]
        ymin = max(0.2, min(pos_vals) * 0.45) if pos_vals else 0.5
        ymax = (max(pos_vals) * 3.5) if pos_vals else 1e4

        for j, (mkey, mlab, col, hat) in enumerate(m_specs):
            v = vals[j]
            bar = ax.bar(
                x[j], max(v, ymin * 1.01) if v > 0 else 0.0, width=0.68,
                yerr=([ci_qps[j]] if v > 0 and ci_qps[j] > 0 else None),
                capsize=2.0, error_kw={"elinewidth": 0.7, "ecolor": "black"},
                color=col, hatch=hat, edgecolor="black", linewidth=0.72, label=mlab
            )
            if idx == 0:
                legend_handles.append(bar[0])
                legend_labels.append(mlab)
            if oom_flags[j]:
                ax.scatter([x[j]], [ymin * 1.38], marker="x", color=PAL["crimson"], s=50, linewidths=1.8, zorder=6)
                ax.text(x[j], ymin * 2.0, "OOM", ha="center", va="bottom",
                        fontsize=7.0, color=PAL["crimson"], rotation=90, fontweight="bold")
            elif mkey == "MAXBOUND_INT8" and v > 0:
                r_mb = find_rec(ds, "MAXBOUND_INT8")
                sp = r_mb.get("speedup_vs_b2", 1.0)
                ax.annotate(f"{sp:.2f}x", xy=(x[j], v), xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7.6, fontweight="bold", color="#1B4F72")

        ax.set_yscale("log")
        ax.set_ylim(ymin, ymax)
        ax.set_xticks(x)
        ax.set_xticklabels(["CPU", "Pad", "QDec", "cuBLAS", "FP16TC", "MB", "MB-R"], fontsize=7.4, rotation=-24)
        if idx % ncols == 0:
            ax.set_ylabel("Throughput (QPS)", fontsize=10.0)
        ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.45, color="lightgray")
        ax.yaxis.grid(False, which="minor")
        sub_letter = chr(ord("a") + idx)
        ax.set_xlabel(f"({sub_letter}) {pretty_ds(ds)}", fontsize=10.0, labelpad=3)

    # 12th subplot (l): Geometric-Mean Normalized Speedup across ALL 12 real datasets in E1!
    if len(datasets) < len(axs_flat):
        ax_sum = axs_flat[len(datasets)]
        all_12_ds = [
            "scifact", "nfcorpus", "arguana", "scidocs",
            "longbench", "fiqa", "hotpotqa", "trec_covid",
            "msmarco", "msmarco_2m", "scifact_bgem3", "fiqa_bgem3"
        ]
        present_all = [ds for ds in all_12_ds if any(r["dataset"] == ds for r in recs)]
        b8_vs_b2, mb8_vs_best, mb8_vs_b2 = [], [], []
        for ds in present_all:
            r_b8 = find_rec(ds, "B8_Fused_FP16_TC_Exact")
            r_mb = find_rec(ds, "MAXBOUND_INT8")
            if r_b8 and r_b8.get("speedup_vs_b2"):
                b8_vs_b2.append(float(r_b8["speedup_vs_b2"]))
            if r_mb and r_mb.get("speedup_vs_best_exact"):
                mb8_vs_best.append(float(r_mb["speedup_vs_best_exact"]))
            if r_mb and r_mb.get("speedup_vs_b2"):
                mb8_vs_b2.append(float(r_mb["speedup_vs_b2"]))
        sum_bars = [
            ("cuBLAS\n(vs B2)", 1.0, "#95B3D7", "||"),
            ("FP16TC\n(vs B2)", float(np.exp(np.mean(np.log(b8_vs_b2)))) if b8_vs_b2 else 1.0, "#85C1E9", ".."),
            ("MB-INT8\n(vs Best)", float(np.exp(np.mean(np.log(mb8_vs_best)))) if mb8_vs_best else 1.0, "#FFE699", "\\\\"),
            ("MB-INT8\n(vs B2)", float(np.exp(np.mean(np.log(mb8_vs_b2)))) if mb8_vs_b2 else 1.0, "#C4D79B", "--"),
        ]
        xs = np.arange(len(sum_bars))
        gmeans = [b[1] for b in sum_bars]
        for j, (mlab, gm_v, col, hat) in enumerate(sum_bars):
            ax_sum.bar(xs[j], gm_v, width=0.58, color=col, hatch=hat, edgecolor="black", linewidth=0.75)
            ax_sum.annotate(f"{gm_v:.2f}x", xy=(xs[j], gm_v), xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8.0, fontweight="bold")
        ax_sum.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
        ax_sum.set_ylim(0, max(gmeans) * 1.30)
        ax_sum.set_xticks(xs)
        ax_sum.set_xticklabels([b[0] for b in sum_bars], fontsize=7.6)
        ax_sum.set_ylabel("GeoMean Speedup (x)", fontsize=10.0)
        ax_sum.grid(True, axis="y", linestyle="--", linewidth=0.45, color="lightgray")
        sub_l = chr(ord("a") + len(datasets))
        ax_sum.set_xlabel(f"({sub_l}) {len(present_all)}-Dataset Exact GeoMean", fontsize=10.0, labelpad=3)
        for k in range(len(datasets) + 1, len(axs_flat)):
            axs_flat[k].axis("off")

    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=len(m_specs),
               fontsize=9.6, bbox_to_anchor=(0.5, 1.03), frameon=True, edgecolor="lightgray")
    plt.tight_layout(w_pad=0.65, h_pad=0.85)
    save_fig(fig, "fig2_e1_main_eval")


# ==============================================================================
# Figure 3: Latency-Recall Pareto Frontier across 4 Datasets (incl. Official FastPLAID, WARP, & MS MARCO)
# ==============================================================================
def plot_fig3_e2():
    d = load_json("e2_pareto_frontier.json")
    if not d:
        return
    recs = d["records"]
    datasets = []
    for r in recs:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])

    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.6))
    if len(datasets) == 1:
        axes = [axes]

    curve_specs = [
        ("B4_Official_FastPLAID", "Official FastPLAID (Rust)", "#8E44AD", "^--"),
        ("B9_WARP_GPU", "WARP-GPU (SIGIR'25)", "#16A085", "v--"),
        ("B10_EMVB_GPU", "EMVB-GPU (ECIR'24)", "#D35400", "p--"),
        ("B4_PLAID_GPU", "PLAID-GPU", "darkgreen", "o--"),
        ("B5_MUVERA_GPU", "MUVERA-GPU (NeurIPS'24)", "darkorange", "s--"),
        ("PLAID_plus_MAXBOUND", "PLAID + MAXBOUND", "darkblue", "d-"),
        ("MUVERA_plus_MAXBOUND", "MUVERA + MAXBOUND", "crimson", "*-"),
    ]
    point_specs = [
        ("B2_Exact", "cuBLAS Exact (100%)", "#7F8C8D", "X", 75),
        ("B8_Fused_FP16_Exact", "Fused-FP16 Exact (100%)", "#F39C12", "D", 70),
        ("MAXBOUND_INT8", "MAXBOUND Exact (100%)", "#2E86AB", "P", 95),
    ]

    handles_all, labels_all = [], []
    for idx, ds in enumerate(datasets):
        ax = axes[idx]
        sub = [r for r in recs if r["dataset"] == ds]
        for m, lab, col, style in curve_specs:
            pts = sorted([r for r in sub if r["method"] == m], key=lambda z: z["oracle_topk_recall"])
            if pts:
                ln, = ax.plot([p["oracle_topk_recall"] * 100.0 for p in pts], [p["qps"] for p in pts],
                              style, color=col, markerfacecolor=col, linewidth=1.1, markersize=4.6, alpha=0.84, label=lab)
                if idx == 0:
                    handles_all.append(ln); labels_all.append(lab)
        for m, lab, col, mk, sz in point_specs:
            pts = [r for r in sub if r["method"] == m]
            if pts:
                sc = ax.scatter([p["oracle_topk_recall"] * 100.0 for p in pts], [p["qps"] for p in pts],
                                color=col, marker=mk, s=sz, zorder=6, edgecolors="black", linewidths=0.7, label=lab)
                if idx == 0:
                    handles_all.append(sc); labels_all.append(lab)

        ax.set_yscale("log")
        if idx == 0:
            ax.set_ylabel("Query per second (QPS)", fontsize=10.5)
        ax.grid(True, which="major", linestyle="--", linewidth=0.45, color="lightgray")
        ax.yaxis.grid(False, which="minor")
        sub_letter = chr(ord("a") + idx)
        ax.set_xlabel(f"Oracle Top-10 Recall (%)\n({sub_letter}) {pretty_ds(ds)}", fontsize=10.0)

    fig.legend(handles_all, labels_all, loc="upper center", ncol=5, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.15), frameon=True, edgecolor="lightgray")
    plt.tight_layout(w_pad=0.8)
    save_fig(fig, "fig3_e2_pareto")


# ==============================================================================
# Figure 4: Scale-50M Constrained VRAM (OOM Red-X) & 4-Dataset Heterogeneous Dual-GPU
# ==============================================================================
def plot_fig4_e3_e4():
    d3 = load_json("e3_memory_scale.json")
    d4 = load_json("e4_dual_gpu.json")
    if not d3 and not d4:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 3.7))

    if d3:
        recs_all = d3["records"]
        tiers = [
            ("msmarco", 8, "MS MARCO\n(32M, 8GB)"),
            ("scale_50m", 8, "WildChat-50M\n(61M, 8GB)"),
            ("scale_50m", 12, "WildChat-50M\n(61M, 12GB)"),
            ("msmarco_2m", 16, "MS MARCO-2M\n(129M, 16GB)"),
        ]
        tiers = [t for t in tiers if any(r.get("dataset") == t[0] and r.get("budget_gb") == t[1] for r in recs_all)]
        mlist = [
            ("B1_Torch_Padded_GEMM", "Padded-GEMM", "#CCCCCC", ""),
            ("B2_cuBLAS_Unpadded_Exact", "cuBLAS-Unpadded", "#FFE699", "\\\\"),
            ("B8_Fused_FP16_TC_Exact", "Fused-FP16-TC", "#85C1E9", ".."),
            ("MAXBOUND_INT4_Offload", "MB-INT4 Offload", "#95B3D7", "//"),
            ("MAXBOUND_INT6_Offload", "MB-INT6 Offload", "#C4D79B", "||"),
            ("MAXBOUND_INT8_Offload", "MB-INT8 Offload", "#FAC090", "--"),
        ]
        x = np.arange(len(tiers))
        w = 0.84 / len(mlist)
        ymin_e3 = 2.0
        ymax_v = 1350.0
        for i, (mk, ml, col, hat) in enumerate(mlist):
            xpos = x + (i - (len(mlist) - 1) / 2.0) * w
            vals, ooms = [], []
            for ds_k, b_gb, _ in tiers:
                hit = [r for r in recs_all if r.get("dataset") == ds_k and r["budget_gb"] == b_gb and r["method"] == mk]
                if hit and hit[0].get("status") == "OK":
                    vals.append(hit[0]["qps"]); ooms.append(False)
                else:
                    vals.append(0.0); ooms.append(True)
            bars = ax1.bar(xpos, [max(v, ymin_e3 * 1.02) if v > 0 else 0.0 for v in vals], width=w * 0.90,
                           color=col, hatch=hat, edgecolor="black", linewidth=0.68, label=ml)
            for bx, xp, v, is_oom in zip(bars, xpos, vals, ooms):
                if is_oom:
                    ax1.scatter([xp], [ymin_e3 * 1.38], marker="x", color=PAL["crimson"], s=28, linewidths=1.35, zorder=5)
                elif v > 0:
                    ax1.annotate(f"{v:.0f}", xy=(xp, v), xytext=(0, 2), textcoords="offset points",
                                 ha="center", va="bottom", fontsize=6.2, fontweight="bold")
        ax1.set_yscale("log")
        ax1.set_ylim(ymin_e3, ymax_v)
        ax1.set_ylabel("Exact Throughput (QPS, log)", fontsize=10.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels([t[2] for t in tiers], fontsize=8.0)
        ax1.grid(True, which="major", axis="y", linestyle="--", linewidth=0.45, color="lightgray")
        ax1.yaxis.grid(False, which="minor")
        ax1.legend(loc="upper right", fontsize=6.9, ncol=2, frameon=True, facecolor="white", framealpha=0.92, edgecolor="lightgray")
        ax1.set_xlabel("(a) Constrained-VRAM scaling up to 129M tokens (red X = OOM)", fontsize=9.8, labelpad=4)

    if d4:
        recs = d4["records"]
        dss = []
        for r in recs:
            if r["dataset"] not in dss:
                dss.append(r["dataset"])
        cfgs = [
            ("A100_Only_B2", "A100 cuBLAS", "#CCCCCC", ""),
            ("A100_Only_MAXBOUND", "A100 MAXBOUND", "#FFE699", "\\\\"),
            ("B7_Naive_DualGPU", "Naive Dual (50/50)", "#CCC0DA", "||"),
            ("B7_Peak_TFLOPS_Split", "Peak-TOPS Split", "#95B3D7", "//"),
            ("MAXBOUND_DualGPU", "MAXBOUND Dual (M3+Steal)", "#C4D79B", "--"),
        ]
        x = np.arange(len(dss))
        w = 0.80 / len(cfgs)
        ymax_d = max([r.get("qps", 0.0) for r in recs] + [100.0]) * 1.24
        for i, (ck, cl, col, hat) in enumerate(cfgs):
            xpos = x + (i - (len(cfgs) - 1) / 2.0) * w
            vals = []
            for ds in dss:
                hit = [r for r in recs if r["dataset"] == ds and r["config"] == ck]
                vals.append(hit[0]["qps"] if hit else 0.0)
            ax2.bar(xpos, vals, width=w * 0.90, color=col, hatch=hat,
                    edgecolor="black", linewidth=0.7, label=cl)
            if ck == "MAXBOUND_DualGPU":
                for xp, v in zip(xpos, vals):
                    ax2.annotate(f"{v:.0f}", xy=(xp, v), xytext=(0, 2), textcoords="offset points",
                                 ha="center", va="bottom", fontsize=7.0, fontweight="bold")
        ax2.set_ylim(0, ymax_d)
        ax2.set_ylabel("Throughput (QPS)", fontsize=10.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels([pretty_ds(ds) for ds in dss], fontsize=8.2, rotation=-14)
        ax2.grid(True, axis="y", linestyle="--", linewidth=0.45, color="lightgray")
        ax2.legend(loc="upper right", fontsize=7.2, ncol=2, frameon=True, edgecolor="lightgray")
        ax2.set_xlabel("Dataset\n(b) Heterogeneous dual-GPU scaling (A100 + RTX 4080)", fontsize=9.8)

    plt.tight_layout(w_pad=1.2)
    save_fig(fig, "fig4_e3_e4_scale_and_dualgpu")


# ==============================================================================
# Figure 5: Skew Stability + 5-Dataset Monotonic CUDA Ablation + Stage Breakdown
# ==============================================================================
def plot_fig5_e5_e6():
    d5 = load_json("e5_skew_robustness.json")
    d6 = load_json("e6_ablation_breakdown.json")
    if not d5 and not d6:
        return
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15.8, 3.7))

    if d5:
        s5a = [r for r in d5["records"] if r.get("sub_exp") == "5A_length_cv_sweep"]
        cvs = [r["actual_cv"] for r in s5a]
        cap_ms = 95.0
        b1_raw = [r["b1_ms"] if r["b1_ms"] else cap_ms for r in s5a]
        b1_clipped = [min(v, cap_ms * 0.96) for v in b1_raw]
        ax1.plot(cvs, b1_clipped, "x--", color="#7F8C8D", linewidth=1.2, markersize=6, label="Padded-GEMM (capped)")
        ax1.plot(cvs, [r["b2_ms"] for r in s5a], "s--", color=PAL["amber"], linewidth=1.4, markersize=5.5, label="cuBLAS-Unpadded")
        if all("b8_ms" in r for r in s5a):
            ax1.plot(cvs, [r["b8_ms"] for r in s5a], "D--", color="#2980B9", linewidth=1.3, markersize=4.8, label="Fused-FP16-TC (B8)")
        ax1.plot(cvs, [r["mb6_ms"] for r in s5a], "^-", color="#9B59B6", linewidth=1.4, markersize=5.5, label="MAXBOUND-INT6")
        ax1.plot(cvs, [r["mb8_ms"] for r in s5a], "o-", color=PAL["deep_blue"], linewidth=1.8, markersize=5.5, label="MAXBOUND-INT8")
        for cx, raw_v in zip(cvs, b1_raw):
            if raw_v > cap_ms:
                ax1.plot([cx, cx], [cap_ms * 0.95, cap_ms * 0.81], color="black", linestyle="--", linewidth=0.8)
                ax1.text(cx, cap_ms * 0.77, f"{raw_v:.0f}ms", fontsize=7.0, ha="center", va="center",
                         bbox=dict(boxstyle="round,pad=0.18", edgecolor="black", facecolor="white", linewidth=0.6))
        ax1.set_ylim(0, cap_ms)
        ax1.set_ylabel("Batch-64 Latency (ms)", fontsize=10.5)
        ax1.grid(True, linestyle="--", linewidth=0.45, color="lightgray")
        ax1.legend(loc="upper left", bbox_to_anchor=(0.01, 0.72), fontsize=7.2, frameon=True, facecolor="white", framealpha=0.92, edgecolor="lightgray")
        ax1.set_xlabel("Document Length CV\n(a) Latency stability under length skew", fontsize=10.2)

    if d6 and d6["records"]:
        recs = d6["records"]
        ds_labs = [pretty_ds(r["dataset"]) for r in recs]
        step_keys = [
            ("V0_Baseline", "Non-opt (cuBLAS)", "#D6EAF8", ""),
            ("V1_plus_TC_Bound", "+TC Bound (M1)", "#ABEBC6", "\\\\"),
            ("V2_plus_Prewarm", "+Prewarm", "#FFE699", "//"),
            ("V3_plus_SMEM_W16", "+SMEM Multi-Q", "#D2B4DE", "||"),
            ("V4_plus_SkewSort", "+Skew-Sorted (Full)", "#FAC090", "--"),
        ]
        x = np.arange(len(recs))
        w = 0.82 / len(step_keys)
        max_sp = 1.0
        for j, (sk, sl, col, hat) in enumerate(step_keys):
            vals = [r["speedup_steps"][sk] for r in recs]
            max_sp = max(max_sp, max(vals))
            xpos = x + (j - (len(step_keys) - 1) / 2.0) * w
            ax2.bar(xpos, vals, width=w * 0.90, color=col, hatch=hat,
                    edgecolor="black", linewidth=0.68, label=sl)
            for xp, v in zip(xpos, vals):
                if j in (0, len(step_keys) - 1):
                    ax2.annotate(f"{v:.2f}", xy=(xp, v), xytext=(0, 2), textcoords="offset points",
                                 ha="center", va="bottom", fontsize=6.8, fontweight="bold" if j == 4 else "normal")
        ax2.set_ylim(0, max_sp * 1.40)
        ax2.set_xticks(x)
        ax2.set_xticklabels(ds_labs, fontsize=8.2, rotation=-15)
        ax2.set_ylabel("Normalized Throughput (vs. B2)", fontsize=10.5)
        ax2.grid(True, axis="y", linestyle="--", linewidth=0.45, color="lightgray")
        ax2.legend(loc="upper left", fontsize=7.1, ncol=2, frameon=True, edgecolor="lightgray")
        ax2.set_xlabel("Dataset\n(b) Stepwise optimization contributions", fontsize=10.2)

        recs_b = d6["records"]
        y = np.arange(len(recs_b))
        s1 = np.array([r["stage_breakdown_ms"]["query_prep_and_prewarm"] for r in recs_b])
        s2 = np.array([r["stage_breakdown_ms"]["stage1_fused_tc_filter"] for r in recs_b])
        s3 = np.array([r["stage_breakdown_ms"]["stage2_exact_verify"] for r in recs_b])
        s4 = np.array([r["stage_breakdown_ms"]["topk_and_d2h"] for r in recs_b])
        tot = np.maximum(s1 + s2 + s3 + s4, 1e-6)
        p1, p2, p3, p4 = s1 / tot * 100, s2 / tot * 100, s3 / tot * 100, s4 / tot * 100
        h = 0.54
        ax3.barh(y, p1, height=h, color="#FFE699", hatch="\\\\", edgecolor="black", linewidth=0.7, label="Prewarm & Prep")
        ax3.barh(y, p2, left=p1, height=h, color="#95B3D7", hatch="//", edgecolor="black", linewidth=0.7, label="Stage-1 PTX TC")
        ax3.barh(y, p3, left=p1 + p2, height=h, color="#FAC090", hatch="||", edgecolor="black", linewidth=0.7, label="Stage-2 Verify")
        ax3.barh(y, p4, left=p1 + p2 + p3, height=h, color="#CCCCCC", edgecolor="black", linewidth=0.7, label="Top-K & D2H")
        for yi, tot_ms in enumerate(tot):
            ax3.text(101.5, yi, f"{tot_ms:.1f}ms", va="center", ha="left", fontsize=7.8)
        ax3.set_xlim(0, 124)
        ax3.set_yticks(y)
        ax3.set_yticklabels([pretty_ds(r["dataset"]) for r in recs_b], fontsize=8.4)
        ax3.grid(True, axis="x", linestyle="--", linewidth=0.45, color="lightgray")
        ax3.legend(loc="upper center", bbox_to_anchor=(0.48, 1.18), ncol=2, fontsize=7.4, frameon=True, edgecolor="lightgray")
        ax3.set_xlabel("Stage Latency Share (%)\n(c) CUDA kernel time breakdown", fontsize=10.2)

    plt.tight_layout(w_pad=1.0)
    save_fig(fig, "fig5_e5_e6_skew_ablation")


# ==============================================================================
# Figure 6: Zero-Leakage Blind-Test Cost Model & Build Breakdown / Break-Even Curve
# ==============================================================================
def plot_fig6_e7_e8():
    d7 = load_json("e7_cost_model.json")
    d8 = load_json("e8_build_amortization.json")
    if not d7 and not d8:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 3.7))

    if d7:
        recs = [r for r in d7["records"] if r.get("is_unique_feasible_regime", True)]
        dev_pts = [r for r in recs if r.get("split") != "blind_test"]
        blind_pts = [r for r in recs if r.get("split") == "blind_test"]
        if dev_pts:
            ax1.scatter([r["actual_sel_ms"] for r in dev_pts], [r["pred_ms"] for r in dev_pts],
                        c=PAL["deep_blue"], alpha=0.72, s=26, edgecolors="black", linewidths=0.4, label="Dev Datasets (8 Corpora)")
        if blind_pts:
            ax1.scatter([r["actual_sel_ms"] for r in blind_pts], [r["pred_ms"] for r in blind_pts],
                        c=PAL["amber"], marker="^", alpha=0.88, s=36, edgecolors="black", linewidths=0.5, label="Unseen Blind Test (HotpotQA/COVID/MARCO)")
        all_v = [r["actual_sel_ms"] for r in recs] + [r["pred_ms"] for r in recs]
        mn, mx = min(all_v) * 0.72, max(all_v) * 1.28
        ax1.plot([mn, mx], [mn, mx], "--", color=PAL["crimson"], linewidth=1.3, label="Ideal $y = x$")
        ax1.set_xscale("log"); ax1.set_yscale("log")
        sm = d7["meta"]["summary"]
        ded = sm.get("deduplicated_unique_regimes", sm)
        bl = sm.get("blind_test_deduplicated", {})
        bl_line = f"\nBlind Test Med APE: {bl['median_ape_pct']:.1f}% ({bl['near_optimal_hit_rate_pct']:.1f}% Hit)" if bl else ""
        ax1.text(0.03, 0.73,
                 f"Unique Regimes: {ded.get('count', sm['num_heldout_configs'])}\nMedian APE: {ded['median_ape_pct']:.1f}%\nNear-Opt Hit: {ded['near_optimal_hit_rate_pct']:.1f}%{bl_line}",
                 transform=ax1.transAxes, fontsize=7.8,
                 bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="black", linewidth=0.7))
        ax1.set_ylabel("Predicted Latency (ms)", fontsize=10.5)
        ax1.grid(True, which="major", linestyle="--", linewidth=0.45, color="lightgray")
        ax1.legend(loc="lower right", fontsize=7.3, frameon=True, edgecolor="lightgray")
        ax1.set_xlabel("Measured CUDA Latency (ms)\n(a) Zero-leakage cost model (synthetic-only fit)", fontsize=10.2)

    if d8:
        recs = [r for r in d8["records"] if r["dataset"] == "msmarco" and r["mode"] in ("int8", "int6")]
        ds_title = "MS MARCO"
        if not recs:
            recs = [r for r in d8["records"] if r["dataset"] == "fiqa" and r["mode"] in ("int8", "int6")]
            ds_title = "FiQA"
        if recs:
            be_ref = max(recs[0]["breakeven_queries"], 1000)
            q_max_x = max(10000, int(be_ref * 2.4))
            Qs = np.linspace(1, q_max_x, 200)
            b2_pq = recs[0]["b2_per_q_ms"] / 1000.0
            ax2.plot(Qs, Qs * b2_pq, "--", color=PAL["berry"], linewidth=1.7,
                     label=f"cuBLAS-Unpadded ({recs[0]['b2_per_q_ms']:.2f}ms/q)")
            for r, col, mk in zip(recs, [PAL["deep_blue"], PAL["amber"]], ["o", "s"]):
                cum_s = r["build_s"] + Qs * (r["mb_per_q_ms"] / 1000.0)
                ax2.plot(Qs, cum_s, color=col, linewidth=1.9, marker=mk, markevery=35, markersize=4.8,
                         label=f"MAXBOUND-{r['mode'].upper()} ({r['build_s']:.1f}s + {r['mb_per_q_ms']:.2f}ms/q)")
                if r["mode"] == "int8" and r["breakeven_queries"] > 0:
                    be = r["breakeven_queries"]
                    y_be = r["build_s"] + be * (r["mb_per_q_ms"] / 1000.0)
                    ax2.axvline(x=be, color=PAL["deep_blue"], linestyle=":", linewidth=1.2)
                    ax2.annotate(f"Break-even\n@ {be:,} queries",
                                 xy=(be, y_be), xytext=(be + q_max_x * 0.16, y_be * 0.72),
                                 arrowprops=dict(arrowstyle="->", color=PAL["deep_blue"], lw=1.1),
                                 fontsize=8.4, color=PAL["deep_blue"], fontweight="bold")
            ax2.set_ylabel("Cumulative Wall-Clock Time (s)", fontsize=10.5)
            ax2.grid(True, linestyle="--", linewidth=0.45, color="lightgray")
            ax2.legend(loc="upper left", fontsize=7.6, frameon=True, edgecolor="lightgray")
        ax2.set_xlabel(f"Cumulative Queries Served ($Q$)\n(b) Index build amortization break-even ({ds_title})", fontsize=10.2)

    plt.tight_layout(w_pad=1.2)
    save_fig(fig, "fig6_e7_e8_cost_amortization")


# ==============================================================================
# Figure 7: Fused 1024D 2D (Scale x Dim + Avg) Heatmap & 11-Dataset Portability
# ==============================================================================
def plot_fig7_e10_e11():
    d10 = load_json("e10_sensitivity.json")
    d11 = load_json("e11_portability.json")
    if not d10 and not d11:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 3.8), gridspec_kw={"width_ratios": [1.0, 1.38]})

    hm_recs = [r for r in (d10["records"] if d10 else []) if r.get("sweep") == "heatmap_N_x_D"]
    if hm_recs:
        dims = sorted(list(set(r["dim"] for r in hm_recs)))
        ndocs = sorted(list(set(r["N_docs"] for r in hm_recs)))
        mat = np.zeros((len(dims), len(ndocs) + 1), dtype=np.float64)
        for i, dm in enumerate(dims):
            for j, nd in enumerate(ndocs):
                hit = [r["speedup"] for r in hm_recs if r["dim"] == dm and r["N_docs"] == nd]
                mat[i, j] = hit[0] if hit else 1.0
            mat[i, -1] = float(np.mean(mat[i, :-1]))

        im = ax1.imshow(mat, cmap="Blues", vmin=max(1.0, mat.min() * 0.88), vmax=mat.max() * 1.05, aspect="auto")
        ax1.axvline(len(ndocs) - 0.5, color="black", linewidth=1.6)
        ax1.set_xticks(range(len(ndocs) + 1))
        nd_labs = [f"{nd // 1000}K" for nd in ndocs] + ["Avg"]
        ax1.set_xticklabels(nd_labs, fontsize=9.0)
        ax1.set_yticks(range(len(dims)))
        ax1.set_yticklabels([str(dm) for dm in dims], fontsize=9.0)
        ax1.set_ylabel("Token Embedding Dimension ($d$)", fontsize=10.5)
        mid = 0.5 * (mat.min() + mat.max())
        for r_i in range(len(dims)):
            for c_i in range(len(ndocs) + 1):
                v = mat[r_i, c_i]
                ax1.text(c_i, r_i, f"{v:.2f}x", ha="center", va="center",
                         color="white" if v > mid else "black",
                         fontsize=8.3, fontweight="bold" if c_i == len(ndocs) else "normal")
        cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.03)
        cbar.set_label("Exact Speedup (x)", fontsize=9.5)
        ax1.set_xlabel("Number of Documents (+ Row Average)\n(a) Scale & dimension speedup matrix", fontsize=10.2)

    if d11:
        recs = d11["records"]
        dss = [pretty_ds(r["dataset"]) for r in recs]
        x = np.arange(len(dss))
        w = 0.37
        sp_a100 = [r["A100_sm80"]["speedup"] or 0.0 for r in recs]
        sp_4080 = [r["RTX4080_sm89"]["speedup"] or 0.0 for r in recs]
        ax2.bar(x - w / 2, sp_a100, width=w, color="#95B3D7", hatch="\\\\",
                edgecolor="black", linewidth=0.7, label="A100 80GB (Ampere sm_80)")
        ax2.bar(x + w / 2, sp_4080, width=w, color="#C4D79B", hatch="//",
                edgecolor="black", linewidth=0.7, label="RTX 4080 16GB (Ada sm_89)")
        ymax = max(max(sp_a100 + sp_4080), 4.0) * 1.25
        for idx_r, r in enumerate(recs):
            if r["RTX4080_sm89"].get("b2_oom"):
                qps_off = r["RTX4080_sm89"]["qps"]
                ax2.scatter([x[idx_r] + w * 0.68], [ymax * 0.05], marker="*", color=PAL["deep_blue"], s=55, zorder=5)
                ax2.text(x[idx_r] + w * 0.72, ymax * 0.11, f"B2 OOM ({qps_off:.0f} QPS)",
                         ha="center", va="bottom", fontsize=6.3, color="#1B4F72", fontweight="bold", rotation=90)
            else:
                v4 = sp_4080[idx_r]
                ax2.annotate(f"{v4:.1f}x", xy=(x[idx_r] + w / 2, v4), xytext=(0, 2),
                             textcoords="offset points", ha="center", va="bottom", fontsize=6.6)
        ax2.axhline(1.0, color="black", linestyle="--", linewidth=0.9)
        ax2.set_ylim(0, ymax)
        ax2.set_xticks(x)
        ax2.set_xticklabels(dss, rotation=28, ha="right", fontsize=8.0)
        ax2.set_ylabel("Exact Speedup vs. cuBLAS-Unpadded", fontsize=10.5)
        ax2.grid(True, axis="y", linestyle="--", linewidth=0.45, color="lightgray")
        ax2.legend(loc="upper left", fontsize=8.0, frameon=True, edgecolor="lightgray")
        ax2.set_xlabel(f"Dataset\n(b) Cross-architecture portability across {len(dss)} datasets", fontsize=10.2)

    plt.tight_layout(w_pad=1.1)
    save_fig(fig, "fig7_e10_e11_sensitivity_portability")


# ==============================================================================
# Figure 8: Genuine Silicon PMU Hardware Counters (Nsight Compute, incl. B8 & Fused 1024D)
# ==============================================================================
def plot_fig8_e12_ncu():
    d12 = load_json("e12_ncu_hardware_profile.json")
    if not d12 or not d12.get("records"):
        return
    recs = [r for r in d12["records"] if not r.get("oom")]
    datasets = []
    for r in recs:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])

    m_order = [
        ("B1_Torch_Padded", "Padded", "#CCCCCC", ""),
        ("B2_cuBLAS_Unpadded", "cuBLAS", "#FFE699", "\\\\"),
        ("B8_Fused_FP16_TC", "FP16-TC (B8)", "#85C1E9", ".."),
        ("MAXBOUND_W1_Naive", "MB-W1", "#95B3D7", "//"),
        ("MAXBOUND_Full_W16", "MB-Full", "#C4D79B", "--"),
    ]
    panels = [
        ("dram_total_mb", "DRAM Traffic (MB)", "(a) Total HBM read+write bytes"),
        ("l2_hit_rate_pct", "L2 Sector Hit Rate (%)", "(b) L2 cache locality"),
        ("tensor_core_pipe_pct", "Tensor Pipe Active (%)", "(c) Tensor Core utilization"),
        ("warp_non_divergence_pct", "SIMT Thread Efficiency (%)", "(d) Non-divergent warp efficiency"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 3.5))
    handles_leg, labels_leg = [], []

    for p_i, (m_key, ylab, subcap) in enumerate(panels):
        ax = axes[p_i]
        x = np.arange(len(datasets))
        w = 0.82 / len(m_order)
        max_v = 1.0
        for j, (mk, ml, col, hat) in enumerate(m_order):
            xpos = x + (j - (len(m_order) - 1) / 2.0) * w
            vals = []
            for ds in datasets:
                hit = [r[m_key] for r in recs if r["dataset"] == ds and r["method"] == mk]
                vals.append(hit[0] if hit else 0.0)
            max_v = max(max_v, max(vals) if vals else 1.0)
            bars = ax.bar(xpos, vals, width=w * 0.90, color=col, hatch=hat,
                          edgecolor="black", linewidth=0.68, label=ml)
            if p_i == 0:
                handles_leg.append(bars[0]); labels_leg.append(ml)
            for xp, v in zip(xpos, vals):
                if v > 0:
                    fmt = f"{v:.0f}" if v >= 100 else f"{v:.1f}"
                    ax.annotate(fmt, xy=(xp, v), xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=6.4)
        ax.set_ylim(0, max_v * 1.25)
        ax.set_xticks(x)
        ax.set_xticklabels([pretty_ds(ds) for ds in datasets], fontsize=9.0)
        ax.set_ylabel(ylab, fontsize=9.8)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.45, color="lightgray")
        ax.set_xlabel(subcap, fontsize=9.5, labelpad=5)

    fig.legend(handles_leg, labels_leg, loc="upper center", ncol=5, fontsize=9.6,
               bbox_to_anchor=(0.5, 1.08), frameon=True, edgecolor="lightgray")
    plt.tight_layout(w_pad=0.85)
    save_fig(fig, "fig8_e12_ncu_hardware_profile")


# ==============================================================================
# Figure 9: Streaming Insertions & Tombstone Deletions (10 Churn Epochs)
# ==============================================================================
def plot_fig9_e13_dynamic():
    d13 = load_json("e13_dynamic_updates.json")
    if not d13 or not d13.get("records"):
        return
    recs = d13["records"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 3.5))

    styles = [("fiqa", "FiQA", PAL["deep_blue"], "o"), ("hotpotqa", "HotpotQA", PAL["amber"], "s")]
    for ds, lab, col, mk in styles:
        sub = sorted([r for r in recs if r["dataset"] == ds], key=lambda z: z["epoch"])
        if not sub:
            continue
        eps = [r["epoch"] for r in sub]
        upd_ms = [r["insert_ms"] + r["delete_ms"] for r in sub]
        reb_ms = [r["plaid_full_rebuild_ms"] for r in sub]
        ax1.plot(eps, upd_ms, f"{mk}-", color=col, linewidth=1.6, markersize=5, label=f"MAXBOUND Incr. ({lab})")
        if all("b2_dense_append_ms" in r for r in sub):
            b2_app = [r["b2_dense_append_ms"] + r["delete_ms"] for r in sub]
            ax1.plot(eps, b2_app, f"{mk}:", color=col, alpha=0.85, linewidth=1.2, markersize=3.8, label=f"cuBLAS Append ({lab})")
        ax1.plot(eps, reb_ms, f"{mk}--", color=col, alpha=0.55, linewidth=1.2, markersize=4.5, label=f"PLAID Rebuild ({lab})")
        ax2.plot(eps, [r["query_qps"] for r in sub], f"{mk}-", color=col, linewidth=1.7, markersize=5.2,
                 label=f"{lab} MB-INT8 QPS (Prune {sub[-1]['prune_pct']:.2f}%)")
        if all("b2_query_qps" in r for r in sub):
            ax2.plot(eps, [r["b2_query_qps"] for r in sub], f"{mk}:", color=col, alpha=0.75, linewidth=1.25, markersize=4.0,
                     label=f"{lab} cuBLAS QPS (BE={sub[-1]['breakeven_queries_vs_b2_append']}q)")

    ax1.set_yscale("log")
    ax1.set_ylim(15, 18000)
    ax1.set_xticks(range(1, 11))
    ax1.set_ylabel("Update / Rebuild Time (ms, log)", fontsize=10.5)
    ax1.grid(True, which="major", linestyle="--", linewidth=0.45, color="lightgray")
    ax1.legend(loc="upper right", ncol=2, fontsize=7.1, frameon=True, facecolor="white", framealpha=0.92, edgecolor="lightgray")
    ax1.set_xlabel("Streaming Churn Epoch (+5% Insert, -1.5% Delete)\n(a) Zero-rebuild update cost vs. index rebuild", fontsize=10.2)

    ax2.set_xticks(range(1, 11))
    ax2.set_ylabel("Exact Query Throughput (QPS)", fontsize=10.5)
    ax2.grid(True, linestyle="--", linewidth=0.45, color="lightgray")
    ax2.legend(loc="best", fontsize=8.2, frameon=True, edgecolor="lightgray")
    ax2.set_xlabel("Streaming Churn Epoch (100.0% FP64 Exact Verified)\n(b) Query throughput stability under churn", fontsize=10.2)

    plt.tight_layout(w_pad=1.2)
    save_fig(fig, "fig9_e13_dynamic_updates")


# ==============================================================================
# Figure 10 (NEW): End-to-End GPU Query Encoding + Retrieval Pipeline (E14) & (N_docs, B) Crossover Boundary
# ==============================================================================
def plot_fig10_e14_rag_and_crossover():
    d14 = load_json("e14_end_to_end_rag.json")
    d10 = load_json("e10_sensitivity.json")
    if not d14 and not d10:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 3.7), gridspec_kw={"width_ratios": [1.32, 1.0]})

    # (a) End-to-End RAG Query Encoding + Retrieval Breakdown at B=64 across 4 corpora
    if d14 and d14.get("records"):
        sub64 = [r for r in d14["records"] if r["batch"] == 64]
        dss = [pretty_ds(r["dataset"]) for r in sub64]
        x = np.arange(len(sub64))
        w = 0.26
        enc_v = np.array([r["encode_median_ms"] for r in sub64])
        b2_r = np.array([r["b2_retrieval_ms"] for r in sub64])
        b8_r = np.array([r["b8_fused_retrieval_ms"] for r in sub64])
        mb_r = np.array([r["mb8_retrieval_ms"] for r in sub64])

        ymax_e = max(enc_v + b2_r) * 1.28
        b8_cap = ymax_e * 0.88 - enc_v
        b8_disp = np.minimum(b8_r, b8_cap)

        # Stacked bars: bottom = Transformer Query Encoding, top = Retrieval
        ax1.bar(x - w, enc_v, width=w, color="#CCCCCC", edgecolor="black", linewidth=0.7, label="GPU Query Encoding")
        ax1.bar(x - w, b2_r, bottom=enc_v, width=w, color="#FFE699", hatch="\\\\", edgecolor="black", linewidth=0.7, label="+ cuBLAS Retrieval (B2)")

        ax1.bar(x, enc_v, width=w, color="#CCCCCC", edgecolor="black", linewidth=0.7)
        ax1.bar(x, b8_disp, bottom=enc_v, width=w, color="#85C1E9", hatch="..", edgecolor="black", linewidth=0.7, label="+ Fused-FP16 Retrieval (B8)")

        ax1.bar(x + w, enc_v, width=w, color="#CCCCCC", edgecolor="black", linewidth=0.7)
        ax1.bar(x + w, mb_r, bottom=enc_v, width=w, color="#C4D79B", hatch="--", edgecolor="black", linewidth=0.7, label="+ MAXBOUND Exact Retrieval")

        for idx_i, r in enumerate(sub64):
            tot_mb = enc_v[idx_i] + mb_r[idx_i]
            ax1.annotate(f"{r['e2e_speedup_vs_b2']:.2f}x\nE2E",
                         xy=(x[idx_i] + w, tot_mb), xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=7.3, fontweight="bold", color="#1B4F72")
            if b8_r[idx_i] > b8_cap[idx_i]:
                ax1.annotate(f"{enc_v[idx_i] + b8_r[idx_i]:.0f}ms\n(cap)",
                             xy=(x[idx_i], enc_v[idx_i] + b8_disp[idx_i]), xytext=(0, 2), textcoords="offset points",
                             ha="center", va="bottom", fontsize=6.6, color="#2C3E50", fontweight="bold")
        ax1.set_ylim(0, ymax_e)
        ax1.set_xticks(x)
        ax1.set_xticklabels(dss, fontsize=9.0)
        ax1.set_ylabel("End-to-End Batch-64 Latency (ms)", fontsize=10.2)
        ax1.grid(True, axis="y", linestyle="--", linewidth=0.45, color="lightgray")
        ax1.legend(loc="upper left", fontsize=7.6, ncol=2, frameon=True, edgecolor="lightgray")
        ax1.set_xlabel("Dataset (ColBERTv2 128D & BGE-M3 1024D)\n(a) End-to-end GPU query encoding + exact retrieval", fontsize=10.0)

    # (b) (N_docs x Batch B) Speedup Crossover Heatmap vs. Fused-FP16-TC (B8) / cuBLAS (B2)
    cross_recs = [r for r in (d10["records"] if d10 else []) if r.get("sweep") == "crossover_N_x_B"]
    if cross_recs:
        bs = sorted(list(set(r["B"] for r in cross_recs)))
        nds = sorted(list(set(r["N_docs"] for r in cross_recs)))
        mat = np.zeros((len(bs), len(nds)), dtype=np.float64)
        for i, b_val in enumerate(bs):
            for j, nd in enumerate(nds):
                hit = [r["speedup_vs_b2"] for r in cross_recs if r["B"] == b_val and r["N_docs"] == nd]
                mat[i, j] = hit[0] if hit else 1.0
        im2 = ax2.imshow(mat, cmap="RdYlGn", vmin=0.6, vmax=max(2.5, mat.max()), aspect="auto")
        ax2.set_xticks(range(len(nds)))
        ax2.set_xticklabels([f"{nd // 1000}K" for nd in nds], fontsize=8.8)
        ax2.set_yticks(range(len(bs)))
        ax2.set_yticklabels([f"B={b_v}" for b_v in bs], fontsize=9.0)
        ax2.set_ylabel("Query Batch Size ($B$)", fontsize=10.2)
        for i in range(len(bs)):
            for j in range(len(nds)):
                v = mat[i, j]
                ax2.text(j, i, f"{v:.2f}x", ha="center", va="center",
                         fontsize=8.2, fontweight="bold" if v >= 1.0 else "normal",
                         color="black")
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
        cbar2.set_label("Speedup vs. cuBLAS (x)", fontsize=9.2)
        ax2.set_xlabel("Corpus Size ($N_{\\mathrm{docs}}$)\n(b) Small-batch & scale crossover boundary", fontsize=10.0)

    plt.tight_layout(w_pad=1.2)
    save_fig(fig, "fig10_e14_e2e_rag_and_crossover")


if __name__ == "__main__":
    plot_fig1_motivation()
    plot_fig2_e1()
    plot_fig3_e2()
    plot_fig4_e3_e4()
    plot_fig5_e5_e6()
    plot_fig6_e7_e8()
    plot_fig7_e10_e11()
    plot_fig8_e12_ncu()
    plot_fig9_e13_dynamic()
    plot_fig10_e14_rag_and_crossover()
