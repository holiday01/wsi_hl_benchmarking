#!/usr/bin/env python3
"""Main-text figures from the S8/secondary outputs (no numbers are typed by hand).
  fig1_inflation.pdf   inflation (cluster SE / tile SE) per task, one point per (teacher, student, seed)      H1
  fig2_verdicts.pdf    verdict heatmap teacher x student per task, ceiling tasks left, non-ceiling right       H2, H3
  fig3_capacity.pdf    delta_min by student size per task; seed SD vs cluster SE                               H4, H6
  fig4_patch.pdf       PanNuke Dice retention per teacher (patch-14 vs patch-16)                               H5
  figS1_margin.pdf     supplement: fraction equivalent vs margin per task (secondary (c))
  figS2_clusters.pdf   supplement: CI half-width vs number of clusters, cluster vs tile bootstrap (secondary (d))
  figS3_teacher_curve.pdf  supplement: same curve for the teacher metric itself (research/teacher_side_secondary_v5, no student)
Usage: make_figures.py --summary outputs/v5/summary --secondary outputs/v5/secondary --root outputs/v5 --out paper_jbhi/figures
"""
import argparse, glob, json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

TASK_ORDER = ["c16", "crc7k", "msi", "center", "c17_loco", "panda", "panda_biopsy_tissue", "pannuke_seg"]
TASK_LABEL = {"c16": "CAMELYON16", "crc7k": "CRC-7K", "msi": "Kather-MSI", "center": "TCGA site", "c17_loco": "CAMELYON17 LOCO",
              "panda": "PANDA tile", "panda_biopsy_tissue": "PANDA biopsy", "pannuke_seg": "PanNuke Dice"}
CEILING = {"c16", "crc7k"}
STATE_COLOR = {"equivalent": "#2b8cbe", "inconclusive": "#bdbdbd", "inequivalent": "#d7301f", "undefined": "white"}
STUDENTS = ["vit-tiny", "vit-small", "vit-base"]
SNAME = {"vit-tiny": "ViT-Ti", "vit-small": "ViT-S", "vit-base": "ViT-B"}
TNAME = {"phikon": "Phikon", "phikon-v2": "Phikon-v2", "conch": "CONCH", "uni": "UNI", "uni2-h": "UNI2-h",
         "prov-gigapath": "Prov-GigaPath", "virchow": "Virchow", "virchow2": "Virchow2", "h-optimus-0": "H-optimus-0",
         "hibou-b": "Hibou-B", "hibou-l": "Hibou-L", "midnight": "Midnight-12k"}   # display names, as in the text


def load(summary):
    rows = []
    for f in glob.glob(f"{summary}/summary_*.json"):
        t = Path(f).stem[len("summary_"):]
        if t == "all":
            continue
        for r in json.load(open(f)):
            r["task"] = t; rows.append(r)
    return pd.DataFrame(rows)


def fig1(df, root, out):
    recs = []
    for f in glob.glob(f"{root}/*/*/seed*/probe_*.json"):
        if f.endswith("_preds.npz") or "pannuke" in f:
            continue
        r = json.load(open(f)); res = r.get("result", {})
        if np.isfinite(res.get("inflation", np.nan)):
            recs.append({"task": r["task"], "inflation": res["inflation"]})
    d = pd.DataFrame(recs)
    tasks = [t for t in TASK_ORDER if t in set(d.task)]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for i, t in enumerate(tasks):
        v = d[d.task == t].inflation.values
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(-0.18, 0.18, len(v)), v, s=6, alpha=0.5, color="#555")
        ax.plot([i - 0.3, i + 0.3], [np.median(v)] * 2, color="#d7301f", lw=1.5)
    ax.axhline(1, color="k", lw=0.5, ls=":"); ax.axhline(1.5, color="#d7301f", lw=0.5, ls="--")
    ax.set_xticks(range(len(tasks))); ax.set_xticklabels([TASK_LABEL[t] for t in tasks], rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("SE ratio (cluster / tile)", fontsize=8); ax.set_yscale("log"); ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(out / "fig1_inflation.pdf"); plt.close(fig)


def fig2(df, out):
    tasks = [t for t in TASK_ORDER if t in set(df.task)]
    teachers = sorted(df.teacher.unique())
    fig, axes = plt.subplots(1, len(tasks), figsize=(0.9 * len(tasks) + 1.2, 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, tasks):
        sub = df[df.task == t]
        for yi, te in enumerate(teachers):
            for xi, st in enumerate(STUDENTS):
                r = sub[(sub.teacher == te) & (sub.student == st)]
                c = STATE_COLOR.get(r.state.iloc[0], "white") if len(r) else "white"
                ax.add_patch(plt.Rectangle((xi, yi), 1, 1, color=c, ec="white"))
                if len(r) and np.isfinite(r.d_pooled.iloc[0]):
                    ax.text(xi + 0.5, yi + 0.5, f"{r.d_pooled.iloc[0]:+.2f}", ha="center", va="center", fontsize=4.5)
        ax.set_xlim(0, 3); ax.set_ylim(0, len(teachers)); ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels(["Ti", "S", "B"], fontsize=7)
        ax.set_title(TASK_LABEL[t] + ("\n(ceiling)" if t in CEILING else ""), fontsize=7)
    axes[0].set_yticks(np.arange(len(teachers)) + 0.5); axes[0].set_yticklabels([TNAME.get(t, t) for t in teachers], fontsize=6)
    fig.tight_layout(); fig.savefig(out / "fig2_verdicts.pdf"); plt.close(fig)


def fig3(df, secondary, root, out):
    dm = pd.read_csv(f"{secondary}/delta_min_pooled.csv") if Path(f"{secondary}/delta_min_pooled.csv").exists() else None
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.4))
    if dm is not None:
        tasks = [t for t in TASK_ORDER if t in set(dm.task)]
        for i, t in enumerate(tasks):
            for j, st in enumerate(STUDENTS):
                v = dm[(dm.task == t) & (dm.student == st)].delta_min.values
                if len(v):
                    axes[0].scatter(np.full(len(v), i + (j - 1) * 0.25), v, s=8, label=SNAME[st] if i == 0 else None, color=["#9ecae1", "#4292c6", "#08306b"][j])
        axes[0].axhline(0.05, color="#d7301f", lw=0.6, ls="--"); axes[0].set_xticks(range(len(tasks)))
        axes[0].set_xticklabels([TASK_LABEL[t] for t in tasks], rotation=35, ha="right", fontsize=6); axes[0].set_ylabel(r"$\delta_{\min}$", fontsize=8)
        axes[0].legend(fontsize=6, frameon=False); axes[0].tick_params(labelsize=7)
    # right panel: seed SD of d against the single-run cluster SE (mean over the three runs' own se_cluster;
    # the summary's se_cluster_seed is the seed-pooled SE and already contains the seed variance)
    sd = df[df.task.isin(["c16", "c17_loco", "panda", "panda_biopsy_tissue", "msi", "center"])].dropna(subset=["d_seed_sd"])
    pts = []
    for r in sd.itertuples():
        se = [json.load(open(f))["result"].get("se_cluster") for f in glob.glob(f"{root}/{r.teacher}/{r.student}/seed*/probe_{r.task}.json")]
        se = [x for x in se if x is not None]
        if se:
            pts.append((r.task, float(np.mean(se)), r.d_seed_sd))
    if pts:
        p = pd.DataFrame(pts, columns=["task", "se1", "sd"])
        for k, t in enumerate(p.task.unique()):
            q = p[p.task == t]
            axes[1].scatter(q.se1, q.sd, s=6, label=TASK_LABEL[t], color=plt.cm.tab10(k))
        lo_, hi_ = min(p.se1.min(), p.sd[p.sd > 0].min()) * 0.8, max(p.se1.max(), p.sd.max()) * 1.2
        axes[1].plot([lo_, hi_], [lo_, hi_], "k:", lw=0.6); axes[1].set_xscale("log"); axes[1].set_yscale("log")
        axes[1].set_xlabel("single-run cluster SE", fontsize=8); axes[1].set_ylabel("seed SD of d", fontsize=8)
        axes[1].legend(fontsize=5, frameon=False, loc="upper left")
    axes[1].tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(out / "fig3_capacity.pdf"); plt.close(fig)


def fig4(df, out):
    pn = df[df.task == "pannuke_seg"]
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    if len(pn):
        pn = pn.assign(retention=pn.s_metric_mean / pn.t_metric)
        teachers = sorted(pn.teacher.unique())
        for j, st in enumerate(STUDENTS):
            v = [pn[(pn.teacher == te) & (pn.student == st)].retention.mean() for te in teachers]
            ax.bar(np.arange(len(teachers)) + (j - 1) * 0.27, v, width=0.27, label=SNAME[st], color=["#9ecae1", "#4292c6", "#08306b"][j])
        ax.set_xticks(range(len(teachers))); ax.set_xticklabels([TNAME.get(t, t) for t in teachers], rotation=45, ha="right", fontsize=6)
        ax.set_ylabel("Dice retention (student / teacher)", fontsize=7); ax.axhline(1, color="k", lw=0.5, ls=":"); ax.legend(fontsize=6, frameon=False)
    ax.tick_params(labelsize=7); fig.tight_layout(); fig.savefig(out / "fig4_patch.pdf"); plt.close(fig)


def figS1(secondary, out):
    f = Path(secondary) / "margin_sweep.csv"
    if not f.exists():
        return
    ms = pd.read_csv(f)
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for t in [t for t in TASK_ORDER if t in set(ms.task)]:
        g = ms[ms.task == t].sort_values("margin")
        ax.plot(g.margin, g.frac_equivalent, marker="o", ms=2.5, lw=1,
                ls="-" if t in CEILING else "--", label=TASK_LABEL[t])
    ax.axvline(0.05, color="k", lw=0.6, ls=":")
    ax.set_xlabel("margin $\\delta$"); ax.set_ylabel("fraction of pairs equivalent"); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=6, frameon=False, ncol=2); ax.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(out / "figS1_margin.pdf"); plt.close(fig)


def log_cluster_axis(ax, n):
    """Log x axis with plain 10/20/50... labels; minor labels off (they collide when the range is under a decade)."""
    ax.set_xscale("log")
    ax.set_xticks([v for v in (10, 20, 50, 100, 200, 500, 1000) if min(n) <= v <= max(n)])
    ax.set_xticklabels([f"{v:g}" for v in ax.get_xticks()]); ax.xaxis.set_minor_formatter(NullFormatter())


def figS2(secondary, out):
    f = Path(secondary) / "cluster_count_curve.csv"
    if not f.exists():
        return
    cc = pd.read_csv(f)
    tasks = [t for t in TASK_ORDER if t in set(cc.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(1.9 * len(tasks) + 0.6, 2.4), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        g = cc[cc.task == t].groupby("n_clusters")[["half_width_cluster", "half_width_naive"]].median().reset_index()
        ax.plot(g.n_clusters, g.half_width_cluster, marker="o", ms=2.5, lw=1, color="#d7301f", label="cluster")
        ax.plot(g.n_clusters, g.half_width_naive, marker="s", ms=2.5, lw=1, color="#2b8cbe", label="tile")
        ax.axhline(0.05, color="k", lw=0.6, ls=":")
        log_cluster_axis(ax, g.n_clusters); ax.set_title(TASK_LABEL[t], fontsize=8); ax.tick_params(labelsize=7)
        ax.set_xlabel("clusters in test set", fontsize=7)
    axes[0][0].set_ylabel("90% CI half-width", fontsize=7); axes[0][0].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(out / "figS2_clusters.pdf"); plt.close(fig)


def figS3(teacher_side, out):
    f = Path(teacher_side) / "cluster_count_curve_teacher.csv"
    if not f.exists():
        return
    cc = pd.read_csv(f)
    tasks = [t for t in TASK_ORDER if t in set(cc.task)]
    fig, axes = plt.subplots(1, len(tasks), figsize=(1.9 * len(tasks) + 0.6, 2.4), squeeze=False)
    for ax, t in zip(axes[0], tasks):
        g = cc[cc.task == t].groupby("n_clusters")[["half_width_cluster", "half_width_tile"]].median().reset_index()
        ax.plot(g.n_clusters, g.half_width_cluster, marker="o", ms=2.5, lw=1, color="#d7301f", label="cluster")
        ax.plot(g.n_clusters, g.half_width_tile, marker="s", ms=2.5, lw=1, color="#2b8cbe", label="tile")
        ax.axhline(0.05, color="k", lw=0.6, ls=":")
        log_cluster_axis(ax, g.n_clusters); ax.set_title(TASK_LABEL[t], fontsize=8); ax.tick_params(labelsize=7)
        ax.set_xlabel("clusters in test set", fontsize=7)
    axes[0][0].set_ylabel("90% CI half-width (teacher metric)", fontsize=7); axes[0][0].legend(fontsize=6, frameon=False)
    fig.tight_layout(); fig.savefig(out / "figS3_teacher_curve.pdf"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="outputs/v5/summary"); ap.add_argument("--secondary", default="outputs/v5/secondary")
    ap.add_argument("--root", default="outputs/v5"); ap.add_argument("--out", default="paper_jbhi/figures")
    ap.add_argument("--teacher_side", default="research/teacher_side_secondary_v5"); ap.add_argument("--teacher_only", action="store_true")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    figS3(a.teacher_side, out)
    if a.teacher_only:
        print("teacher-side figure written to", out); return
    df = load(a.summary)
    fig1(df, a.root, out); fig2(df, out); fig3(df, a.secondary, a.root, out); fig4(df, out)
    figS1(a.secondary, out); figS2(a.secondary, out)
    print("figures written to", out)


if __name__ == "__main__":
    main()
