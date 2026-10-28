#!/usr/bin/env python3
"""Generate supplement tables that depend only on teacher-side files.

  tables/s_bugs.tex     from research/teacher_bug_impact_v5.json + outputs/v5/teacher_reference/*.json
  tables/s_teacher.tex  from outputs/v5/teacher_reference/*.json (CLS, correct preprocessing)
  tables/s_teacher_side.tex  from research/teacher_side_secondary_v5/*.csv (teacher-metric inflation, ICC, ranking resolution)
  tables/s_teacher_extra.tex from research/teacher_side_extra_v5/*.csv (teacher pair equivalence, cluster-size and probe-C robustness)
  tables/s_pannuke_class.tex from outputs/v5/*/*/seed*/probe_pannuke_seg.json (per-class Dice, seed-pooled; written only if any exist)

Usage: python scripts/eval_v5/make_supp_tables.py [--out paper_jbhi/supplement/tables]
"""
import argparse, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
ORDER = ["phikon", "phikon-v2", "conch", "uni", "uni2-h", "prov-gigapath", "virchow", "virchow2",
         "h-optimus-0", "hibou-b", "hibou-l", "midnight"]
NAME = {"phikon": "Phikon", "phikon-v2": "Phikon-v2", "conch": "CONCH", "uni": "UNI", "uni2-h": "UNI2-h",
        "prov-gigapath": "Prov-GigaPath", "virchow": "Virchow", "virchow2": "Virchow2",
        "h-optimus-0": "H-optimus-0", "hibou-b": "Hibou-B", "hibou-l": "Hibou-L", "midnight": "Midnight-12k"}
TASK_NAME = {"c16": "CAMELYON16", "c17_loco": "CAMELYON17 (LOCO)", "panda": "PANDA (tile)",
             "panda_biopsy": "PANDA (biopsy)", "msi": "Kather-MSI", "center": "TCGA-BRCA site",
             "crc7k": "CRC-VAL-HE-7K", "pannuke": "PanNuke (Dice)"}
TASK_ORDER = ["c16", "c17_loco", "panda", "panda_biopsy", "msi", "center", "crc7k", "pannuke"]


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def load_ref(ref_dir):
    out = {}
    for p in sorted(ref_dir.glob("*.json")):
        d = json.loads(p.read_text())
        out[d["task"]] = d
    return out


# Failure 1 in the earlier version: H-optimus-0, CONCH and Midnight throughout; Hibou-B/L in their first runs, until
# the _HIBOU_MEAN fix (git show 4922682:scripts/compare_seed_variance.py, CONFOUNDED), after which Hibou was excluded.
LEGACY_BUG1 = {"h-optimus-0", "conch", "midnight", "hibou-b", "hibou-l"}


def bugs_table(bug, refs):
    t = bug["teachers"]
    # downstream effect: correct - legacy, CLS embedding, per task
    tasks = [k for k in TASK_ORDER if k in refs]
    eff = {k: {r["teacher"]: r for r in refs[k].get("paired", [])
               if r.get("embedding") == "cls" and r.get("contrast") == "correct_minus_legacy"} for k in tasks}
    cols = "@{}lrrrr" + "r" * len(tasks) + "@{}"
    L = [r"\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
         r"\caption{Per-teacher effect of the two preprocessing failures on %d held-out tiles. "
         r"F1: cosine between the CLS target computed with ImageNet statistics and the correct one (mean; 5th percentile). "
         r"F2: cosine between the truncated and the correctly resampled patch target (mean over the $14\times14$ cells; "
         r"fraction of cells below 0.9). $\Delta$: change in the teacher's own linear-probe metric, correct minus legacy "
         r"preprocessing (CLS embedding, same split; 90\,\%% cluster interval excludes 0 marked $^{*}$). "
         r"-- = teacher not affected. Hibou-B and Hibou-L were mis-normalised only in the earlier version's first runs, "
         r"corrected partway through and excluded from that version's results.}" % bug["n_tiles"],
         r"\label{tab:s_bugs}",
         r"\begin{tabular}{%s}" % cols, r"\toprule",
         "Teacher & F1 CLS cos & F1 cos p05 & F2 patch cos & F2 frac$<$0.9 & " +
         " & ".join(r"$\Delta$ " + TASK_NAME[k] for k in tasks) + r"\\", r"\midrule"]
    for k in ORDER:
        v = t[k]
        b1 = v["affected_by_bug1"] and k in LEGACY_BUG1
        row = [NAME[k], f3(v.get("bug1_cls_cos_mean") if b1 else None), f3(v.get("bug1_cls_cos_p05") if b1 else None),
               f3(v.get("bug2_cos_mean")), f3(v.get("bug2_frac_cells_below_0p9"))]
        for task in tasks:
            r = eff[task].get(k)
            if r is None or not b1:
                row.append("--")
            else:
                lo, hi = r["ci90"]
                star = "$^{*}$" if (lo > 0 or hi < 0) else ""
                # |d| < 0.0005 prints as an unsigned 0.000 rather than a signed zero
                row.append(("0.000" if abs(r["d"]) < 5e-4 else f"{r['d']:+.3f}") + star)
        L.append(" & ".join(row) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def teacher_table(refs):
    tasks = [k for k in TASK_ORDER if k in refs]
    rows = {k: {r["teacher"]: r for r in refs[k]["rows"]
                if r["embedding"] == "cls" and r["preprocessing"] == "correct"} for k in tasks}
    L = [r"\begin{table}[h]\centering\small",
         r"\caption{Teacher reference values: CLS embedding, correct preprocessing, same split, probe and "
         r"cluster bootstrap as the students (90\,\% percentile interval).}",
         r"\label{tab:s_teacher}",
         r"\begin{tabular}{@{}l" + "c" * len(tasks) + "@{}}", r"\toprule",
         "Teacher & " + " & ".join(TASK_NAME[k] for k in tasks) + r"\\", r"\midrule"]
    for k in ORDER:
        cells = []
        for task in tasks:
            r = rows[task].get(k)
            cells.append("--" if r is None else f"{r['metric']:.3f} [{r['ci90'][0]:.3f}, {r['ci90'][1]:.3f}]")
        L.append(NAME[k] + " & " + " & ".join(cells) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def teacher_side_table(d, secondary=None):
    import csv
    def rd(name):
        with open(d / name, newline="") as f:
            return list(csv.DictReader(f))
    met, icc, rank = rd("teacher_metric.csv"), rd("icc.csv"), {r["task"]: r for r in rd("ranking_resolution.csv")}
    # ranking resolution is registered analysis (e); print the post-S8 run (outputs/v5/secondary, B=1000) so this table,
    # Table S-ranking and the main-text macros agree (the pre-S7 teacher-side run used B=500 and differs by one pair on c16)
    sec = Path(secondary) / "ranking_resolution.csv" if secondary else None
    if sec and sec.exists():
        with open(sec, newline="") as f:
            for r in csv.DictReader(f):
                if r["task"] in rank:
                    rank[r["task"]] = {"n_teacher_pairs": r["n_teacher_pairs"], "significant_pairs_tile": r["significant_pairs_naive"],
                                       "significant_pairs_cluster": r["significant_pairs_cluster"]}
    import statistics as st
    L = [r"\begin{table}[h]\centering\small",
         r"\caption{Teacher-side uncertainty (no student involved): ratio of the cluster to the tile bootstrap standard error of the teacher metric "
         r"(median and range over the 12 teachers), ICC(1) of the teacher's correct-class probability within clusters, design effect, and the number of the "
         r"66 teacher pairs whose difference is significant under the tile versus the cluster 90\,\% interval (from the pre-specified ranking analysis, \texttt{atlas.pdf}); the CRC-VAL-HE-7K SE ratio is over the 9 teachers with non-zero tile SE. SE ratio, ICC(1) and design effect come from a separate teacher-only run; the pre-specified ranking analysis uses a different bootstrap draw (CAMELYON16 design effect 68.8 in \texttt{atlas.pdf}; the teacher-only run separates 51/28 pairs).}\label{tab:s_teacher_side}",
         r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Task & SE ratio & ICC(1) & design effect & sig.\ pairs (tile) & sig.\ pairs (cluster)\\", r"\midrule"]
    for t in ["c16", "msi", "center", "crc7k"]:
        inf = [float(r["inflation"]) for r in met if r["task"] == t and r["inflation"] not in ("", "nan")]
        ic = [float(r["icc1"]) for r in icc if r["task"] == t and r["icc1"] not in ("", "nan")]
        de = [float(r["design_effect"]) for r in icc if r["task"] == t and r["design_effect"] not in ("", "nan")]
        rk = rank.get(t, {})
        L.append(f"{TASK_NAME[t]} & {st.median(inf):.1f} ({min(inf):.1f}--{max(inf):.1f}) & {fnum(st.median(ic)) if ic else '--'} & "
                 f"{fnum(st.median(de), '{:.0f}') if de else '--'} & {rk.get('significant_pairs_tile', '--')}/{rk.get('n_teacher_pairs', '--')} & "
                 f"{rk.get('significant_pairs_cluster', '--')}/{rk.get('n_teacher_pairs', '--')}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def fnum(x, fmt="{:.2f}"):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    return "--" if v != v else fmt.format(v)

PANNUKE_CLASSES = ["neoplastic", "inflammatory", "connective", "dead", "epithelial", "background"]
STUDENT_NAME = {"vit-tiny": "ViT-Ti", "vit-small": "ViT-S", "vit-base": "ViT-B"}


def pannuke_class_table(root):
    """Per-class Dice (seed-pooled mean) per (teacher, student); teacher Dice on its own row. Returns None if no probe exists."""
    import glob, statistics
    recs = {}
    for f in glob.glob(f"{root}/*/*/seed*/probe_pannuke_seg.json"):
        r = json.loads(Path(f).read_text()); st = Path(f).parents[1].name
        recs.setdefault((r["teacher"], st), []).append(r["per_class"])
    if not recs:
        return None
    teachers = [t for t in ORDER if any(k[0] == t for k in recs)]
    L = [r"\begin{longtable}{@{}llrrrrrr@{}}",
         r"\caption{PanNuke per-class Dice (linear probe on frozen tokens; mean over seeds; background shown for completeness). "
         r"The main text and the deposited atlas use the per-image mean over the foreground classes present in the image or prediction, averaged over images, which is not the mean of these per-class columns.}\label{tab:s_pannuke_class}\\",
         r"\toprule Teacher & Model & " + " & ".join(c.capitalize() for c in PANNUKE_CLASSES) + r"\\ \midrule", r"\endfirsthead",
         r"\toprule Teacher & Model & " + " & ".join(c.capitalize() for c in PANNUKE_CLASSES) + r"\\ \midrule", r"\endhead",
         r"\bottomrule", r"\endlastfoot"]
    for t in teachers:
        pcs = [pc for (tt, _), lst in recs.items() if tt == t for pc in lst]
        td = [statistics.mean(pc[c]["teacher_dice"] for pc in pcs) for c in PANNUKE_CLASSES]
        L.append(f"{NAME.get(t, t)} & teacher & " + " & ".join(f"{v:.3f}" for v in td) + r"\\")
        for st in ["vit-tiny", "vit-small", "vit-base"]:
            if (t, st) not in recs:
                raise KeyError(f"pannuke per-class: no runs for {(t, st)}")
            sd = [statistics.mean(pc[c]["student_dice"] for pc in recs[(t, st)]) for c in PANNUKE_CLASSES]
            L.append(f" & {STUDENT_NAME[st]} ($n$={len(recs[(t, st)])}) & " + " & ".join(f"{v:.3f}" for v in sd) + r"\\")
        L.append(r"\addlinespace")
    L.append(r"\end{longtable}")
    return "\n".join(L) + "\n"


def teacher_extra_table(d):
    """Teacher-only extras (research/teacher_side_extra_v5): pair equivalence at delta=0.05, slide-balanced metric shift, probe-C sensitivity."""
    import csv, statistics
    eq = [r for r in csv.DictReader(open(d / "teacher_pair_equivalence.csv")) if abs(float(r["margin"]) - 0.05) < 1e-9]
    cs = list(csv.DictReader(open(d / "cluster_size_sensitivity.csv")))
    ps = list(csv.DictReader(open(d / "probe_sensitivity.csv")))
    pr = list(csv.DictReader(open(d / "probe_sensitivity_ranking.csv")))
    L = [r"\begin{table}[h]", r"\centering",
         r"\caption{Teacher-only robustness checks (no student involved). Pairs: of the 66 teacher pairs, those equivalent at $\delta=0.05$ under tile and cluster 90\% intervals, and those inconclusive under cluster intervals. "
         r"Balanced: largest absolute change in the teacher metric when every slide contributes at most the median number of tiles (no patient on Kather-MSI and no slide on the site task exceeds the median, so the check is not exercised there). "
         r"Probe $C$: largest absolute change of a teacher's metric when the logistic regularisation is moved from $C=1$ to $0.1$ or $10$, and Kendall $\tau$ of the teacher ranking against $C=1$.}",
         r"\label{tab:s_teacher_extra}", r"\begin{tabular}{@{}lrrrrrrr@{}}", r"\toprule",
         r"Task & Eq.\ tile & Eq.\ cluster & Inconcl.\ cluster & Balanced $|\Delta|_{\max}$ & Probe $|\Delta|_{\max}$ & $\tau(C{=}0.1)$ & $\tau(C{=}10)$\\", r"\midrule"]
    for task in [t for t in TASK_ORDER if any(r["task"] == t for r in eq)]:
        e = next(r for r in eq if r["task"] == task)
        bal = [abs(float(r["delta_balanced_minus_all"])) for r in cs if r["task"] == task and r["delta_balanced_minus_all"] not in ("", "nan")]
        base = {r["teacher"]: float(r["metric"]) for r in ps if r["task"] == task and float(r["C"]) == 1.0}
        dmax = max(abs(float(r["metric"]) - base[r["teacher"]]) for r in ps if r["task"] == task and float(r["C"]) != 1.0)
        tau = {float(r["C"]): float(r["kendall_tau_vs_C1"]) for r in pr if r["task"] == task}
        L.append(f"{TASK_NAME[task]} & {e['equivalent_tile']} & {e['equivalent_cluster']} & {e['inconclusive_cluster']} & "
                 f"{(f'{max(bal):.3f}' if bal else '--')} & {dmax:.3f} & {tau[0.1]:.2f} & {tau[10.0]:.2f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "paper_jbhi" / "supplement" / "tables"))
    ap.add_argument("--bug", default=str(ROOT / "research" / "teacher_bug_impact_v5.json"))
    ap.add_argument("--ref", default=str(ROOT / "outputs" / "v5" / "teacher_reference"))
    ap.add_argument("--teacher_side", default=str(ROOT / "research" / "teacher_side_secondary_v5"))
    ap.add_argument("--root", default=str(ROOT / "outputs" / "v5"), help="run root for probe_pannuke_seg.json")
    ap.add_argument("--teacher_extra", default=str(ROOT / "research" / "teacher_side_extra_v5"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    refs = load_ref(Path(a.ref))
    bug = json.loads(Path(a.bug).read_text())
    (out / "s_bugs.tex").write_text(bugs_table(bug, refs))
    (out / "s_teacher.tex").write_text(teacher_table(refs))
    if (Path(a.teacher_side) / "teacher_metric.csv").exists():
        (out / "s_teacher_side.tex").write_text(teacher_side_table(Path(a.teacher_side), ROOT / "outputs" / "v5" / "secondary"))
    if (Path(a.teacher_extra) / "probe_sensitivity.csv").exists():
        (out / "s_teacher_extra.tex").write_text(teacher_extra_table(Path(a.teacher_extra)))
    pk = pannuke_class_table(a.root)
    if pk:
        (out / "s_pannuke_class.tex").write_text(pk)
    print("wrote", out / "s_bugs.tex", out / "s_teacher.tex", "tasks:", [k for k in TASK_ORDER if k in refs])


if __name__ == "__main__":
    main()
