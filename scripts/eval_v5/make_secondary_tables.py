#!/usr/bin/env python3
"""Supplement tables for the external pair and the secondary analyses.

  <out>/external.tex       from outputs/v5_external/summary/summary_all.json (H-optimus-0 -> H0-mini), rows = tasks
  <out>/s_three_state.tex  from outputs/v5/secondary/three_state.csv
  <out>/s_icc.tex          icc.csv summarised per task (median ICC(1), cluster size, design effect over teachers)
  <out>/s_calibration.tex  calibration_simulation.csv
  <out>/s_ranking.tex      ranking_resolution.csv + headroom_teacher_dispersion.csv
  <macros>                 \\extState<Task>, \\extD<Task>, \\extPrediction (held / not held), \\extLo/\\extHiPannuke;
                           margin sensitivity at delta 0.02 / 0.10 (\\sensCeil*, \\sensB/S/Ti*) from margin_sweep.csv
                           and delta_min_pooled.csv

Usage: python scripts/eval_v5/make_secondary_tables.py [--external outputs/v5_external/summary]
         [--secondary outputs/v5/secondary] [--out paper_jbhi/supplement/tables] [--macros paper_jbhi/macros_external.tex]
"""
import argparse, csv, json, statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TASK_NAME = {"c16": "CAMELYON16", "c17_loco": "CAMELYON17 LOCO", "panda": "PANDA tile", "panda_biopsy_tissue": "PANDA biopsy",
             "panda_biopsy": "PANDA biopsy, annotated tiles", "msi": "Kather-MSI", "center": "TCGA-BRCA site", "crc7k": "CRC-VAL-HE-7K", "pannuke_seg": "PanNuke Dice"}
# main-text tasks (whole-biopsy PANDA replaced the annotation-selected variant, analysis plan 7.1); EXTRA rows follow a rule
TASK_ORDER = ["c16", "c17_loco", "panda", "panda_biopsy_tissue", "msi", "center", "crc7k", "pannuke_seg"]
EXTRA = ["panda_biopsy"]
MACRO_KEY = {"c16": "CSixteen", "c17_loco": "CSeventeen", "panda": "Panda", "panda_biopsy_tissue": "PandaBiopsy", "panda_biopsy": "PandaRoi",
             "msi": "Msi", "center": "Center", "crc7k": "Crc", "pannuke_seg": "Pannuke"}


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x, fmt="{:.3f}"):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    if v != v or v in (float("inf"), float("-inf")):
        return "--"
    return fmt.format(v)


def external_table(summary):
    L = [r"\begin{table}[h]\centering\small",
         r"\caption{External pair H-optimus-0 $\rightarrow$ H0-mini under the identical probe, split, margin and cluster bootstrap "
         r"(single released checkpoint, so no seed pooling). Pre-specified prediction: equivalent on CAMELYON16; "
         r"not equivalent on at least one of CAMELYON17 and PANDA.}\label{tab:external}",
         r"\begin{tabular}{@{}lrrrrrc@{}}", r"\toprule",
         r"Task & $m_T$ & $m_S$ & $d$ & 90\,\% CI & $\delta_{\min}$ & verdict\\", r"\midrule"]
    macros = []
    for task in TASK_ORDER + EXTRA:
        rows = [r for r in summary.get(task, []) if r["student"] == "h0-mini"]
        if not rows:
            continue
        if task in EXTRA:
            L.append(r"\midrule")
        r = rows[0]; k = MACRO_KEY[task]
        dmin = max(abs(r["lo"]), abs(r["hi"]))
        L.append(f"{TASK_NAME[task]} & {r['t_metric']:.3f} & {r['s_metric_mean']:.3f} & {r['d_pooled']:+.3f} & "
                 f"[{r['lo']:+.3f}, {r['hi']:+.3f}] & {dmin:.3f} & {r['state']}\\\\")
        macros += [f"\\newcommand{{\\extState{k}}}{{{r['state']}}}", f"\\newcommand{{\\extD{k}}}{{{r['d_pooled']:+.3f}}}"]
        if task == "pannuke_seg":
            macros += [f"\\newcommand{{\\extLoPannuke}}{{{r['lo']:+.4f}}}", f"\\newcommand{{\\extHiPannuke}}{{{r['hi']:+.4f}}}"]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    st = {t: next((r["state"] for r in summary.get(t, []) if r["student"] == "h0-mini"), None) for t in ("c16", "c17_loco", "panda")}
    if all(st.values()):
        held = st["c16"] == "equivalent" and (st["c17_loco"] != "equivalent" or st["panda"] != "equivalent")
        macros.append(f"\\newcommand{{\\extPrediction}}{{{'held' if held else 'not held'}}}")
    return "\n".join(L) + "\n", macros


def three_state_table(rows):
    L = [r"\begin{table}[h]\centering\small",
         r"\caption{Three-state decomposition per task (seed-pooled verdicts, $\delta=0.05$): counts, median and maximum "
         r"$|d|$, median interval half-width, and inconclusive pairs with $|d|<0.02$.}\label{tab:s_three_state}",
         r"\begin{tabular}{@{}lrrrrrrrr@{}}", r"\toprule",
         r"Task & pairs & E & I & ? & med $|d|$ & max $|d|$ & med half-width & ? with $|d|<0.02$\\", r"\midrule"]
    by = {r["task"]: r for r in rows}
    for t in TASK_ORDER:
        r = by.get(t)
        if r is None:
            continue
        L.append(f"{TASK_NAME[t]} & {r['n_pairs']} & {r['equivalent']} & {r['inequivalent']} & {r['inconclusive']} & "
                 f"{fnum(r['abs_d_median'])} & {fnum(r['abs_d_max'])} & {fnum(r['half_width_median'])} & {r['inconclusive_with_small_d']}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def icc_table(rows):
    L = [r"\begin{table}[h]\centering\small",
         r"\caption{Within-cluster correlation of the teacher's correct-class probability: median over teachers of ICC(1), "
         r"ANOVA-adjusted cluster size $n_0$ and design effect $1+(n_0-1)\,\mathrm{ICC}$.}\label{tab:s_icc}",
         r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule", r"Task & clusters & ICC(1) & $n_0$ & design effect\\", r"\midrule"]
    for t in TASK_ORDER:
        rs = [r for r in rows if r["task"] == t]
        if not rs:
            continue
        def med(key):
            v = [float(r[key]) for r in rs if r[key] not in ("", "nan")]
            v = [x for x in v if x == x]
            return statistics.median(v) if v else None
        L.append(f"{TASK_NAME[t]} & {rs[0]['n_clusters']} & {fnum(med('icc1'))} & {fnum(med('mean_cluster_size'), '{:.1f}')} & "
                 f"{fnum(med('design_effect'), '{:.1f}')}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def calibration_table(rows):
    L = [r"\begin{table}[h]\centering\small",
         r"\caption{Calibration simulation: fraction of 200,000 simulated estimates declared equivalent when the true $d$ is $0$, $\delta/2$ or $\delta$ "
         r"and the noise equals the median cluster SE, using the tile-level SE versus the cluster SE (CRC-VAL-HE-7K, without cluster identifiers, and PanNuke, without a tile bootstrap, omitted).}\label{tab:s_calibration}",
         r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
         r"Task & true $d$ & SE$_{\text{cluster}}$ & SE$_{\text{tile}}$ & equiv.\ rate (tile) & equiv.\ rate (cluster)\\", r"\midrule"]
    for t in TASK_ORDER:
        if t == "crc7k":                    # no cluster: both SEs round to zero and the simulation is degenerate
            continue
        for r in [r for r in rows if r["task"] == t]:
            L.append(f"{TASK_NAME[t]} & {fnum(r['true_d'])} & {fnum(r['se_cluster_median'], '{:.4f}')} & {fnum(r['se_naive_median'], '{:.4f}')} & "
                     f"{fnum(r['equiv_rate_naive'], '{:.2f}')} & {fnum(r['equiv_rate_cluster'], '{:.2f}')}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def headroom_tau_table(head, tau):
    """Registered analyses (f) headroom-normalised differences and the student part of (e), teacher-vs-student Kendall tau."""
    med = lambda xs: statistics.median(xs) if xs else float("nan")
    def big(v, fmt):
        return ">100" if v == v and abs(v) > 100 else fnum(v, fmt)
    L = [r"\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}",
         r"\caption{Headroom and ranking agreement. Headroom $1-m_T$, the teacher--student difference and the margin relative to it "
         r"(medians over the 36 pairs), and Kendall's $\tau$ between the teacher ranking and the ranking of their students of each size "
         r"(12 teachers). PanNuke has no ranking analysis.}\label{tab:s_headroom}",
         r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Task & $1-m_T$ & $d/(1-m_T)$ & $\delta/(1-m_T)$ & $\tau$ ViT-B & $\tau$ ViT-S & $\tau$ ViT-Ti\\", r"\midrule"]
    notes = []
    for t in TASK_ORDER:
        rows = [r for r in head if r["task"] == t]
        if not rows:
            continue
        f = lambda k: [float(r[k]) for r in rows if r[k] not in ("", "nan")]
        n_fin = len(f("d_over_headroom"))
        if n_fin < len(rows):            # teacher metric exactly 1: headroom 0, ratio undefined
            notes.append(f"{TASK_NAME[t]}: {n_fin} of {len(rows)} pairs with non-zero headroom")
        tt = {r["student"]: r["kendall_tau"] for r in tau if r["task"] == t}
        L.append(f"{TASK_NAME[t]} & {fnum(med(f('headroom')), '{:.3f}')} & {big(med(f('d_over_headroom')), '{:+.2f}')} & "
                 f"{big(med(f('margin_over_headroom')), '{:.2f}')} & " +
                 " & ".join(fnum(tt.get(k), '{:+.2f}') if k in tt else "--" for k in ("vit-base", "vit-small", "vit-tiny")) + r"\\")
    if notes:
        L[1] = L[1].replace(r"PanNuke has no ranking analysis.}", r"PanNuke has no ranking analysis; " + "; ".join(notes) + ".}")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


def ranking_table(rank, disp):
    d = {r["task"]: r for r in disp}
    L = [r"\begin{table}[h]\centering\small",
         r"\caption{Ranking resolution and teacher dispersion: number of teacher pairs whose difference is significant under "
         r"the tile-level versus the cluster 90\,\% interval, and the spread of teacher metrics relative to the margin.}\label{tab:s_ranking}",
         r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Task & teachers & sig.\ pairs (tile) & sig.\ pairs (cluster) & $\bar m_T$ & range($m_T$) & $\delta$/range\\", r"\midrule"]
    for t in TASK_ORDER:
        r = next((x for x in rank if x["task"] == t), None)
        if r is None:
            continue
        g = d.get(t, {})
        L.append(f"{TASK_NAME[t]} & {r['n_teachers']} & {r['significant_pairs_naive']}/{r['n_teacher_pairs']} & "
                 f"{r['significant_pairs_cluster']}/{r['n_teacher_pairs']} & {fnum(g.get('t_metric_mean'))} & "
                 f"{fnum(g.get('t_metric_range'))} & {fnum(g.get('margin_over_range'), '{:.1f}')}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(L) + "\n"


CEILING = ("c16", "crc7k", "panda", "c17_loco")
SENS = {"Two": 0.02, "Ten": 0.10}


def sensitivity_macros(sweep, dmin):
    """Verdict counts at the two alternative margins: ceiling-task equivalents (of 144) and, per student size,
    equivalents over the seven tasks other than CRC-VAL-HE-7K (of 84; same pairs as the capacity paragraph)."""
    macros = []
    size_key = {"vit-base": "B", "vit-small": "S", "vit-tiny": "Ti"}
    for name, delta in SENS.items():
        ceil = sum(int(r["equivalent"]) for r in sweep if r["task"] in CEILING and abs(float(r["margin"]) - delta) < 1e-9)
        macros.append(f"\\newcommand{{\\sensCeil{name}}}{{{ceil}}}")
        for size, k in size_key.items():
            n = sum(1 for r in dmin if r["student"] == size and r["task"] in TASK_ORDER and r["task"] != "crc7k" and float(r["delta_min"]) < delta)
            macros.append(f"\\newcommand{{\\sens{k}{name}}}{{{n}}}")
    return macros


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", default=str(ROOT / "outputs" / "v5_external" / "summary"))
    ap.add_argument("--secondary", default=str(ROOT / "outputs" / "v5" / "secondary"))
    ap.add_argument("--out", default=str(ROOT / "paper_jbhi" / "supplement" / "tables"))
    ap.add_argument("--macros", default=str(ROOT / "paper_jbhi" / "macros_external.tex"))
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ext = Path(a.external) / "summary_all.json"
    macros = []
    if ext.exists():
        tex, macros = external_table(json.loads(ext.read_text()))
        (out / "external.tex").write_text(tex)
        print("external ->", out / "external.tex")
    sec = Path(a.secondary)
    if (sec / "margin_sweep.csv").exists() and (sec / "delta_min_pooled.csv").exists():
        macros += sensitivity_macros(read_csv(sec / "margin_sweep.csv"), read_csv(sec / "delta_min_pooled.csv"))
    if macros:
        Path(a.macros).write_text("% generated by scripts/eval_v5/make_secondary_tables.py -- do not edit\n" + "\n".join(macros) + "\n")
        print("macros ->", a.macros)
    if (sec / "three_state.csv").exists():
        (out / "s_three_state.tex").write_text(three_state_table(read_csv(sec / "three_state.csv")))
        (out / "s_icc.tex").write_text(icc_table(read_csv(sec / "icc.csv")))
        (out / "s_calibration.tex").write_text(calibration_table(read_csv(sec / "calibration_simulation.csv")))
        (out / "s_ranking.tex").write_text(ranking_table(read_csv(sec / "ranking_resolution.csv"),
                                                         read_csv(sec / "headroom_teacher_dispersion.csv")))
        (out / "s_headroom.tex").write_text(headroom_tau_table(read_csv(sec / "headroom_pairs.csv"),
                                                              read_csv(sec / "ranking_kendall_tau.csv")))
        print("secondary ->", out)


if __name__ == "__main__":
    main()
