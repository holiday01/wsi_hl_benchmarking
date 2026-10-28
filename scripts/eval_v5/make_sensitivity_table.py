#!/usr/bin/env python3
"""Supplementary tables for the probe and resampling sensitivity analyses (run_sensitivity_v5b.sh,
aggregate.py --seed_mode mean, c17_per_centre.py):
  paper_jbhi/supplement/tables/s_sensitivity.tex   verdict counts per task under each variant
  paper_jbhi/supplement/tables/s_c17_centre.tex    CAMELYON17-LOCO per held-out centre
  paper_jbhi/supplement/tables/s_panda_variants.tex whole-biopsy (tissue tiles) vs annotation-selected PANDA biopsy task (plan 7.1)"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_jbhi" / "supplement" / "tables"
TASKS = [("c16", "CAMELYON16"), ("c17_loco", "CAMELYON17 LOCO"), ("panda", "PANDA tile"), ("crc7k", "CRC-VAL-HE-7K"),
         ("panda_biopsy_tissue", "PANDA biopsy"), ("msi", "Kather-MSI"), ("center", "TCGA-BRCA site"), ("pannuke_seg", "PanNuke"),
         ("panda_biopsy", "PANDA biopsy, annotated tiles")]
VARIANTS = [("Main", "outputs/v5/summary/summary_all.json"),
            ("Seed mean", "outputs/v5_sens/seed_mean/summary_all.json"),
            ("$C=0.1$", "outputs/v5_sens/C0.1/summary/summary_all.json"),
            ("$C=10$", "outputs/v5_sens/C10/summary/summary_all.json"),
            ("Site by patient", "outputs/v5_sens/site_patient/summary/summary_all.json"),
            ("Centred", "outputs/v5_sens/basic/summary_all.json")]


def cell(rows):
    e = sum(r["state"] == "equivalent" for r in rows)
    lo = sum(r["state"] == "inequivalent" and r["hi"] < 0 for r in rows)
    hi = sum(r["state"] == "inequivalent" and r["lo"] > 0 for r in rows)
    inc = sum(r["state"] == "inconclusive" for r in rows)
    return f"{e}/{lo}/{hi}/{inc}"


def main():
    data = [(name, json.load(open(ROOT / p))) for name, p in VARIANTS]
    L = [r"\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
         r"\caption{Sensitivity of the seed-pooled verdicts at $\delta=0.05$ (36 pairs per task; equivalent/inferior/superior/inconclusive). "
         r"Main: pre-specified analysis (probe $C=1$, one seed drawn per bootstrap replicate, site folds by slide). Seed mean: interval for the mean of the three seeds' $d$. "
         r"$C=0.1$, $C=10$: every classification probe refitted (PanNuke not refitted). Site by patient: site-task folds grouped by patient. "
         r"Centred: the same bootstrap replicates centred on the estimate (basic interval, centred tail $p$-values) instead of the pre-specified percentile interval. "
         r"These analyses were run after the main results. --: not run.}\label{tab:s_sensitivity}",
         r"\begin{tabular}{@{}l" + "r" * len(data) + r"@{}}", r"\toprule",
         "Task & " + " & ".join(n for n, _ in data) + r"\\", r"\midrule"]
    for key, label in TASKS:
        if key == "panda_biopsy":
            L.append(r"\midrule")
        L.append(label + " & " + " & ".join(cell(s[key]) if s.get(key) else "--" for _, s in data) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "s_sensitivity.tex").write_text("\n".join(L) + "\n")
    panda_variants(data)

    c = json.load(open(ROOT / "outputs/v5/secondary/c17_per_centre.json"))
    L = [r"\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\caption{CAMELYON17 leave-one-centre-out, per held-out centre (ten slides each): teacher AUROC range over the 12 teachers and the seed-mean "
         r"teacher--student difference $d$ over the 36 pairs, and verdicts at $\delta=0.05$ from a per-centre paired slide bootstrap (equivalent/inconclusive; median 90\,\% half-width). With ten slides per centre the bootstrap intervals are themselves imprecise.}\label{tab:s_c17_centre}",
         r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
         r"Centre & Tiles & Teacher AUROC & median $d$ & range of $d$ & E/Inc. & half-width\\", r"\midrule"]
    for k in sorted(c):
        r = c[k]
        L.append(f"{int(k[-1])} & {r['n_test_tiles']:,} & {r['teacher_min']:.3f}--{r['teacher_max']:.3f} & {r['d_median']:+.3f} & "
                 f"{r['d_min']:+.3f} to {r['d_max']:+.3f} & {r['n_equivalent']}/{r['n_inconclusive']} & {r['halfwidth_median']:.3f}\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "s_c17_centre.tex").write_text("\n".join(L) + "\n")


def panda_variants(data):
    """Side by side, as plan section 7.1 requires: teacher AUROC range, verdict counts, per-pair d difference, H0-mini."""
    main = dict(data)["Main"]
    hier = json.load(open(ROOT / "outputs/v5/secondary/hier_bh_margins.json"))["per_task"]
    ext = json.load(open(ROOT / "outputs/v5_external/summary/summary_all.json"))
    tis = {(r["teacher"], r["student"]): r for r in main["panda_biopsy_tissue"]}
    roi = {(r["teacher"], r["student"]): r for r in main["panda_biopsy"]}
    dd = sorted(tis[k]["d_pooled"] - roi[k]["d_pooled"] for k in roi)
    same = sum(tis[k]["state"] == roi[k]["state"] for k in roi)
    L = [r"\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{3pt}",
         r"\caption{PANDA biopsy grading with label-independent tissue tiles (main text) and with annotation-selected Gleason tiles (replaced variant): "
         r"teacher macro-AUROC range over the 12 teachers, verdicts at $\delta=0.05$ (equivalent/inferior/superior/inconclusive) under the main analysis and the sensitivity variants, "
         r"equivalent pairs after hierarchical correction (teacher-on-task families), median $\delta_{\min}$ and H0-mini against H-optimus-0. "
         r"%d of the 36 pairs receive the same verdict under both; the per-pair $d$ (tissue minus annotated) differs by a median of %+.3f (range %+.3f to %+.3f).}\label{tab:s_panda_variants}"
         % (same, dd[len(dd) // 2] if len(dd) % 2 else (dd[len(dd) // 2 - 1] + dd[len(dd) // 2]) / 2, dd[0], dd[-1]),
         r"\begin{tabular}{@{}lrrrrrrrl@{}}", r"\toprule",
         r"Tiles & Teacher AUROC & Main & E$_{\mathrm{H}}$ & Seed mean & $C=0.1$ & $C=10$ & $\tilde\delta_{\min}$ & H0-mini $d$ [90\,\% CI]\\", r"\midrule"]
    for key, lab in (("panda_biopsy_tissue", "Tissue"), ("panda_biopsy", "Annotated")):
        rows = main[key]; t = [r["t_metric"] for r in rows]
        dm = sorted(max(abs(r["lo"]), abs(r["hi"])) for r in rows); n = len(dm)
        med = dm[n // 2] if n % 2 else (dm[n // 2 - 1] + dm[n // 2]) / 2
        e = next(r for r in ext[key] if r["student"] == "h0-mini")
        cells = [cell(dict(data)[v][key]) if dict(data)[v].get(key) else "--" for v in ("Main", "Seed mean", "$C=0.1$", "$C=10$")]
        L.append(f"{lab} & {min(t):.3f}--{max(t):.3f} & {cells[0]} & {hier[key]['equivalent_hier_task']} & {cells[1]} & {cells[2]} & {cells[3]} & {med:.3f} & "
                 f"{e['d_pooled']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}]\\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / "s_panda_variants.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
