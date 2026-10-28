#!/usr/bin/env python3
"""Completes the planned hierarchical Benjamini-Hochberg procedure (teacher as family, q=0.05) and
tabulates verdicts over a range of margins, from the seed-pooled summaries (aggregate.py).

Hierarchical BH (Benjamini & Bogomolov 2014): stage 1 tests each teacher family with its Simes
p-value and applies BH across the 12 families at q; within each of the R selected families the
student hypotheses are tested by BH at q*R/12. Two family definitions are reported:
  task:  a family is one teacher on one task (3 student sizes), the scope of the stage-1 BH
         already in aggregate.py
  atlas: a family is one teacher over all eight tasks (24 hypotheses), the stricter reading
Margins: verdict counts at delta in {0.01, 0.02, 0.03, 0.05, 0.10} from the stored 90% intervals.
Writes outputs/v5/secondary/hier_bh_margins.json and the supplementary table s_margins_bh.tex."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
Q = 0.05
TASKS = [("c16", "CAMELYON16"), ("c17_loco", "CAMELYON17 LOCO"), ("panda", "PANDA tile"), ("crc7k", "CRC-VAL-HE-7K"),
         ("panda_biopsy_tissue", "PANDA biopsy"), ("msi", "Kather-MSI"), ("center", "TCGA-BRCA site"), ("pannuke_seg", "PanNuke")]
# annotation-selected biopsy variant (replaced in the main text, analysis plan 7.1): task families only, outside the atlas families
EXTRA = [("panda_biopsy", "PANDA biopsy, annotated tiles")]
DELTAS = [0.01, 0.02, 0.03, 0.05, 0.10]


def bh(p, q):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p)
    ok = p[o] <= q * np.arange(1, n + 1) / n
    rej = np.zeros(n, bool)
    if ok.any():
        rej[o[:np.max(np.flatnonzero(ok)) + 1]] = True
    return rej


def simes(p):
    p = np.sort(np.asarray(p, float)); n = len(p)
    return float(np.min(n * p / np.arange(1, n + 1)))


def hierarchical(fams, q=Q):
    """fams: {family: [(key, p), ...]} -> set of rejected keys."""
    names = sorted(fams)
    sel = bh([simes([p for _, p in fams[f]]) for f in names], q)
    R = int(sel.sum()); out = set()
    for f, s in zip(names, sel):
        if s:
            keys, ps = zip(*fams[f])
            out |= {k for k, r in zip(keys, bh(ps, q * R / len(names))) if r}
    return out


def main():
    summ = json.load(open(ROOT / "outputs/v5/summary/summary_all.json"))
    rows = {(t, r["teacher"], r["student"]): r for t, _ in TASKS + EXTRA for r in summ[t]}
    main_tasks = {t for t, _ in TASKS}
    pv = {k: (r["p_tost"] if np.isfinite(r["p_tost"]) else 1.0) for k, r in rows.items()}
    fam_task, fam_atlas = defaultdict(list), defaultdict(list)
    for k, p in pv.items():
        fam_task[(k[0], k[1])].append((k, p))
        if k[0] in main_tasks:
            fam_atlas[k[1]].append((k, p))
    # task-level families: run the hierarchy within each task over its 12 teachers
    rej_task = set()
    for t, _ in TASKS + EXTRA:
        rej_task |= hierarchical({f: v for f, v in fam_task.items() if f[0] == t})
    rej_atlas = hierarchical(fam_atlas)
    res = {"q": Q, "per_task": {}}
    for t, _ in TASKS + EXTRA:
        rr = [r for k, r in rows.items() if k[0] == t]
        eq = [k for k, r in rows.items() if k[0] == t and r["state"] == "equivalent"]
        d = {"equivalent_unadjusted": len(eq),
             "equivalent_stage1": sum(1 for k in eq if rows[k].get("bh_reject")),
             "equivalent_hier_task": sum(1 for k in eq if k in rej_task),
             "equivalent_hier_atlas": sum(1 for k in eq if k in rej_atlas) if t in main_tasks else None, "margins": {}}
        for dl in DELTAS:
            d["margins"][str(dl)] = {
                "E": sum(r["lo"] > -dl and r["hi"] < dl for r in rr),
                "I-": sum(r["hi"] < -dl for r in rr), "I+": sum(r["lo"] > dl for r in rr)}
            d["margins"][str(dl)]["Inc"] = len(rr) - sum(d["margins"][str(dl)].values())
        res["per_task"][t] = d
        print(t, {k: v for k, v in d.items() if k != "margins"}, {k: v["E"] for k, v in d["margins"].items()})
    for size in ("vit-base", "vit-small", "vit-tiny"):
        res[f"capacity_hier_atlas_{size}"] = sum(1 for k in rej_atlas if k[2] == size and k[0] in main_tasks - {"crc7k"} and rows[k]["state"] == "equivalent")
        res[f"capacity_hier_task_{size}"] = sum(1 for k in rej_task if k[2] == size and k[0] in main_tasks - {"crc7k"} and rows[k]["state"] == "equivalent")
    print({k: v for k, v in res.items() if k.startswith("capacity")})
    Path(ROOT / "outputs/v5/secondary").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(ROOT / "outputs/v5/secondary/hier_bh_margins.json", "w"), indent=1)

    L = [r"\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}",
         r"\caption{Multiplicity and margin. Left: equivalent pairs (of 36) unadjusted, after the first-stage Benjamini--Hochberg correction within teacher and task, "
         r"and after the full two-stage hierarchical procedure (Simes-combined teacher families, BH across families, BH within selected families at $qR/m$; $q=0.05$) "
         r"with a family being one teacher on one task (task) or one teacher over all eight tasks (atlas). "
         r"The annotation-selected PANDA biopsy variant is outside the atlas families. "
         r"Right: verdicts at other margins from the same 90\,\% intervals (equivalent/inferior/superior/inconclusive).}\label{tab:s_margins_bh}",
         r"\begin{tabular}{@{}lrrrr" + "r" * len(DELTAS) + r"@{}}", r"\toprule",
         r"& \multicolumn{4}{c}{Equivalent at $\delta=0.05$} & \multicolumn{%d}{c}{Verdicts at $\delta$}\\" % len(DELTAS),
         r"\cmidrule(lr){2-5}\cmidrule(l){6-%d}" % (5 + len(DELTAS)),
         r"Task & Unadj. & Stage 1 & Task & Atlas & " + " & ".join(f"{x:.2f}" for x in DELTAS) + r"\\", r"\midrule"]
    for t, lab in TASKS + EXTRA:
        d = res["per_task"][t]; m = d["margins"]
        if t not in main_tasks:
            L.append(r"\midrule")
        L.append(f"{lab} & {d['equivalent_unadjusted']} & {d['equivalent_stage1']} & {d['equivalent_hier_task']} & {d['equivalent_hier_atlas'] if t in main_tasks else '--'} & "
                 + " & ".join(f"{m[str(x)]['E']}/{m[str(x)]['I-']}/{m[str(x)]['I+']}/{m[str(x)]['Inc']}" for x in DELTAS) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (ROOT / "paper_jbhi/supplement/tables/s_margins_bh.tex").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
