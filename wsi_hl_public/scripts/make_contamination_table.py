#!/usr/bin/env python3
"""Supplement Table S1: teacher corpus and disclosed overlap with the evaluation cohorts.

Six cohort columns are parsed from research/contamination_verification_2026-07-05.md and the CAMELYON17 and
PANDA columns from research/contamination_verification_2026-10-18_c17_panda.md; both files were verified against
the developers' papers and model cards (marks and verbatim sentences recorded there).
Output: paper_jbhi/supplement/tables/s_sources.tex
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "research" / "contamination_verification_2026-07-05.md"
SRC2 = ROOT / "research" / "contamination_verification_2026-10-18_c17_panda.md"
OUT = ROOT / "paper_jbhi" / "supplement" / "tables" / "s_sources.tex"
ORDER = ["Phikon", "Phikon-v2", "CONCH", "UNI", "UNI2-H", "Prov-GigaPath", "Virchow", "Virchow2",
         "H-Optimus-0", "Hibou-B", "Hibou-L", "Midnight"]
SHOW = {"UNI2-H": "UNI2-h", "H-Optimus-0": "H-optimus-0", "Midnight": "Midnight-12k"}
COHORTS = ["C16", "TCGA-BRCA", "NCT-CRC", "Kather-MSI", "PanNuke", "NCT-VAL"]
SYM = {"✓": r"$\bullet$", "↔": r"$\circ$", "✗": "--", "?": "?"}
CORPUS = {  # one-line corpus descriptions, from the Corpus: lines of the source file
    "Phikon": "TCGA, 6,093 WSIs, 16 cancer types", "Phikon-v2": "PANCAN-XL, 58,359 WSIs, 136 public and internal sets",
    "CONCH": "in-house 21,442 WSIs; 1.17M image--caption pairs", "UNI": "Mass-100K (MGH, BWH, GTEx)",
    "UNI2-H": "Mass General Brigham, $>$350k WSIs", "Prov-GigaPath": "Providence, 171,189 WSIs",
    "Virchow": "MSKCC, 1.49M WSIs", "Virchow2": "MSKCC and partner institutions, 3.1M WSIs",
    "H-Optimus-0": "proprietary, $>$500k WSIs", "Hibou-B": "HistAI proprietary, $>$1M WSIs",
    "Hibou-L": "HistAI proprietary, $>$1M WSIs", "Midnight": "TCGA, 12k WSIs, 32 cancer types"}


def parse():
    marks = {t: {} for t in ORDER}
    for src, cohorts in ((SRC, COHORTS), (SRC2, ["C17", "PANDA"])):
        txt = src.read_text()
        for t in ORDER:
            m = re.search(r"^## %s\n(.*?)(?=^## |\Z)" % re.escape(t), txt, re.S | re.M)
            block = m.group(1)
            marks[t].update({c: re.search(r"\*\*%s\*\*: (\S)" % re.escape(c), block).group(1) for c in cohorts})
    return marks


def main():
    marks = parse()
    cols = ["C16", "C17", "PANDA", "NCT-CRC", "NCT-VAL", "Kather-MSI", "PanNuke", "TCGA-BRCA"]
    L = [r"\begin{table}[h]\centering\footnotesize\setlength{\tabcolsep}{2.5pt}",
         r"\caption{Teacher pretraining corpora and disclosed overlap with the evaluation cohorts. "
         r"$\bullet$ = cohort disclosed as part of the corpus; $\circ$ = cohort derived from TCGA slides that are in the corpus; "
         r"-- = corpus disclosed as disjoint; ? = proprietary or undisclosed. All marks were verified against the developers' "
         r"papers and model cards; several teachers used CAMELYON- or PANDA-derived patches for evaluation only. Overlap may raise the "
         r"teacher's absolute score and the task ceiling and so bias $d$ against the student (not measured).}\label{tab:s_sources}",
         r"\begin{tabular}{@{}llcccccccc@{}}", r"\toprule",
         "Teacher & Corpus (as disclosed) & " + " & ".join(cols) + r"\\", r"\midrule"]
    for t in ORDER:
        m = marks[t]
        row = {"C16": SYM[m["C16"]], "C17": SYM[m["C17"]], "PANDA": SYM[m["PANDA"]],
               "NCT-CRC": SYM[m["NCT-CRC"]], "NCT-VAL": SYM[m["NCT-VAL"]], "Kather-MSI": SYM[m["Kather-MSI"]],
               "PanNuke": SYM[m["PanNuke"]], "TCGA-BRCA": SYM[m["TCGA-BRCA"]]}
        L.append(f"{SHOW.get(t, t)} & {CORPUS[t]} & " + " & ".join(row[c] for c in cols) + r"\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print("wrote", OUT)
    for t in ORDER:
        print(f"{t:14s}", " ".join(f"{c}:{marks[t][c]}" for c in COHORTS + ["C17", "PANDA"]))


if __name__ == "__main__":
    main()
