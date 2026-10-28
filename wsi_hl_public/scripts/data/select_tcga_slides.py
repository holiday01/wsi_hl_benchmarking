#!/usr/bin/env python3
"""Deterministic, patient-level selection of TCGA slides for the v5 redo.

Roles (redo_execution_plan_2026-10-16.md §2):
  brca/distill   300 slides   distillation corpus only
  brca/eval      ~15% of patients   centre-probe evaluation only (disjoint patients)
  lung/distill   150 slides   distillation corpus only
  lihc/distill    80 slides   distillation corpus only
Assignment uses sha256 of "<cohort>:<patient>" (NOT Python's salted hash()), so the
manifest is reproducible from the metadata table alone. Slides without an MPP are excluded.

  python scripts/data/select_tcga_slides.py --meta data/manifests/tcga_svs_meta_2026-10-16.tsv \
      --out data/manifests/tcga_slides_v5.json
"""
import argparse, csv, hashlib, json, os
from collections import defaultdict

PARAMS = {"brca_eval_frac": 0.15, "brca_distill": 300, "lung_distill": 150, "lihc_distill": 80,
          "mpp_min": 0.2, "mpp_max": 0.6,
          # stain filter from survey_stain.py: IHC (DAB) slides hide among TCGA "DX" files
          "max_brown_frac": 0.2, "min_pink_frac": 0.5}


def bucket(cohort: str, patient: str) -> float:
    h = hashlib.sha256(f"{cohort}:{patient}".encode()).hexdigest()
    return int(h, 16) / float(1 << 256)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="TSV: path, mpp, appmag, levels, dims, vendor")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stain", help="TSV from survey_stain.py; slides failing the H&E rule are excluded")
    args = ap.parse_args()

    bad_stain = {}
    if args.stain:
        with open(args.stain) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                name = os.path.basename(r["path"])
                if r["status"] != "ok":
                    bad_stain[name] = "survey:" + r["note"]
                elif float(r["brown_frac"]) > PARAMS["max_brown_frac"] or float(r["pink_frac"]) < PARAMS["min_pink_frac"]:
                    bad_stain[name] = f"not H&E (brown={r['brown_frac']}, pink={r['pink_frac']})"

    by_cohort = defaultdict(list)
    n_skipped = 0
    with open(args.meta) as f:
        for row in csv.reader(f, delimiter="\t"):
            path = row[0]
            cohort = path.split("/")[0]
            try:
                mpp = float(row[1])
            except (ValueError, IndexError):
                n_skipped += 1; continue
            if not (PARAMS["mpp_min"] <= mpp <= PARAMS["mpp_max"]):
                n_skipped += 1; continue
            name = os.path.basename(path)
            if name in bad_stain:
                n_skipped += 1; continue
            patient = name[:12]                 # TCGA-XX-XXXX
            tss = name.split("-")[1]
            by_cohort[cohort].append({"path": os.environ.get("WSI_TCGA_DIR", "/mnt/nas1/wsi_datasets/tcga") + "/" + path, "name": name,
                                      "cohort": cohort, "patient": patient, "tss": tss,
                                      "mpp": mpp, "bucket": bucket(cohort, patient)})

    selected = []

    def take(cohort, n_target, role, exclude_patients=frozenset(), min_bucket=0.0):
        """Patients in bucket order until n_target slides are reached (whole patients)."""
        rows = sorted((r for r in by_cohort[cohort] if r["bucket"] >= min_bucket
                       and r["patient"] not in exclude_patients),
                      key=lambda r: (r["bucket"], r["name"]))
        out, pats = [], set()
        for r in rows:
            if len(out) >= n_target and r["patient"] not in pats:
                break
            out.append(dict(r, role=role)); pats.add(r["patient"])
        return out

    # BRCA: evaluation patients are bucket < frac; distillation drawn from the rest.
    brca_eval = [dict(r, role="eval_center_probe") for r in by_cohort["brca"]
                 if r["bucket"] < PARAMS["brca_eval_frac"]]
    eval_pats = {r["patient"] for r in brca_eval}
    brca_dist = take("brca", PARAMS["brca_distill"], "distill", exclude_patients=eval_pats,
                     min_bucket=PARAMS["brca_eval_frac"])
    assert not ({r["patient"] for r in brca_dist} & eval_pats)
    selected += brca_eval + brca_dist
    selected += take("lung", PARAMS["lung_distill"], "distill")
    selected += take("lihc", PARAMS["lihc_distill"], "distill")

    selected.sort(key=lambda r: (r["cohort"], r["role"], r["name"]))
    body = {"params": PARAMS, "n_skipped_no_or_bad_mpp_or_stain": n_skipped,
            "excluded_stain": bad_stain, "slides": selected}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    body["sha256"] = digest
    with open(args.out, "w") as f:
        json.dump(body, f, indent=1)

    from collections import Counter
    c = Counter((r["cohort"], r["role"]) for r in selected)
    for k, v in sorted(c.items()):
        pats = len({r["patient"] for r in selected if (r["cohort"], r["role"]) == k})
        print(f"  {k[0]:5s} {k[1]:18s} slides={v:4d} patients={pats}")
    tss_eval = Counter(r["tss"] for r in brca_eval)
    print(f"  brca eval TSS with >=3 slides: {sum(1 for v in tss_eval.values() if v >= 3)} "
          f"(of {len(tss_eval)}), slides in those: {sum(v for v in tss_eval.values() if v >= 3)}")
    print(f"  manifest sha256 {digest[:16]}  -> {args.out}")


if __name__ == "__main__":
    main()
