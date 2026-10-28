#!/usr/bin/env python3
"""Sensitivity of the seed-pooled verdicts to the bootstrap interval: pre-specified percentile interval with uncentred
tail p-values (outputs/v5/summary) versus the basic interval centred on the estimate (outputs/v5_sens/basic/<task>,
same replicates, aggregate.py --interval basic). Writes outputs/v5_sens/basic/compare.json and prints per-task counts."""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASKS = ["c16", "c17_loco", "panda", "panda_biopsy_tissue", "msi", "center", "crc7k", "pannuke_seg", "panda_biopsy"]
SHORT = {"equivalent": "E", "inequivalent": "N", "inconclusive": "I", "undefined": "U"}


def counts(rows):
    c = Counter(r["state"] for r in rows)
    return f"{c['equivalent']}/{c['inequivalent']}/{c['inconclusive']}"


def main():
    out = {}
    for t in TASKS:
        f = ROOT / f"outputs/v5_sens/basic/{t}/summary_{t}.json"
        if not f.exists():
            continue
        a = json.load(open(ROOT / f"outputs/v5/summary/summary_{t}.json")); b = json.load(open(f))
        key = lambda r: (r["teacher"], r["student"])
        bb = {key(r): r for r in b}
        changed = [(r["teacher"], r["student"], SHORT[r["state"]] + "->" + SHORT[bb[key(r)]["state"]],
                    round(r["d_pooled"], 4), round(r["lo"], 4), round(r["hi"], 4),
                    round(bb[key(r)]["lo"], 4), round(bb[key(r)]["hi"], 4))
                   for r in a if bb[key(r)]["state"] != r["state"]]
        shift = [abs((r["lo"] + r["hi"]) / 2 - r["d_pooled"]) for r in a]
        out[t] = {"percentile": counts(a), "basic": counts(b), "n_changed": len(changed), "changed": changed,
                  "max_asymmetry": max(shift), "equiv_bh_percentile": sum(r["bh_reject"] for r in a),
                  "equiv_bh_basic": sum(r["bh_reject"] for r in b)}
        print(t, out[t]["percentile"], "->", out[t]["basic"], "changed", len(changed),
              "max |mid-d|", round(max(shift), 4), changed)
    json.dump(out, open(ROOT / "outputs/v5_sens/basic/compare.json", "w"), indent=1)


if __name__ == "__main__":
    main()
