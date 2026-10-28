#!/usr/bin/env python3
"""Determinism tests for the v5 data pipeline (no GPU, no datasets on disk)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "scripts" / "data"))
sys.path.insert(0, str(ROOT / "scripts" / "eval_v5"))

from build_corpus import h01, subsample  # noqa: E402
from select_tcga_slides import bucket  # noqa: E402
from run_probes import split_for, cluster_tost  # noqa: E402


def test_sha_buckets_are_stable_and_uniform():
    assert bucket("brca", "TCGA-A1-A0SB") == bucket("brca", "TCGA-A1-A0SB")
    assert bucket("brca", "TCGA-A1-A0SB") != bucket("lung", "TCGA-A1-A0SB")
    vals = np.array([h01(f"x{i}") for i in range(20000)])
    assert 0.48 < vals.mean() < 0.52 and vals.min() >= 0 and vals.max() < 1


def test_subsample_is_order_independent():
    items = [{"path": f"p{i}", "index": i} for i in range(500)]
    a = subsample(list(items), 50, 42, key=lambda it: it["path"])
    b = subsample(list(reversed(items)), 50, 42, key=lambda it: it["path"])
    assert [it["path"] for it in a] == [it["path"] for it in b]
    c = subsample(list(items), 50, 43, key=lambda it: it["path"])
    assert [it["path"] for it in a] != [it["path"] for it in c]


def _fake_lab(n=600, n_groups=30, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    g = np.array([f"s{i % n_groups:03d}" for i in range(n)])
    y = rng.integers(0, n_classes, n)
    return {"labels": y, "groups": g, "subgroup": np.array([f"center{int(x[1:]) % 5}" for x in g]),
            "is_test": (np.arange(n) % 3 == 0).astype(int)}


def test_splits_are_deterministic_and_group_disjoint():
    lab = _fake_lab()
    for task in ("c17_loco", "msi", "panda", "crc7k", "center"):
        f1 = split_for(task, lab, 42); f2 = split_for(task, lab, 42)
        for (tr1, te1, n1), (tr2, te2, n2) in zip(f1, f2):
            assert n1 == n2 and np.array_equal(tr1, tr2) and np.array_equal(te1, te2)
            assert not set(tr1) & set(te1)
            if task != "crc7k":
                assert not set(lab["groups"][tr1]) & set(lab["groups"][te1]), task
    # leave-one-centre-out covers every tile exactly once
    cov = np.zeros(len(lab["labels"]), int)
    for _, te, _ in split_for("c17_loco", lab, 42):
        cov[te] += 1
    assert (cov == 1).all()


def test_cluster_tost_states():
    rng = np.random.default_rng(1)
    n = 2000; y = rng.integers(0, 2, n); g = np.array([f"s{i % 40}" for i in range(n)])
    sig = y + rng.normal(0, 1.0, n)
    pt = np.stack([1 - sig, sig], 1); ps = pt.copy()
    r = cluster_tost(y, pt, ps, g, np.array([0, 1]), "auroc", 0.05, n_boot=200)
    assert r["state"] == "equivalent" and abs(r["d"]) < 1e-9
    ps_bad = np.stack([rng.random(n), rng.random(n)], 1)        # student at chance
    r2 = cluster_tost(y, pt, ps_bad, g, np.array([0, 1]), "auroc", 0.05, n_boot=200)
    assert r2["state"] == "inequivalent" and r2["d"] < -0.1
