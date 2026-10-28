#!/usr/bin/env python3
"""Study-design schematic (fig0_design.pdf): no numbers from results; also usable as a graphical abstract.
Usage: make_schematic.py --out paper_jbhi/figures"""
import argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def box(ax, x, y, w, h, title, lines, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03", fc=fc, ec="0.3", lw=0.8))
    ax.text(x + w / 2, y + h - 0.09, title, ha="center", va="top", fontsize=8, weight="bold")
    ax.text(x + w / 2, y + h / 2 - 0.08, "\n".join(lines), ha="center", va="center", fontsize=6.2, linespacing=1.3)

def arrow(ax, x0, x1, y, text=None):
    ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=10, lw=0.9, color="0.3"))
    if text:
        ax.text((x0 + x1) / 2, y + 0.05, text, ha="center", va="bottom", fontsize=5.8, color="0.3")

def main(out):
    fig, ax = plt.subplots(figsize=(7.1, 2.1))
    ax.set_xlim(0, 7.1); ax.set_ylim(0, 2.1); ax.axis("off")
    y, h = 0.35, 1.45
    box(ax, 0.1, y, 1.45, h, "12 teachers", ["public pathology FMs", "86 M to 1.1 B params", "patch 14 and 16", "one tested wrapper"], "#deebf7")
    box(ax, 2.0, y, 1.45, h, "Students", ["ViT-Ti / S / B", "token-wise feature", "distillation, fp32", "3 seeds, 108 runs"], "#e5f5e0")
    box(ax, 3.9, y, 1.5, h, "8 tasks", ["CAMELYON16/17", "PANDA tile/biopsy", "Kather-MSI, CRC-7K", "TCGA site, PanNuke", "frozen linear probes"], "#fff7bc")
    box(ax, 5.85, y, 1.15, h, "Verdict", ["TOST, pre-specified", "margin", "cluster CI", "E / I / inconclusive"], "#fee0d2")
    arrow(ax, 1.58, 1.98, y + h / 2, "distil")
    arrow(ax, 3.48, 3.88, y + h / 2, "probe")
    arrow(ax, 5.43, 5.83, y + h / 2, "test")
    ax.text(3.55, 0.08, "Analysis plan fixed before student training; tile-level intervals are replaced by cluster bootstraps wherever cluster identifiers exist.",
            ha="center", va="bottom", fontsize=6.5, style="italic", color="0.25")
    fig.savefig(f"{out}/fig0_design.pdf", bbox_inches="tight"); fig.savefig(f"{out}/fig0_design.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="paper_jbhi/figures"); main(ap.parse_args().out)
