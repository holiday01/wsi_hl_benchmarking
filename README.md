# Distilling pathology foundation models into ViT students

Code for distilling twelve public pathology foundation models (teachers) into ViT-Ti, ViT-S and ViT-B students
and for testing each teacher-student pair for equivalence (two one-sided tests, cluster bootstrap, margin 0.05)
on eight frozen-probe tasks. The run records, pooled summaries and per-pair atlas are in the data deposit
(see *Data*); this repository holds only code.

## Contents

| Path | What it does |
|---|---|
| `scripts/distill_wsi_model.py` | Teacher wrappers (model IDs, input normalisation, token layout) and the ViT student |
| `scripts/data/` | Slide selection, tile extraction (TCGA, CAMELYON17, PANDA), distillation-corpus build |
| `scripts/cache_teacher_targets.py` | Runs each teacher once over the corpus and caches its class and patch-token targets |
| `scripts/distill_v5.py` | Trains one student against one cached teacher |
| `scripts/eval_v5/extract_features.py` | Frozen features of a teacher or student on an evaluation tile set |
| `scripts/eval_v5/run_probes.py` | Linear probe, teacher-student difference, cluster bootstrap and TOST verdict for one task |
| `scripts/eval_v5/pannuke_seg.py` | PanNuke patch-token cache and segmentation probe |
| `scripts/eval_v5/aggregate.py` | Seed-pooled verdict per teacher-student pair and task |
| `scripts/run_v5_pipeline.sh` | The whole study end to end (corpus, teacher caches, 108 runs, features, probes, summary) |
| `scripts/eval_v5/*.py`, `scripts/figures/` | Secondary and sensitivity analyses, tables and figures |
| `configs/corpus_v5.json` | Sources of the distillation corpus |
| `data/manifests/` | Slide and biopsy lists with checksums used for selection |
| `tests/` | CPU unit tests of the teacher preprocessing (no weights needed) |

## 1. Environment

Tested with Python 3.12, PyTorch 2.11 (CUDA 13.0) on one NVIDIA RTX 5090 (32 GB) under Linux.

```bash
python3.12 -m venv ~/venvs/kd && source ~/venvs/kd/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pytest -q tests          # 37 tests, CPU only
```

Hardware used for the study: a ViT-B student needs about 18 GB of GPU memory in fp32 at batch size 128 (ViT-S about
10 GB, ViT-Ti about 6 GB). The twelve teacher target caches take about 0.9 TB of disk.

## 2. Teacher weights

Most teachers are gated on the Hugging Face Hub. Request access on each model page, accept its licence
(several are restricted to non-commercial research), then log in once:

```bash
huggingface-cli login
```

| Key | Hub model |
|---|---|
| `phikon`, `phikon-v2` | `owkin/phikon`, `owkin/phikon-v2` |
| `uni`, `uni2-h` | `MahmoodLab/UNI`, `MahmoodLab/UNI2-h` |
| `conch` | `MahmoodLab/CONCH` (also needs the CONCH package, see `requirements.txt`) |
| `virchow`, `virchow2` | `paige-ai/Virchow`, `paige-ai/Virchow2` |
| `h-optimus-0` | `bioptimus/H-optimus-0` |
| `prov-gigapath` | `prov-gigapath/prov-gigapath` |
| `hibou-b`, `hibou-l` | `histai/hibou-b`, `histai/hibou-L` |
| `midnight` | `kaiko-ai/midnight` |
| `h0-mini` (external pair) | `bioptimus/H0-mini` |

## 3. Data locations

Paths are read from environment variables; the defaults are the paths of the original machine. Set them once:

```bash
export WSI_V5_ROOT=/your/fast/disk/kd        # tiles (patches/), corpus, teacher caches, features
export WSI_PATCH_CACHE=/your/fast/disk/kd/patches_legacy   # CAMELYON16 and Kather-MSI tile files
export WSI_EXTERNAL_DIR=/your/data/external  # nct_crc/, pannuke/, kather_msi/ as downloaded
export WSI_TCGA_DIR=/your/data/tcga          # TCGA diagnostic slides by cohort
export C17_RAW=/your/data/camelyon17         # CAMELYON17 download (slides and annotations)
export PANDA_RAW=/your/data/panda             # PANDA download (train.csv, images, masks)
export PY=$(which python)
```

Absolute source paths in `configs/corpus_v5.json` must be edited to match.

All cohorts are public: TCGA (BRCA, LUAD, LUSC, LIHC; Genomic Data Commons), CAMELYON16 and CAMELYON17, PANDA
(Radboud subset), NCT-CRC-HE-100K and CRC-VAL-HE-7K, Kather-MSI (TCGA CRC MSI/MSS tiles) and PanNuke.
Download helpers: `scripts/downloads/download_tcga.py` (needs `gdc-client`), `scripts/download_camelyon16.py`,
`scripts/data/download_camelyon17.sh`, `scripts/data/download_panda.sh` (Kaggle API) and
`scripts/downloads/download_external.py` (NCT-CRC, Kather-MSI, PanNuke).

## 4. Tutorial: train one student

The steps below distil Phikon into a ViT-S student with seed 42. Every other teacher, student size and seed is the
same command with different arguments.

**4.1 Extract the corpus tiles** (224 x 224 px at 0.5 µm/px). The TCGA slide lists are in `data/manifests/`.

```bash
bash scripts/data/run_tcga_extraction_v5.sh      # writes $WSI_V5_ROOT/patches/tcga_*_distill_v5.h5
```

**4.2 Build the distillation corpus** (memory-mapped tiles plus a 10 % group-level validation split).

```bash
$PY scripts/data/build_corpus.py --config configs/corpus_v5.json --out_dir $WSI_V5_ROOT/corpus --workers 24
```

**4.3 Cache the teacher's targets.** Each teacher runs once over the corpus; the cache is checked against a fresh
forward pass on `--verify_n` tiles before it is marked verified.

```bash
$PY scripts/cache_teacher_targets.py --corpus $WSI_V5_ROOT/corpus --teacher phikon \
    --out_root $WSI_V5_ROOT/teacher_cache --batch_size 128 --num_workers 8
```

**4.4 Train the student.** Defaults are the study's recipe: 30 epochs, early stopping with patience 5, batch 128,
AdamW at 1e-4 with cosine decay and weight decay 1e-4, class-token loss weight 1.0 and patch-token loss weight 0.5, fp32.

```bash
$PY scripts/distill_v5.py --corpus $WSI_V5_ROOT/corpus --cache_root $WSI_V5_ROOT/teacher_cache \
    --teacher phikon --student vit_small_patch16_224 --seed 42 \
    --output outputs/v5/phikon/vit-small/seed42 --num_workers 5
```

Students: `vit_tiny_patch16_224`, `vit_small_patch16_224`, `vit_base_patch16_224`; seeds used: 42, 2025, 7.
The run directory receives `best.pt`, `training_log.json` (per-epoch losses; `"finished": true` when done) and
`train.log`. On the RTX 5090 a ViT-B epoch took about 6.5 minutes when running alone.
`--exclude_tags pannuke` leaves a corpus source out (used for the PanNuke-exposure ablation).

**4.5 Evaluate the pair on one task.** Extract frozen features of teacher and student on the task's tile set,
then fit the probe and test equivalence.
The evaluation tile sets are built once:

```bash
$PY scripts/build_kather_msi_h5.py --base $WSI_EXTERNAL_DIR/kather_msi --out $WSI_PATCH_CACHE/patches_kather_msi.h5
$PY scripts/extract_c16_annotated.py --wsi_dir /your/data/camelyon16 --ann_dir /your/data/camelyon16/annotations \
    --out $WSI_PATCH_CACHE/patches_c16_annotated_270.h5 --seed 42
bash scripts/data/run_camelyon17_extraction_v5.sh
bash scripts/data/run_panda_extraction_v5.sh
```

The CAMELYON16 lesion annotations (`tumor_001.xml` to `tumor_111.xml`) are at
`https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/annotations/`. CRC-VAL-HE-7K is read directly from its folder
under `$WSI_EXTERNAL_DIR/nct_crc/`.


```bash
$PY scripts/eval_v5/extract_features.py --dataset kather_msi --model teacher:phikon
$PY scripts/eval_v5/extract_features.py --dataset kather_msi --model student:outputs/v5/phikon/vit-small/seed42
$PY scripts/eval_v5/run_probes.py --task msi --teacher phikon \
    --student_key student__v5_phikon_vit-small_seed42 --out outputs/v5/phikon/vit-small/seed42/probe_msi.json
```

Tasks: `c16`, `c17_loco`, `panda`, `panda_biopsy_tissue`, `msi`, `center`, `crc7k` (and `pannuke_seg` through
`scripts/eval_v5/pannuke_seg.py cache|probe`). The JSON holds the teacher and student metric, `d`, the 90 % cluster
bootstrap interval and the verdict at the 0.05 margin.

**4.6 Pool the seeds** once all runs of a pair are evaluated:

```bash
$PY scripts/eval_v5/aggregate.py --root outputs/v5 --out outputs/v5/summary
```

## 5. The full study

`scripts/run_v5_pipeline.sh` runs every stage for all 12 teachers x 3 students x 3 seeds (108 runs), skipping
finished stages, and launches student runs only when enough GPU memory is free. Before computing any probe
statistic it checks that the analysis plan is committed; in this code-only repository set
`ANALYSIS_PLAN_GATE=0` (the analysis plan is in the data deposit).

```bash
ANALYSIS_PLAN_GATE=0 bash scripts/run_v5_pipeline.sh
```

On one RTX 5090 the 108 runs took about nine days. The secondary, sensitivity and ablation analyses are run by
`scripts/eval_v5/run_v5_additions.sh`, `run_teacher_side_v5.sh`, `run_sensitivity_v5b.sh`,
`run_panda_tissue_v5b.sh`, `run_sensitivity_panda_tissue.sh` and `run_ablation_no_pannuke.sh`. The table and figure
generators (`scripts/eval_v5/make_*.py`, `scripts/figures/`) write into a `paper_jbhi/` directory and read the
analysis plan and teacher-side analyses from `research/`; both are provided by the data deposit, not this repository.

## Data

Run records, seed-pooled summaries, secondary, sensitivity and ablation analyses, the analysis plan and the per-pair
atlas: DOI to be added.

## Licence

Code: MIT (see `LICENSE`). Teacher weights and datasets remain under their own licences and terms of use.
