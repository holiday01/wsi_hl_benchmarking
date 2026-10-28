#!/usr/bin/env python3
"""Evaluation tile sets for the v5 redo. Each entry yields uint8 224x224 tiles plus
labels and cluster ids (slide / biopsy / patient) from a fixed on-disk source.
None of these sources is part of the distillation corpus (configs/corpus_v5.json).

  id                     source                                       label              cluster
  c16_annotated_270      patches_c16_annotated_270.h5 (270 slides)    tumour/normal      slide
  camelyon17_annotated50 camelyon17_annotated50_v5.h5 (50 slides)     tumour/normal      slide (+centre)
  kather_msi             patches_kather_msi.h5 (is_test split)        MSI/MSS            slide
  panda_radboud          panda_radboud_v5.h5                          benign/G3/G4/G5    biopsy
  tcga_brca_center       tcga_brca_eval_v5.h5                         TSS code           slide
  panda_radboud_tissue   panda_radboud_tissue_v5b.h5 (same biopsies)  none (ISUP/biopsy) biopsy
  crc_val_7k             CRC-VAL-HE-7K folder                         9 tissue classes   tile (none)
"""
import hashlib, json, os
from pathlib import Path

import h5py
import numpy as np

# data roots; override with environment variables (see README)
H5_OLD = os.environ.get("WSI_PATCH_CACHE", "/data3/cache/patches")
H5_V5 = os.environ.get("WSI_V5_ROOT", "/data/wsi_hl_v5") + "/patches"
EXTERNAL = os.environ.get("WSI_EXTERNAL_DIR", "/mnt/nas1/wsi_datasets/external")
CRC_CLASSES = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]

REGISTRY = {
    "c16_annotated_270":      {"kind": "h5", "path": f"{H5_OLD}/patches_c16_annotated_270.h5"},
    "camelyon17_annotated50": {"kind": "h5", "path": f"{H5_V5}/camelyon17_annotated50_v5.h5"},
    "kather_msi":             {"kind": "h5", "path": f"{H5_OLD}/patches_kather_msi.h5"},
    "panda_radboud":          {"kind": "h5", "path": f"{H5_V5}/panda_radboud_v5.h5"},
    "tcga_brca_center":       {"kind": "h5", "path": f"{H5_V5}/tcga_brca_eval_v5.h5"},
    "panda_radboud_tissue":   {"kind": "h5", "path": f"{H5_V5}/panda_radboud_tissue_v5b.h5"},
    "crc_val_7k":             {"kind": "folder",
                               "root": f"{EXTERNAL}/nct_crc/CRC-VAL-HE-7K_unz/CRC-VAL-HE-7K"},
}


def _dec(x):
    return x.decode() if isinstance(x, bytes) else str(x)


def file_fingerprint(path):
    st = os.stat(path)
    return {"path": str(path), "size": st.st_size, "mtime": int(st.st_mtime)}


class TileSet:
    """n, labels [n] int64, groups [n] str, extras dict of per-tile arrays, read(idx)->uint8."""

    def __init__(self, ds_id):
        spec = REGISTRY[ds_id]
        self.id, self.kind = ds_id, spec["kind"]
        self.extras = {}
        if self.kind == "h5":
            self.path = spec["path"]
            with h5py.File(self.path, "r") as h:
                self.labels = h["labels"][:].astype(np.int64)
                sid = h["slide_ids"][:].astype(np.int64)
                names = [_dec(x) for x in h.attrs.get("slide_names", [])]
                self.groups = np.array([names[i] if names else f"slide{i}" for i in sid])
                # slide_groups holds the per-slide sub-group written by extract_patches_v5
                # (centre for CAMELYON17, TSS code for TCGA-BRCA); never overwrite "groups".
                for key, out_key in (("slide_groups", "subgroup"), ("slide_patients", "patient")):
                    if key in h.attrs:
                        arr = [_dec(x) for x in h.attrs[key]]
                        self.extras[out_key] = np.array([arr[i] for i in sid])
                if "is_test" in h:
                    self.extras["is_test"] = h["is_test"][:].astype(np.int64)
            self.n = len(self.labels)
            self.fingerprint = file_fingerprint(self.path)
            self._h5 = None
        elif self.kind == "folder":
            root = Path(spec["root"])
            files = sorted(str(p) for p in root.rglob("*.tif"))
            self.files = files
            self.labels = np.array([CRC_CLASSES.index(Path(f).parent.name) for f in files], dtype=np.int64)
            self.groups = np.array([Path(f).stem for f in files])     # no patient ids: each tile its own cluster
            self.n = len(files)
            self.fingerprint = {"root": str(root), "n": self.n,
                                "sha256_of_list": hashlib.sha256("\n".join(files).encode()).hexdigest()}
        else:
            raise ValueError(self.kind)

    def read(self, idx):
        """idx: sorted int array -> uint8 [k,224,224,3]."""
        if self.kind == "h5":
            if self._h5 is None:
                self._h5 = h5py.File(self.path, "r", swmr=True)
            return self._h5["patches"][np.asarray(idx)]
        from PIL import Image
        out = np.empty((len(idx), 224, 224, 3), dtype=np.uint8)
        for k, i in enumerate(idx):
            im = Image.open(self.files[i]).convert("RGB")
            if im.size != (224, 224):
                im = im.resize((224, 224), Image.BILINEAR)
            out[k] = np.asarray(im)
        return out

    def label_bundle(self):
        d = {"labels": self.labels, "groups": self.groups}
        d.update(self.extras)
        return d
