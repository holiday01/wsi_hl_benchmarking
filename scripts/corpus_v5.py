#!/usr/bin/env python3
"""Shared readers for the v5 distillation corpus and the teacher-target cache.

Corpus directory (scripts/data/build_corpus.py):
  corpus_u8.npy   uint8 [N,224,224,3]  (np.save format, memmap-able)
  index.npz       tag / group / split per sample
  meta.json       n, memmap_sha256, manifest_sha256, config
Teacher cache directory (scripts/cache_teacher_targets.py), one per teacher:
  cls.npy         float16 [N,D]      CLS token
  patch14.npy     float16 [N,196,D]  patch tokens resampled to the student's 14x14 grid
  native_mean.npy float16 [N,D]      mean of the native patch tokens (Midnight official
                                     embedding = concat(cls, native_mean))
  meta.json       teacher, D, N, corpus hashes, verification stats
Images are ImageNet-normalised on the GPU with normalize_u8(); this is byte-identical
to multi_source_dataset._to_tensor for 224x224 inputs.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_u8(x_u8: torch.Tensor) -> torch.Tensor:
    """uint8 (B,224,224,3) or (B,3,224,224) on any device -> ImageNet-normalised float32 (B,3,224,224)."""
    if x_u8.shape[-1] == 3:
        x_u8 = x_u8.permute(0, 3, 1, 2)
    x = x_u8.float().div_(255.0)
    return (x - _MEAN.to(x.device)) / _STD.to(x.device)


class Corpus:
    def __init__(self, corpus_dir):
        self.dir = Path(corpus_dir)
        self.meta = json.load(open(self.dir / "meta.json"))
        idx = np.load(self.dir / "index.npz")
        self.tag, self.group, self.split = idx["tag"], idx["group"], idx["split"]
        self.n = int(self.meta["n"])
        assert len(self.tag) == self.n
        self._images = None

    @property
    def images(self) -> np.memmap:
        if self._images is None:  # opened lazily so DataLoader workers each get their own map
            self._images = np.load(self.dir / "corpus_u8.npy", mmap_mode="r")
        return self._images

    def indices(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)


class TeacherCache:
    def __init__(self, cache_dir, corpus: "Corpus | None" = None):
        self.dir = Path(cache_dir)
        self.meta = json.load(open(self.dir / "meta.json"))
        if corpus is not None:
            if self.meta["corpus_memmap_sha256"] != corpus.meta["memmap_sha256"]:
                raise ValueError(f"teacher cache {self.dir} was built for a different corpus")
        self.dim = int(self.meta["dim"])
        self.n = int(self.meta["n"])
        self._cls = self._patch = self._nm = None

    def _open(self):
        if self._cls is None:
            self._cls = np.load(self.dir / "cls.npy", mmap_mode="r")
            self._patch = np.load(self.dir / "patch14.npy", mmap_mode="r")
            self._nm = np.load(self.dir / "native_mean.npy", mmap_mode="r")

    def cls(self, i):
        self._open(); return self._cls[i]

    def patch14(self, i):
        self._open(); return self._patch[i]

    def native_mean(self, i):
        self._open(); return self._nm[i]


class CorpusImages(Dataset):
    """uint8 images only (for teacher extraction / evaluation). Returns (img_u8 HWC, idx)."""
    def __init__(self, corpus: Corpus, indices=None):
        self.c = corpus
        self.idx = np.arange(corpus.n) if indices is None else np.asarray(indices)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        return torch.from_numpy(np.ascontiguousarray(self.c.images[i])), i


class DistillPairs(Dataset):
    """(img_u8 HWC, teacher cls fp16 [D], teacher patch14 fp16 [196,D]) for distillation."""
    def __init__(self, corpus: Corpus, cache: TeacherCache, indices):
        self.c, self.t = corpus, cache
        self.idx = np.asarray(indices)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = int(self.idx[k])
        return (torch.from_numpy(np.ascontiguousarray(self.c.images[i])),
                torch.from_numpy(np.ascontiguousarray(self.t.cls(i))),
                torch.from_numpy(np.ascontiguousarray(self.t.patch14(i))))
