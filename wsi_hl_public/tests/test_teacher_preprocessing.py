#!/usr/bin/env python3
"""CPU unit tests for the teacher preprocessing contract (two preprocessing bugs fixed in this version). No model weights are needed.

Bug 1: every dataset feeds ImageNet-normalised tensors; teachers with their own
       statistics must be re-normalised exactly inside TeacherModel.forward.
Bug 2: patch-14 teachers emit a 16x16 patch grid behind CLS + register tokens;
       the old code took the LAST 196 tokens, which is spatially misaligned with
       the student's 14x14 grid. Patch tokens must be sliced by prefix count and
       resampled 16x16 -> 14x14 with cell centres aligned.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from distill_wsi_model import (  # noqa: E402
    FM_REGISTRY, TEACHER_INPUT_NORM, TEACHER_TOKEN_LAYOUT, TeacherModel,
    _CLIP_MEAN, _CLIP_STD, _HALF_MEAN, _HALF_STD, _HIBOU_MEAN, _HIBOU_STD,
    _HOPTIMUS_MEAN, _HOPTIMUS_STD, _IMAGENET_MEAN, _IMAGENET_STD,
    _renormalize, _resample_patch_grid,
)

NON_IMAGENET = {"h-optimus-0", "conch", "midnight", "hibou-b", "hibou-l", "h0-mini"}


# ---------------------------------------------------------------- bug 1: normalisation
def test_registry_coverage():
    """Every teacher except the gated h-optimus-1 has an explicit input norm; every
    teacher has a token layout."""
    for k in FM_REGISTRY:
        assert k in TEACHER_TOKEN_LAYOUT, k
        if k != "h-optimus-1":
            assert k in TEACHER_INPUT_NORM, k


@pytest.mark.parametrize("name", sorted(TEACHER_INPUT_NORM))
def test_renormalisation_matches_direct(name):
    """ImageNet-normalised input -> _renormalize == direct normalisation with the
    teacher's own statistics, to float32 precision."""
    torch.manual_seed(0)
    raw = torch.rand(4, 3, 32, 32)                        # x/255 tiles in [0, 1]
    x_imagenet = (raw - _IMAGENET_MEAN) / _IMAGENET_STD   # what the dataset feeds
    mean, std = TEACHER_INPUT_NORM[name]
    got = _renormalize(x_imagenet, _IMAGENET_MEAN, _IMAGENET_STD, mean, std)
    want = (raw - mean) / std
    assert (got - want).abs().max().item() < 1e-5, name


def test_official_statistics_values():
    """Pin the constants to the values read from the official sources on 2026-10-16."""
    assert _HOPTIMUS_MEAN.flatten().tolist() == pytest.approx([0.707223, 0.578729, 0.703617])
    assert _HOPTIMUS_STD.flatten().tolist() == pytest.approx([0.211883, 0.230117, 0.177517])
    assert _CLIP_MEAN.flatten().tolist() == pytest.approx([0.48145466, 0.4578275, 0.40821073])
    assert _CLIP_STD.flatten().tolist() == pytest.approx([0.26862954, 0.26130258, 0.27577711])
    assert _HALF_MEAN.flatten().tolist() == [0.5, 0.5, 0.5]
    assert _HALF_STD.flatten().tolist() == [0.5, 0.5, 0.5]
    assert _HIBOU_MEAN.flatten().tolist() == pytest.approx([0.7068, 0.5755, 0.7220])
    assert _HIBOU_STD.flatten().tolist() == pytest.approx([0.1950, 0.2316, 0.1816])
    for k, (m, s) in TEACHER_INPUT_NORM.items():
        is_imagenet = torch.allclose(m, _IMAGENET_MEAN) and torch.allclose(s, _IMAGENET_STD)
        assert is_imagenet == (k not in NON_IMAGENET), k


# ---------------------------------------------------------------- bug 2: token layout
def test_token_layout_values():
    patch14 = {"uni2-h", "virchow", "virchow2", "h-optimus-0", "h-optimus-1",
               "hibou-b", "hibou-l", "midnight", "h0-mini"}
    for k, (ps, npref) in TEACHER_TOKEN_LAYOUT.items():
        assert ps == (14 if k in patch14 else 16), k
        assert 224 % ps == 0
    assert TEACHER_TOKEN_LAYOUT["uni2-h"] == (14, 9)
    for k in ("virchow2", "h-optimus-0", "hibou-b", "hibou-l"):
        assert TEACHER_TOKEN_LAYOUT[k] == (14, 5), k
    for k in ("virchow", "midnight"):
        assert TEACHER_TOKEN_LAYOUT[k] == (14, 1), k
    for k in ("phikon", "phikon-v2", "uni", "conch", "prov-gigapath"):
        assert TEACHER_TOKEN_LAYOUT[k] == (16, 1), k


def test_resample_identity_and_shape():
    x = torch.randn(2, 196, 8)
    assert _resample_patch_grid(x, 14, 14) is x
    y = _resample_patch_grid(torch.randn(2, 256, 8), 16, 14)
    assert y.shape == (2, 196, 8)
    with pytest.raises(ValueError):
        _resample_patch_grid(torch.randn(2, 196, 8), 16, 14)


def test_resample_is_cell_centre_aligned():
    """A delta at teacher cell (p, q) on the 16x16 grid must land at student cell
    round((p + 0.5) * 14 / 16 - 0.5) on the 14x14 grid (align_corners=False)."""
    for p, q in [(0, 0), (15, 15), (0, 15), (7, 8), (3, 12)]:
        x = torch.zeros(1, 256, 1)
        x[0, p * 16 + q, 0] = 1.0
        y = _resample_patch_grid(x, 16, 14)[0, :, 0].reshape(14, 14)
        i, j = divmod(int(y.argmax()), 14)
        ei = int(round((p + 0.5) * 14 / 16 - 0.5))
        ej = int(round((q + 0.5) * 14 / 16 - 0.5))
        assert (i, j) == (ei, ej), ((p, q), (i, j), (ei, ej))
    # a constant field stays constant
    c = _resample_patch_grid(torch.full((1, 256, 3), 2.5), 16, 14)
    assert torch.allclose(c, torch.full((1, 196, 3), 2.5))


def _bare_teacher(name, patch_grid=14):
    """TeacherModel without loading weights: only the layout fields are needed."""
    t = TeacherModel.__new__(TeacherModel)
    torch.nn.Module.__init__(t)
    t.teacher_name = name
    t.patch_grid = patch_grid
    t.patch_size, t.n_prefix_tokens = TEACHER_TOKEN_LAYOUT[name]
    t.native_grid = 224 // t.patch_size
    return t


@pytest.mark.parametrize("name", sorted(TEACHER_TOKEN_LAYOUT))
def test_extract_patch_tokens_drops_prefix_and_resamples(name):
    t = _bare_teacher(name)
    g, npref = t.native_grid, t.n_prefix_tokens
    D = 4
    seq = torch.zeros(1, npref + g * g, D)
    seq[0, :npref, 0] = -100.0                       # poison prefix tokens
    seq[0, npref:, 1] = torch.arange(g * g).float()  # patch index ramp
    out = t.extract_patch_tokens(seq)
    assert out.shape == (1, 196, D)
    assert (out[..., 0] > -1).all(), "prefix tokens leaked into patch_feats"
    # raster order is preserved: the ramp is non-decreasing except at row breaks
    r = out[0, :, 1]
    assert (r[1:] >= r[:-1] - 1e-4).sum() >= 196 - 14, "raster order not preserved"
    # native grid is returned untouched when patch_grid=None
    tn = _bare_teacher(name, patch_grid=None)
    assert tn.extract_patch_tokens(seq).shape == (1, g * g, D)
    # wrong token count (e.g. non-224 input) fails loudly
    with pytest.raises(RuntimeError):
        t.extract_patch_tokens(torch.zeros(1, npref + g * g + 1, D))


def test_old_last196_slice_is_misaligned_for_patch14():
    """Document the bug: on a 256-token grid, the last 196 tokens start at raster
    index 60 = row 3, col 12, not at the top-left corner."""
    g = 16
    idx = torch.arange(g * g)
    old = idx[-196:]
    assert int(old[0]) == 60 and divmod(60, g) == (3, 12)
    seq = torch.zeros(1, 5 + 256, 1)
    seq[0, 5:, 0] = idx.float()
    new = _bare_teacher("h-optimus-0").extract_patch_tokens(seq)[0, :, 0]
    assert new[0] < 2.0   # new slicing starts at the top-left patch


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
