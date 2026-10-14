"""Unit tests for the PORTAL construction (no model downloads required).

These exercise the math on synthetic statistics: the closed-form map should
match the target's mean and covariance on the top-k subspace, produce the
documented LLaVA key/shape layout, and keep W2 fixed to the identity.
"""

import torch

from portal import build_portal, make_projector_state_dict, pca_stats, random_baseline
from portal.io import PROJECTOR_KEYS, validate_projector
from portal.targets import load_prompts

W1, B1, W2, B2 = PROJECTOR_KEYS


def _gaussian(n, d, mean, cov_sqrt, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g)
    return z @ cov_sqrt.T + mean


def test_pca_stats_shapes_and_flattening():
    feats = torch.randn(50, 7, 16)  # per-patch style input (N, P, d)
    mu, v, sigma = pca_stats(feats, k=8)
    assert mu.shape == (16,)
    assert v.shape == (8, 16)
    assert sigma.shape == (8,)
    # Orthonormal rows.
    gram = v @ v.T
    assert torch.allclose(gram, torch.eye(8), atol=1e-4)


def test_rank_is_clamped_to_available_dimensions():
    feats = torch.randn(10, 16)  # economy SVD yields min(N, d) = 10 components
    _, v, sigma = pca_stats(feats, k=512)
    assert v.shape[0] == 10  # clamped from the requested k=512
    assert sigma.shape[0] == v.shape[0]


def test_state_dict_keys_and_shapes():
    source = pca_stats(torch.randn(2000, 32), k=16)
    target = pca_stats(torch.randn(2000, 48), k=16)
    sd = build_portal(source, target, max_scale_ratio=None)
    validate_projector(sd)  # raises on any mismatch
    assert sd[W1].shape == (48, 32)
    assert sd[B1].shape == (48,)
    assert sd[W2].shape == (48, 48)
    assert sd[B2].shape == (48,)


def test_w2_is_identity():
    source = pca_stats(torch.randn(2000, 32), k=16)
    target = pca_stats(torch.randn(2000, 48), k=16)
    sd = build_portal(source, target, max_scale_ratio=None)
    assert torch.allclose(sd[W2].float(), torch.eye(48))
    assert torch.allclose(sd[B2].float(), torch.zeros(48))


def test_portal_matches_target_moments_on_subspace():
    # Full-rank Gaussians so the top-k subspace covers the whole space.
    d = 24
    g = torch.Generator().manual_seed(1)
    mean_x = torch.randn(d, generator=g)
    mean_y = torch.randn(d, generator=g) * 3 + 5
    cov_sqrt_x = torch.randn(d, d, generator=g)
    cov_sqrt_y = torch.randn(d, d, generator=g) * 2
    x = _gaussian(40000, d, mean_x, cov_sqrt_x, seed=2)
    y = _gaussian(40000, d, mean_y, cov_sqrt_y, seed=3)

    sd = build_portal(pca_stats(x, k=d), pca_stats(y, k=d), max_scale_ratio=None)
    w1 = sd[W1].float()
    b1 = sd[B1].float()
    projected = x @ w1.T + b1  # pre-activation output

    # Mean and covariance of the mapped source should match the target.
    assert torch.allclose(projected.mean(0), y.mean(0), atol=0.1)
    cov_proj = torch.cov(projected.T)
    cov_tgt = torch.cov(y.T)
    rel_err = (cov_proj - cov_tgt).norm() / cov_tgt.norm()
    assert rel_err < 0.05, f"covariance mismatch, rel_err={rel_err:.4f}"


def test_max_scale_ratio_caps_amplification():
    # Source with a near-degenerate direction would otherwise blow up the scale.
    source_feats = torch.randn(5000, 20)
    source_feats[:, 0] *= 1e-3  # tiny variance along axis 0
    source = pca_stats(source_feats, k=20)
    target = pca_stats(torch.randn(5000, 20) * 10, k=20)
    sd = build_portal(source, target, max_scale_ratio=50.0)
    # Spectral norm is bounded by the max scale (50) for orthonormal bases;
    # allow ~1% slack for bfloat16 storage rounding.
    assert torch.linalg.matrix_norm(sd[W1].float(), ord=2) <= 50.0 * 1.01


def test_random_baseline_layout_and_determinism():
    a = random_baseline(vision_dim=64, llm_dim=128, seed=7)
    b = random_baseline(vision_dim=64, llm_dim=128, seed=7)
    validate_projector(a)
    assert a[W1].shape == (128, 64)
    for key in PROJECTOR_KEYS:
        assert torch.equal(a[key], b[key])  # same seed -> identical weights


def test_load_prompts_skips_blank_and_comment_lines(tmp_path):
    f = tmp_path / "prompts.txt"
    f.write_text("# a comment\n\na photo of a dog\n  cat  \n#another\n")
    assert load_prompts(f) == ["a photo of a dog", "cat"]


def test_make_projector_state_dict_is_bfloat16():
    sd = make_projector_state_dict(
        torch.randn(8, 4), torch.randn(8), torch.eye(8), torch.zeros(8)
    )
    assert all(v.dtype == torch.bfloat16 for v in sd.values())
