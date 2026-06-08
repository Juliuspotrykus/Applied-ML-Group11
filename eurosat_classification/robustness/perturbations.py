"""
Perturbation functions for model robustness evaluation.

RGB functions operate on [3, H, W] float tensors in [0, 1].
MS functions operate on [13, H, W] z-score normalised tensors.
All functions are pure: they do not modify the input tensor.
"""
from __future__ import annotations

import torch
import torchvision.transforms.functional as TF


# ── RGB perturbations ─────────────────────────────────────────────────────────


def rotate_rgb(img: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate by a fixed angle; border pixels filled with black."""
    return TF.rotate(img, degrees, fill=0)


def noise_rgb(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add zero-mean Gaussian noise; output clipped to [0, 1]."""
    return torch.clamp(img + torch.randn_like(img) * sigma, 0.0, 1.0)


def brightness_rgb(img: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale all pixel values by factor; simulates lighting change."""
    return TF.adjust_brightness(img, factor)


def salt_pepper_rgb(img: torch.Tensor, density: float) -> torch.Tensor:
    """Replace a random fraction of pixels with 0 (pepper) or 1 (salt)."""
    out = img.clone()
    mask = torch.rand_like(img)
    out[mask < density / 2] = 0.0
    out[mask > 1.0 - density / 2] = 1.0
    return out


def contrast_rgb(img: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale pixel values around the image mean; factor > 1 increases contrast."""
    mean = img.mean()
    return torch.clamp((img - mean) * factor + mean, 0.0, 1.0)


def hflip(img: torch.Tensor) -> torch.Tensor:
    return TF.hflip(img)


def vflip(img: torch.Tensor) -> torch.Tensor:
    return TF.vflip(img)


# ── MS perturbations  ─────────────────────────────────────────

def rotate_ms(img: torch.Tensor, degrees: float) -> torch.Tensor:
    """Rotate by a fixed angle; border filled with 0 (≈ band mean in z-score space)."""
    return TF.rotate(img, degrees, fill=0)


def noise_ms(img: torch.Tensor, sigma: float) -> torch.Tensor:
    """Add Gaussian noise in normalised z-score space."""
    return img + torch.randn_like(img) * sigma


def brightness_ms(img: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale all z-scores by factor; approximates uniform sensor-gain change."""
    return img * factor


def salt_pepper_ms(img: torch.Tensor, density: float) -> torch.Tensor:
    """Replace a fraction of spatial pixels with ±3 z-score values (dead/saturated pixels)."""
    out = img.clone()
    spatial_mask = torch.rand(*img.shape[1:]) < density
    n_affected = int(spatial_mask.sum())
    if n_affected > 0:
        polarity = torch.where(
            torch.rand(img.shape[0], n_affected) > 0.5,
            torch.full((img.shape[0], n_affected), 3.0),
            torch.full((img.shape[0], n_affected), -3.0),
        )
        out[:, spatial_mask] = polarity
    return out


def contrast_ms(img: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale each band around its spatial mean; factor > 1 increases contrast."""
    mean = img.mean(dim=(-2, -1), keepdim=True)
    return (img - mean) * factor + mean


def band_dropout_ms(img: torch.Tensor, band_idx: int) -> torch.Tensor:
    """Zero out a single band (sets z-scores to 0, equivalent to replacing with band mean)."""
    out = img.clone()
    out[band_idx] = 0.0
    return out


def band_n_dropout_ms(img: torch.Tensor, n: int) -> torch.Tensor:
    """Zero out the first n bands in index order (0 … n-1)."""
    out = img.clone()
    out[:n] = 0.0
    return out
