"""Denoising filters to compare before the tissue threshold (applied to a crop around the tract region)."""
import numpy as np
import scipy.ndimage as ndi
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means


def noise_sigma(a):
    """Robust noise level from neighbour differences (median absolute deviation); avoids the PyWavelets dependency."""
    d = np.diff(a, axis=2).ravel()
    return 1.4826 * float(np.median(np.abs(d - np.median(d)))) / np.sqrt(2)


def perona_malik(a, n_iter=10, kappa=0.08, step=0.12):
    """Anisotropic diffusion: smooths within regions, stops at edges (intensity steps larger than ~kappa)."""
    u = a.astype(np.float32).copy()
    g = lambda d: np.exp(-(d / kappa) ** 2) * d
    for _ in range(n_iter):
        upd = np.zeros_like(u)
        for ax in range(3):
            d = np.diff(u, axis=ax)                                   # u[i+1] - u[i]
            f = g(d)
            pad_hi = [(0, 1) if i == ax else (0, 0) for i in range(3)]
            pad_lo = [(1, 0) if i == ax else (0, 0) for i in range(3)]
            upd += np.pad(f, pad_hi) - np.pad(f, pad_lo)              # forward minus backward flux, zero at borders
        u += step * upd
    return u


FILTERS = {
    'none':        ('No filter', lambda a: a),
    'gauss07':     ('Gaussian σ 0.9 mm (current)', lambda a: ndi.gaussian_filter(a, 0.7)),
    'gauss12':     ('Gaussian σ 1.5 mm', lambda a: ndi.gaussian_filter(a, 1.2)),
    'median3':     ('Median 3×3×3', lambda a: ndi.median_filter(a, 3)),
    'median5':     ('Median 5×5×5', lambda a: ndi.median_filter(a, 5)),
    'tv':          ('Total variation', lambda a: denoise_tv_chambolle(a, weight=0.05)),
    'nlm':         ('Non-local means', lambda a: denoise_nl_means(a, patch_size=3, patch_distance=4, fast_mode=True,
                                                                  h=1.5 * noise_sigma(a))),
    'aniso':       ('Anisotropic diffusion', perona_malik),
}
