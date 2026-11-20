"""Denoising filters to compare before the tissue threshold (applied to a crop around the tract region)."""
import numpy as np
import scipy.ndimage as ndi
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, estimate_sigma


def perona_malik(a, n_iter=10, kappa=0.08, step=0.12):
    """Anisotropic diffusion: smooths within regions, stops at edges (gradient > ~kappa)."""
    u = a.astype(np.float32).copy()
    for _ in range(n_iter):
        flux = np.zeros_like(u)
        for ax in range(3):
            d = np.diff(u, axis=ax, append=np.take(u, [-1], axis=ax))      # forward difference
            c = np.exp(-(d / kappa) ** 2)
            f = c * d
            flux += f - np.roll(f, 1, axis=ax) * (np.arange(u.shape[ax]) > 0).reshape([-1 if i == ax else 1 for i in range(3)])
        u += step * flux
    return u


FILTERS = {
    'none':        ('No filter', lambda a: a),
    'gauss07':     ('Gaussian σ 0.9 mm (current)', lambda a: ndi.gaussian_filter(a, 0.7)),
    'gauss12':     ('Gaussian σ 1.5 mm', lambda a: ndi.gaussian_filter(a, 1.2)),
    'median3':     ('Median 3×3×3', lambda a: ndi.median_filter(a, 3)),
    'median5':     ('Median 5×5×5', lambda a: ndi.median_filter(a, 5)),
    'tv':          ('Total variation', lambda a: denoise_tv_chambolle(a, weight=0.05)),
    'nlm':         ('Non-local means', lambda a: denoise_nl_means(a, patch_size=3, patch_distance=4, fast_mode=True,
                                                                  h=0.8 * float(np.mean(estimate_sigma(a))))),
    'aniso':       ('Anisotropic diffusion', perona_malik),
}
