"""Consistent internal geometry across scans and tissue thresholds.

1. Rigid registration of every scan to a reference scan (29_clench_r1) using only the upper head (rows above the
   palate: skull, brain, nose, upper face), which moves rigidly. Jaw, tongue and neck are not used.
2. Per scan: emptiness = fraction of the tissue thresholds THRS (0.20-0.50) at which a voxel is empty (tissue_mesh.voids).
3. Resample emptiness (and intensity) into the reference frame and average over scans.
   consistency ~1: empty in every scan at every threshold (fixed tubes: nasal cavity, nasopharynx, pharynx);
   intermediate: empty only for some articulations (oral cavity) or thresholds.
Outputs out/consensus/consensus.npz (mean_empty, mean_img, std_empty, scans, params) and per-scan transforms.
"""
import sys, os, glob
import numpy as np
import scipy.ndimage as ndi
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation
import fit_gnm as fg
import tissue_mesh as tm

REF = '29_clench_r1'
THRS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
REG_ROW_MAX = 70          # registration mask: head above this row (palate ~ row 62)


def transform(p, shape):
    """p = rotvec (rad, 3) + translation (voxels, 3); maps reference voxel coords -> moving voxel coords."""
    c = (np.array(shape) - 1) / 2
    R = Rotation.from_rotvec(p[:3]).as_matrix()
    return R, c - R @ c + p[3:]


def warp(vol, p, order=1):
    R, o = transform(p, vol.shape)
    return ndi.affine_transform(vol, R, offset=o, order=order, mode='constant', cval=0.0)


def register(ref, mov, mask, p0=None):
    """Rigid registration (normalised cross-correlation inside mask), coarse (2x down) then full resolution."""
    p = np.zeros(6) if p0 is None else p0.copy()
    for f in (2, 1):
        r = ndi.zoom(ref, 1 / f, order=1); m = ndi.zoom(mov, 1 / f, order=1)
        k = ndi.zoom(mask.astype(np.float32), 1 / f, order=1) > 0.5
        rv = r[k]; rv = (rv - rv.mean()) / rv.std()
        def cost(q):
            w = warp(m, q)[k]
            return -np.mean(rv * (w - w.mean()) / (w.std() + 1e-6))
        q0 = p.copy(); q0[3:] /= f
        res = minimize(cost, q0, method='Powell', options=dict(xtol=1e-3, ftol=1e-5, maxfev=1500))
        p = res.x.copy(); p[3:] *= f
    return p, -res.fun


if __name__ == '__main__':
    recon_dir, out_dir = sys.argv[1:3]
    os.makedirs(os.path.join(out_dir, 'reg'), exist_ok=True)
    scans = sorted(os.path.basename(f)[10:-10] for f in glob.glob(os.path.join(recon_dir, 'sub001_3d_*_recon.mat')))
    ref = fg.load_volume(os.path.join(recon_dir, f'sub001_3d_{REF}_recon.mat'))
    mask = ndi.binary_dilation(fg.head_mask(ref), iterations=2); mask[REG_ROW_MAX:] = False
    sum_e = np.zeros(ref.shape, np.float32); sum_e2 = np.zeros_like(sum_e); sum_i = np.zeros_like(sum_e)
    sum_t = np.zeros((len(THRS),) + ref.shape, np.float32)          # per-threshold sums, to re-pick the range later
    params, scores = {}, {}
    for s in scans:
        a = fg.load_volume(os.path.join(recon_dir, f'sub001_3d_{s}_recon.mat'))
        cache = os.path.join(out_dir, 'reg', f'{s}.npz')               # registration is the slow part: cache it
        if s == REF:
            p, ncc = np.zeros(6), 1.0
        elif os.path.exists(cache):
            c_ = np.load(cache); p, ncc = c_['p'], float(c_['ncc'])
        else:
            p, ncc = register(ref, a, mask)
            np.savez(cache, p=p, ncc=ncc)
        et = np.stack([warp(tm.voids(a, t)[0].astype(np.float32), p) for t in THRS])
        sum_t += et
        ew = et.mean(0); aw = warp(a, p)
        sum_e += ew; sum_e2 += ew ** 2; sum_i += aw
        params[s], scores[s] = p, ncc
        print(f'{s:22s} ncc {ncc:.3f}  rot {np.degrees(np.linalg.norm(p[:3])):4.1f} deg  shift {np.round(p[3:] * fg.VOX, 1)} mm', flush=True)
    n = len(scans)
    mean_e = sum_e / n
    np.savez_compressed(os.path.join(out_dir, 'consensus.npz'), mean_empty=mean_e, std_empty=np.sqrt(np.maximum(sum_e2 / n - mean_e ** 2, 0)),
                        mean_img=sum_i / n, per_thr=(sum_t / n * 255).round().astype(np.uint8), ref_img=ref, scans=np.array(scans), params=np.array([params[s] for s in scans]),
                        ncc=np.array([scores[s] for s in scans]), thrs=np.array(THRS), ref=REF)


def mirror_params(ref):
    """Rigid params p with warp(flip_lr(ref), p) ~= ref, i.e. the reflection about the head's own midline plane
    (registration on the whole head mask, not just the upper head)."""
    mask = ndi.binary_dilation(fg.head_mask(ref), iterations=2)
    return register(ref, ref[:, :, ::-1].copy(), mask)


def symmetrise(M, p):
    """Average a map with its mirror image about the plane found by mirror_params."""
    return 0.5 * (M + warp(M[:, :, ::-1].copy(), p))
