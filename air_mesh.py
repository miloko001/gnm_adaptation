"""General mesh of the air spaces inside the head (nasal cavity/sinuses, oral cavity, pharynx, larynx).

No lip / glottis / velum cuts -- just "dark inside the head":
  inside  head mask, with an extra coronal hole fill so an open mouth counts as inside; eroded 2 voxels
  box     rows >= 30 (drops the brain, which is fairly dark in this GRE) and columns < 98 (drops the spine / neck)
  air     hysteresis threshold on the lightly smoothed volume: grow below HIGH from seeds below LOW
  keep    components > 0.5 ml whose centroid is within 28 mm of the midline (drops ear canals / mastoid air)
"""
import sys, os
import numpy as np
import scipy.ndimage as ndi
from skimage import measure
import trimesh
import fit_gnm as fg

LOW, HIGH = 0.10, 0.16
ROW_MIN, COL_MAX = 30, 98
MIN_ML, MAX_ABS_X = 0.5, 28.0
VOXML = fg.VOX ** 3 / 1000


def inside_mask(a):
    m = fg.head_mask(a)
    for j in range(m.shape[1]):                      # coronal slices: closes the oral cavity when the lips are open
        m[:, j, :] = ndi.binary_fill_holes(m[:, j, :])
    return ndi.binary_erosion(m, iterations=2)


def internal_air(a):
    s = ndi.gaussian_filter(a, 0.7)
    box = inside_mask(a); box[:ROW_MIN] = False; box[:, COL_MAX:] = False
    lab, n = ndi.label((s < HIGH) & box)
    idx = np.arange(1, n + 1)
    size = ndi.sum(np.ones_like(lab), lab, idx) * VOXML
    seeded = ndi.maximum(s < LOW, lab, idx) > 0
    xc = (ndi.mean(np.indices(lab.shape)[2], lab, idx) - (a.shape[2] - 1) / 2) * fg.VOX
    keep = idx[seeded & (size > MIN_ML) & (np.abs(xc) < MAX_ABS_X)]
    return np.isin(lab, keep)


def air_mesh(air, faces=20000):
    v, f, _, _ = measure.marching_cubes(ndi.gaussian_filter(np.pad(air, 2).astype(np.float32), 0.8), 0.5)
    m = trimesh.Trimesh(fg.vox_to_world(v - 2, air.shape[2]), f)
    trimesh.smoothing.filter_taubin(m, iterations=10)
    if len(m.faces) > faces:
        m = m.simplify_quadric_decimation(face_count=faces)
    return m


if __name__ == '__main__':
    recon_dir, out_dir = sys.argv[1:3]
    for scan in sys.argv[3:]:
        a = fg.load_volume(os.path.join(recon_dir, f'sub001_3d_{scan}_recon.mat'))
        air = internal_air(a)
        m = air_mesh(air)
        m.export(os.path.join(out_dir, f'{scan}_air.stl'))
        print(f'{scan}: air {air.sum() * VOXML:.1f} ml, {ndi.label(air)[1]} parts, mesh {len(m.faces)} faces', flush=True)
