"""Inner surface of the solid tissue, built the same way as the skin outline (head_mesh_fit.mri_head_mesh).

solid   lightly smoothed volume > THR (adjustable: tissue_mesh(a, thr)), 1-voxel opening. THR is higher than the skin threshold (0.12) because narrow
        parts of the tract (e.g. the oropharynx for /a/) only drop to ~0.25 through partial volume, and 0.12 seals them.
voids   non-solid voxels inside the head, within a box below the brain (rows >= 30) and in front of the spine
        (cols < 98), and within NECK_HALF_MM of the midline below row 100 (the side of the neck is dark from coil
        fall-off); kept if > 0.5 ml, near the midline and containing some really dark (< 0.10) voxels
mesh    marching cubes of the solid/void boundary; faces on the box / FOV cut planes are removed (like the slab cut
        faces on the skin mesh), so only real tissue boundary is left
"""
import sys, os
import numpy as np
import scipy.ndimage as ndi
from skimage import measure
import trimesh
import fit_gnm as fg
import air_mesh as am

THR = 0.25
ROW_MIN, COL_MAX = 30, 98
NECK_ROW, NECK_HALF_MM = 100, 22.0
MIN_ML, MAX_ABS_X, SEED = 0.5, 28.0, 0.10


def region(a):
    box = am.inside_mask(a)
    box[:ROW_MIN] = False; box[:, COL_MAX:] = False
    x = (np.arange(a.shape[2]) - (a.shape[2] - 1) / 2) * fg.VOX
    box[NECK_ROW:, :, np.abs(x) > NECK_HALF_MM] = False
    return box


def voids(a, thr=THR):
    s = ndi.gaussian_filter(a, 0.7)
    box = region(a)
    solid = ndi.binary_opening((s > thr) | ~box, iterations=1)
    lab, n = ndi.label(~solid & box)
    idx = np.arange(1, n + 1)
    size = ndi.sum(np.ones_like(lab), lab, idx) * am.VOXML
    seeded = ndi.maximum(s < min(SEED, thr), lab, idx) > 0
    xc = (ndi.mean(np.indices(lab.shape)[2], lab, idx) - (a.shape[2] - 1) / 2) * fg.VOX
    return np.isin(lab, idx[seeded & (size > MIN_ML) & (np.abs(xc) < MAX_ABS_X)]), box


def tissue_mesh(a, thr=THR, faces=40000):
    v_, box = voids(a, thr)
    v, f, _, _ = measure.marching_cubes(ndi.gaussian_filter(np.pad(v_, 2).astype(np.float32), 0.7), 0.5)
    v -= 2
    # faces lying on the region boundary are box cuts, not tissue
    edge = ndi.distance_transform_edt(np.pad(box, 1))[1:-1, 1:-1, 1:-1]
    cut = ndi.map_coordinates(edge, v.T, order=1) < 1.5
    m = trimesh.Trimesh(fg.vox_to_world(v, a.shape[2]), f[~cut[f].any(1)])
    m.remove_unreferenced_vertices()
    if len(m.faces) == 0:
        return m, v_
    trimesh.smoothing.filter_taubin(m, iterations=10)
    if len(m.faces) > faces:
        m = m.simplify_quadric_decimation(face_count=faces)
    return m, v_


if __name__ == '__main__':
    recon_dir, out_dir = sys.argv[1:3]
    for scan in sys.argv[3:]:
        a = fg.load_volume(os.path.join(recon_dir, f'sub001_3d_{scan}_recon.mat'))
        m, v_ = tissue_mesh(a)
        m.export(os.path.join(out_dir, f'{scan}_tissue.stl'))
        print(f'{scan}: enclosed space {v_.sum() * am.VOXML:.0f} ml in {ndi.label(v_)[1]} parts, mesh {len(m.faces)} faces', flush=True)
