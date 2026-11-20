"""Segment the vocal-tract airway (lips -> glottis) from a 3D speech MRI volume.

Works on a 2x-upsampled crop. Cuts:
  lips    - coronal plane through the fitted GNM inner-lip loop (from fit_gnm.py)
  nasal   - airway above the hard-palate level is dropped (per-speaker constant)
  glottis - minimum-area axial slice of the laryngeal airway in a search window
"""
import sys
import numpy as np
import scipy.ndimage as ndi
from skimage import measure, filters, graph
import trimesh
from scipy.spatial import cKDTree
from fit_gnm import load_volume, head_mask, VOX, vox_to_world

# Per-speaker landmarks in original voxel rows/cols (sub001, read off mid-sagittal).
CROP = dict(i=(40, 140), j=(25, 98))      # j upper bound ~ posterior pharyngeal wall + margin
HALF_WIDTH_MM = 30       # lateral corridor around the mid-sagittal plane
PALATE_ROW = 62          # nasopharynx above this row is excluded
GLOTTIS_SEARCH = (108, 132)
UP = 2                   # upsampling factor
LOW, HIGH = 0.12, 0.22   # hysteresis thresholds on normalised magnitude


def gnm_lip_loop(gnm_mesh, groups):
    """World-mm vertices of the open boundary loop joining GNM's mouth sock to the lips."""
    faces = gnm_mesh.faces
    sock = faces[groups['mouth_sock'][faces].all(1)]
    uniq, counts = np.unique(np.sort(sock[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), 1), axis=0, return_counts=True)
    return gnm_mesh.vertices[np.unique(uniq[counts == 1])]


def lip_centre(loop, half_width=8.0):
    """Centre of the inner-lip loop's midline section (corners sit far behind it when the mouth is open)."""
    mid = loop[np.abs(loop[:, 0] - np.median(loop[:, 0])) < half_width]
    return mid.mean(0)


def dense_group_points(mesh, vmask, spacing=0.5):
    """Surface samples (world mm) of the faces whose vertices are all in vmask."""
    sub = trimesh.Trimesh(mesh.vertices, mesh.faces[vmask[mesh.faces].all(1)], process=False)
    n = max(1000, int(sub.area / spacing ** 2))
    return np.vstack([mesh.vertices[vmask], trimesh.sample.sample_surface(sub, n, seed=0)[0]])


def teeth_exclusion_points(gnm_mesh, groups, R, root_mm=12.0, step=1.5):
    """Dense GNM crown samples swept along the tooth axis to cover roots/alveolar bone (dark in MRI)."""
    up = R @ np.array([0.0, 1.0, 0.0])               # GNM model 'up' in world frame
    pts = []
    for grp, sign in (('upper_teeth_and_gums', 1.0), ('lower_teeth_and_gums', -1.0)):
        crown = dense_group_points(gnm_mesh, groups['teeth'] & groups[grp])
        for d in np.arange(0.0, root_mm + 1e-6, step):
            pts.append(crown + sign * d * up)
    return np.vstack(pts)


def segment(a, lip_centre, teeth_pts=None, teeth_radius=2.5):
    lip_z = lip_centre[2]
    i0, i1 = CROP['i']; j0, j1 = CROP['j']
    nz = a.shape[2]
    gi = np.arange(i0, i1, 1 / UP); gj = np.arange(j0, j1, 1 / UP); gk = np.arange(0, nz - 1 + 1e-6, 1 / UP)
    I, J, K = np.meshgrid(gi, gj, gk, indexing='ij')
    s = ndi.map_coordinates(ndi.gaussian_filter(a, 0.5), [I, J, K], order=3, mode='nearest').astype(np.float32)
    inside = ndi.map_coordinates(ndi.binary_erosion(head_mask(a), iterations=2).astype(np.float32),
                                 [I, J, K], order=1) > 0.5
    del I, J, K

    air = filters.apply_hysteresis_threshold(-s, -HIGH, -LOW)
    # geometric cuts
    z = -(gj * VOX)                                   # world z of each column
    air[:, z > lip_z, :] = False                       # anterior to lip plane
    air[gi < PALATE_ROW, :, :] = False                 # nasal cavity / nasopharynx
    air &= inside                                      # never outside the head
    x = (gk - (nz - 1) / 2) * VOX
    air[:, :, np.abs(x) > HALF_WIDTH_MM] = False       # lateral corridor

    allowed = inside.copy()                            # same geometric cuts, without the threshold
    allowed[:, z > lip_z, :] = False
    allowed[gi < PALATE_ROW] = False
    allowed[:, :, np.abs(x) > HALF_WIDTH_MM] = False
    teeth = np.zeros_like(air)
    if teeth_pts is not None:                          # rough dental-arch exclusion from fitted GNM teeth
        ii, jj, kk = np.nonzero(allowed)
        P = vox_to_world(np.stack([gi[ii], gj[jj], gk[kk]], 1), nz)
        d, _ = cKDTree(teeth_pts).query(P, distance_upper_bound=teeth_radius)
        teeth[ii[np.isfinite(d)], jj[np.isfinite(d)], kk[np.isfinite(d)]] = True
        air &= ~teeth

    # seed: darkest voxel in the oropharynx, mid-sagittal
    kc = len(gk) // 2
    win = (slice(int((85 - i0) * UP), int((100 - i0) * UP)), slice(int((78 - j0) * UP), int((95 - j0) * UP)))
    sub = s[win[0], win[1], kc]
    si, sj = np.unravel_index(sub.argmin(), sub.shape)
    seed = (win[0].start + si, win[1].start + sj, kc)

    lab, _ = ndi.label(air)
    if lab[seed] == 0:
        raise RuntimeError(f'pharynx seed {seed} is not in the air mask')
    vt = lab == lab[seed]

    # Minimal-intensity path pharynx -> lips keeps narrow constrictions connected.
    ci = (-lip_centre[1] / VOX - i0) * UP
    cj = (-(lip_z - 2.0) / VOX - j0) * UP              # 2 mm behind the lip plane
    ck = (lip_centre[0] / VOX + (nz - 1) / 2) * UP
    end = tuple(int(round(c)) for c in (ci, cj, ck))
    cost = np.where(allowed, (s + 0.02) ** 3, 1e3)
    path, _ = graph.route_through_array(cost, seed, end, fully_connected=True, geometric=True)
    path = np.array(path)
    path_max = float(s[tuple(path.T)].max())
    tube = np.zeros_like(vt); tube[tuple(path.T)] = True
    tube = ndi.binary_dilation(tube, iterations=1)
    touched = np.unique(lab[tube & air]); touched = touched[touched > 0]
    vt |= (tube & allowed) | np.isin(lab, touched)

    # glottis: minimum airway area per axial slice within the search window
    area = vt.sum((1, 2)).astype(float)
    rows = np.where((gi >= GLOTTIS_SEARCH[0]) & (gi <= GLOTTIS_SEARCH[1]))[0]
    g_row = rows[area[rows].argmin()]
    vt[g_row:] = False
    lab, _ = ndi.label(vt)
    vt = lab == lab[seed]
    vt = ndi.binary_fill_holes(vt)
    allowed[g_row:] = False
    info = dict(allowed=allowed, teeth=teeth, seed=seed, path_max=path_max, glottis_row=gi[g_row], area=area, rows=gi)
    return vt, s, (gi, gj, gk), info


def mask_to_mesh(vt, grids, smooth_iter=15):
    gi, gj, gk = grids
    pad = 3                                            # pad so masks touching the crop edge still close
    v, f, _, _ = measure.marching_cubes(ndi.gaussian_filter(np.pad(vt, pad).astype(np.float32), 0.8), 0.5)
    v -= pad
    ijk = np.stack([gi[0] + v[:, 0] / UP, gj[0] + v[:, 1] / UP, gk[0] + v[:, 2] / UP], 1)
    m = trimesh.Trimesh(vox_to_world(ijk, len(gk) // UP + 1), f)
    trimesh.smoothing.filter_taubin(m, iterations=smooth_iter)
    return m


if __name__ == '__main__':
    vol, fit_prefix, groups_npz, out = sys.argv[1:5]
    a = load_volume(vol)
    gm = trimesh.load(fit_prefix + '_gnm.obj', process=False); G = np.load(groups_npz)
    loop = gnm_lip_loop(gm, G)
    R = np.load(fit_prefix + '_fit.npz')['R']
    vt, s, grids, info = segment(a, lip_centre(loop), teeth_exclusion_points(gm, G, R))
    m = mask_to_mesh(vt, grids)
    vol_ml = vt.sum() * (VOX / UP) ** 3 / 1000
    print(f'path max intensity {info["path_max"]:.2f}  glottis row {info["glottis_row"]:.1f}  volume {vol_ml:.1f} ml  '
          f'mesh verts {len(m.vertices)} watertight {m.is_watertight}')
    m.export(out + '_vt.stl')
    np.save(out + '_liploop.npy', loop)
    np.save(out + '_vtmask.npy', np.packbits(vt))
