"""Fit the GNM head model to the skin surface of a 3D speech MRI volume.

World frame used throughout (millimetres), chosen to match GNM's axes:
  x = dim2 (left-right, slab centre at 0; handedness unverified)
  y = up      = -dim0 * 1.25
  z = forward = -dim1 * 1.25
GNM is linear at zero joint pose: v = T + Bi @ id + Be @ ex, so each shape
update is a closed-form ridge solve.
"""
import sys
import numpy as np
import scipy.io as sio
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure
import trimesh

VOX = 1.25
ID_HEAD = np.arange(0, 170)          # head identity components
EX_LOWER = np.arange(200, 350)       # lower-face expression components


def load_volume(path):
    a = np.abs(sio.loadmat(path)['recon']).astype(np.float32)
    return a / np.percentile(a, 99)


def vox_to_world(ijk, nz):
    i, j, k = ijk[:, 0], ijk[:, 1], ijk[:, 2]
    return np.stack([(k - (nz - 1) / 2) * VOX, -i * VOX, -j * VOX], 1)


def head_mask(a, thr=0.12):
    """Solid head mask (airways filled) in original voxel grid."""
    s = ndi.gaussian_filter(a, 1.0)
    m = s > thr
    lab, _ = ndi.label(m)
    m = lab == np.bincount(lab.ravel())[1:].argmax() + 1
    m = ndi.binary_closing(m, ndi.generate_binary_structure(3, 1), iterations=3)
    for k in range(m.shape[2]):                      # close airways per sagittal slice
        m[:, :, k] = ndi.binary_fill_holes(m[:, :, k])
    for i in range(m.shape[0]):                      # and per axial slice
        m[i] = ndi.binary_fill_holes(m[i])
    return m


def skin_surface(a, thr=0.12, edge=3):
    """Outer skin surface as (points, normals) in world mm, slab-edge points dropped."""
    m = head_mask(a, thr)
    v, f, n, _ = measure.marching_cubes(ndi.gaussian_filter(m.astype(np.float32), 1.0), 0.5)
    sh = np.array(m.shape)
    keep = np.all((v > edge) & (v < sh - 1 - edge), 1)
    pts = vox_to_world(v[keep], sh[2])
    nrm = np.stack([n[keep, 2], -n[keep, 0], -n[keep, 1]], 1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    if np.mean(np.einsum('ij,ij->i', nrm, pts - pts.mean(0))) < 0:  # make normals point outward
        nrm = -nrm
    return pts, nrm


def kabsch(src, dst):
    cs, cd = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - cs).T @ (dst - cd))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cd - R @ cs


def fit(g, pts, nrm, n_outer=15, verbose=True, fixed_id=None, lam_final=5.0):
    """Return dict with id, ex, R, t (model->world) and fitted world vertices (mm)."""
    T = g.template_vertex_positions * 1000
    Bi = g.vertex_identity_basis[ID_HEAD] * 1000
    Be = g.expression_basis[EX_LOWER] * 1000
    G = dict(zip(g.vertex_group_names, g.vertex_groups.astype(bool)))
    sel = G['skin_exterior'] & ~G['ears'] & (T[:, 2] > T[:, 2].max() - 110)
    vidx = np.where(sel)[0]
    tree = cKDTree(pts)

    idp = np.zeros(len(ID_HEAD)) if fixed_id is None else fixed_id.copy()
    ex = np.zeros(len(EX_LOWER))
    # Initialise by matching nose tips (most anterior point near the midline).
    mid = np.abs(pts[:, 0]) < 10
    nose_w = pts[mid][pts[mid][:, 2].argmax()]
    nose_m = T[sel][T[sel][:, 2].argmax()]
    R, t = np.eye(3), nose_w - nose_m

    for it in range(n_outer):
        cutoff = 15.0 if it < 4 else 6.0
        lam = 6 * lam_final if it < 5 else lam_final
        V = T + np.tensordot(idp, Bi, 1) + np.tensordot(ex, Be, 1)
        Vw = V[vidx] @ R.T + t
        d, nn = tree.query(Vw)
        Vn = trimesh.Trimesh(V @ R.T + t, g.triangles, process=False).vertex_normals[vidx]
        ok = (d < cutoff) & (np.einsum('ij,ij->i', Vn, nrm[nn]) > 0.5)
        # rigid update (point-to-point Kabsch on inliers)
        R, t = kabsch(V[vidx][ok], pts[nn[ok]])
        # shape update in model frame: point-to-plane + weak point-to-point
        q = (pts[nn[ok]] - t) @ R            # targets in model frame
        nq = nrm[nn[ok]] @ R
        vi = vidx[ok]
        r0 = q - T[vi]
        if fixed_id is not None:              # identity is held fixed, so it is part of the base shape
            r0 = r0 - np.tensordot(idp, Bi[:, vi], 1)
        cols = [Be[:, vi]] if fixed_id is not None else [Bi[:, vi], Be[:, vi]]
        B = np.concatenate(cols, 0)          # (P, Nv, 3)
        A_pl = np.einsum('pvc,vc->vp', B, nq)
        b_pl = np.einsum('vc,vc->v', r0, nq)
        A_pt = 0.1 * B.transpose(1, 2, 0).reshape(-1, B.shape[0])
        b_pt = 0.1 * r0.reshape(-1)
        A = np.vstack([A_pl, A_pt]); b = np.concatenate([b_pl, b_pt])
        p = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ b)
        if fixed_id is None:
            idp, ex = p[:len(ID_HEAD)], p[len(ID_HEAD):]
        else:
            ex = p
        if verbose:
            print(f'it {it:2d} inliers {ok.sum():5d} median {np.median(d[ok]):.2f} mm  p90 {np.percentile(d[ok], 90):.2f} mm')

    V = T + np.tensordot(idp, Bi, 1) + np.tensordot(ex, Be, 1)
    Vw = V @ R.T + t
    d, _ = tree.query(Vw[vidx])
    return dict(id=idp, ex=ex, R=R, t=t, verts=Vw, faces=g.triangles, groups=G,
                face_dist=d, face_vidx=vidx)


if __name__ == '__main__':
    from gnm.shape import gnm_numpy
    path = sys.argv[1]
    out = sys.argv[2]
    a = load_volume(path)
    pts, nrm = skin_surface(a)
    print('skin points', len(pts))
    g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    res = fit(g, pts, nrm)
    np.savez(out + '_fit.npz', id=res['id'], ex=res['ex'], R=res['R'], t=res['t'])
    trimesh.Trimesh(res['verts'], res['faces'], process=False).export(out + '_gnm.obj')
    np.save(out + '_skinpts.npy', pts.astype(np.float32))
