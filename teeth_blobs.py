"""Fit the GNM template teeth (wisdom teeth removed, teeth otherwise unmodified) to tooth crowns detected
as dark 3-D blobs in the MRI.

1. Template teeth -> individual teeth (connected components); drop the rearmost tooth on each side of each arch.
2. Detect dark blobs (3-D Laplacian of Gaussian, tooth-sized) in a band around the profile-posed template arch.
3. Rigid (optionally + scale) trimmed ICP between template crown centroids and blob centres.
"""
import sys, os, json
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from skimage.feature import blob_log
import trimesh
import fit_gnm as fg
import jaw_teeth as jt

UP = 2   # upsampling factor for blob detection


def template_teeth(tp):
    """Per-tooth vertex index arrays for each arch, wisdom teeth removed. Returns {arch: [idx, ...]}."""
    F = tp.g.triangles
    out = {}
    for arch, grp in (('upper', 'upper_teeth_and_gums'), ('lower', 'lower_teeth_and_gums')):
        idx = np.where(tp.G['teeth'] & tp.G[grp])[0]
        m = trimesh.Trimesh(tp.T, F[np.isin(F, idx).all(1)], process=False)
        comps = [np.array(c) for c in trimesh.graph.connected_components(m.edges, nodes=np.unique(m.faces)) if len(c) >= 20]
        cen = np.array([tp.T[c].mean(0) for c in comps])
        keep = np.ones(len(comps), bool)
        for side in (cen[:, 0] < 0, cen[:, 0] >= 0):        # rearmost (min z) tooth on each side = third molar
            ii = np.where(side)[0]
            keep[ii[np.argmin(cen[ii, 2])]] = False
        out[arch] = [c for c, k in zip(comps, keep) if k]
    return out


def detect_blobs(a, roi_pts, margin_mm=8.0, r_mm=(2.0, 4.5), thr=0.06):
    """Dark tooth-sized blobs (world mm centres, radius mm) near roi_pts (world mm)."""
    nz = a.shape[2]
    head = ndi.binary_erosion(fg.head_mask(a), iterations=2)
    ijk = np.stack([-roi_pts[:, 1] / 1.25, -roi_pts[:, 2] / 1.25, roi_pts[:, 0] / 1.25 + (nz - 1) / 2], 1)
    lo = np.maximum(np.floor(ijk.min(0) - margin_mm / 1.25).astype(int), 0)
    hi = np.minimum(np.ceil(ijk.max(0) + margin_mm / 1.25).astype(int) + 1, a.shape)
    sub = ndi.zoom(a[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], UP, order=3)
    hsub = ndi.zoom(head[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].astype(np.float32), UP, order=0) > 0.5
    inv = np.where(hsub, np.clip(1 - sub / 0.7, 0, 1), 0).astype(np.float32)
    sig = np.array(r_mm) / 1.25 * UP / np.sqrt(3)
    b = blob_log(inv, min_sigma=sig[0], max_sigma=sig[1], num_sigma=8, threshold=thr, overlap=0.5)
    c = b[:, :3] / UP + lo
    P = np.stack([(c[:, 2] - (nz - 1) / 2) * 1.25, -c[:, 0] * 1.25, -c[:, 1] * 1.25], 1)
    rad = b[:, 3] * np.sqrt(3) / UP * 1.25
    near = cKDTree(roi_pts).query(P)[0] < margin_mm
    val = ndi.map_coordinates(ndi.gaussian_filter(a, 0.7), c.T, order=1)
    keep = near & (val < 0.2) & (rad >= 2.5)          # strong, tooth-sized voids only (weak/small = clutter)
    return P[keep], rad[keep], val[keep]


def fitted_teeth_mesh(tp, z):
    """Wisdom-free template teeth placed by a saved fit (profile pose, then per-arch rigid transform)."""
    teeth = template_teeth(tp)
    F = tp.g.triangles
    V = float(z['s0']) * tp.T @ z['R0'].T + z['t0']
    out = V.copy(); faces = []
    for k, comps in teeth.items():
        idx = np.concatenate(comps)
        out[idx] = float(z[f'{k}_s']) * V[idx] @ z[f'{k}_R'].T + z[f'{k}_t']
        faces.append(F[np.isin(F, idx).all(1)])
    m = trimesh.Trimesh(out, np.vstack(faces), process=False); m.remove_unreferenced_vertices()
    return m


def similarity(src, dst, with_scale):
    return jt.umeyama(src, dst, with_scale=with_scale)


def coarse_search(src, dst, dx=(-4, 4), dy=(-4, 18), dz=(-10, 8), pitch=(-10, 10), tol=3.0):
    """Grid search translation (+pitch about the src centroid) maximising template teeth with a blob within tol mm."""
    tree = cKDTree(dst); c = src.mean(0)
    best = (-1, np.inf, None)
    for p in np.arange(pitch[0], pitch[1] + 0.1, 2.5):
        Rp = jt.rot_x(np.radians(p))
        S = (src - c) @ Rp.T + c
        for x in np.arange(dx[0], dx[1] + 0.1, 1.0):
            for y in np.arange(dy[0], dy[1] + 0.1, 1.0):
                for z in np.arange(dz[0], dz[1] + 0.1, 1.0):
                    d, _ = tree.query(S + [x, y, z])
                    n = int((d < tol).sum()); m = float(np.mean(np.minimum(d, tol)))
                    if n > best[0] or (n == best[0] and m < best[1]):
                        best = (n, m, (Rp, c - Rp @ c + np.array([x, y, z])))
    return best


def icp(src, dst, with_scale=False, n_iter=40, cut=(6.0, 3.5), init=None):
    """Trimmed ICP src -> dst (both world mm). Returns s, R, t, matched mask, distances."""
    tree = cKDTree(dst)
    s, R, t = 1.0, np.eye(3), np.zeros(3)
    if init is not None:
        R, t = init
    for it in range(n_iter):
        X = s * src @ R.T + t
        d, nn = tree.query(X)
        c = cut[0] if it < n_iter // 2 else cut[1]
        ok = d < c
        # one-to-one: keep the closest template tooth per blob
        order = np.argsort(d)
        used, ok2 = set(), np.zeros_like(ok)
        for i in order:
            if ok[i] and nn[i] not in used:
                ok2[i] = True; used.add(nn[i])
        if ok2.sum() < 4:
            break
        ds, dR, dt = similarity(X[ok2], dst[nn[ok2]], with_scale)
        s, R, t = ds * s, dR @ R, ds * dR @ t + dt
    X = s * src @ R.T + t
    d, nn = tree.query(X)
    return s, R, t, d < cut[1], d


if __name__ == '__main__':
    from gnm.shape import gnm_numpy
    recon_dir, out_dir, scan = sys.argv[1:4]
    with_scale = len(sys.argv) > 4 and sys.argv[4] == 'scale'
    g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    tp = jt.Template(g)
    teeth = template_teeth(tp)
    a = fg.load_volume(os.path.join(recon_dir, f'sub001_3d_{scan}_recon.mat'))
    pts, _ = fg.skin_surface(a)
    s0, R0, t0, _ = jt.profile_pose(tp, pts, scale=0.932)
    place = lambda P: s0 * P @ R0.T + t0
    cents = {k: np.array([place(tp.T[c]).mean(0) for c in v]) for k, v in teeth.items()}
    allv = place(tp.T[np.concatenate([np.concatenate(v) for v in teeth.values()])])
    B, rad, val = detect_blobs(a, allv, margin_mm=10.0)
    print(f'{scan}: {sum(len(v) for v in teeth.values())} template teeth (wisdom removed), {len(B)} candidate blobs')
    src = np.vstack([cents['upper'], cents['lower']])
    n0, m0, init = coarse_search(src, B)
    print(f'  coarse search: {n0}/{len(src)} teeth within 3 mm of a blob')
    s, R, t, ok, d = icp(src, B, with_scale=with_scale, init=init, cut=(4.0, 3.0))
    print(f'  ICP ({"rigid+scale" if with_scale else "rigid"}): matched {ok.sum()}/{len(src)} teeth, '
          f'median match {np.median(d[ok]):.2f} mm, extra scale {s:.3f}, centroid shift {np.round(s * src.mean(0) @ R.T + t - src.mean(0), 1)} mm, '
          f'rotation {np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))):.1f} deg')
    # per-arch refinement from the joint fit (template bite need not match the speaker's)
    arch = {}
    for k in ('upper', 'lower'):
        sa, Ra, ta, oka, da = icp(cents[k], B, with_scale=with_scale, init=(R, t), cut=(3.5, 3.0))
        arch[k] = (sa, Ra, ta)
        print(f'  {k} arch: matched {oka.sum()}/{len(cents[k])}, median {np.median(da[oka]):.2f} mm, '
              f'shift vs joint {np.round(sa * cents[k].mean(0) @ Ra.T + ta - (s * cents[k].mean(0) @ R.T + t), 1)} mm')
    np.savez(os.path.join(out_dir, f'{scan}_teeth_blobs{"_scale" if with_scale else ""}.npz'),
             s0=s0, R0=R0, t0=t0, s=s, R=R, t=t, blobs=B, blob_r=rad, matched=ok, dist=d,
             **{f'{k}_{n}': v for k in arch for n, v in zip(('s', 'R', 't'), arch[k])})


def head_teeth_model(g, tp, fit):
    """Wisdom-free teeth in GNM model mm under a head fit (identity/expression applied), per arch."""
    V = (tp.T + np.tensordot(fit['id'], g.vertex_identity_basis[fg.ID_HEAD] * 1000, 1)
         + np.tensordot(fit['ex'], g.expression_basis[fg.EX_LOWER] * 1000, 1))
    return V, template_teeth(tp)


def search_arch(cent_model, B, s, R, t, dy=(-6, 16), dz=(-3, 3), dx=(-2, 2), pitch=(-10, 10), tol=3.0):
    """Grid search (model frame, about the arch centroid): pitch + translation maximising teeth matched to blobs."""
    tree = cKDTree(B); c0 = cent_model.mean(0)
    best = (-1, np.inf, None)
    for p in np.arange(pitch[0], pitch[1] + 0.1, 2.0):
        Rp = jt.rot_x(np.radians(p))
        S0 = (cent_model - c0) @ Rp.T + c0
        for x in np.arange(dx[0], dx[1] + 0.1, 1.0):
            for y in np.arange(dy[0], dy[1] + 0.1, 0.5):
                for z in np.arange(dz[0], dz[1] + 0.1, 0.5):
                    W = s * (S0 + [x, y, z]) @ R.T + t
                    d, _ = tree.query(W)
                    n = int((d < tol).sum()); m = float(np.mean(np.minimum(d, tol)))
                    if n > best[0] or (n == best[0] and m < best[1]):
                        best = (n, m, dict(pitch=p, tau=np.array([x, y, z], float), c0=c0))
    return best


def arch_world(V_model, idx, A, s, R, t):
    Rp = jt.rot_x(np.radians(A['pitch']))
    return s * ((V_model[idx] - A['c0']) @ Rp.T + A['c0'] + A['tau']) @ R.T + t
