"""Fit the GNM head to a skin-surface mesh extracted from the 3-D MRI volume.

MRI mesh: marching cubes on the solid head mask; faces on the slab / field-of-view cut planes are removed,
so the mesh only contains real skin. Fit: scale fixed from the mid-sagittal profile (0.932 for sub001);
rotation/translation + GNM head identity (170) + lower-face expression (150), ridge-regularised, with
*bidirectional* point-to-plane correspondences:
    A  GNM skin vertices inside the scanned region -> MRI surface
    B  MRI surface points -> GNM skin (this is what pulls in parts of GNM that stick out, e.g. the jaw)
"""
import sys, os, json
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure
import trimesh
import fit_gnm as fg
import jaw_teeth as jt

VOX = 1.25


def mri_head_mesh(a, faces=40000, edge_vox=2.0):
    m = fg.head_mask(a)
    v, f, _, _ = measure.marching_cubes(ndi.gaussian_filter(np.pad(m, 2).astype(np.float32), 1.0), 0.5)
    v -= 2
    sh = np.array(m.shape)
    on_cut = np.any((v < edge_vox) | (v > sh - 1 - edge_vox), 1)          # slab / FOV boundary
    keep = ~on_cut[f].any(1)
    mesh = trimesh.Trimesh(fg.vox_to_world(v, sh[2]), f[keep])
    mesh.remove_unreferenced_vertices()
    if len(mesh.faces) > faces:
        mesh = mesh.simplify_quadric_decimation(face_count=faces)
    mesh.fix_normals()
    if np.mean(np.einsum('ij,ij->i', mesh.vertex_normals, mesh.vertices - mesh.vertices.mean(0))) < 0:
        mesh.invert()                                             # open mesh: make normals point out of the head
    return mesh


def in_scan(P, nk=80, ni=160, nj=160, margin=3.0):
    """World points inside the scanned box (slab +-50 mm, FOV 200 mm), with a margin."""
    half = (nk - 1) / 2 * VOX
    return ((np.abs(P[:, 0]) < half - margin) & (P[:, 1] < -margin) & (P[:, 1] > -(ni - 1) * VOX + margin)
            & (P[:, 2] < -margin) & (P[:, 2] > -(nj - 1) * VOX + margin))


def fit_head_mesh(g, mesh, s, R, t, lam=0.15, n_iter=25, verbose=True):
    T = g.template_vertex_positions * 1000
    Bi = g.vertex_identity_basis[fg.ID_HEAD] * 1000
    Be = g.expression_basis[fg.EX_LOWER] * 1000
    B = np.concatenate([Bi, Be], 0)                              # (P, V, 3)
    G = dict(zip(g.vertex_group_names, g.vertex_groups.astype(bool)))
    skin = np.where(G['skin_exterior'])[0]
    skin_faces = g.triangles[G['skin_exterior'][g.triangles].all(1)]
    Q = mesh.vertices; NQ = mesh.vertex_normals
    qtree = cKDTree(Q)
    p = np.zeros(B.shape[0])
    for it in range(n_iter):
        cut = 10.0 if it < 6 else 5.0
        V = T + np.tensordot(p, B, 1)
        W = s * V @ R.T + t
        gn = trimesh.Trimesh(W, skin_faces, process=False).vertex_normals
        vis = skin[in_scan(W[skin])]
        # A: GNM -> MRI
        dA, nA = qtree.query(W[vis])
        okA = (dA < cut) & (np.einsum('ij,ij->i', gn[vis], NQ[nA]) > 0.3)
        ia, qa, na = vis[okA], Q[nA[okA]], NQ[nA[okA]]
        # B: MRI -> GNM (nearest visible GNM vertex)
        gtree = cKDTree(W[vis])
        dB, nB = gtree.query(Q)
        okB = (dB < cut) & (np.einsum('ij,ij->i', gn[vis][nB], NQ) > 0.3)
        ib, qb, nb = vis[nB[okB]], Q[okB], gn[vis][nB[okB]]
        # rigid update (fixed scale) from A-pairs
        _, R_new, _ = jt.umeyama(V[ia], qa, with_scale=False)
        R = R_new; t = qa.mean(0) - s * R @ V[ia].mean(0)
        W = s * V @ R.T + t
        # linear solve for p: residual n.(s R (T + B p) + t - q)
        rows, rhs = [], []
        for idx, q, n in ((ia, qa, na), (ib, qb, nb)):
            nm = n @ R                                           # normals in model frame
            A_ = s * np.einsum('pvc,vc->vp', B[:, idx], nm)
            b_ = np.einsum('vc,vc->v', n, q - t) - s * np.einsum('vc,vc->v', nm, T[idx])
            w = 1.0 / np.sqrt(len(idx))
            rows.append(A_ * w); rhs.append(b_ * w)
        A = np.vstack(rows); b = np.concatenate(rhs)
        p = np.linalg.solve(A.T @ A + lam * np.eye(len(p)), A.T @ b)
        if verbose and (it % 5 == 0 or it == n_iter - 1):
            print(f'  it {it:2d}  A: {okA.sum():5d} pts median {np.median(dA[okA]):.2f} mm | '
                  f'B: {okB.sum():5d} pts median {np.median(dB[okB]):.2f} mm | coef rms {np.sqrt((p ** 2).mean()):.2f} max {np.abs(p).max():.1f}')
    V = T + np.tensordot(p, B, 1)
    return dict(id=p[:len(fg.ID_HEAD)], ex=p[len(fg.ID_HEAD):], s=s, R=R, t=t, verts=s * V @ R.T + t)


if __name__ == '__main__':
    from gnm.shape import gnm_numpy
    recon_dir, out_dir, scan = sys.argv[1:4]
    lam = float(sys.argv[4]) if len(sys.argv) > 4 else 0.15
    g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    tp = jt.Template(g)
    a = fg.load_volume(os.path.join(recon_dir, f'sub001_3d_{scan}_recon.mat'))
    mesh = mri_head_mesh(a)
    pts, _ = fg.skin_surface(a)
    s, R, t, _ = jt.profile_pose(tp, pts, scale=0.932)
    print(f'{scan}: MRI head mesh {len(mesh.vertices)} verts, lam {lam}')
    r = fit_head_mesh(g, mesh, s, R, t, lam=lam)
    os.makedirs(out_dir, exist_ok=True)
    mesh.export(os.path.join(out_dir, f'{scan}_mri_head.stl'))
    np.savez(os.path.join(out_dir, f'{scan}_head_fit.npz'), id=r['id'], ex=r['ex'], s=r['s'], R=r['R'], t=r['t'])
    trimesh.Trimesh(r['verts'], g.triangles, process=False).export(os.path.join(out_dir, f'{scan}_gnm_head.obj'))
