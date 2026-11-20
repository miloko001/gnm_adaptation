"""Template GNM teeth placed by a similarity fit to the upper face, lower arch parented to a jaw hinge.

Per scan:
  1. similarity (s, R, t) of the *template* GNM upper face (forehead, nose, cheekbones) to the MRI skin
  2. speaker scale = median s over scans; refit R, t with that scale fixed
  3. jaw: rotation about a lateral hinge axis near the TMJ, fitted to chin + lower-lip skin
Teeth are the unmodified template teeth; upper arch follows the head, lower arch follows the jaw.
"""
import sys, os, glob, json
import numpy as np
from scipy.spatial import cKDTree
import trimesh
import fit_gnm as fg

SLAB_HALF = 50.0   # mm, half-width of the MRI slab (80 x 1.25 mm)


def umeyama(src, dst, with_scale=True):
    ms, md = src.mean(0), dst.mean(0)
    A, B = src - ms, dst - md
    U, S, Vt = np.linalg.svd(B.T @ A / len(src))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (A ** 2).sum(1).mean() if with_scale else 1.0
    return s, R, md - s * R @ ms


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


class Template:
    def __init__(self, g):
        self.g = g
        self.T = g.template_vertex_positions * 1000
        G = self.G = dict(zip(g.vertex_group_names, g.vertex_groups.astype(bool)))
        upper_regions = ['forehead_region', 'nose_region', 'left_orbital_region', 'right_orbital_region',
                         'left_infraorbital_region', 'right_infraorbital_region', 'left_zygomatic_region',
                         'right_zygomatic_region', 'middle_brow_region', 'left_brow_region', 'right_brow_region']
        self.upper = np.where(np.any([G[k] for k in upper_regions], 0) & G['skin_exterior'])[0]
        self.jaw_skin = np.where(G['chin_region'] & G['skin_exterior'])[0]   # chin only: lips move independently of the jaw
        self.lower_teeth = np.where(G['lower_teeth_and_gums'] & (G['teeth'] | G['gums']))[0]
        self.upper_teeth = np.where(G['upper_teeth_and_gums'] & (G['teeth'] | G['gums']))[0]
        # Jaw hinge: lateral axis through a point ~10 mm anterior and ~5 mm below the ear centroid (approx. TMJ).
        ears = self.T[G['ears']]
        self.hinge = np.array([0.0, ears[:, 1].mean() - 5.0, ears[:, 2].mean() + 10.0])
        self.normals = trimesh.Trimesh(self.T, g.triangles, process=False).vertex_normals


def fit_head(tp, pts, nrm, scale=None, n_iter=30):
    tree = cKDTree(pts)
    src = tp.T[tp.upper]
    mid = np.abs(pts[:, 0]) < 10
    s = 1.0 if scale is None else scale
    R = np.eye(3)
    t = pts[mid][pts[mid][:, 2].argmax()] - s * tp.T[tp.upper][tp.T[tp.upper][:, 2].argmax()]
    for it in range(n_iter):
        cut = 12.0 if it < 8 else 5.0
        X = s * src @ R.T + t
        d, nn = tree.query(X)
        inside = (np.abs(X[:, 0]) < SLAB_HALF - 6) & (X[:, 1] < -6)   # within the MRI slab / below the FOV top
        ok = inside & (d < cut) & (np.einsum('ij,ij->i', tp.normals[tp.upper] @ R.T, nrm[nn]) > 0.5)
        s_new, R, t = umeyama(src[ok], pts[nn[ok]], with_scale=scale is None)
        s = s_new if scale is None else scale
        if scale is not None:   # umeyama without scale assumed s=1: redo translation for fixed s
            t = pts[nn[ok]].mean(0) - s * R @ src[ok].mean(0)
    X = s * src @ R.T + t
    d, _ = tree.query(X)
    d = d[(np.abs(X[:, 0]) < SLAB_HALF - 6) & (X[:, 1] < -6)]
    return s, R, t, float(np.median(d[d < 5])), float((d < 5).mean())


def sim2d(A, B):
    ma, mb = A.mean(0), B.mean(0); a, b = A - ma, B - mb
    U, Sg, Vt = np.linalg.svd(b.T @ a / len(A)); D = np.diag([1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt
    return np.trace(np.diag(Sg) @ D) / (a ** 2).sum(1).mean(), R, mb - np.trace(np.diag(Sg) @ D) / (a ** 2).sum(1).mean() * R @ ma


def profile_pose(tp, pts, scale=None, n_iter=40):
    """Head pose from the mid-sagittal upper-face profile (forehead -> subnasale).

    2-D similarity in the (y, z) plane gives scale, pitch and (y, z) translation; yaw = roll = 0 (head
    assumed straight in the scanner); x translation from the nose region's lateral centre.
    """
    T = tp.T
    mid = tp.G['skin_exterior'] & (np.abs(T[:, 0]) < 3)
    nose = T[mid][T[mid][:, 2].argmax()]
    sel = mid & (T[:, 1] < nose[1] + 70) & (T[:, 1] > nose[1] - 12) & (T[:, 2] > nose[2] - 45)
    src = T[sel][:, [1, 2]]
    P = pts[np.abs(pts[:, 0]) < 2.5][:, [1, 2]]
    tree = cKDTree(P)
    s, R2, t2 = 1.0, np.eye(2), P[P[:, 1].argmax()] - nose[[1, 2]]
    for it in range(n_iter):
        X = s * src @ R2.T + t2
        d, nn = tree.query(X); ok = d < (8 if it < 10 else 3)
        s_fit, R2, t2 = sim2d(src[ok], P[nn[ok]])
        if scale is None:
            s = s_fit
        else:
            s = scale; t2 = P[nn[ok]].mean(0) - s * R2 @ src[ok].mean(0)
    d, _ = tree.query(s * src @ R2.T + t2)
    tip = pts[np.argmax(pts[:, 2])]
    x0 = pts[np.linalg.norm(pts - tip, axis=1) < 12][:, 0].mean()
    R = np.eye(3); R[1:, 1:] = R2
    return s, R, np.array([x0 - s * nose[0], t2[0], t2[1]]), float(np.median(d))


def fit_jaw(tp, pts, nrm, s, R, t, n_iter=20):
    """1-DOF hinge rotation (radians, + = opening) about the template TMJ axis, fitted to chin/lower-lip skin."""
    tree = cKDTree(pts)
    src = tp.T[tp.jaw_skin] - tp.hinge
    angles = np.radians(np.arange(-10, 35.5, 0.5))
    def cost(a):
        X = s * ((src @ rot_x(a).T) + tp.hinge) @ R.T + t
        X = X[np.abs(X[:, 0]) < SLAB_HALF - 6]
        d, _ = tree.query(X)
        return np.mean(np.minimum(d, 6.0) ** 2)
    c = [cost(a) for a in angles]
    a0 = angles[int(np.argmin(c))]
    fine = np.radians(np.arange(np.degrees(a0) - 0.5, np.degrees(a0) + 0.55, 0.05))
    cf = [cost(a) for a in fine]
    return float(fine[int(np.argmin(cf))]), float(np.sqrt(min(cf)))


def teeth_world(tp, s, R, t, jaw):
    V = tp.T.copy()
    lo = tp.lower_teeth
    V[lo] = (V[lo] - tp.hinge) @ rot_x(jaw).T + tp.hinge
    return s * V @ R.T + t


if __name__ == '__main__':
    from gnm.shape import gnm_numpy
    recon_dir, out_dir = sys.argv[1:3]
    os.makedirs(out_dir, exist_ok=True)
    g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    tp = Template(g)
    files = sorted(glob.glob(os.path.join(recon_dir, '*_recon.mat')))
    names = [os.path.basename(f).split('_3d_')[1].replace('_recon.mat', '') for f in files]
    # Speaker scale from the mid-sagittal upper-face profile (forehead -> subnasale), median over scans: 0.932 for sub001.
    S = float(sys.argv[3]) if len(sys.argv) > 3 else 0.932
    surf = {n: fg.skin_surface(fg.load_volume(f)) for f, n in zip(files, names)}
    print(f'speaker scale (fixed): {S:.3f}')
    faces = g.triangles[(tp.G['teeth'] | tp.G['gums'])[g.triangles].all(1)]
    res = {}
    for n in names:
        pts, nrm = surf[n]
        _, R, t, med = profile_pose(tp, pts, scale=S); frac = 1.0
        jaw, jrms = fit_jaw(tp, pts, nrm, S, R, t)
        V = teeth_world(tp, S, R, t, jaw)
        m = trimesh.Trimesh(V, faces, process=False); m.remove_unreferenced_vertices()
        m.export(os.path.join(out_dir, f'{n}_teeth.stl'))
        res[n] = dict(upper_face_median_mm=round(med, 2), upper_face_inlier_frac=round(frac, 2),
                      jaw_deg=round(np.degrees(jaw), 1), jaw_rms_mm=round(jrms, 2))
        np.savez(os.path.join(out_dir, f'{n}_pose.npz'), s=S, R=R, t=t, jaw=jaw, hinge=tp.hinge)
        print(f'{n:18s} upper-face median {med:.2f} mm (inliers {frac:.2f})  jaw {np.degrees(jaw):5.1f} deg  chin rms {jrms:.2f} mm', flush=True)
    json.dump(dict(scale=S, scans=res), open(os.path.join(out_dir, 'jaw_teeth.json'), 'w'), indent=1)
