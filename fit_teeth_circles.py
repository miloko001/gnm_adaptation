"""Rigid per-arch fit of the unmodified GNM template teeth to the tooth cross-sections in the MRI.

Teeth are signal voids: in axial slices each crown is a dark, roughly circular blob separated from its
neighbours by brighter tissue. Objective (per arch, 6-DOF rigid about the arch centroid, model scale fixed):
    points inside each crown  -> dark   (residual  s)
    ring just outside crowns  -> bright (residual  s_ref - s)
A coarse search over vertical / anterior-posterior offsets precedes Gauss-Newton refinement.
"""
import sys, os, json
import numpy as np
import scipy.ndimage as ndi
import trimesh
import fit_gnm as fg
import fit_teeth as ft
import jaw_teeth as jt


def tooth_samples(tp, arch_idx, shell_mm=1.5):
    """Per-tooth interior and outer-ring samples (template mm, before scaling), plus tooth labels."""
    T = tp.T
    F = tp.g.triangles
    m = trimesh.Trimesh(T, F[np.isin(F, arch_idx).all(1)], process=False)
    comps = trimesh.graph.connected_components(m.edges, nodes=np.unique(m.faces))
    inner, outer, lab = [], [], []
    for ci, c in enumerate(comps):
        v = T[c]
        if len(v) < 20:
            continue
        cen = v.mean(0)
        d = v - cen; n = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
        for a in (0.25, 0.55):
            inner.append(cen + a * d)
        outer.append(v + shell_mm * n)
        lab += [ci] * len(v)
    return np.vstack(inner), np.vstack(outer), len(comps)


class ArchFitter:
    def __init__(self, tp, a, s, R, t):
        self.tp, self.s, self.R, self.t = tp, s, R, t
        self.head = ndi.binary_erosion(fg.head_mask(a), iterations=2)
        self.a = np.where(self.head, a, 1.0)          # outside air must not look like teeth
        self.ref = float(np.percentile(a[self.head], 70))

    def world(self, P, w, tau, c0):
        return self.s * (((P - c0) @ ft.rotvec(w).T + c0 + tau) @ self.R.T) + self.t

    def cost(self, samp, inner, outer, w, tau, c0):
        vi, _ = samp(self.world(inner, w, tau, c0))
        vo, _ = samp(self.world(outer, w, tau, c0))
        return np.mean(vi ** 2) + np.mean(np.clip(self.ref - vo, 0, None) ** 2)

    def fit(self, inner, outer, c0, search=True, n_iter=30, lam_w=200.0, lam_t=0.05,
            dy=(-6, 16), dz=(-10, 10), pitch=(-10, 10), w0=None, tau0=None):
        """Rigid fit of one sample set (an arch, or both arches together). Offsets in model mm / degrees."""
        w = np.zeros(3) if w0 is None else np.array(w0, float)
        tau = np.zeros(3) if tau0 is None else np.array(tau0, float)
        samp = ft.Sampler(self.a, 1.2)
        if search:
            best = (np.inf, None, None)
            for p in np.arange(pitch[0], pitch[1] + 0.1, 2.5):
                wp = w + np.array([np.radians(p), 0, 0])
                for y in np.arange(dy[0], dy[1] + 0.1, 1.0):
                    for zz in np.arange(dz[0], dz[1] + 0.1, 1.0):
                        tp_ = tau + np.array([0, y, zz])
                        c = self.cost(samp, inner, outer, wp, tp_, c0)
                        if c < best[0]:
                            best = (c, wp, tp_)
            _, w, tau = best
        hist = [self.cost(samp, inner, outer, w, tau, c0)]
        for it in range(n_iter):
            if it == n_iter // 2:
                samp = ft.Sampler(self.a, 0.7)
            rows, res = [], []
            for P, sign, ref in ((inner, 1.0, 0.0), (outer, -1.0, self.ref)):
                X = self.world(P, w, tau, c0)
                val, gw = samp(X)
                gm = self.s * (gw @ self.R)
                vm = (P - c0) @ ft.rotvec(w).T
                Jw = -np.einsum('nc,ncd->nd', gm, ft.skew(vm))
                J = np.hstack([Jw, gm]) * sign
                r = val if sign > 0 else np.clip(ref - val, 0, None)
                if sign < 0:
                    J = J * (r > 0)[:, None]
                rows.append(J / np.sqrt(len(P))); res.append(r / np.sqrt(len(P)))
            J = np.vstack(rows); r = np.concatenate(res)
            lam = np.r_[np.full(3, lam_w), np.full(3, lam_t)] * 1e-3
            th0 = np.r_[w, tau] if w0 is None else np.r_[w - w0, tau - tau0]
            step = -np.linalg.solve(J.T @ J + np.diag(lam), J.T @ r + lam * th0)
            w, tau = w + step[:3], tau + step[3:]
            hist.append(self.cost(samp, inner, outer, w, tau, c0))
        return w, tau, hist


ARCH_GROUPS = {'upper': 'upper_teeth_and_gums', 'lower': 'lower_teeth_and_gums'}


def arch_idx(tp, name):
    return np.where(tp.G['teeth'] & tp.G[ARCH_GROUPS[name]])[0]


def fit_clench(tp, a, s, R, t):
    """Both arches as one rigid body (template occlusion is correct when the teeth are clenched)."""
    fitter = ArchFitter(tp, a, s, R, t)
    si = [tooth_samples(tp, arch_idx(tp, k)) for k in ('upper', 'lower')]
    inner = np.vstack([x[0] for x in si]); outer = np.vstack([x[1] for x in si])
    c0 = tp.T[np.r_[arch_idx(tp, 'upper'), arch_idx(tp, 'lower')]].mean(0)
    c_init = fitter.cost(ft.Sampler(fitter.a, 0.7), inner, outer, np.zeros(3), np.zeros(3), c0)
    w, tau, hist = fitter.fit(inner, outer, c0)
    A = dict(w=w, tau=tau, c0=c0, cost_init=float(c_init), cost_final=float(hist[-1]), n_teeth=sum(x[2] for x in si))
    return {'upper': dict(A), 'lower': dict(A)}


def fit_other(tp, a, s, R, t, clench_arches):
    """Upper arch: clench result, small refinement. Lower arch: searched from the clench pose (jaw motion)."""
    fitter = ArchFitter(tp, a, s, R, t)
    out = {}
    for name, search, dy, dz, pitch in (('upper', True, (-3, 3), (-3, 3), (-2.5, 2.5)),
                                        ('lower', True, (-14, 3), (-8, 5), (-5, 25))):
        inner, outer, n = tooth_samples(tp, arch_idx(tp, name))
        A = clench_arches[name]
        c_init = fitter.cost(ft.Sampler(fitter.a, 0.7), inner, outer, A['w'], A['tau'], A['c0'])
        w, tau, hist = fitter.fit(inner, outer, A['c0'], search=search, dy=dy, dz=dz, pitch=pitch,
                                  w0=A['w'], tau0=A['tau'])
        out[name] = dict(w=w, tau=tau, c0=A['c0'], n_teeth=n, cost_init=float(c_init), cost_final=float(hist[-1]))
    return out


def teeth_world_rigid(tp, s, R, t, arches):
    """Template teeth+gums with each arch moved rigidly (model frame), then scaled/posed into the MRI."""
    V = tp.T.copy()
    for name, grp in (('upper', 'upper_teeth_and_gums'), ('lower', 'lower_teeth_and_gums')):
        idx = np.where(tp.G[grp] & (tp.G['teeth'] | tp.G['gums']))[0]
        A = arches[name]
        V[idx] = (V[idx] - A['c0']) @ ft.rotvec(A['w']).T + A['c0'] + A['tau']
    return s * V @ R.T + t


if __name__ == '__main__':
    from gnm.shape import gnm_numpy
    recon_dir, out_dir = sys.argv[1], sys.argv[2]
    scans = sys.argv[3:]
    g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    tp = jt.Template(g)
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    clench_arches = None
    for scan in scans:
        a = fg.load_volume(os.path.join(recon_dir, f'sub001_3d_{scan}_recon.mat'))
        pts, _ = fg.skin_surface(a)
        s, R, t, med = jt.profile_pose(tp, pts, scale=0.932)
        if 'clench' in scan:
            arches = fit_clench(tp, a, s, R, t); clench_arches = arches
        else:
            arches = fit_other(tp, a, s, R, t, clench_arches)
        np.savez(os.path.join(out_dir, f'{scan}_teeth_rigid.npz'), s=s, R=R, t=t,
                 **{f'{k}_{f}': arches[k][f] for k in arches for f in ('w', 'tau', 'c0')})
        summary[scan] = {k: dict(shift_mm=np.round(arches[k]['tau'] * s, 1).tolist(),
                                 rot_deg=round(float(np.degrees(np.linalg.norm(arches[k]['w']))), 1),
                                 cost=f"{arches[k]['cost_init']:.3f}->{arches[k]['cost_final']:.3f}",
                                 n_teeth=arches[k]['n_teeth']) for k in arches}
        print(scan, json.dumps(summary[scan]), flush=True)
    json.dump(summary, open(os.path.join(out_dir, 'teeth_rigid_summary.json'), 'w'), indent=1)
