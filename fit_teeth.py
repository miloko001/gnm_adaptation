"""Fit GNM teeth (80 identity coefs + rigid offsets for upper/lower arch) to the dark dental arch.

Teeth are signal voids in MRI, so the objective is the (smoothed) image intensity
sampled at the teeth vertices: Gauss-Newton pulls them into dark voxels. Best run
on a clenched-teeth scan, where little air touches the teeth.

Rigid offsets are expressed in GNM model space (about each arch's centroid), so the
upper-arch result transfers to other scans of the speaker via their face-fit (R, t).
"""
import sys
import numpy as np
import scipy.ndimage as ndi
from fit_gnm import load_volume, head_mask, VOX, ID_HEAD, EX_LOWER

ID_TEETH = np.arange(173, 253)


def skew(v):
    """Batched cross-product matrices, (N,3) -> (N,3,3)."""
    z = np.zeros(len(v))
    return np.stack([np.stack([z, -v[:, 2], v[:, 1]], 1),
                     np.stack([v[:, 2], z, -v[:, 0]], 1),
                     np.stack([-v[:, 1], v[:, 0], z], 1)], 1)


def rotvec(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th; K = skew(k[None])[0]
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


class Sampler:
    """Trilinear samples + world-frame gradients of a smoothed volume, world mm in."""
    def __init__(self, a, sigma):
        self.s = ndi.gaussian_filter(a, sigma)
        self.g = np.gradient(self.s)
        self.nz = a.shape[2]

    def __call__(self, P):
        c = [-P[:, 1] / VOX, -P[:, 2] / VOX, P[:, 0] / VOX + (self.nz - 1) / 2]
        val = ndi.map_coordinates(self.s, c, order=1, mode='nearest')
        gi, gj, gk = (ndi.map_coordinates(g, c, order=1, mode='nearest') for g in self.g)
        return val, np.stack([gk / VOX, -gi / VOX, -gj / VOX], 1)


def teeth_vertices(g, fit, coefs, arches):
    """World-mm teeth vertices for each arch given teeth coefs and per-arch (w, tau, c0)."""
    T = g.template_vertex_positions * 1000
    Bi = g.vertex_identity_basis * 1000
    V = (T + np.tensordot(fit['id'], Bi[ID_HEAD], 1)
         + np.tensordot(fit['ex'], g.expression_basis[EX_LOWER] * 1000, 1)
         + np.tensordot(coefs, Bi[ID_TEETH], 1))
    out = {}
    for name, (idx, w, tau, c0) in arches.items():
        vm = (V[idx] - c0) @ rotvec(w).T + c0 + tau
        out[name] = vm @ fit['R'].T + fit['t']
    return out


def fit_teeth(g, a, fit, n_iter=40, lam_c=10.0, lam_w=1e4, lam_t=5.0, verbose=True):
    G = dict(zip(g.vertex_group_names, g.vertex_groups.astype(bool)))
    a = np.where(ndi.binary_erosion(head_mask(a), iterations=2), a, 1.0)   # outside air must not attract teeth
    Bi = g.vertex_identity_basis[ID_TEETH] * 1000          # (80, V, 3), model frame
    T0 = g.template_vertex_positions * 1000
    R = fit['R']
    idx = {'upper': np.where(G['teeth'] & G['upper_teeth_and_gums'])[0],
           'lower': np.where(G['teeth'] & G['lower_teeth_and_gums'])[0]}
    c = np.zeros(len(ID_TEETH))
    st = {k: [np.zeros(3), np.zeros(3), T0[v].mean(0)] for k, v in idx.items()}   # w, tau, c0
    hist = []
    for it in range(n_iter):
        if it in (0, 15, 30):
            samp = Sampler(a, {0: 2.0, 15: 1.2, 30: 0.8}[it])
        arches = {k: (idx[k], *st[k]) for k in idx}
        P = teeth_vertices(g, fit, c, arches)
        rows, res = [], []
        for ai, k in enumerate(idx):
            val, gw = samp(P[k])
            gm = gw @ R                                        # gradient in model frame
            Rd = rotvec(st[k][0])
            # d p_model / d coefs = Rd @ Bi ;  d/dw ~ -[Rd(v - c0)]x ;  d/dtau = I
            Jc = np.einsum('nc,cd,pnd->np', gm, Rd, Bi[:, idx[k]])
            vm = (P[k] - fit['t']) @ R - st[k][2] - st[k][1]  # Rd(v - c0) in model frame
            Jw = -np.einsum('nc,ncd->nd', gm, skew(vm))
            Jt = gm
            J = np.zeros((len(val), len(c) + 12))
            J[:, :len(c)] = Jc
            J[:, len(c) + 6 * ai: len(c) + 6 * ai + 3] = Jw
            J[:, len(c) + 6 * ai + 3: len(c) + 6 * ai + 6] = Jt
            rows.append(J); res.append(val)
        J = np.vstack(rows); r = np.concatenate(res)
        theta = np.concatenate([c, st['upper'][0], st['upper'][1], st['lower'][0], st['lower'][1]])
        lam = np.concatenate([np.full(len(c), lam_c), np.full(3, lam_w), np.full(3, lam_t),
                              np.full(3, lam_w), np.full(3, lam_t)])
        H = J.T @ J + np.diag(lam)
        step = -np.linalg.solve(H, J.T @ r + lam * theta)
        theta = theta + step
        c = theta[:len(c)]
        st['upper'][0], st['upper'][1] = theta[80:83], theta[83:86]
        st['lower'][0], st['lower'][1] = theta[86:89], theta[89:92]
        hist.append(r.mean())
        if verbose and (it % 5 == 0 or it == n_iter - 1):
            print(f'it {it:2d} mean intensity at teeth {r.mean():.3f}  '
                  f'upper shift {np.round(st["upper"][1], 1)} mm rot {np.degrees(np.linalg.norm(st["upper"][0])):.1f} deg  '
                  f'lower shift {np.round(st["lower"][1], 1)} mm rot {np.degrees(np.linalg.norm(st["lower"][0])):.1f} deg')
    return dict(coefs=c, upper=st['upper'], lower=st['lower'], idx=idx, hist=hist)


if __name__ == '__main__':
    from gnm.shape import gnm_numpy
    vol, fit_prefix, out = sys.argv[1:4]
    g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    fit = dict(np.load(fit_prefix + '_fit.npz'))
    a = load_volume(vol)
    res = fit_teeth(g, a, fit)
    np.savez(out + '_teeth.npz', coefs=res['coefs'],
             upper_w=res['upper'][0], upper_tau=res['upper'][1], upper_c0=res['upper'][2],
             lower_w=res['lower'][0], lower_tau=res['lower'][1], lower_c0=res['lower'][2])
