"""Vocal tract -> area function -> vowel synthesis, per scan.

1. Regularised GNM face fit (internal only: places teeth and the lip opening).
2. Speaker teeth from the clench scan; lower arch re-fitted rigidly per scan (follows the jaw).
3. Airway segmentation (segment_vt).
4. Area function A(x): voxel count in shells of geodesic distance from the glottis.
5. Frequency-domain lossy transmission line (yielding walls, viscous loss, radiation load)
   driven by a Rosenberg glottal pulse train -> WAV.
"""
import sys, os, json
import numpy as np
import scipy.ndimage as ndi
from scipy.io import wavfile
from scipy.signal import find_peaks
from skimage import graph
import trimesh
import fit_gnm as fg
import fit_teeth as ft
import segment_vt as sv

RHO, C, MU = 1.14e-3, 3.5e4, 1.86e-4          # CGS: g/cm^3, cm/s, poise
WALL_M, WALL_R, WALL_K = 1.5, 1600.0, 3.0e5    # yielding-wall mass, resistance, stiffness per area
A_MIN = 0.05                                   # cm^2 floor for (near-)closed sections


def fit_lower_arch(g, a, fit, teeth, n_iter=25, lam_w=300.0, lam_t=1.0):
    """Rigid Gauss-Newton fit of the lower arch only (teeth shape fixed) into the dark arch."""
    a = np.where(ndi.binary_erosion(fg.head_mask(a), iterations=2), a, 1.0)
    R = fit['R']; idx = teeth['idx']['lower']
    w, tau, c0 = [np.array(x, float) for x in teeth['lower']]
    up = teeth['upper']
    for it in range(n_iter):
        if it in (0, 12):
            samp = ft.Sampler(a, 1.5 if it == 0 else 0.8)
        P = ft.teeth_vertices(g, fit, teeth['coefs'], {'lower': (idx, w, tau, c0)})['lower']
        val, gw = samp(P)
        gm = gw @ R
        vm = (P - fit['t']) @ R - c0 - tau
        J = np.hstack([-np.einsum('nc,ncd->nd', gm, ft.skew(vm)), gm])
        th = np.concatenate([w - teeth['lower'][0], tau - teeth['lower'][1]])
        lam = np.r_[np.full(3, lam_w), np.full(3, lam_t)]
        step = -np.linalg.solve(J.T @ J + np.diag(lam), J.T @ val + lam * th)
        w, tau = w + step[:3], tau + step[3:]
    return dict(teeth, lower=(w, tau, c0))


def world_mesh(g, fit, teeth):
    T0 = g.template_vertex_positions * 1000
    V = (T0 + np.tensordot(fit['id'], g.vertex_identity_basis[fg.ID_HEAD] * 1000, 1)
         + np.tensordot(fit['ex'], g.expression_basis[fg.EX_LOWER] * 1000, 1)
         + np.tensordot(teeth['coefs'], g.vertex_identity_basis[ft.ID_TEETH] * 1000, 1))
    for k in ('upper', 'lower'):
        idx = teeth['idx'][k]; w, tau, c0 = teeth[k]
        V[idx] = (V[idx] - c0) @ ft.rotvec(w).T + c0 + tau
    return trimesh.Trimesh(V @ fit['R'].T + fit['t'], g.triangles, process=False)


def area_function(vt, s, grids, info, lip_z, dx_mm=2.5, band_vox=3):
    """Area (cm^2) vs distance from glottis (cm) using fractional air content.

    Voxels within `band_vox` of the binary airway (inside the allowed region, not teeth) count
    f = clip((s_tis - s) / (s_tis - s_air), 0, 1), so sub-voxel channels contribute their true
    area. Distance = geodesic distance from the glottis within that band.
    """
    gi, gj, gk = grids
    vox_mm = sv.VOX / sv.UP
    band = (ndi.binary_dilation(vt, iterations=band_vox) & info['allowed'] & ~info['teeth']) | vt
    s_air = float(np.median(s[vt]))
    s_tis = float(np.percentile(s[ndi.binary_dilation(vt, iterations=8) & info['allowed'] & ~vt], 60))
    frac = np.where(vt, 1.0, np.clip((s_tis - s) / (s_tis - s_air), 0, 1)) * band
    bottom = np.where(vt.any((1, 2)))[0].max()
    start = np.argwhere(vt & (np.arange(vt.shape[0])[:, None, None] == bottom))
    mcp = graph.MCP_Geometric(np.where(band, 1.0, np.inf))
    dist, _ = mcp.find_costs([tuple(p) for p in start])
    dist = np.where(band, dist * vox_mm, np.inf)
    z = -(gj * sv.VOX)
    lip_vox = vt & (z[None, :, None] > lip_z - 3.0)
    dl = dist[lip_vox & np.isfinite(dist)]
    L = np.median(dl) if dl.size else dist[np.isfinite(dist)].max()
    edges = np.arange(0, L + dx_mm, dx_mm)
    ok = np.isfinite(dist) & (dist <= L)
    w, _ = np.histogram(dist[ok], edges, weights=frac[ok])
    x = 0.5 * (edges[1:] + edges[:-1])
    return x / 10, np.maximum(w * vox_mm ** 3 / dx_mm / 100, A_MIN), L / 10, (s_air, s_tis)


def transfer_function(x_cm, A_cm2, f):
    """Volume-velocity transfer U_lips/U_glottis and lip radiation impedance for a lossy tube chain."""
    dx = np.gradient(x_cm) if len(x_cm) > 1 else np.array([x_cm[0]])
    w = 2 * np.pi * f
    K = np.tile(np.eye(2, dtype=complex), (len(f), 1, 1))
    for A, l in zip(A_cm2, dx):
        S = 2 * np.sqrt(np.pi * A)                                    # perimeter of an equivalent circle
        Zs = 1j * w * RHO / A + S / A ** 2 * np.sqrt(w * RHO * MU / 2)
        Yw = S / (1j * w * WALL_M + WALL_R + WALL_K / (1j * w))
        Y = 1j * w * A / (RHO * C ** 2) + Yw
        gam = np.sqrt(Zs * Y); Zc = np.sqrt(Zs / Y)
        ch, sh = np.cosh(gam * l), np.sinh(gam * l)
        M = np.stack([np.stack([ch, Zc * sh], -1), np.stack([sh / Zc, ch], -1)], -2)
        K = K @ M
    Al = A_cm2[-1]; ka = w / C * np.sqrt(Al / np.pi)
    Zr = RHO * C / Al * (ka ** 2 / 2 + 1j * 0.8488 * ka)
    H = 1.0 / (K[:, 1, 0] * Zr + K[:, 1, 1])
    return H


def formants(f, H, n=4):
    mag = 20 * np.log10(np.abs(H) + 1e-12)
    pk, _ = find_peaks(mag, prominence=3)
    pk = pk[f[pk] > 150]
    return [float(f[p]) for p in pk[:n]]


def synthesize(x_cm, A_cm2, fs=22050, dur=1.2, f0=115.0):
    n = 8192
    f = np.fft.rfftfreq(n, 1 / fs); f[0] = 1.0
    H = transfer_function(x_cm, A_cm2, f) * 1j * 2 * np.pi * f   # lip radiation ~ d/dt
    H[0] = 0
    h = np.fft.irfft(H, n); h *= np.hanning(2 * n)[n:]           # taper the tail
    t = np.arange(int(dur * fs)) / fs
    f0t = f0 * (1 + 0.04 * np.sin(2 * np.pi * 5 * t)) * np.interp(t, [0, dur], [1.05, 0.92])
    phase = np.cumsum(f0t / fs) % 1.0
    op, cl = 0.4, 0.16                                            # Rosenberg open/closing quotients
    u = np.where(phase < op, 0.5 * (1 - np.cos(np.pi * phase / op)),
                 np.where(phase < op + cl, np.cos(np.pi * (phase - op) / (2 * cl)), 0.0))
    y = np.convolve(u - u.mean(), h)[:len(t)]
    env = np.minimum(1, np.minimum(t / 0.06, (dur - t) / 0.12))
    y = y * env
    return (y / (np.abs(y).max() + 1e-9) * 0.8 * 32767).astype(np.int16), fs


if __name__ == '__main__':
    from gnm.shape import gnm_numpy
    recon_dir, out_dir = sys.argv[1], sys.argv[2]
    scans = sys.argv[3:]
    g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
    G = dict(zip(g.vertex_group_names, g.vertex_groups.astype(bool)))
    tz = np.load(os.path.join(out_dir, 'speaker_teeth.npz'))
    idx = {'upper': np.where(G['teeth'] & G['upper_teeth_and_gums'])[0],
           'lower': np.where(G['teeth'] & G['lower_teeth_and_gums'])[0]}
    teeth0 = dict(coefs=tz['coefs'], idx=idx, upper=(tz['upper_w'], tz['upper_tau'], tz['upper_c0']),
                  lower=(tz['lower_w'], tz['lower_tau'], tz['lower_c0']))
    results = {}
    for scan in scans:
        a = fg.load_volume(os.path.join(recon_dir, f'sub001_3d_{scan}_recon.mat'))
        fp = os.path.join(out_dir, f'{scan}_fit.npz')
        if os.path.exists(fp):
            fit = dict(np.load(fp))
        else:
            pts, nrm = fg.skin_surface(a)
            r = fg.fit(g, pts, nrm, verbose=False, lam_final=150.0)
            fit = dict(id=r['id'], ex=r['ex'], R=r['R'], t=r['t']); np.savez(fp, **fit)
        teeth = fit_lower_arch(g, a, fit, teeth0)
        gm = world_mesh(g, fit, teeth)
        lc = sv.lip_centre(sv.gnm_lip_loop(gm, G))
        vt, s, grids, info = sv.segment(a, lc, sv.teeth_exclusion_points(gm, G, fit['R']))
        x, A, L, lev = area_function(vt, s, grids, info, lc[2])
        f = np.linspace(50, 5000, 2000)
        H = transfer_function(x, A, f)
        F = formants(f, H)
        y, fs = synthesize(x, A)
        wavfile.write(os.path.join(out_dir, f'{scan}.wav'), fs, y)
        m = sv.mask_to_mesh(vt, grids); m = m.simplify_quadric_decimation(face_count=15000)
        m.export(os.path.join(out_dir, f'{scan}_vt.stl'))
        tm = trimesh.Trimesh(gm.vertices, g.triangles[(G['teeth'] | G['gums'])[g.triangles].all(1)], process=False)
        tm.remove_unreferenced_vertices(); tm.export(os.path.join(out_dir, f'{scan}_teeth.stl'))
        jaw = float(np.linalg.norm(teeth['lower'][1] - teeth0['lower'][1]))
        results[scan] = dict(length_cm=round(L, 2), vol_ml=round(float(vt.sum() * (sv.VOX / sv.UP) ** 3 / 1000), 1),
                             min_area_cm2=round(float(A.min()), 3), formants_hz=[round(v) for v in F],
                             path_max=round(info['path_max'], 2), glottis_row=float(info['glottis_row']),
                             lower_arch_shift_mm=round(jaw, 1), x_cm=x.round(3).tolist(), A_cm2=A.round(3).tolist())
        print(f'{scan:14s} air/tissue {lev[0]:.2f}/{lev[1]:.2f}  L {L:5.2f} cm  vol {results[scan]["vol_ml"]:5.1f} ml  minA {A.min():.3f}  '
              f'F {results[scan]["formants_hz"]}  jaw shift {jaw:.1f} mm  glottis row {info["glottis_row"]}', flush=True)
    json.dump(results, open(os.path.join(out_dir, 'synth_results.json'), 'w'))
