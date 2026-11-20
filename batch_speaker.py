"""Run the GNM face fit, teeth transfer and VT segmentation over every 3D volume of one speaker.

Pass 1: free identity+expression fit per scan -> speaker identity = median.
Pass 2: refit each scan with identity fixed (expression + rigid only).
Teeth:  80 teeth coefs + arch offsets fitted once on the clench scan, reused everywhere.
Then segment each VT and export an STL. Writes summary.csv and a mid-sagittal contact sheet.
"""
import sys, os, glob, csv, time
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gnm.shape import gnm_numpy
import fit_gnm as fg
import fit_teeth as ft
import segment_vt as sv

recon_dir, out_dir = sys.argv[1:3]
os.makedirs(out_dir, exist_ok=True)
files = sorted(glob.glob(os.path.join(recon_dir, '*_recon.mat')))
names = [os.path.basename(f).split('_3d_')[1].replace('_recon.mat', '') for f in files]

g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
G = dict(zip(g.vertex_group_names, g.vertex_groups.astype(bool)))
T0 = g.template_vertex_positions * 1000

# ---- pass 1: free fits -------------------------------------------------------
t0 = time.time()
surf, ids = {}, []
for f, n in zip(files, names):
    pts, nrm = fg.skin_surface(fg.load_volume(f))
    surf[n] = (pts.astype(np.float32), nrm.astype(np.float32))
    r = fg.fit(g, pts, nrm, verbose=False)
    ids.append(r['id'])
    print(f'[pass1] {n:20s} median {np.median(r["face_dist"]):.2f} mm', flush=True)
ids = np.array(ids)
speaker_id = np.median(ids, 0)
np.save(os.path.join(out_dir, 'speaker_identity.npy'), speaker_id)
print(f'pass1 {time.time() - t0:.0f}s; identity spread (mean |id - median|) {np.abs(ids - speaker_id).mean():.3f}')

# ---- pass 2: fixed identity -------------------------------------------------
fits = {}
for n in names:
    pts, nrm = surf[n]
    r = fg.fit(g, pts.astype(float), nrm.astype(float), verbose=False, fixed_id=speaker_id)
    fits[n] = dict(id=speaker_id, ex=r['ex'], R=r['R'], t=r['t'],
                   med=float(np.median(r['face_dist'])), p90=float(np.percentile(r['face_dist'], 90)))
    np.savez(os.path.join(out_dir, f'{n}_fit.npz'), **{k: fits[n][k] for k in ('id', 'ex', 'R', 't')})
del surf

# ---- teeth from clench --------------------------------------------------------
clench = [n for n in names if 'clench' in n][0]
teeth = ft.fit_teeth(g, fg.load_volume(files[names.index(clench)]), fits[clench], verbose=False)
np.savez(os.path.join(out_dir, 'speaker_teeth.npz'), coefs=teeth['coefs'],
         upper_w=teeth['upper'][0], upper_tau=teeth['upper'][1], upper_c0=teeth['upper'][2],
         lower_w=teeth['lower'][0], lower_tau=teeth['lower'][1], lower_c0=teeth['lower'][2])
print(f'teeth fitted on {clench}: mean intensity {teeth["hist"][0]:.3f} -> {teeth["hist"][-1]:.3f}')


def world_mesh(fit):
    """Full fitted GNM mesh in world mm, teeth replaced by the speaker's fitted teeth."""
    V = (T0 + np.tensordot(fit['id'], g.vertex_identity_basis[fg.ID_HEAD] * 1000, 1)
         + np.tensordot(fit['ex'], g.expression_basis[fg.EX_LOWER] * 1000, 1)
         + np.tensordot(teeth['coefs'], g.vertex_identity_basis[ft.ID_TEETH] * 1000, 1))
    for k in ('upper', 'lower'):
        idx = teeth['idx'][k]; w, tau, c0 = teeth[k]
        V[idx] = (V[idx] - c0) @ ft.rotvec(w).T + c0 + tau
    return trimesh.Trimesh(V @ fit['R'].T + fit['t'], g.triangles, process=False)


# ---- segmentation ----------------------------------------------------------------
rows, sheet = [], []
for f, n in zip(files, names):
    fit = fits[n]
    row = dict(scan=n, face_median_mm=round(fit['med'], 2), face_p90_mm=round(fit['p90'], 2))
    try:
        a = fg.load_volume(f)
        gm = world_mesh(fit)
        loop = sv.gnm_lip_loop(gm, G)
        vt, s, grids, info = sv.segment(a, sv.lip_centre(loop), sv.teeth_exclusion_points(gm, G, fit['R']))
        gi, gj, gk = grids
        v = (sv.VOX / sv.UP) ** 3 / 1000
        J = gj[None, :, None]
        row.update(vt_ml=round(vt.sum() * v, 1), oral_ml=round((vt & (J < 78)).sum() * v, 1),
                   pharynx_ml=round((vt & (J >= 78)).sum() * v, 1),
                   path_max=round(info['path_max'], 2), glottis_row=info['glottis_row'], error='')
        m = sv.mask_to_mesh(vt, grids)
        row['watertight'] = m.is_watertight
        m.export(os.path.join(out_dir, f'{n}_vt.stl'))
        kc = len(gk) // 2
        sheet.append((n, s[:, :, kc], vt[:, :, kc], (gj[0], gj[-1], gi[-1], gi[0])))
    except Exception as e:  # keep going; record the failure
        row['error'] = repr(e)[:120]
    rows.append(row)
    print(f'[seg] {n:20s} {row}', flush=True)

keys = ['scan', 'face_median_mm', 'face_p90_mm', 'vt_ml', 'oral_ml', 'pharynx_ml', 'path_max', 'glottis_row', 'watertight', 'error']
with open(os.path.join(out_dir, 'summary.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, keys); w.writeheader(); [w.writerow({k: r.get(k, '') for k in keys}) for r in rows]

cols = 11
fig, axs = plt.subplots(int(np.ceil(len(sheet) / cols)), cols, figsize=(cols * 2.2, np.ceil(len(sheet) / cols) * 2.6))
for ax in axs.ravel():
    ax.axis('off')
for ax, (n, img, mk, ext) in zip(axs.ravel(), sheet):
    ax.imshow(img, cmap='gray', vmax=1, extent=ext)
    ax.contour(np.linspace(ext[0], ext[1], img.shape[1]), np.linspace(ext[3], ext[2], img.shape[0]), mk, [0.5], colors='r', linewidths=0.6)
    ax.set_title(n, fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'contact_sheet.png'), dpi=60)
print(f'done in {time.time() - t0:.0f}s')
