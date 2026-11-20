"""Overlay template teeth (profile pose = cyan, rigid-fitted = yellow) on axial slices through the dental region
plus two sagittal slices. Usage: show_teeth_rigid.py <recon_dir> <fit_dir> <scan> <png>"""
import sys
import numpy as np
import scipy.ndimage as ndi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gnm.shape import gnm_numpy
import fit_gnm as fg
import jaw_teeth as jt
import fit_teeth_circles as fc

recon_dir, fit_dir, scan, png = sys.argv[1:5]
g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
tp = jt.Template(g); G = tp.G
a = fg.load_volume(f'{recon_dir}/sub001_3d_{scan}_recon.mat'); nz = a.shape[2]
z = np.load(f'{fit_dir}/{scan}_teeth_rigid.npz')
s, R, t = float(z['s']), z['R'], z['t']
fitted = {k: dict(w=z[f'{k}_w'], tau=z[f'{k}_tau'], c0=z[f'{k}_c0']) for k in ('upper', 'lower')}
init = {k: dict(w=np.zeros(3), tau=np.zeros(3), c0=z[f'{k}_c0']) for k in ('upper', 'lower')}
teeth = G['teeth']
sets = [('c', fc.teeth_world_rigid(tp, s, R, t, init)), ('y', fc.teeth_world_rigid(tp, s, R, t, fitted))]
rows = list(range(68, 100, 3))
fig, axs = plt.subplots(2, 7, figsize=(26, 8.5))
for ax, r in zip(axs.ravel()[:len(rows)], rows):
    y = -r * 1.25
    ax.imshow(ndi.zoom(a[r, 25:95, :], 2, order=1), cmap='gray', vmax=1.0, extent=(-0.5, 79.5, 94.5, 24.5))
    for col, V in sets:
        P = V[teeth]; P = P[np.abs(P[:, 1] - y) < 0.9]
        ax.plot(P[:, 0] / 1.25 + (nz - 1) / 2, -P[:, 2] / 1.25, '.', color=col, ms=1.6)
    ax.set_title(f'{scan} axial row {r}', fontsize=9); ax.set_xlim(0, 79); ax.set_ylim(94, 25)
for ax, k in zip(axs.ravel()[len(rows):], (nz // 2, nz // 2 - 16, nz // 2 + 16)):
    x = (k - (nz - 1) / 2) * 1.25
    ax.imshow(a[50:125, 20:110, k], cmap='gray', vmax=1.1, extent=(19.5, 109.5, 124.5, 49.5))
    for col, V in sets:
        P = V[teeth]; P = P[np.abs(P[:, 0] - x) < 1.0]
        ax.plot(-P[:, 2] / 1.25, -P[:, 1] / 1.25, '.', color=col, ms=1.6)
    ax.set_title(f'sagittal x={x:.0f} mm', fontsize=9)
plt.tight_layout(); plt.savefig(png, dpi=50)
