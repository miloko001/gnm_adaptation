"""Overlay GNM teeth (initial = cyan, fitted = yellow) on MRI: mid-sagittal, coronal and axial slices."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_gnm import load_volume, VOX
import fit_teeth as ft

vol, fit_prefix, teeth_npz, png = sys.argv[1:5]
from gnm.shape import gnm_numpy
g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
G = dict(zip(g.vertex_group_names, g.vertex_groups.astype(bool)))
fit = dict(np.load(fit_prefix + '_fit.npz'))
a = load_volume(vol); nz = a.shape[2]
idx = {'upper': np.where(G['teeth'] & G['upper_teeth_and_gums'])[0],
       'lower': np.where(G['teeth'] & G['lower_teeth_and_gums'])[0]}
T0 = g.template_vertex_positions * 1000
init = ft.teeth_vertices(g, fit, np.zeros(80), {k: (v, np.zeros(3), np.zeros(3), T0[v].mean(0)) for k, v in idx.items()})
sets = [('c', init)]
if teeth_npz != '-':
    t = np.load(teeth_npz)
    sets.append(('y', ft.teeth_vertices(g, fit, t['coefs'], {
        k: (idx[k], t[k + '_w'], t[k + '_tau'], t[k + '_c0']) for k in idx})))

allp = np.vstack(list(init.values()))
yu, yl = init['upper'][:, 1].mean(), init['lower'][:, 1].mean()
zc = allp[:, 2].mean()
views = [('sag x=0', a[:, :, nz // 2], lambda P: (-P[:, 2] / VOX, -P[:, 1] / VOX), np.abs, 0),
         ('sag x=-15', a[:, :, nz // 2 - 12], lambda P: (-P[:, 2] / VOX, -P[:, 1] / VOX), lambda x: np.abs(x + 15), 0),
         ('cor', a[:, int(round(-zc / VOX)), :], lambda P: (P[:, 0] / VOX + (nz - 1) / 2, -P[:, 1] / VOX), lambda x: np.abs(x - zc), 2),
         ('ax upper', a[int(round(-yu / VOX))], lambda P: (P[:, 0] / VOX + (nz - 1) / 2, -P[:, 2] / VOX), lambda x: np.abs(x - yu), 1),
         ('ax lower', a[int(round(-yl / VOX))], lambda P: (P[:, 0] / VOX + (nz - 1) / 2, -P[:, 2] / VOX), lambda x: np.abs(x - yl), 1)]
fig, axs = plt.subplots(1, 5, figsize=(25, 6))
for ax, (title, img, proj, dist, axis) in zip(axs, views):
    ax.imshow(img, cmap='gray', vmax=1.1)
    for col, P in sets:
        Q = np.vstack(list(P.values()))
        near = dist(Q[:, axis]) < 1.5
        u, v = proj(Q[near]); ax.plot(u, v, '.', color=col, ms=2)
    Q = np.vstack(list(init.values())); u, v = proj(Q)
    ax.set_xlim(u.min() - 25, u.max() + 25); ax.set_ylim(v.max() + 25, v.min() - 25); ax.set_title(title)
plt.tight_layout(); plt.savefig(png, dpi=60)
