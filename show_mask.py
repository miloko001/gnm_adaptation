"""Show VT mask contours on upsampled crop: 4 sagittal slices + 4 axial slices."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_gnm import load_volume
import segment_vt as sv
import trimesh

vol, fit_prefix, groups_npz, png = sys.argv[1:5]
a = load_volume(vol)
gm = trimesh.load(fit_prefix + '_gnm.obj', process=False); G = np.load(groups_npz)
loop = sv.gnm_lip_loop(gm, G)
vt, s, (gi, gj, gk), info = sv.segment(a, sv.lip_centre(loop), sv.teeth_exclusion_points(gm, G, np.load(fit_prefix + '_fit.npz')['R']))
print('volume ml', vt.sum() * (sv.VOX / sv.UP) ** 3 / 1000, 'glottis', info['glottis_row'], 'path max', info['path_max'])
kc = len(gk) // 2
sag = [kc, kc - 16, kc + 16, kc - 32]
ax_rows = [np.searchsorted(gi, r) for r in (68, 80, 95, 105)]
fig, axs = plt.subplots(2, 4, figsize=(20, 10))
ext = (gj[0], gj[-1], gi[-1], gi[0])
for ax, k in zip(axs[0], sag):
    ax.imshow(s[:, :, k], cmap='gray', vmax=1, extent=ext)
    ax.contour(gj, gi, vt[:, :, k], [0.5], colors='r', linewidths=0.8)
    if info['teeth'][:, :, k].any(): ax.contour(gj, gi, info['teeth'][:, :, k], [0.5], colors='c', linewidths=0.6)
    ax.set_title(f'sag k={gk[k]:.1f}')
for ax, r in zip(axs[1], ax_rows):
    ax.imshow(s[r], cmap='gray', vmax=1, extent=(gk[0], gk[-1], gj[-1], gj[0]))
    ax.contour(gk, gj, vt[r], [0.5], colors='r', linewidths=0.8)
    if info['teeth'][r].any(): ax.contour(gk, gj, info['teeth'][r], [0.5], colors='c', linewidths=0.6)
    ax.set_title(f'axial row={gi[r]:.1f}')
plt.tight_layout(); plt.savefig(png, dpi=55)
