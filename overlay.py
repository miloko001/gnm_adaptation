"""Overlay GNM mesh cross-sections (skin=red, lips=yellow, teeth/gums=cyan, tongue=magenta) on MRI slices."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import trimesh
from fit_gnm import load_volume, VOX

vol_path, mesh_path, png = sys.argv[1:4]
groups = np.load(sys.argv[4]) if len(sys.argv) > 4 else None
a = load_volume(vol_path)
nz = a.shape[2]
mesh = trimesh.load(mesh_path, process=False)

def to_px(P, plane):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    k = x / VOX + (nz - 1) / 2; i = -y / VOX; j = -z / VOX
    return {'sag': (j, i), 'cor': (k, i), 'ax': (k, j)}[plane]

planes = [('sag', a[:, :, nz // 2], [0.3, 0, 0], [1, 0, 0]),
          ('sag', a[:, :, nz // 2 - 12], [-12 * VOX, 0, 0], [1, 0, 0]),
          ('cor', a[:, 70, :], [0, 0, -70 * VOX], [0, 0, 1]),
          ('ax', a[80, :, :], [0, -80 * VOX, 0], [0, 1, 0])]
colors = {'skin': 'r', 'upper_lip': 'y', 'lower_lip': 'y', 'teeth': 'c', 'gums': 'c', 'tongue': 'm', 'mouth_sock': 'orange'}

fig, axs = plt.subplots(1, 4, figsize=(20, 6))
for ax, (pl, img, o, n) in zip(axs, planes):
    ax.imshow(img, cmap='gray', vmax=1.1)
    # Edge/plane intersections (robust to seams that break trimesh.section).
    E = mesh.edges_unique
    sd = (mesh.vertices - np.asarray(o, float)) @ np.asarray(n, float)
    cross = np.sign(sd[E[:, 0]]) != np.sign(sd[E[:, 1]])
    for name in (['all'] if groups is None else colors):
        ok = cross if name == 'all' else cross & groups[name][E[:, 0]] & groups[name][E[:, 1]]
        e = E[ok]
        w = sd[e[:, 0]] / (sd[e[:, 0]] - sd[e[:, 1]])
        P = mesh.vertices[e[:, 0]] + w[:, None] * (mesh.vertices[e[:, 1]] - mesh.vertices[e[:, 0]])
        u, v = to_px(P, pl)
        ax.plot(u, v, '.', color=colors.get(name, 'r'), ms=1.5)
    ax.set_title(f'{pl} {o}'); ax.set_xlim(0, img.shape[1]); ax.set_ylim(img.shape[0], 0)
plt.tight_layout(); plt.savefig(png, dpi=70)
