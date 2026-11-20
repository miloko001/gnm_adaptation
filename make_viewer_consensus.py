"""Viewer for consensus.py. Ships the scan-averaged emptiness map for each tissue threshold (and its left-right mirror
image about the head's own midline plane), cropped to the tract region; the page builds the surface live (surface
nets) from four sliders: lowest / highest tissue threshold, agreement across scans, symmetry blend.
Writes <out>/index.html + vols.js + maps.js."""
import sys, os, json, base64
import numpy as np
import scipy.ndimage as ndi
import trimesh
import fit_gnm as fg
import tissue_mesh as tm
import consensus as cs

cons_npz, v5_dir, html_out = sys.argv[1:4]
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
z = np.load(cons_npz)
ref_img, mean_img, ref = z['ref_img'], z['mean_img'], str(z['ref'])
thrs = [float(t) for t in z['thrs']]
out_dir = os.path.dirname(html_out); os.makedirs(out_dir, exist_ok=True)

mp_file = os.path.join(os.path.dirname(cons_npz), 'mirror_params.npy')
if os.path.exists(mp_file):
    mp = np.load(mp_file)
else:
    mp, _ = cs.mirror_params(ref_img); np.save(mp_file, mp)

# per-threshold maps: smoothed (linear, so averaging thresholds / blending mirrors in the page stays exact), + mirror
maps = [ndi.gaussian_filter(z['per_thr'][n].astype(np.float32) / 255, 0.6) for n in range(len(thrs))]
mirs = [cs.warp(M[:, :, ::-1].copy(), mp) for M in maps]
box = tm.region(ref_img)
cut = ndi.distance_transform_edt(np.pad(box, 1))[1:-1, 1:-1, 1:-1] < 1.5        # region-box boundary: not anatomy
cut |= ndi.distance_transform_edt(np.pad(cs.warp(box[:, :, ::-1].astype(np.float32), mp) > 0.5, 1))[1:-1, 1:-1, 1:-1] < 1.5

# crop to where anything is non-zero (+ margin)
nz = np.argwhere(np.max(maps, 0) + np.max(mirs, 0) > 0.02)
lo = np.maximum(nz.min(0) - 2, 0); hi = np.minimum(nz.max(0) + 3, ref_img.shape)
sl = tuple(slice(a, b) for a, b in zip(lo, hi))
u8 = lambda a: (np.clip(a, 0, 1) * 255).round().astype(np.uint8)
with open(os.path.join(out_dir, 'maps.js'), 'w') as fh:
    fh.write('window.CMAP = ' + json.dumps(dict(raw=[b64(u8(M[sl])) for M in maps], mir=[b64(u8(M[sl])) for M in mirs],
                                                 cut=b64(cut[sl].astype(np.uint8)))) + ';')
with open(os.path.join(out_dir, 'vols.js'), 'w') as fh:
    fh.write('window.CVOL = ' + json.dumps(dict(mean=b64(u8(mean_img / 1.1)), ref=b64(u8(ref_img / 1.1)))) + ';')

gnm = trimesh.load(os.path.join(v5_dir, f'{ref}_gnm_head.obj'), process=False)
centre = gnm.vertices.mean(0)
Q = 10.0
q = lambda V: b64(np.round((V - centre) * Q).astype(np.int16))
gs = gnm.simplify_quadric_decimation(face_count=20000)
P = z['params']
data = dict(q=Q, centre=centre.tolist(), ref=ref, vol=dict(ni=160, nj=160, nk=80, vox=1.25), thrs=thrs,
            crop=dict(lo=lo.tolist(), shape=[int(b - a) for a, b in zip(lo, hi)]),
            mirror=dict(offset_mm=float(mp[5] * fg.VOX / 2), tilt_deg=float(np.degrees(np.linalg.norm(mp[:3])) / 2)),
            gnm=dict(v=q(gs.vertices), f=b64(gs.faces.astype(np.uint16))),
            stats=dict(n=int(len(z['scans'])), ncc=float(np.median(z['ncc'])), rot=float(np.median(np.degrees(np.linalg.norm(P[:, :3], axis=1)))),
                       shift=float(np.median(np.linalg.norm(P[:, 3:], axis=1) * fg.VOX))))
here = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(here, 'consensus_template.html')).read().replace('/*__SN__*/', open(os.path.join(here, 'surface_nets.js')).read())
open(html_out, 'w').write(tpl.replace('/*__DATA__*/null', json.dumps(data)))
print(f'crop {data["crop"]}, wrote {html_out} ({os.path.getsize(html_out) / 1e6:.1f} MB), '
      f'maps.js {os.path.getsize(os.path.join(out_dir, "maps.js")) / 1e6:.1f} MB')
