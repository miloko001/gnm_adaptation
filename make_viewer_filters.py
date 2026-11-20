"""Filter comparison viewer for one scan: each denoising filter (filters.py) applied to a crop around the tract region;
the page runs the whole tissue_mesh pipeline live (threshold -> opening -> empty-space pieces -> size / dark-seed /
midline checks -> surface nets) with every parameter on a slider. Writes <out>/index.html + data.js.
Usage: python make_viewer_filters.py <recon_dir> <scan> <out_dir>/index.html"""
import sys, os, json, base64, time
import numpy as np
import scipy.ndimage as ndi
import fit_gnm as fg
import tissue_mesh as tm
import filters as F

recon_dir, scan, html_out = sys.argv[1:4]
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
SCALE = 1.2                                                   # uint8 code 255 = intensity 1.2
a = fg.load_volume(os.path.join(recon_dir, f'sub001_3d_{scan}_recon.mat'))
box = tm.region(a)
nz = np.argwhere(box)
lo = np.maximum(nz.min(0) - 6, 0); hi = np.minimum(nz.max(0) + 7, a.shape)
sl = tuple(slice(x, y) for x, y in zip(lo, hi))
crop = a[sl]
edge = ndi.distance_transform_edt(np.pad(box, 1))[1:-1, 1:-1, 1:-1]
cut = (edge < 1.5)[sl]
u8 = lambda v: (np.clip(v / SCALE, 0, 1) * 255).round().astype(np.uint8)
vols, labels = {}, {}
for k, (name, f) in F.FILTERS.items():
    t = time.time(); vols[k] = b64(u8(f(crop))); labels[k] = name
    print(f'{k:8s} {time.time() - t:5.1f}s', flush=True)
out_dir = os.path.dirname(html_out); os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, 'data.js'), 'w') as fh:
    fh.write('window.FDATA = ' + json.dumps(dict(vols=vols, box=b64(box[sl].astype(np.uint8)), cut=b64(cut.astype(np.uint8)),
                                                  raw=b64(u8(a)))) + ';')
data = dict(scan=scan, scale=SCALE, vol=dict(ni=a.shape[0], nj=a.shape[1], nk=a.shape[2], vox=fg.VOX),
            crop=dict(lo=lo.tolist(), shape=list(crop.shape)), filters=labels, default='gauss07',
            params=dict(thr=tm.THR, seed=tm.SEED, min_ml=tm.MIN_ML, max_abs_x=tm.MAX_ABS_X))
here = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(here, 'filters_template.html')).read().replace('/*__SN__*/', open(os.path.join(here, 'surface_nets.js')).read())
open(html_out, 'w').write(tpl.replace('/*__DATA__*/null', json.dumps(data)))
print(f'wrote {html_out}; data.js {os.path.getsize(os.path.join(out_dir, "data.js")) / 1e6:.1f} MB')
