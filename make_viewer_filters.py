"""Filter comparison viewer (scan picker): each denoising filter (filters.py) applied to a crop around the tract region;
optionally blended with its left-right mirror image about the head's midline plane;
the page runs the whole tissue_mesh pipeline live (threshold -> opening -> empty-space pieces -> size / dark-seed /
midline checks -> surface nets) with every parameter on a slider. Writes <out>/index.html.
Usage: python make_viewer_filters.py <recon_dir> <out_dir>/index.html [scan ...]   (default: the 15 example scans)
Per-scan data goes to <out_dir>/data/<scan>.js (cached; delete a file to rebuild it)."""
import sys, os, json, base64, time
import numpy as np
import scipy.ndimage as ndi
import fit_gnm as fg
import tissue_mesh as tm
import filters as F
import consensus as cs

recon_dir, html_out = sys.argv[1:3]
SCANS = sys.argv[3:] or ['06_pot_r1', '29_clench_r1', '01_beet_r1', '01_beet_r2', '02_bit_r1', '06_pot_r2', '08_bought_r1', '10_boot_r1',
                         '10_boot_r2', '12_bird_r1', '18_asa_r1', '20_asha_r1', '23_ama_r1', '26_ala_r1', '31_yawn_r1']
LABEL = {'beet': 'beet /i/', 'bit': 'bit /ɪ/', 'pot': 'pot /ɑ/', 'bought': 'bought /ɔ/', 'boot': 'boot /u/', 'bird': 'bird /ɝ/',
         'asa': 'asa /s/', 'asha': 'asha /ʃ/', 'ama': 'ama /m/', 'ala': 'ala /l/', 'clench': 'clench', 'yawn': 'yawn'}
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
SCALE = 1.2                                                   # uint8 code 255 = intensity 1.2
u8 = lambda v: (np.clip(v / SCALE, 0, 1) * 255).round().astype(np.uint8)
out_dir = os.path.dirname(html_out); os.makedirs(os.path.join(out_dir, 'data'), exist_ok=True)


def build_scan(scan):
    a = fg.load_volume(os.path.join(recon_dir, f'sub001_3d_{scan}_recon.mat'))
    box = tm.region(a)
    nz = np.argwhere(box)
    lo = np.maximum(nz.min(0) - 6, 0); hi = np.minimum(nz.max(0) + 7, a.shape)
    sl = tuple(slice(x, y) for x, y in zip(lo, hi))
    crop = a[sl]
    edge = ndi.distance_transform_edt(np.pad(box, 1))[1:-1, 1:-1, 1:-1]
    # left-right mirror about this scan's own midline plane (flipped scan registered to itself; cached)
    mp_file = os.path.join(out_dir, 'data', f'{scan}_mirror.npy')
    if os.path.exists(mp_file):
        mp = np.load(mp_file)
    else:
        mp, _ = cs.mirror_params(a); np.save(mp_file, mp)
    mirror = lambda full: cs.warp(full[:, :, ::-1].copy(), mp)
    vols, mvols = {}, {}
    for k, (name, f) in F.FILTERS.items():
        fc = f(crop); vols[k] = b64(u8(fc))
        full = a.copy(); full[sl] = fc                        # mirror needs the filtered values on the other side
        mvols[k] = b64(u8(mirror(full)[sl]))
    mbox = mirror(box.astype(np.float32)) > 0.5
    medge = ndi.distance_transform_edt(np.pad(mbox, 1))[1:-1, 1:-1, 1:-1]
    d = dict(crop=dict(lo=lo.tolist(), shape=list(crop.shape)),
             mirror=dict(offset_mm=float(mp[5] * fg.VOX / 2), tilt_deg=float(np.degrees(np.linalg.norm(mp[:3])) / 2)),
             vols=vols, mvols=mvols, box=b64(box[sl].astype(np.uint8)), cut=b64((edge < 1.5)[sl].astype(np.uint8)),
             mbox=b64(mbox[sl].astype(np.uint8)), mcut=b64((medge < 1.5)[sl].astype(np.uint8)), raw=b64(u8(a)))
    with open(os.path.join(out_dir, 'data', f'{scan}.js'), 'w') as fh:
        fh.write(f'window.FDATA = window.FDATA || {{}}; window.FDATA["{scan}"] = ' + json.dumps(d) + ';')


for scan in SCANS:
    if not os.path.exists(os.path.join(out_dir, 'data', f'{scan}.js')):
        t = time.time(); build_scan(scan); print(f'{scan}: {time.time() - t:.0f}s', flush=True)
labels = {s: LABEL[s.split('_')[1]] + (' · r2' if s.endswith('r2') else '') for s in SCANS}
data = dict(scale=SCALE, vol=dict(ni=160, nj=160, nk=80, vox=fg.VOX), scans=SCANS, labels=labels, first=SCANS[0],
            filters={k: n for k, (n, _) in F.FILTERS.items()}, default='gauss07',
            params=dict(thr=tm.THR, seed=tm.SEED, min_ml=tm.MIN_ML, max_abs_x=tm.MAX_ABS_X))
here = os.path.dirname(os.path.abspath(__file__))
tpl = open(os.path.join(here, 'filters_template.html')).read().replace('/*__SN__*/', open(os.path.join(here, 'surface_nets.js')).read())
open(html_out, 'w').write(tpl.replace('/*__DATA__*/null', json.dumps(data)))
print(f'wrote {html_out}; {len(SCANS)} scans in {out_dir}/data/')
