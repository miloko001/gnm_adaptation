"""Head-fit viewer: MRI head surface (wireframe) + fitted GNM head coloured by distance to the MRI surface
+ full MRI volume planes. Writes out/viewer_head/index.html (+ reuses/creates vol/*.js)."""
import sys, os, json, base64, shutil
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from gnm.shape import gnm_numpy
import fit_gnm as fg
import head_mesh_fit as hm
import tissue_mesh as tmesh

recon_dir, v5_dir, html_out = sys.argv[1:4]
ORDER = ['29_clench_r1', '01_beet_r1', '01_beet_r2', '02_bit_r1', '06_pot_r1', '06_pot_r2', '08_bought_r1', '10_boot_r1',
         '10_boot_r2', '12_bird_r1', '18_asa_r1', '20_asha_r1', '23_ama_r1', '26_ala_r1', '31_yawn_r1']
LABEL = {'beet': 'beet /i/', 'bit': 'bit /ɪ/', 'pot': 'pot /ɑ/', 'bought': 'bought /ɔ/', 'boot': 'boot /u/', 'bird': 'bird /ɝ/',
         'asa': 'asa /s/', 'asha': 'asha /ʃ/', 'ama': 'ama /m/', 'ala': 'ala /l/', 'clench': 'clench', 'yawn': 'yawn'}
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
G = dict(zip(g.vertex_group_names, g.vertex_groups.astype(bool)))
skinF = g.triangles[G['skin'][g.triangles].all(1)]
out_dir = os.path.dirname(html_out); os.makedirs(os.path.join(out_dir, 'vol'), exist_ok=True); os.makedirs(os.path.join(out_dir, 'mesh'), exist_ok=True)
LEVELS = np.round(np.arange(0.10, 0.801, 0.05), 2)                 # tissue brightness thresholds for the slider
old_vol = os.path.join(os.path.dirname(out_dir), 'viewer', 'vol')

heads = {s: trimesh.load(os.path.join(v5_dir, f'{s}_gnm_head.obj'), process=False) for s in ORDER}
centre = np.mean([h.vertices[G['skin']].mean(0) for h in heads.values()], 0)
Q = 10.0
q = lambda V: b64(np.round((V - centre) * Q).astype(np.int16))
scans = []
for s in ORDER:
    vf = os.path.join(out_dir, 'vol', f'{s}.js')
    if not os.path.exists(vf):
        if os.path.exists(os.path.join(old_vol, f'{s}.js')):
            shutil.copy(os.path.join(old_vol, f'{s}.js'), vf)
        else:
            a = fg.load_volume(os.path.join(recon_dir, f'sub001_3d_{s}_recon.mat'))
            with open(vf, 'w') as fh:
                fh.write(f'window.VOL = window.VOL || {{}}; window.VOL["{s}"] = "{b64((np.clip(a / 1.1, 0, 1) * 255).astype(np.uint8))}";')
    mri = trimesh.load(os.path.join(v5_dir, f'{s}_mri_head.stl'))
    mri_s = mri.simplify_quadric_decimation(face_count=12000)
    # inner tissue surface at several brightness thresholds -> mesh/<scan>.js sidecar (loaded per scan)
    mf_ = os.path.join(out_dir, 'mesh', f'{s}.js')
    lv = {}
    if os.path.exists(mf_):                                     # reuse cached levels
        t_ = open(mf_).read(); lv = json.loads(t_[t_.index('= {') + 2:-1])
    missing = [thr for thr in LEVELS if f'{thr:.2f}' not in lv]
    if missing:
        a = fg.load_volume(os.path.join(recon_dir, f'sub001_3d_{s}_recon.mat'))
        for thr in missing:
            m, _ = tmesh.tissue_mesh(a, thr=thr, faces=25000)
            lv[f'{thr:.2f}'] = dict(v=q(m.vertices), f=b64(m.faces.astype(np.uint16))) if len(m.faces) else None
        with open(mf_, 'w') as fh:
            fh.write(f'window.TIS = window.TIS || {{}}; window.TIS["{s}"] = {json.dumps(lv)};')
    V = heads[s].vertices
    # distance of each GNM skin vertex to the MRI surface (only meaningful inside the scanned box)
    d, _ = cKDTree(mri.vertices).query(V)
    inside = hm.in_scan(V)
    code = np.where(inside & G['skin'], np.clip(np.round(d * 10), 0, 254), 255).astype(np.uint8)   # 0.1 mm units; 255 = no data
    vis = inside & G['skin']
    dv = d[vis]
    # MRI -> GNM distance (coverage of the MRI surface by GNM)
    dm, _ = cKDTree(V[G['skin']]).query(mri.vertices)
    scans.append(dict(id=s, label=LABEL[s.split('_')[1]] + (' · r2' if s.endswith('r2') else ''),
                      v=q(V), d=b64(code), mv=q(mri_s.vertices), mf=b64(mri_s.faces.astype(np.uint16)),
                      stats=dict(gnm_med=round(float(np.median(dv)), 2), gnm_p90=round(float(np.percentile(dv, 90)), 2),
                                 within3=round(float((dv < 3).mean() * 100)), mri_med=round(float(np.median(dm)), 2),
                                 mri_within3=round(float((dm < 3).mean() * 100)))))
    print(s, scans[-1]['stats'], flush=True)

data = dict(q=Q, levels=[f'{l:.2f}' for l in LEVELS], level0=f'{tmesh.THR:.2f}', centre=centre.tolist(), vol=dict(ni=160, nj=160, nk=80, vox=1.25),
            skinf=b64(skinF.astype(np.uint16)), scans=scans)
tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viewer4_template.html')).read()
open(html_out, 'w').write(tpl.replace('/*__DATA__*/null', json.dumps(data)))
print(f'wrote {html_out} ({os.path.getsize(html_out) / 1e6:.1f} MB)')
