"""Viewer v3: airway mesh + unmodified GNM template teeth (scale/pose from the mid-sagittal face profile,
lower arch closed) + optional template head + full MRI volume planes (vol/*.js sidecar files)."""
import sys, os, json, base64
import numpy as np
import trimesh
from gnm.shape import gnm_numpy
import fit_gnm as fg
import jaw_teeth as jt
import teeth_blobs as tb

recon_dir, v2_dir, v3_dir, html_out = sys.argv[1:5]
ORDER = ['01_beet_r1', '01_beet_r2', '02_bit_r1', '06_pot_r1', '06_pot_r2', '08_bought_r1', '10_boot_r1', '10_boot_r2',
         '12_bird_r1', '18_asa_r1', '20_asha_r1', '23_ama_r1', '26_ala_r1', '29_clench_r1', '31_yawn_r1']
LABEL = {'beet': 'beet /i/', 'bit': 'bit /ɪ/', 'pot': 'pot /ɑ/', 'bought': 'bought /ɔ/', 'boot': 'boot /u/', 'bird': 'bird /ɝ/',
         'asa': 'asa /s/', 'asha': 'asha /ʃ/', 'ama': 'ama /m/', 'ala': 'ala /l/', 'clench': 'clench', 'yawn': 'yawn'}
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
tp = jt.Template(g); G = tp.G
F = g.triangles
teeth_faces = F[(G['teeth'] | G['gums'])[F].all(1)]
skin_faces = F[G['skin'][F].all(1)]
R = json.load(open(os.path.join(v2_dir, 'synth_results.json')))

vts = {s: trimesh.load(os.path.join(v2_dir, f'{s}_vt.stl')) for s in ORDER}
centre = np.vstack([m.vertices for m in vts.values()]).mean(0)
Q = 10.0
q = lambda V: b64(np.round((V - centre) * Q).astype(np.int16))
vol_dir = os.path.join(os.path.dirname(html_out), 'vol'); os.makedirs(vol_dir, exist_ok=True)
scans = []
for s in ORDER:
    a = fg.load_volume(os.path.join(recon_dir, f'sub001_3d_{s}_recon.mat'))
    vf = os.path.join(vol_dir, f'{s}.js')
    if not os.path.exists(vf):
        with open(vf, 'w') as fh:
            fh.write(f'window.VOL = window.VOL || {{}}; window.VOL["{s}"] = "{b64((np.clip(a / 1.1, 0, 1) * 255).astype(np.uint8))}";')
    pts, _ = fg.skin_surface(a)
    S_, Rm, t, med = jt.profile_pose(tp, pts, scale=0.932)
    V = jt.teeth_world(tp, S_, Rm, t, 0.0)               # lower arch closed: jaw not tracked yet
    th = trimesh.Trimesh(V, teeth_faces, process=False); th.remove_unreferenced_vertices()
    flags = ['lower teeth shown closed (jaw not tracked yet)'] if 'clench' not in s else []
    bf = os.path.join(os.path.dirname(html_out), '..', 'v4', f'{s}_teeth_blobs.npz')
    if os.path.exists(bf):   # teeth fitted to detected tooth crowns (wisdom teeth removed)
        th = tb.fitted_teeth_mesh(tp, np.load(bf))
        flags = ['teeth fitted to detected tooth crowns (wisdom teeth removed); upper arch only 5/14 teeth matched']
    r = R[s]
    if r['glottis_row'] <= 108:
        flags.append('glottis cut at search-window edge (tract likely too short)')
    if r['path_max'] > 0.75:
        flags.append('closure in the MRI: oral channel forced open')
    scans.append(dict(id=s, label=LABEL[s.split('_')[1]] + (' · r2' if s.endswith('r2') else ''),
                      vtv=q(vts[s].vertices), vtf=b64(vts[s].faces.astype(np.uint16)),
                      thv=q(th.vertices), thf=b64(th.faces.astype(np.uint16)),
                      skv=q(S_ * tp.T @ Rm.T + t),
                      L=r['length_cm'], vol=r['vol_ml'], profile_mm=round(med, 2), flags=flags))
    print(f'{s}: profile fit {med:.2f} mm', flush=True)

data = dict(q=Q, centre=centre.tolist(), vol=dict(ni=160, nj=160, nk=80, vox=1.25),
            skinf=b64(skin_faces.astype(np.uint16)), scans=scans)
tpl = open(os.path.join(os.path.dirname(__file__), 'viewer3_template.html')).read()
open(html_out, 'w').write(tpl.replace('/*__DATA__*/null', json.dumps(data)))
print(f'wrote {html_out} ({os.path.getsize(html_out) / 1e6:.1f} MB)')
