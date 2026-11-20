"""Self-contained viewer: reconstructed vocal-tract mesh + teeth + MRI mid-sagittal slice, per scan,
with area function and (experimental) synthesised vowel audio."""
import sys, os, io, json, base64
import numpy as np
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fit_gnm as fg

recon_dir, v2_dir, html_out = sys.argv[1:4]
R = json.load(open(os.path.join(v2_dir, 'synth_results.json')))
ORDER = ['01_beet_r1', '01_beet_r2', '02_bit_r1', '06_pot_r1', '06_pot_r2', '08_bought_r1', '10_boot_r1', '10_boot_r2',
         '12_bird_r1', '18_asa_r1', '20_asha_r1', '23_ama_r1', '26_ala_r1', '29_clench_r1', '31_yawn_r1']
LABEL = {'beet': 'beet /i/', 'bit': 'bit /ɪ/', 'pot': 'pot /ɑ/', 'bought': 'bought /ɔ/', 'boot': 'boot /u/', 'bird': 'bird /ɝ/',
         'asa': 'asa /s/', 'asha': 'asha /ʃ/', 'ama': 'ama /m/', 'ala': 'ala /l/', 'clench': 'clench', 'yawn': 'yawn'}
TYPICAL = {'beet': (270, 2290), 'bit': (390, 1990), 'pot': (730, 1090), 'bought': (570, 840), 'boot': (300, 870), 'bird': (490, 1350)}
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()

meshes = {s: (trimesh.load(os.path.join(v2_dir, f'{s}_vt.stl')), trimesh.load(os.path.join(v2_dir, f'{s}_teeth.stl'))) for s in ORDER}
centre = np.vstack([m[0].vertices for m in meshes.values()]).mean(0)
Q = 10.0
scans = []
for s in ORDER:
    vt, th = meshes[s]
    a = fg.load_volume(os.path.join(recon_dir, f'sub001_3d_{s}_recon.mat'))
    vol8 = (np.clip(a / 1.1, 0, 1) * 255).astype(np.uint8)
    os.makedirs(os.path.join(os.path.dirname(html_out), 'vol'), exist_ok=True)
    with open(os.path.join(os.path.dirname(html_out), 'vol', f'{s}.js'), 'w') as fh:
        fh.write(f'window.VOL = window.VOL || {{}}; window.VOL["{s}"] = "{b64(vol8)}";')
    word = s.split('_')[1]
    r = R[s]
    flags = []
    if r['path_max'] > 0.75:
        flags.append('closure in the MRI: oral channel forced open')
    if r['glottis_row'] <= 108:
        flags.append('glottis cut at search-window edge (tract likely too short)')
    if r['min_area_cm2'] <= 0.05:
        flags.append('area function hits the 0.05 cm² floor (artificial constriction)')
    wav = open(os.path.join(v2_dir, f'{s}.wav'), 'rb').read()
    scans.append(dict(
        id=s, label=LABEL[word] + (' · r2' if s.endswith('r2') else ''),
        vtv=b64(np.round((vt.vertices - centre) * Q).astype(np.int16)), vtf=b64(vt.faces.astype(np.uint16)),
        thv=b64(np.round((th.vertices - centre) * Q).astype(np.int16)), thf=b64(th.faces.astype(np.uint16)),
        wav=base64.b64encode(wav).decode(),
        x=r['x_cm'], A=r['A_cm2'], F=r['formants_hz'][:3], L=r['length_cm'], vol=r['vol_ml'],
        typical=TYPICAL.get(word), flags=flags))

# MRI slice plane geometry (world mm): rows i -> y = -1.25 i, cols j -> z = -1.25 j, mid-sagittal x = 0.625
data = dict(q=Q, centre=centre.tolist(), vol=dict(ni=160, nj=160, nk=80, vox=1.25), scans=scans)
tpl = open(os.path.join(os.path.dirname(__file__), 'viewer2_template.html')).read()
open(html_out, 'w').write(tpl.replace('/*__DATA__*/null', json.dumps(data)))
print(f'wrote {html_out} ({os.path.getsize(html_out) / 1e6:.1f} MB)')
