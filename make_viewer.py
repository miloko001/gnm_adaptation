"""Build a self-contained HTML viewer: fitted GNM head + MRI vocal-tract mesh for a set of scans.

Vertices are quantised to int16 at 0.1 mm and embedded as base64 in the page.
Also writes a GLB of the first scan for use in other 3D tools.
"""
import sys, os, csv, json, base64
import numpy as np
import trimesh
from gnm.shape import gnm_numpy
import fit_gnm as fg
import fit_teeth as ft

out_dir, html_out, glb_out = sys.argv[1:4]
SCANS = ['01_beet_r1', '02_bit_r1', '06_pot_r1', '08_bought_r1', '10_boot_r1', '12_bird_r1',
         '18_asa_r1', '20_asha_r1', '23_ama_r1', '26_ala_r1', '29_clench_r1', '31_yawn_r1']
LABELS = {'01_beet_r1': 'beet /i/', '02_bit_r1': 'bit /ɪ/', '06_pot_r1': 'pot /ɑ/', '08_bought_r1': 'bought /ɔ/',
          '10_boot_r1': 'boot /u/', '12_bird_r1': 'bird /ɝ/', '18_asa_r1': 'asa /s/', '20_asha_r1': 'asha /ʃ/',
          '23_ama_r1': 'ama /m/', '26_ala_r1': 'ala /l/', '29_clench_r1': 'clench', '31_yawn_r1': 'yawn'}

g = gnm_numpy.GNM.from_remote(version=gnm_numpy.GNMMajorVersion.V3, variant=gnm_numpy.GNMVariant.HEAD)
G = dict(zip(g.vertex_group_names, g.vertex_groups.astype(bool)))
T0 = g.template_vertex_positions * 1000
F = g.triangles
teeth = np.load(os.path.join(out_dir, 'speaker_teeth.npz'))
tidx = {'upper': np.where(G['teeth'] & G['upper_teeth_and_gums'])[0],
        'lower': np.where(G['teeth'] & G['lower_teeth_and_gums'])[0]}
summary = {r['scan']: r for r in csv.DictReader(open(os.path.join(out_dir, 'summary.csv')))}

# Face partition (each face goes to the first part whose group holds all 3 vertices).
part_groups = [('tongue', G['tongue']), ('mouth', G['mouth_sock']),
               ('teeth', G['teeth']), ('gums', G['gums']), ('eyes', G['eyes']), ('skin', np.ones(len(T0), bool))]
assigned = np.zeros(len(F), bool); parts = {}
for name, m in part_groups:
    sel = m[F].all(1) & ~assigned
    parts[name] = F[sel]; assigned |= sel


def world_verts(fit):
    V = (T0 + np.tensordot(fit['id'], g.vertex_identity_basis[fg.ID_HEAD] * 1000, 1)
         + np.tensordot(fit['ex'], g.expression_basis[fg.EX_LOWER] * 1000, 1)
         + np.tensordot(teeth['coefs'], g.vertex_identity_basis[ft.ID_TEETH] * 1000, 1))
    for k, idx in tidx.items():
        V[idx] = (V[idx] - teeth[k + '_c0']) @ ft.rotvec(teeth[k + '_w']).T + teeth[k + '_c0'] + teeth[k + '_tau']
    return V @ fit['R'].T + fit['t']


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


all_v = []
scans = []
for n in SCANS:
    fit = dict(np.load(os.path.join(out_dir, f'{n}_fit.npz')))
    V = world_verts(fit)
    vt = trimesh.load(os.path.join(out_dir, f'{n}_vt.stl'))
    vt = vt.simplify_quadric_decimation(face_count=12000)
    all_v.append(V); all_v.append(vt.vertices)
    s = summary[n]
    flags = []
    if float(s['path_max']) > 0.75:
        flags.append('closure: oral channel forced open')
    if float(s['glottis_row']) <= 108:
        flags.append('glottis cut at search-window edge')
    scans.append(dict(id=n, label=LABELS[n], V=V, vtV=vt.vertices, vtF=vt.faces,
                      info=dict(vt_ml=s['vt_ml'], oral_ml=s['oral_ml'], pharynx_ml=s['pharynx_ml'],
                                face_mm=s['face_median_mm'], flags=flags)))

centre = np.vstack(all_v).mean(0)
Q = 10.0  # int16 units per mm
data = dict(
    q=Q, parts={k: b64(v.astype(np.uint16)) for k, v in parts.items()},
    scans=[dict(id=s['id'], label=s['label'], info=s['info'],
                v=b64(np.round((s['V'] - centre) * Q).astype(np.int16)),
                vtv=b64(np.round((s['vtV'] - centre) * Q).astype(np.int16)),
                vtf=b64(s['vtF'].astype(np.uint16 if len(s['vtV']) < 65536 else np.uint32)),
                vtf32=bool(len(s['vtV']) >= 65536)) for s in scans])

tpl = open(os.path.join(os.path.dirname(__file__), 'viewer_template.html')).read()
html = tpl.replace('/*__DATA__*/null', json.dumps(data))
open(html_out, 'w').write(html)
print(f'wrote {html_out} ({os.path.getsize(html_out) / 1e6:.1f} MB)')

# GLB of the first scan: skin semi-transparent, teeth, eyes, VT; GNM tongue/mouth sock omitted.
s0 = scans[0]
cols = {'skin': [230, 190, 170, 110], 'teeth': [245, 245, 235, 255], 'gums': [215, 120, 125, 255], 'eyes': [240, 240, 240, 255]}
scene = trimesh.Scene()
for name, c in cols.items():
    m = trimesh.Trimesh(s0['V'] - centre, parts[name], process=False); m.remove_unreferenced_vertices()
    m.visual.face_colors = c; scene.add_geometry(m, geom_name=name)
vtm = trimesh.Trimesh(s0['vtV'] - centre, s0['vtF']); vtm.visual.face_colors = [200, 40, 50, 255]
scene.add_geometry(vtm, geom_name='vocal_tract')
scene.export(glb_out)
print(f'wrote {glb_out} ({os.path.getsize(glb_out) / 1e6:.1f} MB)')
