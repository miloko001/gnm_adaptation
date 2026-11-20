# Vocal tract + GNM head/teeth pipeline (sub001)

Data: `~/phd/data/sub001` — USC 75-speaker speech MRI (paper: `2102.07896v1.pdf` in that folder).
3D volumes 160×160×80, **1.25 mm isotropic** (confirmed by the paper and by registration to the T2w DICOMs).
sub001: F, 24, L1 Mandarin (`metafile_public_20210129.json`).

## Status (end of session, 2026-11-20)
- **Head fit — done, confirmed good by user.** `head_mesh_fit.py`: MRI skin mesh → GNM template at scale
  0.932 (from the mid-sagittal face profile) + rigid + head identity (170) + lower-face expression (150),
  λ = 0.15, bidirectional point-to-plane. Fitted for 15 scans in `out/v5/`. Viewer: `out/viewer_head/index.html`.
  Face/jaw within ~1–2 mm; poor areas are the eye region, slab edges (temples) and lower neck.
- **Teeth — paused.** GNM template teeth (wisdom teeth removed, option chosen by user) are oriented roughly right,
  but front–back placement is unresolved: blob/intensity fits drift back (or onto the mandible cortex);
  the head-fit pose puts the incisors ~6 mm too far forward. Idea for next time: sliders in the viewer so the
  user aligns the arches by eye on clench, then reuse those offsets.
- **Airway segmentation — paused.** Leaks into spine and buccal/sublingual air; glottis cut unreliable.
- **Synthesis — shelved** until the airway is right (earlier results were not meaningful).

## Consistent internal geometry across scans
- `consensus.py`: aligns all 66 scans to 29_clench_r1 (upper head only; alignments cached in `out/consensus/reg/`),
  thresholds each at tissue levels 0.20–0.50 (`tissue_mesh.voids`, box unchanged), averages → `out/consensus/consensus.npz`.
  Restart / resume (skips cached alignments): `nohup ./.venv/bin/python consensus.py ~/phd/data/sub001/3d/recon out/consensus > out/consensus_log.txt 2>&1 &`
- Then build the viewer: `./.venv/bin/python make_viewer_consensus.py out/consensus/consensus.npz out/v5 out/viewer_consensus/index.html`
  (sliders: lowest/highest tissue threshold, agreement across scans, left–right symmetry; surface built live in the page by
  `surface_nets.js`; mirror plane in `out/consensus/mirror_params.npy`). Done for all 66 scans; viewer built.

## Setup (new computer)
```
git clone https://github.com/miloko001/gnm_adaptation.git vt_pipeline && cd vt_pipeline
./setup.sh                    # venv + Google GNM source + model (downloads)
./setup.sh /path/to/bundle    # or: offline, using the GNM source/model from the bundle
```
`requirements-lock.txt` pins the exact versions used (Python 3.14, macOS arm64); `requirements.txt` is the loose list.
GNM Head model cache: `~/.cache/gnm/models/v3_0/gnm_head.npz`.

**Not in this repo** (participant data, kept out of git on purpose): the MRI data (USC 75-speaker dataset; scripts take the
`recon` folder as an argument, e.g. `~/phd/data/sub001/3d/recon`) and everything in `out/` (fits, meshes, viewers with
embedded MRI). Those travel in the bundle: unpack it and copy its `out/` into the repo to keep the cached alignments/fits.

## Main scripts
| Script | Purpose |
|---|---|
| `head_mesh_fit.py` | **Current head fit** (`mri_head_mesh`, `fit_head_mesh`). `python head_mesh_fit.py <recon_dir> out/v5 <scan> 0.15` |
| `tissue_mesh.py` | Inner surface of the solid tissue (threshold 0.25, like the skin outline) around the tract → `out/v5/<scan>_tissue.stl`. The head viewer precomputes it at thresholds 0.10–0.80 (`out/viewer_head/mesh/`, ~2.7 MB per scan, cached; delete a file to regenerate) with a slider |
| `consensus.py`, `make_viewer_consensus.py` + `consensus_template.html`, `surface_nets.js` | Cross-scan consistency structure and live-slider viewer (see above) |
| `filters.py` | Denoising filters to compare before thresholding (draft, untested) |
| `air_mesh.py` | Earlier: thresholded air inside the head (narrow parts close up) → `out/v5/<scan>_air.stl` |
| `make_viewer4.py` + `viewer4_template.html` | **Head-fit viewer** → `out/viewer_head/` (keep `vol/` and `mesh/` beside `index.html`); shows `_tissue.stl` as an orange wireframe if present |
| `fit_gnm.py` | MRI skin surface / head mask helpers; older one-directional GNM fit |
| `jaw_teeth.py` | `Template`, `profile_pose` (scale/pose from mid-sagittal upper-face profile), `umeyama`, hinge jaw (not working) |
| `teeth_blobs.py` | Wisdom-free template teeth, 3-D dark-blob tooth detection, ICP / grid search |
| `fit_teeth_circles.py`, `show_teeth_rigid.py` | Intensity-contrast teeth fit (drifted to the mandible) + overlays |
| `fit_teeth.py` | Older GNM teeth-coefficient fit |
| `segment_vt.py`, `vt_synth.py`, `batch_speaker.py` | Airway segmentation, area function/synthesis, old batch (paused / obsolete) |
| `make_viewer3.py` + `viewer3_template.html` | Airway + teeth viewer → `out/viewer/` |
| `overlay.py`, `show_mask.py`, `show_teeth.py` | Diagnostic slice overlays |

Rebuild the head-fit viewer:
```
./.venv/bin/python make_viewer4.py ~/phd/data/sub001/3d/recon out/v5 out/viewer_head/index.html
```

## Outputs
- `out/v5/` — **head fits**: `<scan>_head_fit.npz` (id, ex, s, R, t), `<scan>_gnm_head.obj`, `<scan>_mri_head.stl`; clench teeth trials
- `out/viewer_head/` — head-fit viewer (GNM coloured by distance to MRI skin, MRI wireframe, full MRI volume planes)
- `out/v4/` — teeth blob/circle fitting trials and diagnostic images (clench)
- `out/viewer/` — airway + teeth viewer (older approach)
- `out/v2/`, `out/v3/` — airway meshes/area functions; profile poses and early teeth meshes
- `out/early/` — first-day exploratory outputs and diagnostic images
- `out/sub001/` — obsolete (over-fitted head era, 233 MB; safe to delete)

## Next steps (suggested)
1. Run the head fit on all 66 scans; optionally one shared identity per speaker, expression per scan.
2. Teeth on top of the head fit (manual alignment sliders on clench, then reuse; lower arch follows the jaw).
3. Airway segmentation using the improved head/teeth for exclusion masks.
4. MRI left/right handedness still unverified (T2w DICOM headers could settle it).
