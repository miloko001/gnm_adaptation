#!/usr/bin/env bash
# Set up the pipeline on a new machine: Python venv, Google GNM source, GNM Head model.
#   ./setup.sh                 # fetch GNM from GitHub; model downloads on first use
#   ./setup.sh <bundle_dir>    # use the GNM source + model from an unpacked bundle (offline)
set -euo pipefail
cd "$(dirname "$0")"
BUNDLE="${1:-}"

if [[ ! -d third_party/gnm ]]; then
  mkdir -p third_party
  if [[ -n "$BUNDLE" && -d "$BUNDLE/third_party/gnm" ]]; then
    cp -R "$BUNDLE/third_party/gnm" third_party/
  else
    git clone --depth 1 https://github.com/google/GNM.git third_party/gnm
  fi
fi

if [[ -n "$BUNDLE" && -d "$BUNDLE/gnm_cache" ]]; then
  mkdir -p ~/.cache/gnm
  cp -R "$BUNDLE/gnm_cache/"* ~/.cache/gnm/
fi

python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt
SITE=$(./.venv/bin/python -c 'import site; print(site.getsitepackages()[0])')
echo "$PWD/third_party/gnm" > "$SITE/gnm_src.pth"

./.venv/bin/python -c "from gnm.shape import gnm_numpy as g; m = g.GNM.from_remote(version=g.GNMMajorVersion.V3, variant=g.GNMVariant.HEAD); print('GNM Head OK:', m.template_vertex_positions.shape)"
echo "Done. Data is not included: point the scripts at your copy of the USC 75-speaker data (e.g. ~/phd/data/sub001/3d/recon)."
