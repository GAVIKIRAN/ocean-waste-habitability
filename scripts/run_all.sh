#!/usr/bin/env bash
# End-to-end pipeline: data -> train -> evaluate -> tests -> launch app.
# Usage: bash scripts/run_all.sh [demo|full] [--low-vram] [--no-app]
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-demo}"; shift || true
LOWVRAM=""; LAUNCH=1
for a in "$@"; do
  case "$a" in
    --low-vram) LOWVRAM="--low-vram" ;;
    --no-app) LAUNCH=0 ;;
  esac
done
if [ -x ".venv/bin/python" ]; then PY=".venv/bin/python"; elif [ -x ".venv/Scripts/python.exe" ]; then PY=".venv/Scripts/python.exe"; else PY="python"; fi

echo "== [1/7] GPU check";            "$PY" scripts/check_gpu.py || echo "(continuing on CPU)"
echo "== [2/7] Datasets";             "$PY" -m src.data.download_data
echo "== [3/7] YOLO dataset";         "$PY" -m src.data.prepare_yolo_dataset
echo "== [4/7] Water model ($MODE)";  "$PY" -m src.water.train --mode "$MODE" && "$PY" -m src.water.evaluate
echo "== [5/7] Waste model ($MODE)";  "$PY" -m src.waste.train --mode "$MODE" $LOWVRAM && "$PY" -m src.waste.evaluate
echo "== [6/7] Tests";                "$PY" -m pytest -q
if [ "$LAUNCH" = "1" ]; then
  echo "== [7/7] Launching Streamlit on http://localhost:8501"
  "$PY" -m streamlit run app/streamlit_app.py
fi
