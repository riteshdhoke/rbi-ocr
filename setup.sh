#!/usr/bin/env bash
# setup.sh — stand the project up from a fresh checkout or a bundle extract.
#
# Runs, in order, the steps docs/AIR-GAPPED-DEPLOY.md sections 1.3, 3.0, 3a/3b
# and 3b.4 describe, and stops at the first one that fails with the section to
# read.  Nothing here is new: it is the guide, executed.
#
#   ./setup.sh                 # online install, then verify
#   ./setup.sh --offline       # install from ./wheelhouse, no network
#   ./setup.sh --with-models   # also fetch model weights (needs internet)
#   ./setup.sh --run-tests     # also run the full suite at the end
#
# Safe to re-run: an existing .venv is reused, not rebuilt.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

OFFLINE=0
WITH_MODELS=0
RUN_TESTS=0
for arg in "$@"; do
  case "$arg" in
    --offline)     OFFLINE=1 ;;
    --with-models) WITH_MODELS=1 ;;
    --run-tests)   RUN_TESTS=1 ;;
    -h|--help)     sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

PY=.venv/bin/python
step()  { printf '\n\033[1m==> %s\033[0m\n' "$1"; }
ok()    { printf '    ok: %s\n' "$1"; }
die()   { printf '\n\033[31mFAILED: %s\033[0m\n' "$1" >&2; exit 1; }

# --------------------------------------------------------------------------- #
step "1/6  Python 3.12 (deploy guide 1.3)"
# --------------------------------------------------------------------------- #
INTERPRETER=""
for candidate in python3.12 python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    version="$("$candidate" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo none)"
    if [ "$version" = "3.12" ]; then INTERPRETER="$candidate"; break; fi
  fi
done
[ -n "$INTERPRETER" ] || die "no Python 3.12 on PATH. The project pins 3.12; see deploy guide 1.3 for install routes.
  On Debian/Ubuntu: sudo apt install python3.12 python3.12-venv"
ok "$INTERPRETER -> $("$INTERPRETER" --version)"

# --------------------------------------------------------------------------- #
step "2/6  Virtual environment (deploy guide 3.0)"
# --------------------------------------------------------------------------- #
if [ -x "$PY" ]; then
  ok "reusing existing .venv ($($PY --version))"
else
  "$INTERPRETER" -m venv .venv \
    || die "venv creation failed. On Debian/Ubuntu: sudo apt install python3.12-venv"
  ok "created .venv ($($PY --version))"
fi
# Never `activate`: it silently stops working in service wrappers and CI, and
# every failure it causes looks like a different bug.  Full path, always.

# --------------------------------------------------------------------------- #
step "3/6  Dependencies (deploy guide 3a online / 3b offline)"
# --------------------------------------------------------------------------- #
REQS=(-r requirements.txt -r requirements-dev.txt -c constraints.txt)
if [ "$OFFLINE" = "1" ]; then
  [ -d wheelhouse ] || die "--offline needs a ./wheelhouse folder. Build it on a connected
  machine of the SAME OS and architecture (deploy guide 3b.1):
    python3.12 -m pip download ${REQS[*]} -d wheelhouse"
  count=$(find wheelhouse -name '*.whl' | wc -l | tr -d ' ')
  ok "wheelhouse has $count wheels"
  "$PY" -m pip install --quiet --upgrade --no-index --find-links wheelhouse pip \
    || ok "pip not upgraded (not in the wheelhouse) — continuing"
  "$PY" -m pip install --no-index --find-links wheelhouse "${REQS[@]}" \
    || die "offline install failed. A wheel for this platform is missing from ./wheelhouse
  — rebuild it on a machine with the same OS, architecture and Python version."
else
  "$PY" -m pip install --quiet --upgrade pip
  "$PY" -m pip install "${REQS[@]}" \
    || die "install failed. Behind a corporate proxy, set HTTPS_PROXY (deploy guide 3a);
  with no network at all, build a wheelhouse and re-run with --offline (3b)."
fi
ok "dependencies installed"

# --------------------------------------------------------------------------- #
step "4/6  Verify the install (deploy guide 3b.4)"
# --------------------------------------------------------------------------- #
"$PY" -m pip check || die "pip reports inconsistent dependencies (see above)"
ok "pip check: no broken requirements"
"$PY" - <<'PYCODE' || die "a heavy library failed to import (see the traceback above)"
import app  # applies the thread caps before any numeric import (invariant 5)
import cv2, ctranslate2, onnxruntime, pypdfium2
print(f"    ok: imports OK  onnxruntime {onnxruntime.__version__}  "
      f"ctranslate2 {ctranslate2.__version__}  cv2 {cv2.__version__}")
PYCODE

# --------------------------------------------------------------------------- #
step "5/6  Configuration and models (deploy guide 5 and 7)"
# --------------------------------------------------------------------------- #
if [ -f .env ]; then
  ok ".env already present — leaving it alone"
else
  cp .env.example .env
  ok "created .env from .env.example — edit PROVIDER_PROFILE before running for real"
fi

if [ "$WITH_MODELS" = "1" ]; then
  "$PY" tools/fetch_models.py || die "model download failed (deploy guide 5.2)"
fi

if [ -f models/ocr/det.onnx ]; then
  n=$(find models -type f | wc -l | tr -d ' ')
  ok "models/ has $n files"
else
  printf '    NOTE: models/ has no weights yet. Either unzip bundle 2, or run:\n'
  printf '          %s tools/fetch_models.py    (needs internet, ~837 MB)\n' "$PY"
fi

# --------------------------------------------------------------------------- #
step "6/6  Smoke test"
# --------------------------------------------------------------------------- #
if [ "$RUN_TESTS" = "1" ]; then
  "$PY" -m pytest -q || die "the test suite failed"
else
  "$PY" -m pytest -q tests/unit/test_settings.py \
    || die "the settings tests failed — the install is not sound"
  ok "settings tests pass (run ./setup.sh --run-tests for the full ~2700)"
fi

cat <<EOF

$(printf '\033[32mSETUP OK\033[0m')

Next:
  Start the service     .venv/bin/python -m uvicorn app.api.app:create_app --factory --port 8000
  Then open             http://127.0.0.1:8000/ui/test_page.html
  Health / error rates  curl localhost:8000/healthz   curl localhost:8000/v1/stats
  Check a GPU box       .venv/bin/python tools/dryrun_gpu.py

Before processing anything real, set PROVIDER_PROFILE in .env:
  laptop    development only, synthetic data only (enforced)
  cpu_node  air-gapped production, fully offline
  gpu_node  air-gapped production on CUDA
EOF
