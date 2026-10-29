<#
.SYNOPSIS
    Stand the project up from a fresh checkout or a bundle extract.

.DESCRIPTION
    Runs, in order, the steps docs/AIR-GAPPED-DEPLOY.md sections 1.3, 3.0,
    3a/3b and 3b.4 describe, and stops at the first one that fails with the
    section to read. Nothing here is new: it is the guide, executed.

    Safe to re-run: an existing .venv is reused, not rebuilt.

.EXAMPLE
    .\setup.ps1
    Online install, then verify.

.EXAMPLE
    .\setup.ps1 -Offline
    Install from .\wheelhouse with the network forbidden.

.EXAMPLE
    .\setup.ps1 -WithModels -RunTests
    Also download the weights, and run the full suite at the end.

.NOTES
    If PowerShell refuses to run this ("running scripts is disabled"), start it
    with:  powershell -ExecutionPolicy Bypass -File .\setup.ps1
    That applies to this one invocation and changes nothing system-wide.
#>
[CmdletBinding()]
param(
    [switch]$Offline,
    [switch]$WithModels,
    [switch]$RunTests
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

$Py = '.venv\Scripts\python.exe'

function Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Ok($text)   { Write-Host "    ok: $text" }
function Die($text)  { Write-Host "`nFAILED: $text" -ForegroundColor Red; exit 1 }

# --------------------------------------------------------------------------- #
Step '1/6  Python 3.12 (deploy guide 1.3)'
# --------------------------------------------------------------------------- #
$interpreter = $null
# `py -3.12` is the reliable selector on Windows; a bare `python` may be a
# Store alias that does nothing useful, so it is tried last.
#
# Version comes from `--version`, not from `-c "print(...)"`. PowerShell
# re-parses arguments on their way to a native executable and eats embedded
# double quotes, so a -c snippet containing a quoted format string arrives at
# Python as a syntax error. Passing no quotes at all sidesteps the whole class
# of problem; the same reason the import probe further down goes in over stdin.
foreach ($candidate in @(@('py', '-3.12'), @('python3.12'), @('python'))) {
    $exe = $candidate[0]
    $prefix = @($candidate | Select-Object -Skip 1)
    if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
    try {
        $reported = & $exe @prefix --version
    } catch { continue }
    if ("$reported" -match 'Python\s+3\.12\.') { $interpreter = @($exe) + $prefix; break }
}
if (-not $interpreter) {
    Die @'
no Python 3.12 on PATH. The project pins 3.12; see deploy guide 1.3.
  Try:  winget install Python.Python.3.12
  In a restricted organisation, ask for "CPython 3.12, 64-bit" from your
  software catalogue first.
'@
}
$exe = $interpreter[0]
$prefix = @($interpreter | Select-Object -Skip 1)
Ok "$($interpreter -join ' ') -> $(& $exe @prefix --version)"

# --------------------------------------------------------------------------- #
Step '2/6  Virtual environment (deploy guide 3.0)'
# --------------------------------------------------------------------------- #
if (Test-Path $Py) {
    Ok "reusing existing .venv ($(& $Py --version))"
} else {
    & $exe @prefix -m venv .venv
    if (-not $?) { Die 'venv creation failed.' }
    Ok "created .venv ($(& $Py --version))"
}
# Never `activate`: it silently stops working in scheduled tasks and service
# wrappers, and every failure it causes looks like a different bug.

# --------------------------------------------------------------------------- #
Step '3/6  Dependencies (deploy guide 3a online / 3b offline)'
# --------------------------------------------------------------------------- #
$reqs = @('-r', 'requirements.txt', '-r', 'requirements-dev.txt', '-c', 'constraints.txt')
if ($Offline) {
    if (-not (Test-Path 'wheelhouse')) {
        Die @'
-Offline needs a .\wheelhouse folder. Build it on a connected machine of the
  SAME OS and architecture (deploy guide 3b.1):
    py -3.12 -m pip download -r requirements.txt -r requirements-dev.txt -c constraints.txt -d wheelhouse
'@
    }
    $count = (Get-ChildItem wheelhouse -Filter *.whl).Count
    Ok "wheelhouse has $count wheels"
    & $Py -m pip install --quiet --upgrade --no-index --find-links wheelhouse pip
    if (-not $?) { Ok 'pip not upgraded (not in the wheelhouse) - continuing' }
    & $Py -m pip install --no-index --find-links wheelhouse @reqs
    if (-not $?) {
        Die @'
offline install failed. A wheel for this platform is missing from .\wheelhouse
  - rebuild it on a machine with the same OS, architecture and Python version.
'@
    }
} else {
    & $Py -m pip install --quiet --upgrade pip
    & $Py -m pip install @reqs
    if (-not $?) {
        Die @'
install failed. Behind a corporate proxy, set HTTPS_PROXY (deploy guide 3a);
  with no network at all, build a wheelhouse and re-run with -Offline (3b).
'@
    }
}
Ok 'dependencies installed'

# --------------------------------------------------------------------------- #
Step '4/6  Verify the install (deploy guide 3b.4)'
# --------------------------------------------------------------------------- #
& $Py -m pip check
if (-not $?) { Die 'pip reports inconsistent dependencies (see above)' }
Ok 'pip check: no broken requirements'

$probe = @'
import app  # applies the thread caps before any numeric import (invariant 5)
import cv2, ctranslate2, onnxruntime, pypdfium2
print(f"    ok: imports OK  onnxruntime {onnxruntime.__version__}  "
      f"ctranslate2 {ctranslate2.__version__}  cv2 {cv2.__version__}")
'@
$probe | & $Py -
if (-not $?) { Die 'a heavy library failed to import (see the traceback above)' }

# --------------------------------------------------------------------------- #
Step '5/6  Configuration and models (deploy guide 5 and 7)'
# --------------------------------------------------------------------------- #
if (Test-Path '.env') {
    Ok '.env already present - leaving it alone'
} else {
    Copy-Item '.env.example' '.env'
    Ok 'created .env from .env.example - edit PROVIDER_PROFILE before running for real'
}

if ($WithModels) {
    & $Py tools\fetch_models.py
    if (-not $?) { Die 'model download failed (deploy guide 5.2)' }
}

if (Test-Path 'models\ocr\det.onnx') {
    $n = (Get-ChildItem models -Recurse -File).Count
    Ok "models\ has $n files"
} else {
    Write-Host '    NOTE: models\ has no weights yet. Either unzip bundle 2, or run:'
    Write-Host "          $Py tools\fetch_models.py    (needs internet, ~837 MB)"
}

# --------------------------------------------------------------------------- #
Step '6/6  Smoke test'
# --------------------------------------------------------------------------- #
if ($RunTests) {
    & $Py -m pytest -q
    if (-not $?) { Die 'the test suite failed' }
} else {
    & $Py -m pytest -q tests\unit\test_settings.py
    if (-not $?) { Die 'the settings tests failed - the install is not sound' }
    Ok 'settings tests pass (run .\setup.ps1 -RunTests for the full ~2700)'
}

Write-Host ''
Write-Host 'SETUP OK' -ForegroundColor Green
Write-Host @"

Next:
  Start the service     .venv\Scripts\python.exe -m uvicorn app.api.app:create_app --factory --port 8000
  Then open             http://127.0.0.1:8000/ui/test_page.html
  Health / error rates  curl localhost:8000/healthz    curl localhost:8000/v1/stats
  Check a GPU box       .venv\Scripts\python.exe tools\dryrun_gpu.py

Before processing anything real, set PROVIDER_PROFILE in .env:
  laptop    development only, synthetic data only (enforced)
  cpu_node  air-gapped production, fully offline
  gpu_node  air-gapped production on CUDA
"@
