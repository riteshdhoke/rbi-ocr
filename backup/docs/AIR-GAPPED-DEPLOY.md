# Air-gapped deployment guide

> Extraction update: open field discovery is now enabled by default. Offline
> results can include additional labeled fields beyond the six-field examples
> below. See [Open field extraction](OPEN-FIELD-EXTRACTION.md) for behavior,
> provenance, configuration, architecture and limits.


**Audience: a developer who has never seen this project.** You have been handed two zip
files and asked to get a document-processing service running on a machine that has no
internet connection. This document is the whole path, from a blank machine to a working
API, in order, with every command written out.

**Time budget:** about 90 minutes on a machine that already has Python, plus however long
it takes your organisation to move ~900 MB of model files onto the air-gapped node.

**Vocabulary, once, in plain language** — every term below is used repeatedly later:

| Term | What it means here |
|---|---|
| **air-gapped** | The machine has no route to the internet. Nothing may be downloaded on it, ever, at install time or at run time. |
| **venv** (virtual environment) | A private folder holding one Python interpreter and the libraries this project needs, so it cannot collide with anything else on the machine. |
| **wheel** (`.whl` file) | A pre-built Python package. Installing from a wheel needs no compiler. |
| **wheelhouse** | A folder full of wheels that you carry to the air-gapped machine so `pip` never needs the network. |
| **OCR** | Optical Character Recognition — turning a picture of text into text. |
| **model / weights / artifact** | A binary file a machine-learning component loads at start-up. This project needs five of them, plus two small character-list files and one coefficient file — thirteen files in all. They are **not** in git; you place them by hand. |
| **envelope** | The JSON result this service returns for one document. Described in `docs/API.md`; its formal contract is `docs/schema/result_envelope.schema.json`. |
| **corpus** | 295 synthetic (computer-generated, fake) test documents used to prove the install works. It contains no real complainant data. |
| **profile** | Which of three deployment shapes you are running: `laptop`, `cpu_node`, `gpu_node`. See section 7. |
| **SLM / VLM** | Small Language Model / Vision Language Model. Optional components that read free text. The service works without them; see section 7. |
| **HF** | Hugging Face — the public site the models were originally downloaded from, on a *connected* machine. |

**One rule that overrides everything else in this document:** on the air-gapped node the
service must never reach the network. If a step here appears to require a download on that
machine, you are on the wrong step — every download happens on a *connected* staging
machine and is carried across.

**Which shell to use.** On Windows, run everything in **PowerShell**, not the legacy
Command Prompt (`cmd.exe`). Several verification commands below are multi-line
`python -c "..."` blocks; PowerShell and any POSIX shell handle them, `cmd.exe` does not.
Every multi-line block in this document was executed in PowerShell before being written
here. On Linux, any shell is fine.

---

## Contents

0. [What you received](#0-what-you-received)
1. [Machine prerequisites](#1-machine-prerequisites)
2. [Unzip layout](#2-unzip-layout)
3. [Create the virtual environment and install dependencies](#3-create-the-virtual-environment-and-install-dependencies)
4. [Verify fonts and text shaping](#4-verify-fonts-and-text-shaping)
5. [Place the model files](#5-place-the-model-files)
6. [Corpus: unzip or regenerate](#6-corpus-unzip-or-regenerate)
7. [Create the `.env` file](#7-create-the-env-file)
8. [Smoke tests](#8-smoke-tests)
9. [Start the service](#9-start-the-service)
10. [Verification checklist](#10-verification-checklist)
11. [Troubleshooting appendix](#11-troubleshooting-appendix)

---

## 0. What you received

Two zip files. Both are stamped with the git commit they were built from, so
`rbi-ocr-code-<sha>.zip` and `rbi-ocr-models-corpus-<sha>.zip` must carry the **same**
`<sha>`. If they do not, stop and ask for a matching pair — the corpus and the code that
scores it are versioned together, and so are the weights and the code that loads them.

**Bundle 2 now carries the model weights as well as the corpus**, so section 5 is a copy
rather than a download.

### What is *not* in either bundle, and what you do about it

| Missing | Why | What to do |
|---|---|---|
| **Python 3.12** | An operating-system component, not ours to ship. | Section 1.3 — check first, and it lists the routes if your organisation blocks `python.org`. |
| **A wheelhouse** (the Python dependencies, ~400–600 MB) | Wheels are specific to the OS, CPU architecture and Python version of the machine that runs them. A wheelhouse built for Windows will not install on Linux, so it cannot be baked into a bundle that does not know its destination. | Section 3b — one `pip download` on a connected *staging* machine of the **same** OS and architecture, then carry the folder across. |

Everything else — code, docs, tests, corpus and every model weight — is in the two zips.
If the deployment machine reaches the internet through a corporate proxy, you do not need
a staging machine at all; use section 3a.

### Bundle 1 — code (`rbi-ocr-code-<sha>.zip`, small — 14 MB of files, less when zipped)

| Contains | Why you need it |
|---|---|
| `app/` | The service itself. |
| `tools/` | Command-line utilities: model downloader, corpus generator, evaluation harness, benchmarks. |
| `tools/assets/fonts/` + `tools/assets/fribidi-0.dll` | Fonts and a text-shaping library used **only** by the corpus generator. See section 4. |
| `tests/` | About 2 600 automated tests. Running them is smoke test 1. |
| `docs/` | This file, `docs/API.md`, `docs/TUNING.md`, `docs/FUNCTIONAL-SPEC.md`, the evaluation report, the JSON schema, and `docs/calibration/` — which holds `fusion.json` (you copy it into `models/` in section 5) and `quality_gate.json` (the reference the section 8.3 check compares against). |
| `design/` | The original specification documents. Background reading, not needed to deploy. |
| `requirements.txt`, `requirements-dev.txt`, `constraints.txt` | The dependency list and the exact versions. Section 3. |
| `.env.example` | Template for configuration. Section 7. |
| `alembic/`, `docker-compose.yml` | Database migrations and an optional container topology. Not needed for the single-node install. |

**Not in bundle 1, deliberately:** `models/` (bundle 2), `.venv/` (you build it —
section 3), corpus images (bundle 2), and `.env` (it may contain a secret; the bundle
builder asserts it is absent).

### Bundle 2 — models + corpus (`rbi-ocr-models-corpus-<sha>.zip`, ~600 MB zipped)

Everything this machine cannot produce or fetch for itself. 614 files, ~904 MB unzipped.

| Contains | Why you need it |
|---|---|
| `models/` | Every weight the service loads: the two OCR recognition heads and the detector (`models/ocr/`), the language identifier (`models/lid/`), the translation engine (`models/mt/indic-en/`, 808 MB of it), the classifier fusion coefficients and `PROVENANCE.json`. 15 files, 837 MB. |
| `corpus/` | 295 generated test documents (`.jpg`/`.png`), one ground-truth sidecar (`.gt.json`) each, an `index.json` describing the set, and an `adversarial/` folder of seven deliberately broken files (plus their `MANIFEST.json`) used to prove the service refuses bad input safely. 599 files. |

The weights are **gitignored**, so no commit lists them and there is no git object id to
check them against. Their provenance is instead the per-file SHA-256 in
`BUNDLE_MANIFEST-assets.json`, which is the guarantee that actually matters here: it says
the bytes you unzip are the bytes that were measured. Section 5.4 checks them.

You can skip the corpus half and regenerate it from bundle 1 instead — section 6 gives
both routes. You cannot skip the weights: an air-gapped machine has no way to fetch them.

> **If `tools/make_bundle.py` is not in your copy of bundle 1**, the two zips were produced
> by hand or by `git archive`. Nothing in this guide changes; the layouts above are what
> the bundles are specified to contain (`PLAN.md`, WS-I item 7).

---

## 1. Machine prerequisites

### 1.1 Operating system

Windows 10/11 or a 64-bit Linux (x86-64). Everything below gives the Windows command first
and the Linux equivalent second.

> **Honest scope note.** Every expected output and every timing in this document was
> produced on **Windows 11 / CPython 3.12.4**. Linux is a supported target — the code is
> path-portable by design and the dependency set has Linux wheels — but the runs behind
> these numbers were all on Windows. Treat the Linux commands as correct and the Linux
> *outputs* as expected-by-translation. The two places the platforms genuinely differ are
> called out where they occur: text shaping (§4) and OpenCV's system libraries (§3b.4).

### 1.2 CPU, memory, disk

| Resource | Minimum | Comfortable |
|---|---|---|
| CPU | 4 physical cores, x86-64 | 16 vCPU (the target production node) |
| RAM | 8 GB | 16 GB |
| Free disk | 4 GB | 8 GB |

No GPU is required. One is used only when you ask for it: the `cpu_node` profile is
CPU-only by construction, and the `gpu_node` profile puts detection, recognition and
translation on CUDA. See section 7.3.

Disk breakdown, measured on the build machine: virtual environment **732 MB**, models
**837 MB**, corpus **67 MB**, source and docs **14 MB**, plus a wheelhouse of roughly
400–600 MB if you take the offline route (deletable after installing).

### 1.3 Python 3.12 — check what you have

**Windows (PowerShell or Command Prompt):**

```
py -3.12 --version
```

Expected output:

```
Python 3.12.4
```

Any `3.12.x` is fine. If `py` is not recognised, try:

```
python --version
```

**Linux:**

```
python3.12 --version
```

Expected output:

```
Python 3.12.4
```

**If you do not have Python 3.12**, and you are inside a restricted organisation where
`python.org` is blocked, these are the routes, in the order to try them:

1. **Your organisation's software catalogue / Software Center / package manager.** Most
   enterprises publish an approved Python build. Ask for "CPython 3.12, 64-bit".
2. **Windows Store** (`winget install Python.Python.3.12` or search "Python 3.12" in the
   Store) — often permitted where direct downloads are not.
3. **Linux distribution packages:** `sudo apt install python3.12 python3.12-venv`
   (Debian/Ubuntu) or `sudo dnf install python3.12` (RHEL/Fedora). On Debian/Ubuntu the
   `-venv` package is separate and you **will** need it.
4. **Ask your platform team for an offline installer.** The Windows installer is a single
   `.exe`; the same "carry it across" logic as the wheelhouse applies.

> **Why 3.12 specifically?** The project is pinned to it (`PLAN.md` decision D5), the whole
> test suite and every published measurement come from 3.12.4, and `constraints.txt` pins
> versions that were resolved for `cp312`. Neither 3.11 nor 3.13 has been tried here, and a
> compiled dependency without a wheel for your interpreter fails at install time with a
> confusing compiler error rather than a clear message. Do not substitute.

**Troubleshooting — `py -3.12` says "can't find"**: Python is installed but not registered
with the launcher. Find the interpreter directly:

```
where python
```

and use that full path everywhere this guide says `py -3.12`.

### 1.4 Do you need internet at all?

Yes — on a **different, connected machine** (called the *staging machine* from here on),
once, to collect the Python **wheels** (section 3b). The air-gapped node itself never
needs it.

The models no longer require the staging machine: bundle 2 carries them (section 5). The
wheels are the one remaining thing that is not in a bundle, because they are specific to
the operating system, CPU architecture and Python version of the machine that will run
them — a wheelhouse built for Windows will not install on Linux, so it cannot be baked
into a bundle that does not know where it is going.

If the deployment machine *does* have internet through a corporate proxy, you can skip the
staging machine and use the online route in section 3a instead.

---

## 2. Unzip layout

Pick a directory with no spaces in its path. Spaces work, but every later command would
need quoting and mistakes are easy.

**Windows:**

```
mkdir C:\rbi-ocr
cd C:\rbi-ocr
tar -xf rbi-ocr-code-<sha>.zip
tar -xf rbi-ocr-models-corpus-<sha>.zip
```

(`tar` ships with Windows 10 1803 and later and handles `.zip`. If it is missing, use
Explorer: right-click → Extract All.)

**Linux:**

```
mkdir -p ~/rbi-ocr
cd ~/rbi-ocr
unzip rbi-ocr-code-<sha>.zip
unzip rbi-ocr-models-corpus-<sha>.zip
```

After both unzips, the directory must look like this. This is the **project root**, and
every command in this document is run from here:

```
C:\rbi-ocr\                      <- project root; cd here for every command below
├── app/                         service code
├── alembic/                     database migrations
├── corpus/                      295 test documents      (bundle 2)
│   ├── adversarial/             8 deliberately broken files
│   ├── complaint_form_en/       ... one folder per document type
│   └── index.json
├── design/                      specifications (background reading)
├── docs/
│   ├── AIR-GAPPED-DEPLOY.md     this file
│   ├── API.md                   the consumer-facing API reference
│   ├── TUNING.md                every tunable knob and its remedy playbook
│   ├── calibration/             fusion.json + quality_gate.json  (needed in section 5)
│   ├── eval/                    the measured evaluation report
│   └── schema/                  result_envelope.schema.json
├── models/                      15 weight files from bundle 2  (verify in section 5.4)
├── tests/
├── tools/
│   └── assets/                  fonts + fribidi-0.dll
├── constraints.txt
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── docker-compose.yml
```

Confirm it:

**Windows:**

```
dir
```

**Linux:**

```
ls -1
```

You should see `app`, `corpus`, `docs`, `models`, `requirements.txt` in the listing. If you
see a single folder named `rbi-ocr-code-<sha>` instead, the zip had a top-level folder —
`cd` into it and treat *that* as the project root.

---

## 3. Create the virtual environment and install dependencies

### 3.0 Create the venv (both routes need this)

From the project root:

**Windows:**

```
py -3.12 -m venv .venv
.venv\Scripts\python.exe --version
```

Expected output:

```
Python 3.12.4
```

**Linux:**

```
python3.12 -m venv .venv
.venv/bin/python --version
```

Expected output:

```
Python 3.12.4
```

> **From here on, always call the interpreter by its full path** —
> `.venv\Scripts\python.exe` on Windows, `.venv/bin/python` on Linux. Do **not** use
> `activate`. Activation is a convenience that silently stops working in scheduled tasks,
> service wrappers and CI, and every failure it causes looks like a different bug.

**Troubleshooting — Linux: `ensurepip is not available`**: the `venv` module is packaged
separately. `sudo apt install python3.12-venv`, then re-run.

**Troubleshooting — Windows: "running scripts is disabled on this system"**: you tried to
run `activate`. You do not need it. Use the full interpreter path as above.

Now choose route **(a)** if this machine can reach the internet through your corporate
proxy, or route **(b)** if it cannot.

---

### 3a. Route A — online, behind a corporate proxy

Corporate networks commonly intercept HTTPS: a proxy terminates the encrypted connection,
inspects it, and re-encrypts it with a certificate signed by an internal certificate
authority (CA). Python's `pip` does not trust that internal CA by default, so downloads
fail with a TLS error even though the browser on the same machine works fine.

Two independent fixes; apply both.

**Step 3a.1 — tell pip which hosts to trust.** Create `.venv/pip.ini` (Windows) or
`.venv/pip.conf` (Linux) with exactly this content:

```
[global]
trusted-host =
    pypi.org
    files.pythonhosted.org
```

**Windows (PowerShell)** — create it in one command. The closing `"@` **must** be at the
very start of its own line, with no indentation:

```
@"
[global]
trusted-host =
    pypi.org
    files.pythonhosted.org
"@ | Out-File -Encoding ascii .venv\pip.ini
```

> **`-Encoding ascii`, not `utf8` — this matters and it was tested.** Windows PowerShell
> 5.1's `-Encoding utf8` writes a byte-order mark (the three bytes `EF BB BF`) at the start
> of the file. pip's config parser does not strip it, so `[global]` arrives as
> `﻿[global]` and pip dies with:
>
> ```
> Configuration file could not be loaded.
> File contains no section headers.
> file: '...\pip.ini', line: 1
> 'ï»¿[global]'
> ```
>
> The content here is pure ASCII, so `-Encoding ascii` is both correct and BOM-free. If you
> create the file in an editor instead, save it as "UTF-8 **without** BOM" or ANSI.

**Linux:**

```
cat > .venv/pip.conf <<'EOF'
[global]
trusted-host =
    pypi.org
    files.pythonhosted.org
EOF
```

Verify that pip actually reads it — this catches the BOM problem and a wrong path in one
step:

```
.venv\Scripts\python.exe -m pip config list
```

Expected output includes:

```
global.trusted-host='\npypi.org\nfiles.pythonhosted.org'
```

If instead you see `Configuration file could not be loaded. File contains no section
headers.`, re-read the BOM note above.

**Step 3a.2 — set the proxy, if your organisation requires an explicit one.** Ask your
network team for the URL. Then, in the shell you are about to install from:

**Windows (PowerShell):**

```
$env:HTTPS_PROXY = "http://proxy.your-org.internal:8080"
$env:HTTP_PROXY  = "http://proxy.your-org.internal:8080"
```

**Linux:**

```
export HTTPS_PROXY=http://proxy.your-org.internal:8080
export HTTP_PROXY=http://proxy.your-org.internal:8080
```

If your proxy needs credentials, **do not put a password in this file or in a script.**
Ask your platform team for a proxy that authenticates by machine identity, or for a
one-time token you paste into the shell only.

**Step 3a.3 — install.**

**Windows:**

```
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt -c constraints.txt
```

**Linux:**

```
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt -c constraints.txt
```

Expected output ends with a long `Successfully installed ...` line naming about 99
packages.

**What `-c constraints.txt` does:** `requirements.txt` says *what* to install;
`constraints.txt` pins *which version* of everything, including packages that arrive as
somebody else's dependency. Passing it is what makes your install identical to the one
every number in `docs/eval/REPORT.md` was measured on. Never omit it.

**Step 3a.4 — the certificate store, for code that downloads later.** Only two tools in
this project ever download anything, and both are dev-only: `tools/fetch_models.py`
(section 5) and `tools/fetch_fonts.py`. Both call `truststore.inject_into_ssl()` first,
which makes Python trust the operating system's certificate store — where your corporate
CA already lives, because your browser works. `truststore` is in
`requirements-dev.txt`, so step 3a.3 already installed it. Confirm:

```
.venv\Scripts\python.exe -c "import truststore; truststore.inject_into_ssl(); print('truststore active')"
```

Expected output:

```
truststore active
```

> TLS verification is never disabled anywhere in this project, and you must not add
> `--trusted-host` for arbitrary hosts or `verify=False` to any code. If a download still
> fails after this, section 11 has the diagnosis.

Now skip to [section 4](#4-verify-fonts-and-text-shaping).

---

### 3b. Route B — fully offline (wheelhouse)

You will build a folder of `.whl` files on a connected **staging machine**, carry it to the
air-gapped node, and install from it with the network switched off.

#### Step 3b.1 — on the STAGING machine

The staging machine needs Python 3.12 and internet. Copy `requirements.txt`,
`requirements-dev.txt` and `constraints.txt` onto it (three small text files — that is all
`pip download` reads).

**If the staging machine has the SAME operating system and CPU architecture as the target**
(the common case: Windows → Windows, or Linux → Linux, both x86-64), this is the whole
command:

Windows staging → Windows target:

```
py -3.12 -m pip download -r requirements.txt -r requirements-dev.txt -c constraints.txt -d wheelhouse
```

Linux staging → Linux target:

```
python3.12 -m pip download -r requirements.txt -r requirements-dev.txt -c constraints.txt -d wheelhouse
```

**If the staging machine's OS differs from the target's** — e.g. you must collect Linux
wheels while sitting at a Windows desktop — you have to name the target platform
explicitly, and you must add `--only-binary=:all:` because pip cannot build a
source package for a platform it is not running on:

Collect **Windows x86-64** wheels from any machine:

```
py -3.12 -m pip download -r requirements.txt -r requirements-dev.txt -c constraints.txt -d wheelhouse --only-binary=:all: --python-version 3.12 --implementation cp --abi cp312 --platform win_amd64
```

Collect **Linux x86-64** wheels from any machine:

```
py -3.12 -m pip download -r requirements.txt -r requirements-dev.txt -c constraints.txt -d wheelhouse --only-binary=:all: --python-version 3.12 --implementation cp --abi cp312 --platform manylinux2014_x86_64
```

Expected output: a stream of `Saved wheelhouse\<name>-<version>-...whl` lines, then
`Successfully downloaded ...`. Count the result:

**Windows (PowerShell):**

```
(Get-ChildItem wheelhouse -Filter *.whl).Count
```

**Linux:**

```
ls wheelhouse/*.whl | wc -l
```

Expect roughly 100 files. Size depends on the target platform; on the order of 400–600 MB,
which unpacks to a 732 MB virtual environment.

**If the target is a `gpu_node`, add two more wheels now.** `requirements.txt` names the
CPU `onnxruntime`, so a wheelhouse built from it alone cannot produce a working GPU node —
and you will not discover that until you are standing at an air-gapped machine that cannot
download anything. Collect both, for CUDA 12:

```
python3.12 -m pip download "onnxruntime-gpu==1.22.*" "nvidia-cudnn-cu12==9.*" -d wheelhouse --only-binary=:all: --python-version 3.12 --implementation cp --abi cp312 --platform manylinux2014_x86_64
```

That adds roughly 850 MB. On the node, install the wheelhouse as normal, then swap the
runtime — uninstall first, because both wheels install a package named `onnxruntime` and
whichever lands second wins:

```bash
.venv/bin/python -m pip uninstall -y onnxruntime
.venv/bin/python -m pip install --no-index --find-links wheelhouse onnxruntime-gpu nvidia-cudnn-cu12
```

Section 7.3 covers the `ldconfig` step that follows, and why skipping it produces a node
that looks healthy and runs on the CPU.

> **`--platform` explained in one sentence:** a wheel filename encodes the operating
> system and CPU it was built for (`...cp312-cp312-win_amd64.whl`), and this flag tells pip
> to fetch that variant instead of the one matching the machine you are typing on.

**Troubleshooting — "Could not find a version that satisfies the requirement X"** during a
cross-platform download: that package publishes no wheel for the target tag. Two common
causes and fixes:

- *Newer manylinux tag.* Retry with `--platform manylinux_2_17_x86_64`, and if that fails,
  `--platform manylinux_2_28_x86_64`. You may pass `--platform` more than once in a single
  command and pip will accept a wheel matching any of them.
- *Pure-Python package with a universal wheel.* These end `-py3-none-any.whl` and are
  downloaded regardless of `--platform`; if pip still refuses, the package genuinely has no
  wheel — report it, do not silently drop the pin.

#### Step 3b.2 — carry the wheelhouse across

Copy the whole `wheelhouse` folder to the project root on the air-gapped node, using
whatever transfer your organisation approves (removable media, a one-way file drop, an
internal artifact repository). The folder must end up at:

```
C:\rbi-ocr\wheelhouse\           (Windows)
~/rbi-ocr/wheelhouse/            (Linux)
```

#### Step 3b.3 — on the AIR-GAPPED node, install with the network forbidden

**Windows:**

```
.venv\Scripts\python.exe -m pip install --no-index --find-links wheelhouse -r requirements.txt -r requirements-dev.txt -c constraints.txt
```

**Linux:**

```
.venv/bin/python -m pip install --no-index --find-links wheelhouse -r requirements.txt -r requirements-dev.txt -c constraints.txt
```

`--no-index` means "never contact a package index"; `--find-links wheelhouse` means "look
only in this folder". Together they guarantee the install is offline. If a wheel is
missing, pip fails loudly naming the package — it will not quietly go online.

Expected output ends with `Successfully installed ...` naming about 99 packages.

#### Step 3b.4 — verify the install (both routes)

```
.venv\Scripts\python.exe -m pip check
```

Expected output:

```
No broken requirements found.
```

Then confirm the four heaviest components actually import:

**Windows:**

```
.venv\Scripts\python.exe -c "import onnxruntime, ctranslate2, cv2, pypdfium2; print('imports OK', onnxruntime.__version__, ctranslate2.__version__, cv2.__version__)"
```

**Linux:** same, with `.venv/bin/python`.

Expected output (versions from `constraints.txt`):

```
imports OK 1.28.0 4.8.1 5.0.0
```

**Troubleshooting — `ImportError: DLL load failed` on Windows** for `onnxruntime` or
`cv2`: the Microsoft Visual C++ Redistributable (x64) is missing. It is a standard
Microsoft package your platform team can push; there is no Python-side workaround.

**Troubleshooting — `libGL.so.1: cannot open shared object file` on Linux**: you installed
the wrong OpenCV. `requirements.txt` pins `opencv-python-headless` precisely to avoid this.
Check with `pip list | grep opencv`; if you see plain `opencv-python`, uninstall it and
reinstall from the wheelhouse.

---

## 4. Verify fonts and text shaping

**What this section is about, in one paragraph.** This project's test corpus contains
Devanagari — the script Hindi and Marathi are written in. Devanagari is a *complex script*:
vowel signs attach above, below and around consonants, and consonant clusters fuse into
single conjunct glyphs. Rendering it correctly requires a *text shaping* engine. Pillow
(the imaging library) ships one called Raqm on Windows, but Raqm loads a helper library
called `fribidi` dynamically at first use, and it only finds it if the containing folder is
on the process search path **before** Pillow is imported. `tools/assets/fribidi-0.dll` is
committed for exactly this reason, and the corpus generator puts `tools/assets` on the path
before importing Pillow (`OPEN-QUESTIONS.md` decision **D6**). If shaping is unavailable,
Devanagari renders as detached vowel marks and dotted circles — which would produce a test
corpus that is quietly wrong rather than obviously broken, so the generator **refuses to
run** instead.

Run the check. One command, from the project root:

**Windows:**

```
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,'tools'); import generate_test_corpus as g; from PIL import features; import PIL; g.require_fonts(); g.require_text_shaping(); print('fonts        :', len(list(g.FONT_DIR.glob('*.ttf'))), 'faces in', g.FONT_DIR); print('raqm shaping :', features.check('raqm')); print('fribidi DLL  :', (g.ASSETS_DIR/'fribidi-0.dll').exists()); print('Pillow       :', PIL.__version__); print('SHAPING OK')"
```

**Linux:** the same command with `.venv/bin/python`.

Expected output:

```
fonts        : 8 faces in C:\rbi-ocr\tools\assets\fonts
raqm shaping : True
fribidi DLL  : True
Pillow       : 12.3.0
SHAPING OK
```

The eight faces are `NotoSans-{Regular,Medium,Bold}`, `NotoSansMono-Regular`,
`NotoSansDevanagari-{Regular,Bold}`, `Kalam-Regular` and `Caveat-Regular`.

**This section is only needed if you intend to regenerate the corpus (section 6, route B).**
The *service* renders nothing and needs neither the fonts nor fribidi. If you are unzipping
bundle 2 and never regenerating, a failure here is not a blocker — note it and move on.

**Troubleshooting — `raqm shaping : False` / `TextShapingUnavailableError`:**

1. Confirm the DLL is present: `dir tools\assets\fribidi-0.dll`. If it is missing, bundle 1
   was assembled incorrectly — the file is committed and must ship.
2. On Linux, `fribidi-0.dll` is a Windows library and is irrelevant. Install the system
   packages instead: `sudo apt install libraqm0 libfribidi0` (Debian/Ubuntu) or
   `sudo dnf install libraqm fribidi` (RHEL/Fedora), then re-run the check.
3. If it is present on Windows and still `False`, something imported Pillow before
   `tools/assets` reached the path — you ran the check from a shell that had already
   imported PIL, or from a different working directory. Open a fresh shell, `cd` to the
   project root, and re-run the exact command above.

**Troubleshooting — `FontUnavailableError: Missing font faces`:** the fonts are not
vendored in git for licence reasons (`design/FONTS.md`). On a connected staging machine run
`python tools/fetch_fonts.py`, then carry `tools/assets/fonts/` across. Do **not** substitute
a different Devanagari face: the generator refuses a substitute on purpose, because wrong
glyphs are worse than no document.

---

## 5. Place the model files

**What a "model" is here.** Five components load a binary weights file at run time: text
detection, Latin text recognition, Devanagari text recognition, language identification,
and Indic→English translation. A sixth, the page classifier, loads a tiny JSON file of
fitted coefficients. None of them are in git (`models/` is in `.gitignore`) because they
are large and separately licensed.

**If you unzipped bundle 2, they are already in place and there is nothing to download.**
Skip to **5.4** and verify them. Section 2's extract puts `models/` exactly where the
service expects it.

The rest of this section — the artifact table, `fetch_models.py`, and the manual route
— is what you need only when you are building a *new* bundle on a connected machine, or
when bundle 2 was not supplied.

**Total on disk: 837 MB, measured.**

### 5.1 The artifact table

Every SHA-256 below was computed from the bytes on the machine that produced
`docs/eval/REPORT.md`. Verify each file after copying (command in 5.4) — a truncated model
file produces confusing, plausible-looking wrong output rather than a clean failure.

**OCR — detection and recognition.** Licence Apache-2.0. Published by the official
`PaddlePaddle` organisation on Hugging Face as one repository per artifact.

| Lands at (under `models/`) | Download from | File in the repo | Size (bytes) | SHA-256 |
|---|---|---|---|---|
| `ocr/det.onnx` | `https://huggingface.co/PaddlePaddle/PP-OCRv5_mobile_det_onnx` | `inference.onnx` | 4 826 518 | `a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d` |
| `ocr/rec_latn.onnx` | `https://huggingface.co/PaddlePaddle/en_PP-OCRv5_mobile_rec_onnx` | `inference.onnx` | 7 848 423 | `b5f833dfc5d0eb71da397b4efa06ebeee9b431b690a47d6af40d77d8eabc557f` |
| `ocr/rec_latn_dict.txt` | *derived* — see 5.3 | from `en_..._rec_onnx/inference.yml` | 1 852 | `8459e5659185f87d62195cb495f2677493bade134182bfb8c54bf21757b28cdf` |
| `ocr/rec_deva.onnx` | `https://huggingface.co/PaddlePaddle/devanagari_PP-OCRv5_mobile_rec_onnx` | `inference.onnx` | 7 912 311 | `cb789212ce96c69d3e74728ae4309d179281d68cb3945d0616b67cafab41c986` |
| `ocr/rec_deva_dict.txt` | *derived* — see 5.3 | from `devanagari_..._rec_onnx/inference.yml` | 2 511 | `f3ac932023e83fb0e995383957c8953cd3e506737308fe44cffab61549deadb3` |

**Language identification.** fastText's compressed 176-language model, downloaded from
Meta's own distribution point (not Hugging Face).

| Lands at | Download from | Size (bytes) | SHA-256 |
|---|---|---|---|
| `lid/lid.176.ftz` | `https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz` | 938 013 | `8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83` |

> **Licence note for your procurement record:** `lid.176.ftz` is published under
> **CC-BY-SA-3.0**, which is outside the code-dependency licence allowlist in `PLAN.md`
> (invariant 12). That allowlist governs *code*; CC-BY-SA is neither GPL/AGPL nor
> non-commercial, so the downloader fetches it by default and flags it loudly. Attribution
> and share-alike apply to the model file. Confirm with your legal owner before production.
> `tools/fetch_models.py --strict-licence` refuses it if you want the harder rule.

**Translation (Indic → English).** IndicTrans2 distilled 200M, pre-converted to the
CTranslate2 format. Licence MIT. Repository
`https://huggingface.co/adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M`, pinned
revision `524aade4f724dccbc6cd5ffb5e3c5c832a2baee4`. All six files come from the
`indic-en-200m-ct2/ctranslate2_model/` directory inside that repository (the two
`model.SRC`/`model.TGT` files are one level deeper, in `.../ctranslate2_model/vocab/`), and
all six land flat in `models/mt/indic-en/`.

| Lands at | File in the repo | Size (bytes) | SHA-256 |
|---|---|---|---|
| `mt/indic-en/model.bin` | `.../ctranslate2_model/model.bin` | 847 167 702 | `a88c1c56918d267c4003279b2b3d4ed47f7e18a859d38ebb8491da6c422e4de8` |
| `mt/indic-en/config.json` | `.../ctranslate2_model/config.json` | 197 | `10874b31f398169314575b99a04560b6d8944fba1c7a0d96eb89f7676bf08979` |
| `mt/indic-en/source_vocabulary.json` | `.../ctranslate2_model/source_vocabulary.json` | 4 532 115 | `26d1ba4b6e918bef2bdccf1e0370130a3d641d30f7a945898a00f56043a9d8ad` |
| `mt/indic-en/target_vocabulary.json` | `.../ctranslate2_model/target_vocabulary.json` | 543 196 | `debadde275c3afb460bf97913d02959537a4e24350376b93f1cba878e3661453` |
| `mt/indic-en/model.SRC` | `.../ctranslate2_model/vocab/model.SRC` | 3 256 903 | `ac9257c8e76b8b607705b959cc3d075656ea33032f7a974e467b8941df6e98d4` |
| `mt/indic-en/model.TGT` | `.../ctranslate2_model/vocab/model.TGT` | 759 425 | `3cedc5cbcc740369b76201942a0f096fec7287fee039b55bdb956f301235b914` |

> `model.bin` is 847 MB on disk because it is stored at full precision. It is loaded with
> `compute_type="int8"`, so it occupies roughly a quarter of that in memory. This is a
> known and accepted trade (`OPEN-QUESTIONS.md` A-8/A-9): the disk cost buys you a file
> that needs no conversion toolchain on the target.

**Page classifier.** Not downloaded — it is **already in bundle 1** and you copy it.

| Lands at | Copy from | Size (bytes) | SHA-256 |
|---|---|---|---|
| `classifier/fusion.json` | `docs/calibration/fusion.json` | 1 442 | `ea03e8671526701f1d42f94f265ef55fc72a22fc2ae287293c17c90169fa5679` |

Copy it now:

**Windows:**

```
mkdir models\classifier
copy docs\calibration\fusion.json models\classifier\fusion.json
```

**Linux:**

```
mkdir -p models/classifier
cp docs/calibration/fusion.json models/classifier/fusion.json
```

> There is no convolutional page-classifier model in this release. `fusion.json` holds the
> fitted coefficients for the *structural* classification path, with every neural-network
> coefficient set to exactly `0.0`, so the decision is provably independent of a model that
> does not exist yet. The service logs
> `page-classifier CNN unavailable ...; using the fitted structural path` on every page.
> That log line is expected and is not an error.

### 5.2 The easy route — `fetch_models.py` on a connected staging machine

If your staging machine can reach Hugging Face, do not download by hand. Copy the code
bundle there, build a venv exactly as in section 3, then:

**Windows:**

```
set HF_HOME=%CD%\.cache\huggingface
.venv\Scripts\python.exe tools\fetch_models.py --models-dir models --hf-home .cache\huggingface
```

**Linux:**

```
export HF_HOME="$PWD/.cache/huggingface"
.venv/bin/python tools/fetch_models.py --models-dir models --hf-home .cache/huggingface
```

Expected output ends with a summary table and a provenance path:

```
[tls] using the OS certificate store (corporate CA honoured)
[hf] HF_HOME=C:\staging\rbi-ocr\.cache\huggingface
[lid] downloading https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
[lid] wrote ...\models\lid\lid.176.ftz (938013 bytes)
[ocr] det: downloading PaddlePaddle/PP-OCRv5_mobile_det_onnx:inference.onnx
...
[mt] wrote ...\models\mt\indic-en

[provenance] ...\models\PROVENANCE.json
  lid  fasttext-lid-176   ok  CC-BY-SA-3.0  <-- OUT OF GATE
  ocr  det                ok  Apache-2.0
  ocr  rec_latn           ok  Apache-2.0
  ocr  rec_deva           ok  Apache-2.0
  ocr  dict_latn          ok  Apache-2.0
  ocr  dict_deva          ok  Apache-2.0
  mt   indictrans2-...    ok  MIT
```

The `<-- OUT OF GATE` marker on `lid` is the CC-BY-SA licence note above, working as
designed. It is not a failure.

**Why `--hf-home` / `HF_HOME` matters** (`PLAN.md` WS-I item 3): the Hugging Face client
otherwise caches a second full copy of everything in `~/.cache/huggingface`, outside your
project, where nobody thinks to clean it up and no bundle carries it. Pointing it at a
folder inside the project keeps every downloaded byte accounted for and lets you delete
`.cache/` when you are done. **Nothing in `app/` ever contacts Hugging Face for weights** —
the service reads `models/` and only `models/`, resolved from the `MODELS_DIR` setting.

Then carry the whole `models/` directory to the air-gapped node so it sits at
`C:\rbi-ocr\models\`.

**Access and credentials:**

- The three PaddlePaddle OCR repositories and the fastText URL are **public**. No account
  is needed.
- The translation repository `adalat-ai/ct2-rotary-indictrans2-indic-en-dist-200M` is an
  MIT-licensed mirror and was fetched with an authenticated account. Its **upstream**,
  `ai4bharat/indictrans2-indic-en-dist-200M`, is a **gated** repository: it returns HTTP
  403 until you sign in to Hugging Face and click through the terms on the model page.
  This bit the build once (`OPEN-QUESTIONS.md` A-9). If any translation download returns
  401 or 403, that is what happened: sign in, open the model page in a browser, accept the
  terms, and re-run.
- **No token ships in either bundle.** `.env` is excluded from the code bundle and the
  bundle builder asserts its absence. If you need an authenticated download on the staging
  machine, set `HF_TOKEN` in that shell for that command only, and never write it into a
  file that travels.

### 5.3 The manual route — the two `*_dict.txt` files need one extra step

Four of the seven OCR files are straight downloads. The two `rec_*_dict.txt` files are
**not published as text files** — each recognition repository ships an `inference.yml`
containing a `character_dict` list, and `fetch_models.py` extracts that list into a
one-character-per-line text file. If you download by hand, you must do the same extraction,
on the staging machine, after downloading both `inference.yml` files:

```
.venv\Scripts\python.exe -c "import sys,yaml,pathlib; src,dst=sys.argv[1],sys.argv[2]; d=yaml.safe_load(pathlib.Path(src).read_text(encoding='utf-8')); stack=[d]; chars=None
while stack and chars is None:
    n=stack.pop()
    if isinstance(n,dict):
        for k,v in n.items():
            if k=='character_dict' and isinstance(v,list): chars=[str(c) for c in v]; break
            stack.append(v)
    elif isinstance(n,list): stack.extend(n)
assert chars, 'no character_dict in '+src
pathlib.Path(dst).write_text('\n'.join(chars)+'\n', encoding='utf-8'); print(len(chars),'characters ->',dst)" en_inference.yml models\ocr\rec_latn_dict.txt
```

Repeat with `devanagari_inference.yml` → `models\ocr\rec_deva_dict.txt`. Expected output is
`<n> characters -> models\ocr\rec_latn_dict.txt`. Then check the SHA-256 against the table.
If it does not match, the upstream repository changed its character list and the recognition
head will decode to the wrong characters — stop and escalate rather than proceeding.

(This snippet re-implements `fetch_models._extract_character_dict`. It finds the one
`character_dict` list wherever it is nested and writes one character per line, UTF-8. It was
executed against a stand-in YAML file to confirm it runs; it was **not** run against the real
`inference.yml` files, which no longer exist on the build machine — the fetch tool deletes
its staging directory.)

**Strong recommendation: use `fetch_models.py` (5.2) instead.** The manual route exists so
that a machine that cannot run the tool is not blocked, not because it is a good idea.

### 5.4 Verify what you placed

**If the weights came from bundle 2, verify against its manifest instead** — it lists a
SHA-256 for every file it shipped, so this checks the transfer rather than a list of
hashes typed into a document:

```
.venv\Scripts\python.exe tools/make_bundle.py --verify --only assets --out <folder-with-the-zips>
```

`BUNDLE OK` means every byte of every model and corpus file matches what was built. A
CRC or SHA-256 failure means the zip was damaged in transit — copy it again.

The hard-coded check below is the fallback for weights placed by hand, and a second
opinion if you want one. From the project root on the air-gapped node:

**Windows / Linux (same command, adjust the interpreter path):**

```
.venv\Scripts\python.exe -c "import hashlib,pathlib; want={'ocr/det.onnx':'a431985659dc921974177a95adcfbb90fd9e51989a5e04d70d0b75f597b6e61d','ocr/rec_latn.onnx':'b5f833dfc5d0eb71da397b4efa06ebeee9b431b690a47d6af40d77d8eabc557f','ocr/rec_deva.onnx':'cb789212ce96c69d3e74728ae4309d179281d68cb3945d0616b67cafab41c986','ocr/rec_latn_dict.txt':'8459e5659185f87d62195cb495f2677493bade134182bfb8c54bf21757b28cdf','ocr/rec_deva_dict.txt':'f3ac932023e83fb0e995383957c8953cd3e506737308fe44cffab61549deadb3','lid/lid.176.ftz':'8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83','mt/indic-en/model.bin':'a88c1c56918d267c4003279b2b3d4ed47f7e18a859d38ebb8491da6c422e4de8','mt/indic-en/config.json':'10874b31f398169314575b99a04560b6d8944fba1c7a0d96eb89f7676bf08979','mt/indic-en/source_vocabulary.json':'26d1ba4b6e918bef2bdccf1e0370130a3d641d30f7a945898a00f56043a9d8ad','mt/indic-en/target_vocabulary.json':'debadde275c3afb460bf97913d02959537a4e24350376b93f1cba878e3661453','mt/indic-en/model.SRC':'ac9257c8e76b8b607705b959cc3d075656ea33032f7a974e467b8941df6e98d4','mt/indic-en/model.TGT':'3cedc5cbcc740369b76201942a0f096fec7287fee039b55bdb956f301235b914','classifier/fusion.json':'ea03e8671526701f1d42f94f265ef55fc72a22fc2ae287293c17c90169fa5679'}; root=pathlib.Path('models'); bad=0
for rel,exp in sorted(want.items()):
    p=root/rel
    if not p.exists(): print('MISSING ',rel); bad+=1; continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda: f.read(1<<20), b''): h.update(c)
    ok = h.hexdigest()==exp
    print(('OK      ' if ok else 'MISMATCH'), rel, p.stat().st_size)
    bad += 0 if ok else 1
print('FAILURES:', bad)"
```

Expected output — thirteen `OK` lines and:

```
FAILURES: 0
```

**Troubleshooting — `MISMATCH`:** the file is truncated or was fetched from a different
revision. Re-copy it. A size mismatch alongside the hash mismatch means the transfer was
cut short; an identical size with a different hash means a different upstream revision, and
you should re-run `fetch_models.py` on the staging machine and use the fresh
`models/PROVENANCE.json` it writes as your record.

**Troubleshooting — `MISSING`:** check the exact folder names. They are lower case and the
recognition heads are `rec_latn` / `rec_deva`, not `rec_latin` / `rec_devanagari`.

> **A known gap in the shipped provenance file.** `models/PROVENANCE.json` records source
> URL, resolved revision, size, SHA-256 and licence for every artifact, but the file is
> **rewritten in full on every run** of `fetch_models.py`. The last run on the build machine
> was `--only mt`, so the committed copy names only the translation artifact. The hashes in
> the tables above were therefore recomputed directly from the bytes on that machine with
> the same algorithm the tool uses — and the one artifact that appears in both,
> `mt/indic-en/model.bin`, agrees exactly, which is the check that the two methods are the
> same method. **Run `fetch_models.py` with no `--only` flag on the staging machine** and
> keep the `PROVENANCE.json` it writes: that file, not this table, is your audit record.

---

## 6. Corpus: unzip or regenerate

The corpus is 295 synthetic documents used to prove the install works end to end. It
contains **no real complainant data** — every name, account number and address is generated.

### Route A — you have bundle 2 (recommended)

You already unzipped it in section 2. Verify:

```
.venv\Scripts\python.exe -c "import json,pathlib; idx=json.loads(pathlib.Path('corpus/index.json').read_text(encoding='utf-8')); print('documents :', len(idx['documents'])); print('generator :', idx['generator_version']); print('seed      :', idx['seed']); print('synthetic :', idx['synthetic'])"
```

Expected output:

```
documents : 295
generator : 1.2.0
seed      : 1729
synthetic : True
```

And count the image files:

**Windows:**

```
(Get-ChildItem corpus -Recurse -Include *.jpg,*.png).Count
```

**Linux:**

```
find corpus -name '*.jpg' -o -name '*.png' | wc -l
```

Expected: `298` — the 295 documents plus the three *image* adversarial files
(`blank_page.png`, `inverted.png`, `tiny_lowres.png`); the other four adversarial inputs are
PDFs and are not counted by this command.

### Route B — regenerate from bundle 1

Requires section 4 to have passed. From the project root:

**Windows:**

```
.venv\Scripts\python.exe tools\generate_test_corpus.py --out corpus --variants 5 --seed 1729
```

**Linux:**

```
.venv/bin/python tools/generate_test_corpus.py --out corpus --variants 5 --seed 1729
```

**Runtime: 2 minutes 22 seconds** measured on the 4-core build machine. Expected output
ends with:

```
295 pages -> C:\rbi-ocr\corpus
  types        10
  variants     5
  profiles     6
  hand shaping cluster
  index        C:\rbi-ocr\corpus\index.json
```

Then run the Route A verification above and confirm the same three numbers (`295`,
`1.2.0`, `1729`).

> **Regeneration is deterministic and was checked.** On the build machine, regenerating
> into an empty directory with `--seed 1729` produced **599 of 599 files byte-identical**
> to the committed corpus. If yours differs, the difference is in your fonts or your
> Pillow version, not in the generator.

> **The seed matters.** `--seed 1729` is what every committed ground-truth file and every
> number in `docs/eval/REPORT.md` was produced with. A different seed produces a different,
> equally valid corpus that will not match the committed evaluation predictions.

**Troubleshooting — `TextShapingUnavailableError`:** go back to section 4. The generator
refuses to render Devanagari without shaping rather than producing a silently wrong corpus.

**Troubleshooting — `FontUnavailableError`:** section 4, second troubleshooting note.

---

## 7. Create the `.env` file

`.env` is a plain text file of `NAME=value` lines that the service reads at start-up. It is
**not** in either bundle (it can hold a secret) and it is in `.gitignore`. You create it
from the template.

**Windows:**

```
copy .env.example .env
```

**Linux:**

```
cp .env.example .env
```

Then open `.env` in a text editor and set the values per the table below.

Three things to know before you edit it:

- **The template is a subset.** `.env.example` contains the handful of variables a
  developer usually changes. The tables below list **every** variable the service reads.
  Anything absent from your `.env` simply uses the default shown here — you do not have to
  write a line for it.
- **The file is read from the project root**, resolved from the code's own location, so it
  works no matter which directory you launch from.
- **Blank the token.** The template ships `HF_TOKEN=hf_your_token_here`. On an air-gapped
  node change it to `HF_TOKEN=` (empty). Leaving a placeholder there is harmless but
  misleading; leaving a *real* token there on a node that never calls out is a finding the
  start-up banner will report.

### 7.1 Every variable, one line each

**Profile and residency — read this block before anything else.**

| Variable | Default | What it does |
|---|---|---|
| `PROVIDER_PROFILE` | `laptop` | Which deployment shape this is: `laptop` (development; language-model calls go to a remote endpoint), `cpu_node` (air-gapped production; everything local), `gpu_node` (local model server). |
| `SYNTHETIC_ONLY` | `true` | Asserts that only generated test data will be processed. Under `laptop` this **must** stay `true`; the service refuses to start otherwise, because that profile sends text off-machine. There is no override. |
| `HF_TOKEN` | *(unset)* | **Optional.** Access token for the remote language-model endpoint. Only the `laptop` profile ever uses it. **Leave it empty on an air-gapped node.** |
| `HF_BASE_URL` | `https://router.huggingface.co/v1` | The OpenAI-compatible endpoint the language model is called through. Pointing this at a local server is the whole change needed to run the language model on-premises. |

> **Air-gapped nodes: set `PROVIDER_PROFILE=cpu_node`.** Under `cpu_node` the residency
> gate does not apply, `SYNTHETIC_ONLY` is irrelevant, no upload has to declare itself
> synthetic, and nothing is sent off-machine. `laptop` is a *development* profile and its
> defining property is that it sends text to a non-Indian endpoint.

**Language model (optional — the service works fully without it).**

| Variable | Default | What it does |
|---|---|---|
| `SLM_MODEL` | `google/gemma-3-4b-it` | The language model used for free-text fields (complainant name, address, relief sought, summary). |
| `SLM_MODEL_FALLBACK` | `meta-llama/Llama-3.1-8B-Instruct` | Tried when the primary model is unreachable. |
| `SLM_MAX_TOKENS` | `512` | Ceiling on one model answer. Must not exceed `JOB_TOKEN_BUDGET`. |
| `SLM_TEMPERATURE` | `0` | Randomness. Must be `0` for extraction; higher values are a tuning experiment only. |
| `JOB_TOKEN_BUDGET` | `8000` | Ceiling on language-model tokens per job. Exceeding it raises a typed error rather than running up a bill. |
| `PROVIDER_CACHE_PATH` | `.cache/provider.sqlite` | Local cache of model answers, keyed on model + prompt + parameters. Makes a repeated run cheap. |

> **If `HF_TOKEN` is empty or the endpoint is unreachable, the document still completes.**
> Every identifier and amount — account number, IFSC, UTR/RRN, MICR, card fragment,
> amounts, dates — is extracted by deterministic validators that never involve a language
> model (`PLAN.md` invariant 7). You lose the free-text fields and get a
> `slm_unavailable:<CODE>` entry in the envelope's `warnings` list. That is a supported,
> documented state, not a failure.

**Threads — do not raise these to "speed things up".**

| Variable | Default | What it does |
|---|---|---|
| `OMP_NUM_THREADS` | `2` | Threads for OpenMP-based numeric libraries. |
| `MKL_NUM_THREADS` | `2` | Threads for Intel MKL. |
| `ORT_INTRA_OP_THREADS` | `2` | Threads inside one ONNX Runtime operation. |
| `CELERY_CONCURRENCY` | `2` | Worker processes, when running with a task queue. |

> Every one of these libraries defaults to "one thread per core" independently. Left alone,
> four of them on a 16-core box request 64 threads for 16 cores and the result is **3–5×
> slower**, not faster. The rule is `CELERY_CONCURRENCY × ORT_INTRA_OP_THREADS ≤ physical
> cores`. On a 16-vCPU node, `CELERY_CONCURRENCY=8` with `ORT_INTRA_OP_THREADS=2` is the
> shape to start from; see `docs/TUNING.md`.

**Recognition and routing.**

| Variable | Default | What it does |
|---|---|---|
| `OCR_BATCH_SIZE` | `4` | Line images fed to the recognition model per call. Measured on this hardware: larger batches were 23–31 % *slower* (`OPEN-QUESTIONS.md` A-10). Re-tune on the real node. |
| `REC_HEAD_CACHE_SIZE` | `3` | How many recognition models stay loaded in memory. |
| `REC_HEAD_CACHE_TTL_SECONDS` | `900` | How long an idle recognition model stays loaded. |
| `ESCALATION_SHADOW_RATE` | `0.03` | Fraction of *passing* pages also sent through the second-tier reader, to measure how often the first tier is silently wrong. |
| `CODE_MIX_THRESHOLD` | `0.15` | Share of lines in a different script before a page is labelled mixed-script. |

**Quality gate** — the thresholds that decide accept / repair / bounce. All were calibrated
against the corpus; `docs/TUNING.md` explains each and `docs/calibration/quality_gate.json`
is the committed reference.

| Variable | Default | What it does |
|---|---|---|
| `QUALITY_BOUNCE_THRESHOLD` | `0.35` | Below this overall quality score the page is refused. |
| `QUALITY_BLUR_LAPLACIAN_FLOOR` | `9500` | Sharpness measure at which the blur sub-score hits 0. |
| `QUALITY_BLUR_LAPLACIAN_TARGET` | `26000` | Sharpness measure at which it hits 1. |
| `QUALITY_CONTRAST_FLOOR` | `17` | Ink-vs-paper contrast at which the contrast sub-score hits 0. |
| `QUALITY_CONTRAST_TARGET` | `70` | Contrast at which it hits 1. |
| `QUALITY_EFFECTIVE_DPI_FLOOR` | `30` | Effective resolution at which the resolution sub-score hits 0. |
| `QUALITY_EFFECTIVE_DPI_TARGET` | `140` | Effective resolution at which it hits 1. |
| `QUALITY_TEXT_COVERAGE_FLOOR` | `0.004` | Minimum share of the page covered by text. |
| `QUALITY_SKEW_REPAIR_DEG` | `0.75` | Tilt at or above which the page is straightened. |
| `QUALITY_SKEW_BOUNCE_DEG` | `25` | Tilt beyond which straightening is untrustworthy and the page is refused. |
| `QUALITY_MIN_PAGE_DIMENSION_PX` | `200` | Short-edge pixel floor below which the image is not a document. |

**Resource ceilings — the defence against a hostile attachment.**

| Variable | Default | What it does |
|---|---|---|
| `MAX_PAGES_PER_DOC` | `200` | Refuse documents with more pages than this. |
| `MAX_TOTAL_PIXELS` | `500000000` | Refuse documents whose pages decode to more pixels than this. |
| `MAX_DECOMPRESSION_RATIO` | `100` | Refuse a file that expands more than this many times when decompressed (a "zip bomb"). |
| `MAX_WALL_CLOCK_SECONDS` | `900` | Give up on one document after this long. |
| `API_MAX_UPLOAD_BYTES` | `67108864` (64 MB) | Refuse an upload larger than this, while reading it. |

**Extraction.**

| Variable | Default | What it does |
|---|---|---|
| `GROUNDING_MAX_EDIT_DISTANCE` | `2` | A language-model answer must appear in the page's own recognised text within this many character edits, or it is rejected and flagged. |
| `IFSC_MASTER_PATH` | `app/resources/ifsc_master_sample.json` | The bank-**branch** reference list, used to resolve IFSC codes. Ships as a small synthetic fixture. **Point this at the real RBI list on a production node** — a `.csv` is treated as authoritative, so an unknown branch code then fails its lookup instead of being reported as unverifiable. This does **not** govern bank *names* — see the next row. |
| `BANK_ALIASES_PATH` | `app/resources/bank_aliases.json` | The bank-**name** table: display names, alias spellings (English, acronym and Devanagari) and the non-party institution list. Separate from `IFSC_MASTER_PATH` because the RBI master carries one Latin spelling per bank and no native-script rendering, which is not what an OCR'd complaint contains. The committed table covers every public-sector and scheduled private bank, the small-finance and payments banks, the largest urban co-operatives and the pre-merger names — override it only if you must add regional co-operatives or RRBs. A missing, malformed or empty file raises `ConfigurationError` rather than falling back. |

**Storage and service.**

| Variable | Default | What it does |
|---|---|---|
| `DATABASE_URL` | `sqlite+pysqlite:///./data/runtime/app.sqlite` | Where jobs, results and the audit trail are stored. SQLite needs no server and creates itself. Swap in a PostgreSQL URL for a multi-node deployment. |
| `REDIS_URL` | `redis://localhost:6379/0` | Message broker, used only when the task queue is enabled. |
| `CELERY_TASK_ALWAYS_EAGER` | `true` | `true` = run work in-process, no broker needed. Leave it `true` for a single node. |
| `CELERY_BROKER_URL` / `CELERY_RESULT_BACKEND` | *(unset)* | Override the broker/result store; default to `REDIS_URL`. |
| `CELERY_TASK_TIME_LIMIT` | `1800` | Hard per-task ceiling; must exceed `MAX_WALL_CLOCK_SECONDS`. |
| `API_TITLE` | `RBI Complaint Document Intelligence` | Name shown in the generated API documentation. |
| `MODELS_DIR` | `models` | Where the weights live. Point it at a mounted volume if the models are not inside the project. |
| `CORPUS_DIR` | `corpus` | Where the test corpus lives. |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. |

**Result delivery (webhooks) — optional.** See `docs/API.md` for the consumer's half.

| Variable | Default | What it does |
|---|---|---|
| `WEBHOOK_URL` | *(unset)* | Where to POST each result version. Unset disables delivery entirely. |
| `WEBHOOK_HMAC_SECRET` | *(unset)* | Shared secret used to sign every delivery. **Required whenever `WEBHOOK_URL` is set** — an unsigned delivery is refused, because the receiver would have no way to tell it came from this service. |
| `WEBHOOK_DELIVERY_MODE` | `thread` | `thread` = in-process background sender; `celery` = hand to a worker; `inline` = send on the request thread (tests); `disabled` = do not send. |
| `WEBHOOK_TIMEOUT_SECONDS` | `10` | Per-attempt HTTP timeout. |
| `WEBHOOK_MAX_ATTEMPTS` | `5` | Attempts before the delivery is parked for manual replay. |
| `WEBHOOK_BACKOFF_BASE_SECONDS` / `WEBHOOK_BACKOFF_MAX_SECONDS` | `1` / `60` | Retry delay, doubling each attempt up to the maximum. |
| `WEBHOOK_SIGNATURE_TOLERANCE_SECONDS` | `300` | How old a signed delivery may be before the receiver should reject it as a replay. |

### 7.2 The two configurations, side by side

**Development laptop** (what `.env.example` gives you):

```
PROVIDER_PROFILE=laptop
SYNTHETIC_ONLY=true
HF_TOKEN=hf_xxx
```

**Air-gapped production node:**

```
PROVIDER_PROFILE=cpu_node
SYNTHETIC_ONLY=false
HF_TOKEN=
CELERY_CONCURRENCY=8
ORT_INTRA_OP_THREADS=2
IFSC_MASTER_PATH=/data/ifsc/ifsc_master.csv
LOG_LEVEL=INFO
```

Leave every other line at its default until `docs/TUNING.md` tells you otherwise.

**`BANK_ALIASES_PATH` is deliberately absent from that block** — the committed table already
covers the banking system, so the default is correct for a production node. Set it only to add
institutions the shipped table does not enumerate (regional co-operative banks, RRBs). Note that
it is a *different* file from `IFSC_MASTER_PATH`: pointing the IFSC setting at the real RBI list
gives you authoritative **branch** lookups and changes nothing about which bank **names** are
recognised.

**Troubleshooting — the service refuses to start with `RESIDENCY_VIOLATION`:** two
configurations produce it, and they have different fixes.

1. `PROVIDER_PROFILE=laptop` together with `SYNTHETIC_ONLY=false`. That combination would
   send real complainant text to a non-Indian endpoint. On an air-gapped node the fix is
   `PROVIDER_PROFILE=cpu_node`, **not** `SYNTHETIC_ONLY=true`.
2. `PROVIDER_PROFILE=gpu_node` with an `HF_BASE_URL` that is not on-premises — most often
   the shipped default, which is the public HuggingFace router. Point it at your in-house
   model server. If that server has a public-looking name, declare it:
   `LOCAL_ENDPOINT_HOSTS=slm.rbi.gov.in`.

### 7.3 GPU node

`gpu_node` is `cpu_node` plus two things: the vision and translation models run on CUDA,
and field extraction calls an OpenAI-compatible model server **on your own network**.

```
PROVIDER_PROFILE=gpu_node
SYNTHETIC_ONLY=false
GPU_EXECUTION=require
HF_BASE_URL=http://localhost:8001/v1     # your vLLM / llama.cpp / Ollama / TGI
                                         # not :8000 — this service binds that
HF_TOKEN=                                # leave empty: a local server has no auth
SLM_MODEL=<whatever your server serves>
```

**Match the wheel to the box's CUDA, not just to "GPU".** `onnxruntime-gpu` is built
against one CUDA major version, and installing the newest wheel on an older CUDA runtime
produces the *worst* possible failure mode: the package imports, `get_available_providers()`
lists `CUDAExecutionProvider`, and then the first session fails to load
`libonnxruntime_providers_cuda.so` and falls back to CPU. Nothing raises. The node works
and is ten times too slow.

Observed on a CUDA 12 machine with the then-current wheel:

```
Failed to load library .../libonnxruntime_providers_cuda.so with error:
libcublasLt.so.13: cannot open shared object file: No such file or directory
```

`libcublasLt.so.13` is CUDA 13's; the box had CUDA 12. The fix is a wheel built for the
CUDA that is actually installed — for CUDA 12, `pip install "onnxruntime-gpu==1.22.*"`.
Check what you have with `nvidia-smi` (driver's CUDA ceiling) and
`ls /usr/local/cuda/lib64/libcublasLt.so.*` (the runtime actually present).

**The wheel also needs cuDNN 9, and a CUDA install does not include it.** This is a
*second, separate* failure with the same symptom, so it is easy to misdiagnose as the one
above and go on reinstalling the wheel. The error names a different library:

```
Failed to load library .../libonnxruntime_providers_cuda.so with error:
libcudnn.so.9: cannot open shared object file: No such file or directory
```

Some vendor images ship cuDNN preinstalled and some do not — a plain Ubuntu 24.04 GPU
image generally does not. Install it into the same virtualenv:

```bash
.venv/bin/python -m pip install "nvidia-cudnn-cu12==9.*"
```

**Then make the loader find it, with `ldconfig` and not `LD_LIBRARY_PATH`.** The pip wheel
puts the libraries inside `site-packages`, where the dynamic loader does not look. Export
`LD_LIBRARY_PATH` and the dry run goes green — and then the service, started by systemd or
`nohup`, does not inherit your shell's environment, silently falls back to CPU, and reports
itself perfectly healthy while running at a third of the speed. Register the directory
system-wide instead, so every process gets it:

```bash
{ .venv/bin/python -c 'import os, nvidia.cudnn; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))'
  echo /usr/local/cuda-12.8/lib64; } > /etc/ld.so.conf.d/rbi-ocr-cudnn.conf && ldconfig
```

Use `nvidia.cudnn.__path__`, not `__file__`: these are PEP 420 namespace packages, so
`__file__` is `None` and the obvious one-liner dies with `TypeError: expected str, bytes
or os.PathLike object, not NoneType`. Adjust the `cuda-12.8` path to your own runtime.

Confirm the loader agrees, then confirm it holds with the environment stripped — the second
command is the one that tells you the service will work, because it is what systemd sees:

```bash
ldconfig -p | grep -E 'libcudnn\.so\.9|libcublasLt\.so\.12'
env -u LD_LIBRARY_PATH .venv/bin/python tools/dryrun_gpu.py --pages 8
```

**`tools/dryrun_gpu.py` exists to catch exactly this** and prints both numbers when it
does. Run it before you believe a GPU node is using its GPU.

**Measured, so you know what "working" looks like.** Two independent runs on a Tesla T4
with `onnxruntime-gpu==1.22`, against the synthetic corpus, on hosts with very different
CPUs:

| host | CPU pages/min | GPU pages/min | ratio |
|---|---|---|---|
| notebook host, CUDA 12.8 / driver 580 | 77 | **247** | 3.19x |
| 12 vCPU cloud node, CUDA 12.8 / driver 570 | 92 | **252** | 2.73x |

Read those two rows together, because the pair says more than either row alone. The GPU
numbers agree within 2% — same card, same throughput, which is what tells you the
measurement is sound rather than an artifact of one host. The *ratio* moved only because
the CPU baseline moved. **The ratio is a property of the host you are comparing against,
not of the GPU.** A faster production CPU will shrink it, and that is not a regression.

Both arms produced byte-identical text, which is the other half of the check — an
accelerator that changes the output is a bug, not an accelerator.

One further caution: this is *recognition* throughput. A real document also pays for the
quality gate, validators and assembly, so the end-to-end gain is smaller than the table.

Four things go wrong here, in order of how often:

- **cuDNN 9 missing, or reachable only through `LD_LIBRARY_PATH`.** Covered above. On a
  fresh cloud GPU image this is the single most likely reason a correct wheel still runs
  on the CPU.
- **`HF_TOKEN` left set, or the base URL left at its default.** The token is only needed
  for a public endpoint. Leaving `HF_BASE_URL` pointing at the HF router on `gpu_node` is
  refused at start-up — that is the check in the troubleshooting note above, and it exists
  because the failure it prevents is silent.
- **The CPU-only `onnxruntime` wheel shadowing `onnxruntime-gpu`.** Both install the same
  package name. Uninstall the CPU one first. `GPU_EXECUTION=require` turns this into a
  start-up error instead of a node that works correctly and is ten times too slow.
- **No model server running.** Detection, recognition, translation, script ID and the
  deterministic validators all still work; only SLM field extraction fails.

Verify the GPU is actually in use after start-up:

```bash
env -u LD_LIBRARY_PATH .venv/bin/python tools/dryrun_gpu.py --pages 32
```

**Do not verify with `onnxruntime.get_available_providers()`.** You will find that snippet
in a lot of advice, including earlier revisions of this guide, and it is worse than
useless here: it reports what the *wheel* was built to offer, not what a session actually
gets. Both failures on this page — the CUDA major mismatch and the missing cuDNN — print
`CUDAExecutionProvider` from that call and then run entirely on the CPU. It answers a
question next to the one you are asking.

`dryrun_gpu.py` builds a real session through `LocalProvider`, the same factory the
service uses, and reports the provider the session actually landed on. It also times both
arms and **fails a speedup between 0.90x and 1.10x**, because that band is what two CPU
arms look like when the GPU one has quietly fallen back.

---

## 8. Smoke tests

Four checks, in order. Each one proves something the next one depends on.

### 8.1 The test suite

**Windows:**

```
set PYTHONIOENCODING=utf-8
.venv\Scripts\python.exe -m pytest -q
```

**Linux:**

```
PYTHONIOENCODING=utf-8 .venv/bin/python -m pytest -q
```

**Runtime: about 2 minutes 20 seconds** on a 4-core laptop.

The suite is **2 599 tests**. On an idle machine the last line reads
`2599 passed, 1 warning in <n>s`. The one warning is a deprecation notice from the HTTP
test client and is harmless.

> **What was actually measured on the build machine, verbatim, and why it is not
> `2599 passed`:**
>
> ```
> =========================== short test summary info ===========================
> FAILED tests/unit/test_pipeline.py::test_the_head_probe_stays_inside_its_latency_budget
> 1 failed, 2598 passed, 1 warning in 136.13s (0:02:16)
> ```
>
> That run happened while another job was using the same cores. The failing test asserts a
> **timing** budget for the script-detection probe, so it is sensitive to CPU contention.
> Re-run alone on the same machine:
>
> ```
> .venv\Scripts\python.exe -m pytest -q "tests/unit/test_pipeline.py::test_the_head_probe_stays_inside_its_latency_budget"
> ```
>
> Output: `1 passed in 8.56s`.
>
> **So: run the suite on an otherwise idle machine.** If that single test is your only
> failure and it passes when run alone, your install is fine. Any *other* failure is real —
> do not proceed.

**Troubleshooting — `UnicodeEncodeError: 'charmap' codec can't encode character`:** you did
not set `PYTHONIOENCODING=utf-8`. The Windows console defaults to the legacy `cp1252`
encoding, which cannot print Devanagari. Set it and re-run; the tests themselves are fine.

**Troubleshooting — many failures mentioning `ModelArtifactMissing`:** the suite is
designed to pass with an empty `models/` directory, so this means something else. Read the
first failure's message, which names the exact path it wanted.

### 8.2 A single document through the real pipeline

This runs the actual service code over two corpus documents and writes the result files to
a scratch folder (not to `docs/eval/`, which holds the committed reference).

**Windows:**

```
.venv\Scripts\python.exe tools\run_corpus_eval.py --limit 2 --no-slm --out .\smoke\predictions --manifest .\smoke\MANIFEST.json --report .\smoke\report.txt --json-out .\smoke\report.json
```

**Linux:**

```
.venv/bin/python tools/run_corpus_eval.py --limit 2 --no-slm --out ./smoke/predictions --manifest ./smoke/MANIFEST.json --report ./smoke/report.txt --json-out ./smoke/report.json
```

`--no-slm` refuses every remote model call, which is exactly the air-gapped condition, so
this is the right smoke test for that node.

**Runtime: about 8 seconds.** Expected output, abbreviated:

```
INFO run_corpus_eval selected 2 documents (split=dev final=False types=all profiles=all limit=2)
INFO app.providers.factory provider profile active | profile=cpu_node ...
INFO app.pipeline pipeline start document=sha256:f12346ae... job=complaint_form_en_v0_clean pages=1 type=image/jpeg
INFO app.stages.s3_classify page-classifier CNN unavailable (MobileNetV3-Small page-classifier ONNX); using the fitted structural path
WARNING app.pipeline SLM unavailable for document=sha256:f12346ae... (PROVIDER_METHOD_NOT_SUPPORTED); deterministic fields stand
INFO app.pipeline pipeline done document=sha256:f12346ae... versions=1 escalated=0
INFO run_corpus_eval [1/2] complaint_form_en_v0_clean status=ok 1.9s  elapsed=2s eta=2s
...
INFO run_corpus_eval sweep done: 2 ok, 0 failed, 4s, cache hit rate 0.000 -> manifest ...\smoke\MANIFEST.json
```

**The two log lines that look alarming and are not:**

- `page-classifier CNN unavailable ... using the fitted structural path` — expected on every
  page. There is no neural page classifier in this release (section 5.1).
- `SLM unavailable ... deterministic fields stand` — expected, because you passed
  `--no-slm`. It is telling you the identifiers were still extracted.

Now look at what came out:

```
.venv\Scripts\python.exe -c "import json,pathlib; d=json.loads(pathlib.Path('smoke/predictions/complaint_form_en_v0_clean.json').read_text(encoding='utf-8')); print('quality  :', d['quality']['verdict'], round(d['quality']['score'],3)); print('lines    :', len(d['pages'][0]['lines'])); print('fields   :', sorted(d['fields'])); print('warnings :', d['warnings'])"
```

Expected output:

```
quality  : accept 1.0
lines    : 25
fields   : ['account_number', 'amount', 'bank_name', 'ifsc_code', 'transaction_date', 'utr_rrn']
warnings : ['slm_unavailable:PROVIDER_METHOD_NOT_SUPPORTED']
```

**That output is the proof the install works**: the page was read (25 text lines), the
quality gate accepted it, and six identifiers were extracted with the deterministic
validators — with no language model involved at all.

Delete `smoke/` when you are done.

**Troubleshooting — `fields : []` and `lines : 0`:** the models are not being found. Re-run
the section 5.4 verification and check `MODELS_DIR` in `.env`.

### 8.3 The quality-gate calibration check

This re-measures every corpus page against the committed calibration and fails if anything
drifted — a strong end-to-end check that your corpus, your imaging libraries and your
thresholds all agree with the reference build.

**Windows:**

```
.venv\Scripts\python.exe tools\calibrate_quality_gate.py --check
```

**Linux:**

```
.venv/bin/python tools/calibrate_quality_gate.py --check
```

**Runtime: about 66 seconds.** Expected output — exactly one line:

```
calibration reproduces the committed file
```

Exit code `0`.

**Troubleshooting — `CALIBRATION DRIFT against docs\calibration\quality_gate.json`** with
a list of differing keys: your corpus is not the reference corpus (regenerated with a
different seed, or bundle 2 came from a different commit than bundle 1), or your OpenCV
version differs from `constraints.txt`. Check the corpus first: section 6's verification
must print generator `1.2.0` and seed `1729`.

### 8.4 Re-score the committed evaluation

Optional, fast, and needs no models and no token — it reads committed result files:

```
.venv\Scripts\python.exe tools\eval_harness.py --corpus corpus --predictions docs\eval\predictions
```

**Runtime: about 1.3 seconds.** Expected output starts:

```
==============================================================================
RBI OCR corpus evaluation — harness 1.1.0
==============================================================================
corpus       corpus  (seed 1729, generator 1.2.0)
predictions  docs\eval\predictions
split        dev  final=False  locked variants [3, 4]
documents    177 scored / 177 selected / 118 withheld
```

If the header shows a different generator version or seed, your corpus does not match the
committed predictions. See `docs/eval/REPORT.md` for what the numbers mean.

---

## 9. Start the service

### 9.1 Start it

From the project root:

**Windows:**

```
.venv\Scripts\python.exe -m uvicorn app.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

**Linux:**

```
.venv/bin/python -m uvicorn app.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

`uvicorn` is the web server; `--factory` tells it that `create_app` is a function that
builds the application rather than the application itself.

Expected output:

```
INFO:     Started server process [11004]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Under the `laptop` profile you will also see one warning line before startup completes,
which is the residency banner doing its job:

```
profile=laptop routes SLM/VLM text to a non-Indian endpoint; synthetic data only (PLAN.md invariant 6)
```

> **Bind address.** `127.0.0.1` means "this machine only" and is right for the first
> smoke test. To let other machines reach it, use `--host 0.0.0.0` — and read the
> authentication note in `docs/API.md` first: **this service has no authentication of its
> own** and must sit behind your gateway.

Leave this window running and open a second one for the next steps.

### 9.2 Check health

```
curl http://127.0.0.1:8000/healthz
```

(`curl` ships with Windows 10 1803 and later. If it is missing, use
`.venv\Scripts\python.exe -c "import urllib.request,sys; sys.stdout.write(urllib.request.urlopen('http://127.0.0.1:8000/healthz').read().decode())"`.)

Expected output (formatted here for reading; the real response is one line):

```json
{
  "status": "ok",
  "profile": "laptop",
  "synthetic_only": true,
  "banner": "profile=laptop synthetic_only=true slm=google/gemma-3-4b-it hf_token=set threads=2x2",
  "provider_profile_gauge": {"laptop": 1.0, "cpu_node": 0.0, "gpu_node": 0.0},
  "models": [
    {"name": "ocr_detection",          "path": "C:\\rbi-ocr\\models\\ocr\\det.onnx",              "present": true},
    {"name": "ocr_rec_latin",          "path": "C:\\rbi-ocr\\models\\ocr\\rec_latn.onnx",         "present": true},
    {"name": "ocr_rec_devanagari",     "path": "C:\\rbi-ocr\\models\\ocr\\rec_deva.onnx",         "present": true},
    {"name": "language_id",            "path": "C:\\rbi-ocr\\models\\lid\\lid.176.ftz",           "present": true},
    {"name": "translation",            "path": "C:\\rbi-ocr\\models\\mt\\indic-en\\model.bin",    "present": true},
    {"name": "page_classifier_fusion", "path": "C:\\rbi-ocr\\models\\classifier\\fusion.json",    "present": true}
  ],
  "database": "ok",
  "webhook_enabled": false,
  "schema_version": "1.0.0"
}
```

That capture is from a `laptop`-profile instance. On the air-gapped node you configured in
section 7, `profile` reads `cpu_node`, `synthetic_only` is whatever you set, and the banner
reports `hf_token=unset` — everything else is identical.

**`status` must be `"ok"` and every `present` must be `true`.** If `status` is `"degraded"`,
find the entry with `"present": false` — the `path` tells you exactly which file did not
land — or read `database`, which carries a short error code when the database is
unreachable. The service reports `degraded` rather than failing, because it can still
ingest, refuse bad input and validate without weights; that is deliberate, not a bug.

### 9.3 Send a document with `curl`

```
curl -X POST "http://127.0.0.1:8000/v1/documents" -F "file=@corpus/complaint_form_en/complaint_form_en_v3_clean.jpg;type=image/jpeg" -F "synthetic=true" -F "sender=complainant@example.in" -F "subject=Unauthorised debit - CMS/2026/401687"
```

The `synthetic=true` field is **required under the `laptop` profile** and ignored under
`cpu_node`. Omitting it on `laptop` returns HTTP 422 `RESIDENCY_VIOLATION`.

Expected: HTTP 200, and a large JSON body. The important parts of a real response to
exactly this command, captured on the build machine:

```
job_id      f54de5494ce74189b1b5053a82002b9b   <- random per submission; yours will differ
document_id sha256:a2b81e39249e9911cb64617b1a12433203933f6f1004efca341c53ac01f9ef7d
                                               <- the sha256 of the file; yours will match
is_new      true
status      completed
bounce      {"decision": "accept", "reason": null, "detail": null}
pages       [(1, 'accept', 25 lines, language 'eng')]
fields      account_number, amount, bank_name, ifsc_code, prior_complaint_reference,
            transaction_date, utr_rrn
warnings    []
```

`prior_complaint_reference` appears only when a language model answered. On an air-gapped
node without one, expect the other six fields and
`warnings: ["slm_unavailable:PROVIDER_METHOD_NOT_SUPPORTED"]` — that is the correct
air-gapped result, not a fault.

Each field carries its own `value`, `confidence`, `source` (which mechanism produced it)
and `evidence` (page, bounding box, character span). `docs/API.md` explains every field.

**On a first request the response takes ~10 seconds** — the recognition models are loaded
lazily on first use. Subsequent requests on the same process are far faster (the measured
median end-to-end is about 1.4 s for a one-page document).

### 9.4 Open the test page in a browser

`app/review_ui/static/test_page.html` is a self-contained page — no framework, no CDN, no
web fonts — that uploads a document to the running API and displays the envelope as
something other than a wall of JSON: bounce banner, per-page quality, extracted fields with
their provenance, and evidence boxes drawn over the uploaded image.

**The page must be served from the same origin as the API.** Opening it by double-clicking
the file (a `file://` URL) puts it on a different origin from `http://127.0.0.1:8000`, and
the browser will block it from reading the API's responses — the upload appears to do
nothing. The application mounts the pages itself, so the server from §9.1 already serves
them; no launcher is needed. Open:

```
http://127.0.0.1:8000/ui/test_page.html
```

**What to do on the page, and what you should see:**

1. **Section 1, Connection.** The "API base" box is pre-filled with
   `http://127.0.0.1:8000`. Click **Check /healthz**. Expected: a status line reporting
   `ok` and the profile banner. If it reports an error, the API base is wrong or the server
   is not running.
2. **Section 2, Document.** Choose
   `corpus\complaint_form_en\complaint_form_en_v0_clean.jpg` and click **Upload to POST
   /v1/documents**. On this air-gapped node leave *"This file is synthetic test data"*
   unticked — the declaration is only required by the `laptop` profile, and the file is
   corpus data either way. On a development laptop you must tick it, and you must not tick
   it for a real complaint: see §7.2.
3. **Expected result:** an "accepted" banner (not a bounce); one page listed with quality
   verdict `accept`, quality score `1.0` and 25 recognised lines; a field table containing
   `bank_name` = *State Bank of India*, `ifsc_code` = *SBIN0904158*, `account_number`,
   `amount`, `transaction_date` and `utr_rrn`, each with a confidence and a source; and the
   uploaded image with a rectangle drawn around the text each field came from.
4. **Try a bounce.** Upload `corpus\bank_statement\bank_statement_v0_bounce.jpg`. Expected:
   a bounce banner reading `QUALITY_SCORE_BELOW_THRESHOLD` with the detail
   `quality score 0.21 is below the 0.35 threshold`, quality flags `SKEW_DETECTED`,
   `SKEW_REPAIRED`, `BLUR_ABOVE_CEILING`, `SHADOW_DETECTED`, and **no** extracted fields —
   a bounced document deliberately carries no content.

> **How much of step 3–4 was verified, and how.** The page's structure and its default API
> base come from reading `test_page.html` and `review_ui.js`; the field values, the quality
> score, the line count and the bounce reason and detail are the **real API responses** to
> those two corpus files, captured on the build machine. The browser rendering itself was
> not exercised in this session — there was no browser on the build machine — so treat the
> visual description as "what the page is built to show", and the values as measured.

**Troubleshooting — the page loads but Upload does nothing and the browser console shows a
CORS error:** you opened the file over `file://`. Use
`http://127.0.0.1:8000/ui/test_page.html` so the page and the API share an origin.

**Troubleshooting — the page reports HTTP 422 `RESIDENCY_VIOLATION` on every upload:** you
are on the `laptop` profile and the *"This file is synthetic test data"* box is unticked.
Tick it **only** if that is true. On an air-gapped `cpu_node` or `gpu_node` deployment the
declaration is not required and this error cannot occur — if you are seeing it there, the
node is running the wrong profile.

**Troubleshooting — you upload a PDF and the image panel is blank:** expected. A browser
cannot rasterise a PDF without a library, and no library may be fetched on an air-gapped
node, so PDFs render their envelope with the image panel disabled and a message saying why.

### 9.5 Watching it run: error rates

Two endpoints, both reading the same in-process counters, neither of which writes anything
to disk. **A restart resets them.** That is deliberate — this node has no metrics store and
should not grow one — so treat both as "since boot" and scrape them if you need history.

`GET /v1/stats` is the one to look at by hand:

```
curl -s http://127.0.0.1:8000/v1/stats
```

```json
{
  "uptime_seconds": 1893.4,
  "requests": 412,
  "client_errors": 7,
  "server_errors": 0,
  "unhandled_exceptions": 0,
  "error_rate": 0.016990,
  "routes": [
    {"route": "/v1/documents", "requests": 268, "client_errors": 5, "server_errors": 0, "error_rate": 0.018657}
  ],
  "documents": {"accepted": 261, "rejected": 7},
  "bounces": {"BLURRED": 4, "RESOLUTION_BELOW_FLOOR": 2}
}
```

How to read it:

| Number | Rising means |
|---|---|
| `server_errors`, `unhandled_exceptions` | **The service is faulty.** Any non-zero value here is worth investigating; check the log for the route named in `routes`. |
| `client_errors` | Callers are sending things the service refuses — wrong content type, oversize uploads, unknown job ids. Usually an integration problem, not a service one. |
| `documents.rejected` and `bounces` | **The mail stream got worse, not the service.** A refused document is a product outcome delivered as HTTP 200 with a bounce envelope and a reason, so it does not appear in `error_rate` at all. A climbing `bounces.BLURRED` is a scanner that needs attention. |
| `<unmatched>` in `routes` | Requests for paths this service does not serve — a misconfigured caller or a scanner. Static pages appear under `/ui`, so this bucket is only ever genuine 404s. |

`GET /metrics` is the same information plus every per-stage counter, in Prometheus text
format, for a scraper. `http_requests_total{status=~"[45].."}` over
`http_requests_total` is the error rate as a PromQL expression.

Neither endpoint carries a document identifier, and route labels are templates
(`/v1/jobs/{job_id}`) rather than resolved paths, so no complainant data reaches a metrics
system.

### 9.6 A note on the database

With the default `DATABASE_URL`, SQLite creates `data/runtime/app.sqlite` (plus `-wal` and
`-shm` companions) the first time the service starts. Nothing to do.

For a PostgreSQL deployment, run the migrations once before first start:

```
.venv\Scripts\python.exe -m alembic upgrade head
```

Do not rely on automatic table creation there: it produces whatever the code says today,
with no record of how it got there, which is the thing a migration history exists to
prevent.

---

## 10. Verification checklist

Tick every row before declaring the deployment done. Each references the section that
produces the evidence.

| # | Check | How | Pass looks like | ☐ |
|---|---|---|---|---|
| 1 | Python 3.12 present | `.venv\Scripts\python.exe --version` | `Python 3.12.4` | ☐ |
| 2 | Virtual environment built | §3.0 | interpreter runs from `.venv` | ☐ |
| 3 | Dependencies installed and consistent | `pip check` | `No broken requirements found.` | ☐ |
| 4 | Heavy libraries import | §3b.4 | `imports OK 1.28.0 4.8.1 5.0.0` | ☐ |
| 5 | Fonts + text shaping (only if regenerating the corpus) | §4 | `raqm shaping : True` and `SHAPING OK` | ☐ |
| 6 | All 13 model artifacts present with correct SHA-256 | §5.4 | 13 × `OK`, `FAILURES: 0` | ☐ |
| 7 | Corpus present and correct | §6 | `295` documents, generator `1.2.0`, seed `1729` | ☐ |
| 8 | `.env` created, profile correct for this node | §7 | `PROVIDER_PROFILE=cpu_node` on an air-gapped node | ☐ |
| 9 | No credential travelled with the bundle | see the command below the table | `HF_TOKEN: empty` on an air-gapped node | ☐ |
| 10 | Test suite green | §8.1 | `2599 passed` on an idle machine (see §8.1 for the one timing-sensitive test) | ☐ |
| 11 | Real pipeline reads a page | §8.2 | `quality accept 1.0`, `lines 25`, 6 fields | ☐ |
| 12 | Quality gate reproduces its calibration | §8.3 | `calibration reproduces the committed file` | ☐ |
| 13 | Service starts | §9.1 | `Application startup complete.` | ☐ |
| 14 | Health is `ok` with every model present | §9.2 | `"status": "ok"`, six × `"present": true` | ☐ |
| 15 | A document round-trips over HTTP | §9.3 | HTTP 200, `status: completed`, fields extracted | ☐ |
| 16 | A bad document bounces cleanly | §9.4 step 4 | bounce banner, machine-readable reason, no fields | ☐ |
| 17 | Metrics endpoint answers | `curl http://127.0.0.1:8000/metrics` | Prometheus text, includes `provider_profile{profile="cpu_node"} 1.0` | ☐ |
| 18 | Error rates readable | `curl http://127.0.0.1:8000/v1/stats` | JSON with `error_rate`, `routes`, `documents` (§9.6) | ☐ |
| 19 | GPU actually in use (gpu_node only) | `env -u LD_LIBRARY_PATH .venv/bin/python tools/dryrun_gpu.py --pages 32` (§7.3) | 9 passed, 0 failed; speedup well above 1.10x | ☐ |

**Row 9's command** — reports whether the token is set without printing it:

**Windows (PowerShell):**

```
.venv\Scripts\python.exe -c "from app.settings import Settings; s=Settings(); print('HF_TOKEN:', 'SET' if s.hf_token_present else 'empty'); print('stray remote credential:', s.has_stray_remote_credential); print(s.startup_banner())"
```

**Linux:** the same with `.venv/bin/python`.

Expected on an air-gapped node:

```
HF_TOKEN: empty
stray remote credential: False
profile=cpu_node synthetic_only=false slm=google/gemma-3-4b-it hf_token=unset threads=2x2
```

`stray remote credential: True` means a token is present on a profile that never reaches
out — a finding to clear before go-live, not a fault that stops the service.

---

## 11. Troubleshooting appendix

Every failure mode seen during the build, with what causes it and what fixes it.

### Installation and environment

**`SSL: CERTIFICATE_VERIFY_FAILED` on any download or model fetch.**
Your network inspects TLS with a corporate certificate authority that Python does not
trust, even though the OS and your browser do. Three layers, in order:
1. `.venv/pip.ini` trusted hosts — §3a.1. Fixes `pip` only.
2. `truststore` — §3a.4. Fixes `tools/fetch_models.py` and `tools/fetch_fonts.py`, which
   call `inject_truststore()` before any request.
3. For anything else, point Python at a bundle containing your CA, **for that process
   only**: `set SSL_CERT_FILE=C:\path\to\bundle.pem`. Build the bundle by concatenating
   `certifi`'s file with your organisation's root certificate. This is exactly what was
   done to produce `docs/eval/REPORT.md` on a machine with Norton TLS inspection
   (`docs/eval/REPORT-2026-08-08-slmlive.md` §12). **Never** disable verification, and never modify the
   machine's certificate store from a script.

**`ModuleNotFoundError` for a package you know you installed.**
You ran `python` instead of `.venv\Scripts\python.exe`. Use the full path, always.

**`ImportError: DLL load failed while importing onnxruntime_pybind11_state` (Windows).**
Missing Microsoft Visual C++ Redistributable (x64). Platform-team install; no workaround.

**`libGL.so.1: cannot open shared object file` (Linux).**
You have `opencv-python` instead of `opencv-python-headless`. Uninstall and reinstall from
the wheelhouse; `requirements.txt` pins the headless build precisely to avoid needing a
graphics stack on a server.

**`pip download` refuses a package during a cross-platform collection.**
See §3b.1 troubleshooting — retry with a newer `manylinux` tag.

**pip says `Configuration file could not be loaded. File contains no section headers.`**
Your `pip.ini` starts with a UTF-8 byte-order mark. Windows PowerShell's
`Out-File -Encoding utf8` adds one and pip cannot parse past it. Rewrite the file with
`-Encoding ascii`, or save it from your editor as "UTF-8 without BOM" — §3a.1.

### Fonts and corpus generation

**`TextShapingUnavailableError` / `features.check("raqm")` is `False`.**
Pillow cannot shape complex scripts, so Devanagari would render as detached vowel marks and
dotted circles. On Windows: `tools/assets/fribidi-0.dll` must exist and `tools/assets` must
be on the DLL search path *before* Pillow is imported (the generator does this itself —
§4). On Linux: install `libraqm` and `libfribidi` from your distribution. The generator
refuses to run rather than produce a quietly wrong corpus — that refusal is the feature.

**`FontUnavailableError: Missing font faces in .../tools/assets/fonts`.**
Fonts are not vendored in git (licences differ — `design/FONTS.md`). Run
`tools/fetch_fonts.py` on a connected machine and carry `tools/assets/fonts/` across. Do
not substitute a different Devanagari face.

### Models

**`ModelArtifactMissing: model artifact '...' is missing at ...\models\... Run
`python tools/fetch_models.py` on a networked machine and copy the resulting models/
directory to this node.`**
Exactly what it says. The message always names the precise path and the remedy. Check
§5.4, and check `MODELS_DIR` in `.env` if the path in the message is not where you put the
files. Common specific causes:
- folder named `rec_latin` instead of `rec_latn`, or `rec_devanagari` instead of `rec_deva`;
- the `models/` directory placed *beside* the project root instead of inside it;
- `mt/indic-en/` files left nested in `ctranslate2_model/` instead of copied flat.

**`ModelArtifactMissing: non-empty character dictionary for 'deva'`.**
`rec_deva_dict.txt` exists but is zero bytes — the extraction in §5.3 produced nothing, or
the copy truncated. Re-derive it and check the SHA-256.

**HTTP 403 or 401 when fetching the translation model.**
The upstream `ai4bharat` repository is *gated*: you must be signed in to Hugging Face and
have clicked through the terms on that model's page. Do that in a browser, then re-run.
This is `OPEN-QUESTIONS.md` A-9.

**HTTP 402 `Payment Required` from the language-model endpoint.**
Not a model-file problem. The account behind `HF_TOKEN` has no inference credit for that
request. It is *per request*, not account-wide — a small request may succeed while a
production-sized one is refused (`docs/eval/REPORT-2026-08-08-slmlive.md` §6). Consequence: free-text fields
are missing and the envelope carries `slm_unavailable:PROVIDER_ERROR` in `warnings`.
Deterministic identifiers are unaffected. **Resolving it means buying inference credit and
is the account owner's decision, not a deployment step.** On an air-gapped node this cannot
occur, because the language model is never called.

**HTTP 400 `model_not_supported` from the second-tier reader.**
Known open issue **Q15**: the configured second-tier vision model is not served by any
enabled provider on the build account. Escalated pages report `tier2_failed` and keep their
first-tier result. Not a deployment fault; see `OPEN-QUESTIONS.md`.

### Running the service

**Start-up fails with `RESIDENCY_VIOLATION`.**
`PROVIDER_PROFILE=laptop` with `SYNTHETIC_ONLY=false`. The refusal happens before the server
accepts a connection, on purpose. On an air-gapped node set `PROVIDER_PROFILE=cpu_node`.

**Every upload returns HTTP 422 `RESIDENCY_VIOLATION` with "the submitter must declare the
data synthetic".**
You are on the `laptop` profile and the upload did not include `synthetic=true`. Either add
the form field (test data) or move to `cpu_node` (real data). There is no bypass flag.

**HTTP 413 `UPLOAD_TOO_LARGE`.**
The file exceeds `API_MAX_UPLOAD_BYTES` (default 64 MB). The cap is enforced while reading,
not from the client-supplied length.

**`/healthz` reports `"status": "degraded"`.**
Read the `models` array for a `"present": false`, or the `database` value for a short error
code. The service answers rather than failing, so that a container orchestrator does not
restart a process whose real problem is a volume that did not mount.

**Every log line ends with `UnicodeEncodeError` / question marks instead of Devanagari.**
The Windows console is using the legacy `cp1252` code page. Set `PYTHONIOENCODING=utf-8` in
the environment the service runs under. This affects display only; stored data is always
UTF-8.

**Sporadic `ENCRYPTED_PDF` bounces on files that are merely corrupt.**
Fixed, and recorded here because the symptom is memorable. The PDF library keeps its
last-error code in a **process-global, sticky** slot and does not clear it on every failure
path, so once one encrypted PDF had passed through a worker, every later malformed PDF
inherited `ENCRYPTED_PDF` — and the complainant was told to remove a password that did not
exist. `app/stages/s1_ingest.py` now plants a known sentinel error immediately before each
load and only trusts the "encrypted" verdict when the code read back is *not* that
sentinel, holding a module lock across the pair (the library is not thread-safe regardless).
If you ever see this pattern return, that guard is what to look at
(`docs/decisions-wse1.md` E1-1).

**One test fails: `test_the_head_probe_stays_inside_its_latency_budget`.**
Timing-sensitive. Re-run it alone on an idle machine — §8.1.

**First request after start-up takes ~10 seconds.**
Expected: recognition models load lazily on first use. Send one warm-up request after
starting the service if your gateway has a short timeout.

---

## Where to go next

| You want to | Read |
|---|---|
| Call the API from your own code | `docs/API.md` |
| Change a threshold, batch size or model | `docs/TUNING.md` |
| Know how accurate it is, and on what evidence | `docs/eval/REPORT.md` |
| Know what is still open or known-broken | `OPEN-QUESTIONS.md` |
| Know why a design decision was made | `docs/decisions-*.md` |
| Understand the result JSON precisely | `docs/schema/result_envelope.schema.json` |
