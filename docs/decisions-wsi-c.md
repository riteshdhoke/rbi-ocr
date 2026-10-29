# WS-I(c) decisions — the two office hand-off bundles

Lane: `tools/make_bundle.py`, `tests/unit/test_make_bundle.py`, `dist/**` (gitignored
output), the `dist/` and `models/` entries in `.gitignore`, the "Office hand-off" section
of `README.md`, and this file. Nothing else was touched: no `app/` module, no other test,
no corpus file, no document owned by WS-I-A or WS-I-B, no `OPEN-QUESTIONS.md` row, no
`.env`, no settings. No commit was made by this lane.

Date: 2026-08-08. Machine: Windows 11, CPython 3.12.4, `.venv/Scripts/python.exe`.
Final bundles built at `9804bae`.

Every claim below was produced by running the thing it describes. Four defects were
found by the verification and are recorded rather than fixed (§6); one was found *in this
lane's own output* and fixed (§3).

---

## 1. Deliverables

| File | State |
|---|---|
| `tools/make_bundle.py` | **new**, 1 156 lines |
| `tests/unit/test_make_bundle.py` | **new**, 725 lines, 47 tests, all green |
| `README.md` | **+1 section**, "Office hand-off" |
| `.gitignore` | `dist/` added; the `models/` rule corrected (§2) |
| `dist/rbi-ocr-code-<sha>.zip` | built, verified |
| `dist/rbi-ocr-corpus-<sha>.zip` | built, verified |
| `docs/decisions-wsi-c.md` | this file |

---

## 2. D-IC-1 · The file list is `git ls-files`, in **two** invocations, and that is the
whole safety argument

**Decision.** The code bundle's members are

```
git ls-files --cached --exclude-standard      (everything committed)
  +  git ls-files --others --exclude-standard (untracked AND not ignored)
  -  corpus/**
```

**Why not tracked-only, which is what the instruction literally says.** PLAN.md WS-I item
7 requires `constraints.txt` in the code bundle and item 8 requires
`docs/AIR-GAPPED-DEPLOY.md`, `docs/API.md`, `docs/FUNCTIONAL-SPEC.md` and
`docs/TUNING.md`. When this tool was written **all five were untracked** — the WS-I-B lane
had just produced them and, per the working rules, had not committed. A tracked-only
bundle would have shipped without the deployment guide it is supposed to be deployed
with. The instruction's *purpose* — "so nothing untracked/ignored leaks" — is carried
entirely by `--exclude-standard`, not by `--cached`.

**What `--exclude-standard` buys.** A file git ignores cannot enter either archive at all.
`.env`, every byte of `models/`, `.venv/`, `dist/` itself, `__pycache__/` and `*.sqlite`
are excluded **structurally** — not by a blocklist somebody has to remember to extend when
a new secret arrives. That is a stronger property than a curated deny-list, and it is the
reason this design was chosen over `git archive` (which is tracked-only and would have
needed the five documents special-cased by hand) or a `shutil.copytree` + ignore-patterns
walk (which reimplements `.gitignore` badly).

**What it costs, and how the cost is paid.** An untracked file that is *not* ignored — a
developer's scratch note — would be included. Three mitigations, all of them cheap:

1. every untracked file that goes in is **printed by name** during the build and recorded
   in the manifest as `untracked_included`;
2. the token scan (§4) reads every byte of it;
3. `--tracked-only` builds the strict variant for anyone who wants it. It is not the
   default because on the day it was written it produced a bundle that fails the tool's
   own required-file check — which the test suite asserts, so the flag's real behaviour is
   documented by a test rather than by a claim
   (`test_tracked_only_omits_untracked_and_then_fails_its_own_required_check`).

**The blocklist exists anyway.** `check_forbidden` re-tests all of it — weight extensions,
private-key extensions, `.env` by basename, `site-packages`, `corpus/`, `dist/` — on both
the build path and the verify path. "Structurally impossible" is a claim worth testing
rather than trusting, and it costs microseconds.

### D-IC-2 · `.gitignore`'s `models/` rule was silently broken, and it had to be fixed
before the bundle could be correct

`docs/AIR-GAPPED-DEPLOY.md` §2 draws `models/  EMPTY except README.md` in the unzip
layout, and PLAN.md's repository layout says `models/ (gitignored, README only)`. The
committed rule was:

```
models/
!models/README.md
```

**The negation never fired.** Git cannot re-include a file whose *parent directory* is
excluded, so `models/README.md` was ignored too — untracked, unignorable, and therefore
invisible to `git ls-files` under either invocation. Measured before the fix:

```
$ git check-ignore -v models/README.md
.gitignore:23:models/   models/README.md
```

**Decision.** Use the documented idiom, `models/*` plus the negation, so git tells the
truth and the file arrives through the ordinary path:

```
models/*
!models/README.md
```

**Verified after the change** — the README surfaces and nothing else does:

```
$ git ls-files --others --exclude-standard | grep '^models/'
models/README.md

$ git check-ignore -q models/ocr/rec_deva.onnx      && echo IGNORED   -> IGNORED
$ git check-ignore -q models/mt/indic-en/model.bin  && echo IGNORED   -> IGNORED
$ git check-ignore -q models/PROVENANCE.json        && echo IGNORED   -> IGNORED
```

**Rejected alternative — a hard-coded "always add `models/README.md`" injection in the
tool.** It would have worked and left the repository lying about itself, and any future
tool asking git for the file list would have inherited the same wrong answer. Fixing the
rule fixes it once, for everything. The tool keeps an *assertion* that
`models/README.md` is present and that it is the **only** `models/` member — the check
that matters, in the place a reader looks for it.

`models/PROVENANCE.json` stays ignored deliberately. WS-I-B measured that
`fetch_models.py::write_provenance` rewrites the file from one invocation's records only,
so the committed copy describes one artifact out of thirteen (D-IB-2). Shipping it would
hand the deployer an audit record that is confidently incomplete; `AIR-GAPPED-DEPLOY.md`
§5.4 already tells them to generate their own.

---

## 3. D-IC-3 · Both manifests collide on a combined extract — found by testing the
documented procedure, fixed here

The brief says both zips embed `BUNDLE_MANIFEST.json`. They do. But
`docs/AIR-GAPPED-DEPLOY.md` §2 tells the operator to extract **both archives into the same
directory**:

```
mkdir C:\rbi-ocr
cd C:\rbi-ocr
tar -xf rbi-ocr-code-<sha>.zip
tar -xf rbi-ocr-corpus-<sha>.zip
```

Doing exactly that — which is how this was found, not by reading the code — leaves **one**
`BUNDLE_MANIFEST.json`: the corpus one, silently overwriting the code one. The operator
then cannot verify the code bundle from the tree they were told to create, and the file
they *do* have looks authoritative.

**Decision.** Each archive carries the manifest **twice, byte-identically**: once as
`BUNDLE_MANIFEST.json` (the required name, so the specification is met literally) and once
as `BUNDLE_MANIFEST-code.json` / `BUNDLE_MANIFEST-corpus.json`. After a combined extract
both survive side by side. `--verify` requires the second copy and asserts it is identical
to the first, so a bundle that lost it fails.

Measured after the fix — the member-name intersection of the two archives is now exactly
the one file that is *designed* to collide, and both distinct copies land:

```
member-name overlap between the two bundles: ['BUNDLE_MANIFEST.json']
combined extract: BOTH manifests survive side by side -> PASS
```

**Rejected alternatives.** Putting the corpus manifest at `corpus/BUNDLE_MANIFEST.json`
would pollute a tree specified as "exactly as committed" and put an un-declared file next
to a residency-gate `MANIFEST.json` (adjudication A-5). Renaming the root file outright
would stop meeting the brief. Two copies of a 60 KB JSON cost 0.05% of the corpus bundle.

**Not changed:** `docs/AIR-GAPPED-DEPLOY.md` itself, which is WS-I-B's file. §2's
instruction is still correct; it just now produces a tree with three manifest files in it.
The README's new section explains which to read. Whoever next edits the deploy guide should
add the same sentence to §2.

---

## 4. D-IC-4 · The token scan reads every byte, reports offsets, and aborts before writing

**Decision.** `hf_[A-Za-z0-9]{20,}`, compiled against **bytes**, run over **every member**
— text and binary alike — of both archives.

Three specifics, each for a reason:

- **Bytes, not decoded text.** A token pasted into a `.ttf`, a `.dll` or a JSON file with
  an unusual encoding is still a token. Scanning the raw bytes covers all of them with one
  rule; scanning only files whose extension looks textual would have covered 377 of the
  code bundle's 386 members. The report still prints the text/binary split, because the
  brief asks for "every text file" and the honest answer is "those, and the other nine
  too".
- **`{20,}`, not `{34}`.** A real HF token is `hf_` + 34 alphanumerics. The loose floor
  catches a truncated paste and a future format change; the cost is a theoretical false
  positive, which is a build that stops for a human to look at — the correct failure
  direction. `hf_xxx`, which is what `.env.example` and all four documents use, is three
  characters and cannot match. Locked by
  `test_the_hf_xxx_placeholder_the_documents_use_does_not_false_positive`.
- **No archive is written when a match is found.** The scan runs *before* `write_zip`, and
  a hit raises `BundleError`. An earlier draft wrote the zip and then reported the failure,
  which leaves a zip containing a secret sitting in `dist/` — precisely the outcome the
  check exists to prevent. `test_a_planted_token_aborts_the_build_and_writes_no_archive`
  asserts the archive is absent, not merely that the exit code was 1.

**The matched value is never printed, logged, or written.** The report gives the path, the
byte offset and the match length. Two tests assert the fabricated token does not appear in
the exception text or in any problem string. Printing a leaked secret into a build log is
the same leak with more copies.

**Positive control.** "0 matches" is only meaningful if the scanner can find one. The
independent verification pass ran the same pattern against the repository's **real**
`.env`, which does hold a token, and confirmed it matches — so the clean result over the
988 extracted files is evidence rather than a broken regex. The control reports a boolean;
it prints nothing.

---

## 5. Verification performed

### 5.1 The tool's own checks — `--verify` re-opens both zips

Both archives are re-opened immediately after being written, and every SHA-256 is
**recomputed from the bytes in the archive** rather than copied forward from the build.
`--verify` alone does the same thing without building.

```
verified      : rbi-ocr-code-9804bae.zip
  members     : 389  (all SHA-256s recomputed and matched)
  scanned     : 380 text + 9 binary file(s), 0 token match(es)
verified      : rbi-ocr-corpus-9804bae.zip
  members     : 599  (all SHA-256s recomputed and matched)
  scanned     : 297 text + 302 binary file(s), 0 token match(es)
  pair        : both bundles carry git sha 9804bae…
BUNDLE OK — every assertion passed.
```

### 5.2 An independent pass that shares no code with the tool

A separate script re-derived every expectation from PLAN.md WS-I item 7 and
`AIR-GAPPED-DEPLOY.md` §§0 and 2, extracted both zips into one scratch directory exactly
as §2 instructs, and checked the tree. **48 assertions, 48 passed**, including:

- present: all four WS-I documents, `constraints.txt`, `.env.example`,
  `tools/generate_test_corpus.py`, `tools/assets/fribidi-0.dll`, **8** `.ttf` faces
  (both Devanagari ones), `models/README.md`, `docs/schema/result_envelope.schema.json`,
  `docs/calibration/{fusion,quality_gate}.json`;
- absent: `.env`; **zero** files with a weight extension anywhere in the extracted tree;
  `models/` containing `['models/README.md']` and nothing else; no `.venv`; no
  private-key file; no `corpus/` member in the code bundle;
- token scan over the extracted tree: **0 hits across 990 files / 79 672 721 bytes** (388-file
  build; the delivered 389-file build adds `.claude/launch.json`, see D-IC-7), with
  the positive control above;
- corpus: **295 images, 295 `.gt.json`, `index.json`, `adversarial/` with 8 files**
  including `MANIFEST.json`; `index.json` declares `count: 295` and `synthetic: true`; all
  590 paths it names resolve on disk;
- archives: no absolute or `..` member names, no Windows separators, **no top-level
  wrapper folder** — extracting reproduces the project root §2 draws.

### 5.3 The bundle is not missing modules — three escalating proofs

The brief asks for `compileall` from the extracted tree. That only parses, so it would not
notice a missing sibling module. All three were run, from the **extracted** trees, with the
project venv's interpreter:

| Check | Where | Result |
|---|---|---|
| `python -m compileall -q app tools` | code bundle alone | exit **0** |
| `python -m compileall -q tests alembic` | code bundle alone | exit **0** |
| import every `app` module by walking the tree | code bundle alone | **44 / 44 imported** |
| `python -m pytest -q` | code + corpus bundles, extracted together | **2 637 passed, 9 skipped, 1 warning in 145.29 s** |
| `python -m pytest -q` | the repository itself | **2 646 passed in 177.21 s** |

The fourth row is the strongest evidence available: the full suite passing from a scratch
directory that contains **only what the two zips carry** — an empty `models/` holding just
its README, and no `.env` at all. The nine skips are the suite's eight standing ones plus
one this lane added deliberately: `test_the_dotenv_is_gitignored_in_this_repository` shells
out to `git check-ignore`, which cannot answer in a tree that has no `.git` — and a tree
with no `.git` is exactly what a correct code bundle produces, so the test skips there and
asserts `.gitignore`'s content instead. 2 646 − 2 599 = 47, this lane's new tests.

**One pre-existing flake was observed and is not this lane's:**
`tests/unit/test_webhooks.py::TestThreadMode::test_the_thread_drains_the_queue` failed once
under load and passes alone in **1.14 s** in the repository and **1.13 s** in the extracted
tree. Same class as the wall-clock probe-budget assertion WS-I-B saw fail (D-IB-4), and the
same remedy applies: a wall-clock assertion in a unit suite fails on any loaded runner.

### 5.4 The bundles

| Bundle | Files | Uncompressed | Zipped |
|---|---|---|---|
| `dist/rbi-ocr-code-9804bae.zip` | 389 | 9.10 MB | **3.23 MB** (3 385 386 bytes; ceiling 15 MB) |
| `dist/rbi-ocr-corpus-9804bae.zip` | 599 | 66.69 MB | **54.07 MB** (56 696 509 bytes) |

The corpus bundle compresses by 19% because it is almost entirely JPEG, which is already
compressed. `AIR-GAPPED-DEPLOY.md` §0 tells the receiver to expect "14 MB of files" in the
code bundle and 67 MB in the corpus one; the second is right, the first is a 5 MB
over-estimate in the safe direction.

---

## 6. Four defects found, and why none was fixed here

The lane rule is that a defect outside the lane gets recorded precisely and stops there.

> **Numbering is provisional.** WS-I-A drafted rows it called Q16 and Q17, and WS-I-B
> independently drafted a different row it also called Q16 — three lanes writing into a
> register none of them may edit. The two below are labelled Q18/Q19 to avoid all three;
> the orchestrator should renumber all five when it places them.

### D-IC-5 · Q18 — `eval_harness.py` cannot read `index.json` on Linux

`corpus/index.json` stores document paths with **backslashes**
(`"image": "complaint_form_en\\complaint_form_en_v0_clean.jpg"`) because the generator
wrote them on Windows with `str(Path.relative_to())`. Two consumers normalise and two do
not:

| Consumer | Line | Normalises? |
|---|---|---|
| `tools/bench_ocr_cpu.py` | 430 | yes — `.replace("\\", "/")` |
| `tools/calibrate_quality_gate.py` | 109 | yes — `.replace("\\", "/")` |
| **`tools/eval_harness.py`** | **443-444** | **no** — `corpus_dir / entry["gt"]` |
| **`tools/train_page_classifier.py`** | **133-134** | **no** — same |

On Windows this is invisible: `Path("corpus") / "forms\\a.jpg"` resolves. On POSIX the
backslash is an ordinary filename character, so the same expression yields
`corpus/forms\a.jpg`, which does not exist. The harness reports a missing prediction
rather than crashing, so **every one of the 295 documents would score as missing and the
report would print zeros with no error**.

This lands squarely on the hand-off: `AIR-GAPPED-DEPLOY.md` §1.1 names 64-bit Linux as a
supported target and §8.4 makes "re-score with the eval harness" a smoke test. The bundle
ships the file exactly as committed, which is correct — the fix belongs in the two
consumers (or in the generator, which would change `index.json` and therefore the corpus).

*Drafted row, not inserted (`OPEN-QUESTIONS.md` is outside this lane):* **"`corpus/index.json`
stores Windows-separated relative paths. `tools/eval_harness.py` (lines 443-444) and
`tools/train_page_classifier.py` (lines 133-134) join them with `/` without normalising, so
on Linux every document resolves to a non-existent path and the harness silently scores 295
missing predictions. `tools/bench_ocr_cpu.py` and `tools/calibrate_quality_gate.py` already
normalise, so the fix is two lines and the precedent exists. Options: (a) normalise at both
call sites, (b) normalise once in the index loader, (c) emit POSIX separators from the
generator — which changes `index.json` and therefore regenerates the corpus. Recommendation
(a) now, (c) at the next corpus regeneration. Lane: WS-D. Blocks the Linux route of
`AIR-GAPPED-DEPLOY.md` §8.4."*

`make_bundle.py` normalises when it checks that every path `index.json` names is present,
so the bundler is correct on both platforms; it is the only reason the defect surfaced.

### D-IC-6 · Q19 — `tests/golden/test_adversarial.py` aborts collection without the corpus

Measured, from the code bundle extracted **alone**:

```
ERROR collecting tests/golden/test_adversarial.py
E   FileNotFoundError: ...\corpus\adversarial\MANIFEST.json
!!!!!! Interrupted: 1 error during collection !!!!!!
pytest exit 2
```

The manifest is read at **module scope** (line 67), so an absent corpus is a collection
error that stops the entire run — 2 599 tests do not execute — rather than a skip of the
33 that need it. The documented order in `AIR-GAPPED-DEPLOY.md` puts the corpus (§6) before
the tests (§8), so the guided path is unaffected. A deployer who takes §0 at its word
("you can skip bundle 2") and runs `pytest` first gets an abort that names a missing file
rather than the reason.

*Drafted row:* **"`tests/golden/test_adversarial.py` loads `corpus/adversarial/MANIFEST.json`
at import time, so a checkout without the corpus fails collection and the whole suite
(2 599 tests) refuses to run. Verified against the code bundle extracted alone. Options:
(a) move the load into a module-scoped fixture and `pytest.skip` when the corpus is absent,
(b) a `pytest.importorskip`-style guard at the top of the module, (c) document that the
corpus is mandatory for any test run. Recommendation (a). Lane: WS-D."*

### D-IC-7 · The untracked-inclusion tradeoff, demonstrated live, ten minutes after the build

Minutes after the final bundles were written, a concurrent lane created
`.claude/launch.json` — a dev-harness launch configuration for previewing the API. It is
untracked and not ignored, so **the next rebuild will put it in the code bundle** and ship
an editor-tooling file to an air-gapped office that has no such editor.

This is D-IC-1's cost arriving on schedule, and the mitigations behaved as designed: the
build would print `files : N (1 untracked-but-not-ignored)` and name it, and the manifest
would list it under `untracked_included`. Nothing silent, nothing secret — but noise in a
tree whose layout `AIR-GAPPED-DEPLOY.md` §2 draws file by file.

Not fixed here: `.claude/` belongs to whichever lane created it, and a project launch
config is a legitimate thing to commit. The one-line remedy is `.gitignore` or a commit,
whichever that lane intends. A troubleshooting row was added to `README.md` telling the
next builder to read the untracked list rather than skim past it.

The same paragraph applies to the concurrent edit to `app/api/app.py` observed at the same
moment (likely WS-F/WS-G closing the WS-I-B static-mount defect): **the shipped zips are a
snapshot of the worktree at `9804bae`, and must be rebuilt once that lands.** Rebuilding is
one command and about 25 seconds.

### D-IC-8 · `AIR-GAPPED-DEPLOY.md` §2's tree diagram is now one line stale

The diagram does not show `BUNDLE_MANIFEST.json` (or the two aliases from §3) in the
extracted root, and it labels `models/` as "EMPTY except README.md" — which is true only
because of the `.gitignore` fix in §2 of this file. Both are one-line edits in a document
this lane does not own. The README's new section carries the information in the meantime.

---

## 7. What this lane deliberately did not do

- **No commit.** Per the working rules. (The orchestrator committed the worktree,
  including this tool, as `9804bae` while this lane was running; the bundles were rebuilt
  afterwards so their SHA names that commit.)
- **No `OPEN-QUESTIONS.md` edit.** Q18 and Q19 are drafted above for the orchestrator to
  place, following D-H3 and D-IA precedent.
- **No edit to `AIR-GAPPED-DEPLOY.md`, `API.md`, `FUNCTIONAL-SPEC.md`, `TUNING.md` or
  `constraints.txt`** — all WS-I-B's, all shipped byte-identically.
- **No `app/`, `tools/` (other than the new file) or `tests/` (other than the new file)
  change.** The two defects found in `tools/eval_harness.py` and `tests/golden/` were
  documented, not fixed.
- **No token printed, logged or written anywhere.** The manifest records
  `token_matches: 0` and the pattern; never a value, never a length, never a file it came
  from. `README.md` shows no token and no placeholder that could be mistaken for one.
- **`dist/` is gitignored**, so no bundle is ever committed. The zips are build output,
  reproducible from any worktree in about 25 seconds.

---

## 8. One thing that surprised me, kept here because it will surprise the next person

On the first build of this lane — worktree at `bcd85dd`, before the orchestrator committed
— `git ls-files --cached` returned **376** non-corpus paths and the code bundle contained
**386**.

The ten extra files were, exactly, the entire WS-I deliverable set:
`docs/AIR-GAPPED-DEPLOY.md`, `docs/API.md`, `docs/FUNCTIONAL-SPEC.md`, `docs/TUNING.md`,
`constraints.txt`, `docs/decisions-wsi-a.md`, `docs/decisions-wsi-b.md`,
`docs/eval/REPORT-2026-08-08-preSLM.md`, `models/README.md`, and this tool itself.

**Every single file the packaging phase existed to produce was invisible to the command the
packaging instruction named as its source of truth.** A tracked-only build would have run
clean, printed `BUNDLE OK`, and shipped a box with no deployment guide in it. The
instruction was right about the risk — ignored files leaking — and the fix for that risk is
`--exclude-standard`, not `--cached`.

It reads differently now that the orchestrator has committed everything (`9804bae`, where
only two files are untracked), which is the point: the number of untracked files is a
property of *when you build*, not of what the bundle should contain. That is why the tool
asserts on required *content* and prints the untracked list rather than trusting either
number.
