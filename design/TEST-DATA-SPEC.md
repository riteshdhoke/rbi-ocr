# Test Data Specification — Synthetic RBI Complaint Corpus

**Companion to** `generate_test_corpus.py` (runnable, verified) · **Version** 1.0.0

---

## 1. Why synthesis, not image generation

The obvious move is to prompt an image model for "a scanned RBI complaint form in handwritten Marathi." Three reasons not to:

1. **No ground truth.** A generated image carries no label. You would then pay humans to transcribe your own synthetic data — the exact expense synthetic data exists to avoid.
2. **Indic script comes out as noise.** Diffusion models render Devanagari as glyph-shaped nonsense. Evaluating a Devanagari recogniser against it produces a number that means nothing, which is worse than having no number.
3. **Not reproducible.** Regression testing needs the identical corpus on every run, versioned in git beside the code.

**The approach instead: render, then degrade.** Compose each page from text we control using real Unicode fonts, recording every string's position as it is drawn. Then apply physical degradation. Because we rendered it, we know exactly what was written and where — *ground truth falls out for free*, including bounding boxes that survive the geometric transforms.

**The honest limit, stated up front:** font-rendered handwriting with per-glyph jitter is not real handwriting. This corpus validates the plumbing, the schema, the escalation router, the quality gate and the throughput numbers. It does **not** validate handwriting accuracy. Any CER claim for handwritten Indic must be measured on `iiit-indic-hw-words` and on real complaint pages. That caveat is written into `index.json` so it travels with the data.

---

## 2. Corpus shape

**10 document types × 5 variants × 6 degradation profiles ≈ 290 labelled pages**, generated in under two minutes, deterministic from a seed.

| # | Type | Class | Script | Modality | Tests |
|---|---|---|---|---|---|
| 1 | `complaint_form_en` | complaint_form | Latn | print | Baseline field extraction, label/value layout |
| 2 | `complaint_form_deva` | complaint_form | Deva | print | Devanagari recognition head, script routing |
| 3 | `complaint_letter_en` | complaint_letter | Latn | handwriting | Free-form layout, extractive summary spans |
| 4 | `complaint_letter_mar` | complaint_letter | Deva | handwriting | Marathi vs Hindi disambiguation within Devanagari |
| 5 | **`bilingual_form`** | complaint_form | Latn + Deva | print labels, handwritten values | **The hard case.** Line-level LID, code-mix flag, selective translation |
| 6 | `bank_statement` | supporting_doc | Latn | print, tabular | Table structure → Tier-2 escalation trigger |
| 7 | `receipt` | supporting_doc | Latn | print, narrow | Non-A4 aspect, dense small type |
| 8 | `email_printout` | supporting_doc | Latn | print | Header/body split, prior-complaint reference |
| 9 | `cheque` | supporting_doc | Latn | mixed | Handwritten amount + MICR line, occluded fields |
| 10 | `photo_evidence` | photo_evidence | Latn | photographed | Perspective, shadow, occlusion — always degraded |

Variants differ in bank, IFSC, account number, amount, complainant, city and complaint ground — five of each, drawn from realistic pools. Every generated identifier is structurally valid: IFSC follows the 4-alpha + `0` + 6-alnum rule with a real bank prefix, so the deterministic validators in Stage 5 are genuinely exercised rather than trivially satisfied.

---

## 3. Degradation matrix

Six profiles, each a named point on the quality axis rather than random noise, so a regression can be attributed to a specific condition.

| Profile | Skew | Blur | Noise | Scale | JPEG q | Artefacts | Expected gate verdict |
|---|---|---|---|---|---|---|---|
| `clean` | — | — | — | 1.00 | 95 | — | accept |
| `light` | ±1.5° | 0.4 | 0.4% | 1.00 | 90 | — | accept |
| `scan` | ±3.5° | 0.8 | 1.2% | 0.85 | 72 | punch holes, fold, stamp | accept (repair) |
| `heavy` | ±8° | 1.5 | 3% | 0.70 | 55 | holes, fold, shadow, stamp | accept (repair), likely escalate |
| `photo` | ±12° | 1.9 | 2% | 0.62 | 45 | shadow, thumb occlusion, exposure shift | accept, escalate |
| `bounce` | ±17° | 3.4 | 7% | 0.42 | 28 | everything | **bounce** — this profile exists to test the reject path |

**Geometric transforms carry the ground truth.** Rotation and downscale apply the same matrix to every bounding box, so a `bounce`-profile page still has pixel-accurate labels. This is what makes the corpus usable for detection evaluation and not just recognition.

---

## 4. Ground-truth sidecar

Every page emits `<doc_id>.gt.json`:

```json
{
  "doc_id": "bilingual_form_v2_scan",
  "doc_type": "bilingual_form",
  "variant": 2,
  "synthetic": true,
  "document_class": "complaint_form",
  "primary_script": "Latn",
  "primary_language": "eng",
  "is_code_mixed": true,
  "degradations": ["skew:-2.8deg", "holes", "fold", "stamp", "blur:0.8", "noise:0.012", "downscale:0.85"],
  "quality_expectation": "degraded",
  "lines": [
    {
      "text": "4. Name of bank",
      "bbox": [110, 534, 341, 534, 341, 566, 110, 566],
      "script": "Latn", "language": "eng",
      "is_handwritten": false, "field": null
    },
    {
      "text": "बैंक ऑफ बड़ौदा",
      "bbox": [720, 526, 968, 526, 968, 567, 720, 567],
      "script": "Deva", "language": "mar",
      "is_handwritten": true, "field": "bank_name"
    }
  ],
  "fields": { "bank_name": "Bank of Baroda", "ifsc_code": "BARB0417293", "account_number": "..." }
}
```

Note the shape: `script`, `language` and `is_handwritten` sit on the **line**, matching finding A2 in the design review. A corpus that labelled language at the page level would make the bilingual case untestable, which is precisely how that bug survives into production.

`index.json` carries the seed, the generator version, the full document list and the handwriting caveat.

---

## 5. Adversarial set

Separate from the accuracy corpus. These test the **ingestion path**, and the pass condition is a typed rejection, never a crashed worker.

| Input | Required behaviour |
|---|---|
| `zero_byte.pdf` | reject `EMPTY_FILE` |
| `truncated.pdf` | reject `MALFORMED_PDF` |
| `not_a_pdf.pdf` (GIF magic bytes) | reject `MIME_MISMATCH` |
| `html_disguised_as_pdf.pdf` | reject `MIME_MISMATCH`; script never evaluated |
| `tiny_lowres.png` | bounce `RESOLUTION_BELOW_FLOOR` |
| `blank_page.png` | drop page `BLANK_PAGE`, do not bill OCR |
| `inverted.png` | repair by auto-invert, then process |
| *generate locally:* zip bomb | decompression-ratio cap triggers |
| *generate locally:* 5,000-page PDF | page cap triggers |
| *generate locally:* encrypted PDF | reject `ENCRYPTED_DOCUMENT` |

The last three are not committed — a zip bomb in a git repo is its own incident.

---

## 6. Fonts

Not vendored; licences differ and must be reviewed. Place in `tools/assets/fonts/`:

| File | Role | Source |
|---|---|---|
| `NotoSans-Regular.ttf`, `NotoSans-Bold.ttf` | Latin print | Google Fonts, OFL |
| `NotoSansDevanagari-Regular.ttf`, `-Bold.ttf` | Devanagari print | Google Fonts, OFL |
| `Kalam-Regular.ttf` | Handwriting, covers **Devanagari and Latin** | Google Fonts, OFL |
| `Caveat-Regular.ttf` | Handwriting, Latin | Google Fonts, OFL |
| `NotoSansMono-Regular.ttf` | Statement and MICR lines | Google Fonts, OFL |

Latin falls back to DejaVuSans if Noto is absent. **Devanagari has no fallback** — the generator exits with an explanatory error rather than rendering tofu boxes, because a corpus full of `□□□□` that silently trains a recogniser is worse than a corpus that failed to build.

---

## 7. Evaluation harness contract

The corpus is only worth what you measure with it. `eval_harness.py` consumes `index.json` and a directory of pipeline outputs, and reports:

| Metric | Cut by |
|---|---|
| CER, WER | script × modality (print/handwritten) × degradation profile |
| Detection IoU / line recall | degradation profile |
| Field precision, recall, F1 | field × source mechanism (validator / lookup / slm) |
| Schema validity rate | overall |
| Escalation precision & recall | using `quality_expectation` and `is_handwritten` as the reference |
| Quality-gate confusion matrix | accept / repair / bounce vs `quality_expectation` |
| Language ID accuracy | **line-level**, not page-level |
| p50 / p95 latency | stage × profile |

Two rules that make the harness trustworthy:

- **Run it on every merge to main.** A model swap that improves the average while destroying Devanagari handwriting must show up as a red cell, not a green average.
- **Never tune on the full corpus.** Hold out two of the five variants per type as a locked test split. Synthetic data makes overfitting to your own generator embarrassingly easy, and the resulting numbers look wonderful right up until real mail arrives.

---

## 8. Usage

```bash
# full corpus, all types, reproducible
python3 tools/generate_test_corpus.py --out ./corpus --variants 5 --seed 1729

# just the hard case, all degradation profiles
python3 tools/generate_test_corpus.py --out ./corpus --types bilingual_form --variants 5

# regenerate a single failing document without rebuilding the corpus
python3 tools/generate_test_corpus.py --out ./corpus --types complaint_letter_mar \
        --variants 5 --profiles bounce
```

Seeding is per `(seed, type, variant, profile)`, so any single document regenerates byte-identically without touching the rest — which matters the first time a CI failure points at one page and you want to look at exactly that page.
