# WS-G decisions — owner-approved extras

Judgment calls, measurements and open items from building `app/replies/**`,
`app/stages/s7_categorise.py`, `app/resources/rbios_categories.json`,
`app/review_ui/**` and their three test files. Owner merges the relevant lines into
`OPEN-QUESTIONS.md`.

Lane discipline: nothing outside that list was created or edited. `app/contracts/**`,
`app/settings.py`, `app/providers/**`, `app/resources/__init__.py`, `tests/conftest.py`
and everything under `app/api`, `app/storage`, `app/pipeline.py`, `app/tasks.py`,
`app/webhooks.py`, `alembic/` were **read only**. Two typed error families
(`ReplyTemplateMissing`/`ReplyTemplateInvalid`, `TaxonomyInvalid`/`CategoriserNotProduction`)
subclass `app.contracts.errors.PipelineError` from inside their own packages, following
the precedent WS-C set with `app.providers.base.ModelArtifactMissing`, rather than
editing the shared contracts module.

| Deliverable | Files | Tests |
|---|---|---|
| Bounce auto-replies (PRD P1-2) | `app/replies/__init__.py`, `bounce.py`, `templates/{eng,hin}/_layout.txt.j2` + 17 codes each | `tests/unit/test_replies.py` — 316 |
| Categorisation stub (PRD P1-4, owner D3) | `app/stages/s7_categorise.py`, `app/resources/rbios_categories.json` | `tests/unit/test_s7_categorise.py` — 55 |
| Reviewer pages (PRD P1-1, WS-I item 1) | `app/review_ui/__init__.py`, `static/{test_page.html,review.html,review_ui.js,review_ui.css}` | `tests/unit/test_review_ui_static.py` — 45 |

**416 tests, 1.5 s.** Ruff clean. No test in this lane loads a model, opens a socket or
reads `.env`, so no `@pytest.mark.local_models` was needed.

---

## 1. Bounce auto-replies

### D-G1 · One template file per (reason, language); the filename **is** the code

`app/replies/templates/<lang>/<BOUNCE_REASON>.txt.j2`. The lookup is therefore total by
construction, and `missing_templates()` reads the directory rather than a hard-coded
list, so a bounce reason added to the contract without a message is a failing test
(`test_every_bounce_reason_has_a_template_in_every_language`) instead of a 500 in the
mail path. All **17** `BounceReason` members are covered in both languages: 34 templates
plus 2 layouts.

A count assertion (`test_the_enum_has_not_grown_without_a_message`) pins 17, so growing
the enum lands here loudly rather than silently widening `missing_templates()`.

### D-G2 · Subject and body in one file, split on a lone `---` line

The alternatives were two files per code (68 files) or Jinja's `Template.blocks`
introspection. A layout that renders `subject\n---\nbody` keeps the reviewable prose in
one place per code, and `_split_subject_and_body` refuses a rendering with zero or two
separators, an empty half, or a multi-line subject. `---` rather than a `Subject:` prefix
so that no line of reply text can ever be mistaken for a mail header.

### D-G3 · `StrictUndefined` **plus** a post-render marker scan

`StrictUndefined` turns a missing context variable into `ReplyTemplateInvalid`. It does
**not** catch a mistyped `{{ reference_id }` — that is a compile error — nor a literal
`{{` that some other path produced. So rendering also scans the output for `{{ }} {% %}`
and refuses. Both paths are tested, including the "template does not compile" case,
which is why `get_template` is wrapped as well as `render`.

The brief's "assert no unfilled placeholders" is therefore a **runtime** guarantee, not
only a test: nothing that still looks like a template can leave this module.

### D-G4 · `BounceDecision.detail` is deliberately **not** rendered

`detail` is free text written by whichever stage bounced the page — thresholds, stage
names, measured values. It is for engineers. A complainant-facing message is built from
the *code* alone, and `test_internal_detail_never_reaches_the_complainant` pins it with a
detail string containing `laplacian_variance=...` and asserts it is absent from the
reply. The review UI displays `detail` and labels it "internal, not sent to the
complainant".

### D-G5 · Context is a closed set of six fields

`ReplyContext(reference_id, attachment_name, page_numbers, max_pages, recommended_dpi)`
with `extra="forbid"`. Every field ends up in front of a member of the public, so
widening it is a decision, not a convenience. No complainant name, no extracted value,
no bank details: a bounce means we could not read the document, so we know nothing about
it worth repeating back.

`reply_for_envelope()` prefers `job_id` over `document_id` for `reference_id` — the
document id is a `sha256:` hash, and asking somebody to quote 64 hex characters is asking
them not to reply — and derives `page_numbers` from the pages that actually bounced, so a
50-page document says "page 31" rather than making the sender rescan everything.

### D-G6 · Hindi is hand-written and committed, never machine-translated

`app/stages/s6_translate.py` exists and could produce Hindi. It is not used here. A
machine-translated apology is an artefact nobody can be held to, and OPEN-QUESTIONS Q7
already records that the MT path drops sentinels on long fields. The Hindi text is
written, reviewed and committed alongside the English.

### D-G7 · Marathi is absent on purpose

The corpus is English/Hindi/Marathi. Only two languages ship. Nobody on this build
reviewed Marathi prose for tone, and `ReplyLanguage.coerce("mar")` raises rather than
falling back to Hindi — a Devanagari message in the wrong language reads as carelessness
about exactly the person we are apologising to. Adding Marathi is 17 files and one enum
member. `test_marathi_is_deliberately_absent` records the choice.

### D-G8 · Both languages are sent by default

`render_bilingual_bounce_reply` defaults to `(eng, hin)`, English first, joined by a
60-dash rule, with subjects joined by `" / "`. Rationale: a document we could not read is
a document whose language we could not identify, so language selection has no evidence to
work from. Order is a parameter for a deployment that wants Hindi first.

### D-G9 · Plain text, `autoescape=False`

These are `text/plain` bodies; HTML escaping would corrupt an ampersand in a sender's own
filename. There is no HTML template. The module docstring states that an HTML variant
must be loaded through a *separate* environment with autoescaping on, and that a mail
layer rendering these as HTML must escape `attachment_name` itself.

### D-G10 · Tone was reviewed, and the review is machine-checked

Every English body must contain "please"; every Hindi body must contain "कृपया"; and
neither may contain any of a discouraged list — *invalid, illegal, error, failure,
failed, reject, violation, unacceptable, corrupt, you must, your fault* /
*अमान्य, त्रुटि, गलती, अस्वीकार, असफल, दोषी*. Three further wording rules are pinned:

* every reply states the complaint has **not** been closed and no decision has been taken;
* every remedy names a concrete action — the twelve quality-and-size codes quote the
  recommended DPI (`test_rescan_remedies_quote_the_dpi` renders with `recommended_dpi=400`
  and asserts 400 appears and 300 does not, so the number is genuinely interpolated);
* `ENCRYPTED_PDF` names "password"/"पासवर्ड" in both the defect and the remedy, and
  explicitly tells the sender **not** to send us the password.

The signature block is literal text in each layout ("Complaint Processing Desk" /
"शिकायत प्रसंस्करण डेस्क"). It deliberately does not claim to be RBI or any named office;
a deployment is expected to review both layouts before the first message goes out.

### D-G11 · This package renders. It never sends.

No SMTP, no queue, no address book. **PRD blocking open question 3** — *"Is a
bounce/auto-reply to the complainant permissible under RB-IOS process rules?"* — is
unanswered. A delivery path built on this must stay behind an explicit switch until the
process owner answers, and if the answer is no, the quality gate degrades to flag-only
and these templates become reviewer-facing text instead. **Recommend the owner carry this
into `OPEN-QUESTIONS.md`.**

`MalwareDetected` is correctly *not* a `BounceReason` and therefore has no template: the
contracts module already explains that naming the signature to the sender is a free oracle
for tuning the next attempt.

---

## 2. Categorisation stub (`s7_categorise.py`)

### D-G12 · The taxonomy is synthetic and says so in five places

`rbios_categories.json` carries `status: "SYNTHETIC_PLACEHOLDER"`, `authoritative: false`,
`not_production: true`, `version: "0.1.0-PLACEHOLDER"`, a `source` beginning "INVENTED FOR
DEVELOPMENT", and a `blocking_dependency` naming the PRD hard dependency. Every one of the
18 category descriptions contains the word PLACEHOLDER, asserted per category. The
`authoritative: false` convention is borrowed from `ifsc_master_sample.json` so the two
synthetic reference tables behave alike.

The model refuses `authoritative: true` together with `not_production: true` as
self-contradictory. Swapping in the real RB-IOS list is a data change: same schema, flip
both flags, and `test_a_real_taxonomy_is_a_data_change_not_a_code_change` proves the
loader and the categoriser accept it unchanged.

### D-G13 · `NotProduction` is a marker class, not a docstring

Owner decision D3 asks for the stub to be *marked*. `class NotProduction`:

* pins `is_production = False` as a class attribute, so the gate can refuse an
  implementation without running it;
* refuses via `__init_subclass__` to let a subclass exist without a
  `not_production_reason`, which is then copied onto every result;
* emits `NotProductionWarning` on construction (`default_categoriser()` is cached so the
  warning fires once per process — a warning nobody can silence is a warning everybody
  filters out).

`require_production_categoriser()` is the startup gate a deployment calls once the real
classifier exists; it raises `CategoriserNotProduction`. Nothing in the pipeline calls it
today, **because today it would refuse to start** — that is the point.

`CategorisationResult` carries the marking downstream: `is_production=False` plus a
mandatory `not_production_reason`, and the model refuses `is_production=True` on a
non-authoritative taxonomy. A consumer that ignores the field cannot say it was not told.

### D-G14 · `Categoriser` is a data protocol; `isinstance` works, `issubclass` does not

Three members (`name`, `is_production`, `categorise`). Under Python 3.12 `isinstance`
against a runtime-checkable protocol with non-method members works and `issubclass`
raises `TypeError`; both behaviours are pinned by tests so a future reader does not
"fix" the working one.

### D-G15 · The baseline counts **distinct** keywords, and ties break on file order

Scoring a category by distinct matched keywords rather than by occurrences stops one
repeated word — or one OCR stutter — outvoting a category that matched three different
ones (`test_repeating_one_word_does_not_outvote_variety`). Ranking is
`(-score, taxonomy index)`, so the tie-break depends on the committed file and never on
dict iteration or on input order (`test_ties_break_on_taxonomy_order_not_input_order`).

ASCII keywords match on word boundaries (`atm` does not fire on `atmosphere`); Devanagari
keywords match as substrings, because a boundary match on a stem written without its
inflection would never fire. A runner-up needs `ceil(0.6 × winner)` hits to be emitted as
a second tag — `ceil`, not truncation, so "60% of a winner with 3 hits" means 2, not 1.

Nothing matched returns the fallback code at confidence 0.0. An honest "not categorised"
is a better answer than the least-bad guess, and PRD Non-Goal 2 forbids the service from
making judgements about a complaint anyway.

`CategoryScore` is match evidence for a reviewer, explicitly not a priority, a severity or
a recommended disposition.

### D-G16 · Output shape is `DocumentResult.supporting_doc_tags`

`result.tags` is a `tuple[str, ...]` that drops into the existing envelope field with no
contract change (`test_tags_drop_into_the_envelope_unchanged`). WS-F wiring is one call:

```python
from app.stages.s7_categorise import categorise_document
tags = categorise_document(document_id, lines, page_class=page_class).tags
```

`categorise_document` accepts `OCRLine` objects or bare strings;
`CategorisationRequest.from_pages` flattens pages in page order. The request carries text
only — no images, no provider, no already-extracted identifiers — so this stage has
nothing to leak, and it makes no network call of any kind.

---

## 3. Reviewer pages

### D-G17 · Four files, not one — "self-contained" means *no external byte*, not *one inode*

The brief asks for one self-contained `test_page.html` **and** for a second page "sharing
the JS". Those pull in opposite directions, and the shared JS wins: duplicating ~600
lines of rendering code into two files is how the two pages drift. Shipped:

```
app/review_ui/static/test_page.html   API test page (WS-I item 1)
app/review_ui/static/review.html      keyboard-first review (PRD P1-1)
app/review_ui/static/review_ui.js     shared, classic script
app/review_ui/static/review_ui.css    shared
```

Self-containment is enforced where it matters: **no external host, no build step, no
framework, no CDN, no font, no icon**. Each page loads exactly two sibling files by
relative path, which works over `file://` (classic scripts and stylesheets load from disk;
ES modules would not, which is why the JS is ES5-style and not a module). Page-specific
wiring is an inline `<script>` at the foot of each page.

### D-G18 · The air-gap rule is mechanical and tested

`tests/unit/test_review_ui_static.py` **parses** both pages with `html.parser` and asserts:

* every `src`/`href`/`action`/`poster`/`srcset`/… value is a relative path to a file that
  exists in the static directory — no scheme, no protocol-relative `//`, no `..`;
* no `integrity`/`crossorigin` attribute (both only make sense for a third-party host);
* no `@import`, no `url(` and no `@font-face` in the CSS;
* **no `https?://` anywhere** in any of the four files, outside comments, except a single
  line carrying the marker `API-BASE-PLACEHOLDER` — and that one URL's host must be
  loopback. Comment stripping deliberately leaves `//` preceded by `:` alone, so a stray
  absolute URL inside a string cannot hide as a line comment.

Three further guard tests, cheap and load-bearing:

* **no markup sinks** — `innerHTML`, `outerHTML`, `insertAdjacentHTML`, `document.write`,
  `eval(`, `new Function(` appear nowhere. Every value rendered comes out of a complaint
  document, and all of it reaches the DOM through `textContent`;
* **no invented endpoints** — every quoted absolute path in the shared script must start
  with one of `/v1/documents`, `/v1/jobs/`, `/healthz`, `/metrics`;
* **every `RUI.q("#id")` in a page's inline script resolves to an id in that page's HTML**
  — a typo'd selector is otherwise a silently dead button.

### D-G19 · The published envelope carries no OCR lines; the page says so

`PageResult` has `line_count` and nothing else; verbatim OCR text reaches a consumer only
through `complaint_summary.spans`. `ocrLines()` reads the spans, tolerates a deployment
that attaches `pages[].lines`, and when there is neither, prints the reason instead of an
empty panel. This is a contract fact, not a gap in the page.

### D-G20 · The page image is client-side only; PDFs are refused honestly

The service never returns page images, so the overlay is drawn from the file the operator
just chose, read in the browser with `FileReader` — which also means the scan never leaves
the reviewer's machine. Boxes are drawn as polygons (all four corners), not rectangles.
Two honesty notes are on the page rather than in a comment:

* bbox coordinates are in the coordinate space of the page **after** the quality gate, so
  a deskewed or rotated page will show an offset against the original upload;
* a PDF cannot be rasterised without a library, and no library may be fetched, so PDF
  uploads render their envelope with the image panel disabled and a message saying why.

### D-G21 · The upload form-field name is an input, not a guess

`POST /v1/documents` is multipart; the field *name* is WS-F's to choose. Rather than
guessing and shipping a page that fails against the real API, the name is a text box
defaulting to `file`. Same reasoning for the API base: default is the serving origin over
http(s), the marked loopback literal over `file://`, and always editable.

### D-G22 · The review page exports decisions; it does not invent a POST route

PRD P1-1 says corrections go to the active-learning store. The published API contract has
no such endpoint, so the page builds a `rbi-ocr-review-decisions` JSON (document id, job
id, result version, and per-field `action`/`value_before`/`value_after`/`page_no`/`bbox`/
`span`/`flags_at_review`/`decided_at`) and downloads it via a `blob:` URL. When the store
gets an endpoint, posting this same payload is a few lines. `test_review_page_exports_
rather_than_inventing_an_endpoint` pins the absence of `/v1/review`, `PATCH` and `PUT`.

Keys: `j`/`↓` next, `k`/`↑` previous, `a` accept, `c` correct, `s` skip, `x` undo,
`e` export, `Enter` save, `Esc` cancel. Bindings are ignored while the reviewer is typing
in a field, except `Escape`. The legend is on the page.

### D-G23 · Hand-off: sessionStorage primary, localStorage as a one-shot carrier

The payload is a complaint — name, address, account number. `sessionStorage` dies with
the tab, so it is the primary store. But the review page opens in a **new tab**, which
does not always inherit the opener's `sessionStorage` (measured: it does not, over
`file://`, where the origin is opaque). So the stash is written to `localStorage` as well
and `loadStash()` **deletes the localStorage copy the moment it reads it**, re-seeding
`sessionStorage` for reloads. Complainant data does not sit on the reviewer's disk after
the hand-over. If the image is too large for the quota, the envelope is handed over alone
and the page asks for the image again.

### D-G24 · `app/review_ui/__init__.py` imports no web framework

WS-F mounts `STATIC_DIR`. This package only resolves paths — and `static_path()` refuses
anything that is not a bare filename, so a future request handler has a resolver that
cannot be walked out of the directory.

---

## 4. Verification performed

Beyond the 416 unit tests, both pages were driven in a real browser over `file://`
(the awkward mode — no server, opaque origin):

| Check | Result |
|---|---|
| `test_page.html` loads, no console errors | pass |
| API base defaults to the loopback literal under `file://` | pass |
| Pasted envelope renders: banner, overview, 2 page cards, 3-row field table, 2 flagged rows highlighted, summary spans, raw JSON | pass |
| `RUI.drawPage` over an 800×700 image: scaled to 603×528, both evidence polygons + labels drawn (4 555 ink pixels measured on the canvas) | pass |
| `RUI.drawCrop` around a bbox: 370×95 at 1.18× zoom, active box outlined | pass |
| Cross-tab hand-off (new tab, empty sessionStorage): envelope **and** page image recovered from localStorage, localStorage copy consumed, sessionStorage re-seeded | pass |
| Review queue: 2 flagged fields, first selected, correction pre-filled, crop drawn | pass |
| Keyboard flow: `a` accepts and advances → `Save correction` records a `correct` → `k` back → `x` undoes; queue states and decision table follow | pass |

**One bug found and fixed by that run:** `drawCrop`/`drawPage` sized the canvas from
`parentNode.clientWidth`, which a panel that has not been laid out (or is in a hidden
pane) reports as a handful of pixels — the evidence crop came out 32×8. Both now floor the
available width at 320 px (`MIN_CANVAS_WIDTH`). A 32-pixel-wide evidence crop is not
evidence.

Also measured: `sessionStorage` and `localStorage` are both usable from `file://` in the
test browser; a 26 KB stash (envelope + a small page image as a data URL) round-trips.

---

## 5. Unresolved / follow-ups for the owner

| # | Item | Recommendation |
|---|---|---|
| G-1 | **PRD open question 3 is still open**: is a bounce auto-reply to the complainant permissible under RB-IOS process rules? The templates exist; nothing sends them. | Owner to route to the RBI process owner. Until answered, any delivery path stays behind an explicit switch. |
| G-2 | The RB-IOS ground taxonomy (PRD hard dependency, blocks P1-4) is absent. `rbios_categories.json` is invented and marked as such; `KeywordCategoriser` accuracy is **unmeasured**. | Do not report a categorisation number in the M7 eval. When the real list arrives: swap the JSON, flip both flags, replace the categoriser, call `require_production_categoriser` at startup. |
| G-3 | The signature block and the "reply to this message" instruction in both layouts assume email delivery from a desk that has not been named. | Deployment must review `templates/eng/_layout.txt.j2` and `templates/hin/_layout.txt.j2` before the first live message. |
| G-4 | `ReplyContext.max_pages` / `recommended_dpi` default to 200/300 in the model rather than reading `Settings`. `reply_for_envelope` takes overrides. | One line in the WS-F call site: pass `max_pages=settings.MAX_PAGES_PER_DOC`. Deliberately not imported here so the renderer stays pure and testable without settings. |
| G-5 | Serving the pages from `file://` against a running API needs the API to allow the `null` origin in CORS. The page detects the failure and explains it, but cannot fix it. | WS-F: either mount `app.review_ui.STATIC_DIR` (preferred — then the origin matches) or allow `null` in a dev-only CORS profile. |
| G-6 | Marathi replies not written (D-G7). | 17 files + one enum member when a Marathi reviewer is available. |
| G-7 | **Not mine, but failing on `main`:** `tests/unit/test_providers_local.py` (5 tests) and `tests/unit/test_s4_ocr.py` (1) fail with `'_FakeSession' object has no attribute 'get_outputs'`. `app/providers/local.py:1141` now reads the rec model's class axis via `session.get_outputs()`, and the committed fake session in the test file was not updated. `tests/unit/test_slm_bakeoff_tool.py` also had a failure mid-run while WS-H was editing it. | WS-C/WS-H lane: add `get_outputs()` to `_FakeSession`. Not touched from this lane. |
