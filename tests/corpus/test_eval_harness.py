"""End-to-end scoring against a synthetic predictions fixture.

The fixture is built from the ground truth with deliberate, counted defects, so every
number the harness reports is known in advance: N correct lines, one corrupted line, one
wrong language, one dropped field, one hallucinated field, one document with no prediction
at all. Every metric path in the contract (TEST-DATA-SPEC section 7) is exercised.

Does NOT handle: real pipeline output. When ``app/`` exists, the contract tests there are
what prove the envelope matches this schema; this file proves the harness reacts correctly
to output that already does.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FORM = "complaint_form_en"
LETTER = "complaint_letter_mar"

SPURIOUS_TEXT = "ZZZZZZZZZZZZZZZZZZZZ"
SPURIOUS_BBOX = [5000, 5000, 5200, 5000, 5200, 5040, 5000, 5040]


# ------------------------------------------------------------------ fixture io


def _read_gt(corpus: Path, doc_id: str, doc_type: str) -> dict:
    return json.loads((corpus / doc_type / f"{doc_id}.gt.json").read_text(encoding="utf-8"))


def _pred_line(gt_line: dict, index: int, *, text: str | None = None, language: str | None = "KEEP",
               dx: float = 0.0) -> dict:
    bbox = [v + dx if i % 2 == 0 else v for i, v in enumerate(gt_line["bbox"])]
    return {
        "text": gt_line["text"] if text is None else text,
        "bbox": [float(v) for v in bbox],
        "confidence": 0.95,
        "script": gt_line["script"],
        "language": gt_line["language"] if language == "KEEP" else language,
        "is_handwritten": bool(gt_line["is_handwritten"]),
        "line_index": index,
    }


def _field(value: str, source: str = "regex_validator") -> dict:
    return {
        "value": value,
        "value_en": None,
        "normalized": value,
        "confidence": 0.99,
        "source": source,
        "validator_passed": True,
        "evidence": {"page_no": 1, "bbox": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0], "span": [0, len(value)]},
        "mt_engine": None,
    }


def _raw_field_values(gt: dict) -> dict[str, str]:
    """field name -> the verbatim string that was rendered for it."""
    out: dict[str, str] = {}
    for line in gt["lines"]:
        name = line.get("field")
        if isinstance(name, str) and name and name not in out:
            out[name] = line["text"]
    return out


def _envelope(doc_id: str, lines: list[dict] | None, fields: dict | None, *, quality: str | None,
              escalated: bool | None, timings: dict | None) -> dict:
    payload: dict = {"doc_id": doc_id, "result_version": 1, "is_final": True}
    if lines is not None:
        payload["pages"] = [{"page_no": 1, "lines": lines}]
    if fields is not None:
        payload["fields"] = fields
    if quality is not None:
        payload["quality"] = {"verdict": quality, "score": 0.88, "flags": []}
    if escalated is not None:
        payload["escalated"] = escalated
    if timings is not None:
        payload["timings_ms"] = timings
    return payload


@pytest.fixture(scope="module")
def fixture_bundle(harness, mini_corpus: Path, tmp_path_factory) -> dict:
    """Write the predictions directory and record the defects injected into it."""
    predictions = tmp_path_factory.mktemp("predictions")

    form_gt = {v: _read_gt(mini_corpus, f"{FORM}_v{v}_clean", FORM) for v in (0, 1, 2)}
    letter_gt = {v: _read_gt(mini_corpus, f"{LETTER}_v{v}_clean", LETTER) for v in (0, 1)}

    form_ref = harness.build_field_reference(form_gt[0])
    letter_ref = harness.build_field_reference(letter_gt[0])
    n_form_fields = len(form_ref)
    n_letter_fields = len(letter_ref)
    assert n_form_fields >= 5 and n_letter_fields >= 4

    # --- doc 1: a flawless full envelope -----------------------------------------
    gt = form_gt[0]
    raw = _raw_field_values(gt)
    payload = _envelope(
        f"{FORM}_v0_clean",
        [_pred_line(ln, i) for i, ln in enumerate(gt["lines"])],
        {name: _field(raw[name]) for name in sorted(form_ref)},
        quality="accept",
        escalated=False,
        timings={"s1_ingest": 10.0, "s4_ocr": 100.0},
    )
    (predictions / f"{FORM}_v0_clean.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # --- doc 2: counted defects ---------------------------------------------------
    gt = form_gt[1]
    raw = _raw_field_values(gt)
    kept = gt["lines"][:-3]  # three deletions
    lines = []
    for i, ln in enumerate(kept):
        if i == 5:
            lines.append(_pred_line(ln, i, text=ln["text"] + "X", dx=2.0))  # one insertion error
        elif i == 7:
            lines.append(_pred_line(ln, i, language="zzz", dx=2.0))  # one LID miss
        else:
            lines.append(_pred_line(ln, i, dx=2.0))
    lines.append(
        {
            "text": SPURIOUS_TEXT,
            "bbox": [float(v) for v in SPURIOUS_BBOX],
            "confidence": 0.4,
            "script": "Latn",
            "language": "eng",
            "is_handwritten": False,
            "line_index": len(lines),
        }
    )
    ordered = sorted(form_ref)
    hallucinated, dropped = ordered[0], ordered[1]
    fields = {
        name: _field(raw[name])
        for name in ordered
        if name != dropped
    }
    fields[hallucinated] = _field(raw[hallucinated] + "ZZ")  # wrong value -> false positive
    payload = _envelope(
        f"{FORM}_v1_clean",
        lines,
        fields,
        quality="bounce",  # wrong: expectation is clean
        escalated=True,  # wrong: nothing here should escalate
        timings={"s1_ingest": 20.0, "s4_ocr": 200.0},
    )
    (predictions / f"{FORM}_v1_clean.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # --- doc 3: the documented intermediate form (scored, but not schema-valid) ----
    gt = form_gt[2]
    raw = _raw_field_values(gt)
    intermediate = {
        "doc_id": f"{FORM}_v2_clean",
        "lines": [_pred_line(ln, i) for i, ln in enumerate(gt["lines"])],
        "fields": {name: raw[name] for name in sorted(form_ref)},
    }
    (predictions / f"{FORM}_v2_clean.json").write_text(
        json.dumps(intermediate, ensure_ascii=False), encoding="utf-8"
    )

    # --- doc 4: Devanagari handwriting, correctly escalated ------------------------
    gt = letter_gt[0]
    raw = _raw_field_values(gt)
    payload = _envelope(
        f"{LETTER}_v0_clean",
        [_pred_line(ln, i) for i, ln in enumerate(gt["lines"])],
        {name: _field(raw[name], source="slm") for name in sorted(letter_ref)},
        quality="accept",
        escalated=True,
        timings={"s1_ingest": 30.0, "s4_ocr": 300.0},
    )
    (predictions / f"{LETTER}_v0_clean.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # --- doc 5: total OCR failure, missed escalation, wrong verdict ----------------
    payload = _envelope(
        f"{LETTER}_v1_clean",
        [],
        None,
        quality="repair",
        escalated=False,
        timings={"s1_ingest": 40.0, "s4_ocr": 400.0},
    )
    (predictions / f"{LETTER}_v1_clean.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    # --- doc 6: complaint_letter_mar_v2_clean is deliberately absent ---------------

    def chars(doc: dict) -> int:
        return sum(len(harness.normalize_for_cer(ln["text"])) for ln in doc["lines"])

    def words(doc: dict) -> int:
        return sum(len(harness.normalize_for_cer(ln["text"]).split(" ")) for ln in doc["lines"])

    return {
        "predictions": predictions,
        "n_form_fields": n_form_fields,
        "n_letter_fields": n_letter_fields,
        "form_lines": len(form_gt[0]["lines"]),
        "letter_lines": len(letter_gt[0]["lines"]),
        # variants differ in account number, amount and reference, so their character
        # totals differ; the micro average must be computed from the real totals.
        "letter_chars": {v: chars(letter_gt[v]) for v in (0, 1)},
        "letter_words": {v: words(letter_gt[v]) for v in (0, 1)},
    }


@pytest.fixture(scope="module")
def report(harness, mini_corpus: Path, fixture_bundle) -> dict:
    return harness.evaluate(mini_corpus, fixture_bundle["predictions"], split="dev")


# --------------------------------------------------------------------- coverage


def test_split_accounting(report, fixture_bundle):
    split = report["split"]
    assert split["documents_selected"] == 6
    assert split["documents_scored"] == 5
    assert split["documents_withheld"] == 4
    assert split["missing_predictions"] == [f"{LETTER}_v2_clean"]


def test_settings_are_recorded_in_the_report(report):
    assert report["settings"]["iou_threshold"] == 0.5
    assert report["settings"]["escalation_reference"] == "quality_or_hand"
    assert report["harness_version"]


# ------------------------------------------------------------------ recognition


def test_recognition_cells_are_cut_by_script_modality_profile(report, fixture_bundle):
    cells = report["recognition"]["by_script_modality_profile"]
    assert "Latn|print|clean" in cells
    assert "Deva|hand|clean" in cells
    assert cells["Latn|print|clean"]["lines"] == 3 * fixture_bundle["form_lines"]
    assert cells["Deva|hand|clean"]["lines"] == 2 * fixture_bundle["letter_lines"]


def test_recognition_rollups_agree_with_the_cells(report):
    overall = report["recognition"]["overall"]
    by_script = report["recognition"]["by_script"]
    assert set(by_script) == {"Latn", "Deva"}
    assert overall["lines"] == sum(v["lines"] for v in by_script.values())
    assert overall["ref_chars"] == sum(v["ref_chars"] for v in by_script.values())
    assert 0.0 < overall["cer"] < 1.0


def test_a_totally_missed_document_scores_cer_one_on_its_lines(report, fixture_bundle):
    """complaint_letter_mar v1 returned no lines; v0 was perfect.

    Every v1 character is a deletion and every v0 character is correct, so the micro
    average is exactly v1's share of the Devanagari reference text — which is near but
    not equal to one half, because the variants differ in account number and amount.
    """
    deva = report["recognition"]["by_script"]["Deva"]
    chars, words = fixture_bundle["letter_chars"], fixture_bundle["letter_words"]
    assert deva["ref_chars"] == chars[0] + chars[1]
    assert deva["cer"] == pytest.approx(chars[1] / (chars[0] + chars[1]), abs=1e-9)
    assert deva["wer"] == pytest.approx(words[1] / (words[0] + words[1]), abs=1e-9)
    assert 0.45 < deva["cer"] < 0.55


def test_latin_cer_reflects_only_the_injected_errors(report, fixture_bundle):
    """One inserted character plus three deleted lines out of three form documents."""
    latin = report["recognition"]["by_script"]["Latn"]
    assert 0.0 < latin["cer"] < 0.5


# -------------------------------------------------------------------- detection


def test_detection_counts(report, fixture_bundle):
    overall = report["detection"]["overall"]
    form_lines, letter_lines = fixture_bundle["form_lines"], fixture_bundle["letter_lines"]
    assert overall["gt_lines"] == 3 * form_lines + 2 * letter_lines
    assert overall["matched"] == 2 * form_lines + (form_lines - 3) + letter_lines
    assert overall["spurious"] == 1
    assert overall["line_recall"] == pytest.approx(overall["matched"] / overall["gt_lines"])
    assert 0.9 < overall["mean_iou"] <= 1.0


def test_detection_is_reported_per_profile(report):
    by_profile = report["detection"]["by_profile"]
    assert set(by_profile) == {"clean"}
    assert by_profile["clean"]["gt_lines"] == report["detection"]["overall"]["gt_lines"]


# ----------------------------------------------------------------------- fields


def test_field_totals_match_the_injected_defects(report, fixture_bundle):
    by_field = report["fields"]["by_field"]
    n_form, n_letter = fixture_bundle["n_form_fields"], fixture_bundle["n_letter_fields"]
    assert sum(v["tp"] for v in by_field.values()) == 3 * n_form - 2 + n_letter
    # one hallucinated value; it costs precision once and recall once
    assert sum(v["fp"] for v in by_field.values()) == 1
    # the hallucinated field plus the dropped one
    assert sum(v["fn"] for v in by_field.values()) == 2


def test_field_precision_recall_f1_are_present_per_field(report):
    for name, cell in report["fields"]["by_field"].items():
        assert cell["tp"] + cell["fp"] + cell["fn"] > 0, name
        if cell["tp"] + cell["fp"]:
            assert 0.0 <= cell["precision"] <= 1.0


def test_fields_are_attributed_to_their_source_mechanism(report, fixture_bundle):
    by_source = report["fields"]["by_source"]
    n_form, n_letter = fixture_bundle["n_form_fields"], fixture_bundle["n_letter_fields"]
    assert set(by_source) == {"regex_validator", "slm", "unspecified"}
    assert by_source["regex_validator"]["tp"] == 2 * n_form - 2
    assert by_source["regex_validator"]["fp"] == 1
    assert by_source["slm"]["tp"] == n_letter
    assert by_source["unspecified"]["tp"] == n_form  # the intermediate bare-string form


def test_recall_by_source_decomposes_overall_recall(report):
    """Each source's recall is its share of the ground truth, so the shares add up."""
    by_field = report["fields"]["by_field"]
    by_source = report["fields"]["by_source"]
    gt_total = sum(report["fields"]["gt_occurrences"].values())
    assert gt_total == sum(cell["tp"] + cell["fn"] for cell in by_field.values())
    assert all(cell["gt_total"] == gt_total for cell in by_source.values())
    overall_tp = sum(cell["tp"] for cell in by_source.values())
    assert sum(cell["recall"] for cell in by_source.values()) == pytest.approx(overall_tp / gt_total)


def test_a_source_that_ignores_most_fields_does_not_score_recall_one(report, fixture_bundle):
    slm = report["fields"]["by_source"]["slm"]
    assert slm["precision"] == pytest.approx(1.0)
    assert slm["recall"] < 0.5  # it only claimed the Devanagari letter's fields


def test_field_by_source_cells_use_that_field_as_the_denominator(report, fixture_bundle):
    cells = report["fields"]["by_field_source"]
    assert any(key.endswith("|slm") for key in cells)
    assert any(key.endswith("|regex_validator") for key in cells)
    assert any(key.endswith("|unspecified") for key in cells)
    for key, cell in cells.items():
        name = key.split("|", 1)[0]
        assert cell["gt_total"] == report["fields"]["gt_occurrences"].get(name, 0)


def test_a_document_with_no_fields_block_is_not_penalised(report, fixture_bundle):
    """complaint_letter_mar v1 emitted no fields; it must not manufacture false negatives."""
    assert sum(v["fn"] for v in report["fields"]["by_field"].values()) == 2
    gt_total = sum(report["fields"]["gt_occurrences"].values())
    n_form, n_letter = fixture_bundle["n_form_fields"], fixture_bundle["n_letter_fields"]
    assert gt_total == 3 * n_form + n_letter  # the fields-less document contributes nothing


def test_a_wrong_value_costs_precision_and_recall(harness, report):
    """The one hallucinated field must show up as an FP and an FN on the same field."""
    offenders = [
        (name, cell) for name, cell in report["fields"]["by_field"].items() if cell["fp"]
    ]
    assert len(offenders) == 1
    cell = offenders[0][1]
    assert cell["fp"] == 1 and cell["fn"] == 1
    assert cell["precision"] < 1.0 and cell["recall"] < 1.0


# ----------------------------------------------------------------------- schema


def test_schema_validity_rate(report):
    schema = report["schema"]
    assert schema["checked"] == 5
    assert schema["valid"] == 4
    assert schema["validity_rate"] == pytest.approx(0.8)


def test_schema_errors_name_the_offending_document(report):
    assert any(e["doc_id"] == f"{FORM}_v2_clean" for e in report["schema"]["examples"])


def test_validate_prediction_accepts_a_complete_envelope(harness):
    payload = _envelope(
        "d",
        [
            {
                "text": "x",
                "bbox": [0, 0, 1, 0, 1, 1, 0, 1],
                "confidence": 0.5,
                "script": "Latn",
                "language": "eng",
                "is_handwritten": False,
                "line_index": 0,
            }
        ],
        {"ifsc_code": _field("BARB0417293")},
        quality="accept",
        escalated=False,
        timings={"s4_ocr": 1.0},
    )
    assert harness.validate_prediction(payload, "d") == []


@pytest.mark.parametrize(
    "mutate,fragment",
    [
        (lambda p: p.pop("pages"), "pages[] missing"),
        (lambda p: p["fields"]["ifsc_code"].pop("evidence"), "evidence"),
        (lambda p: p["fields"]["ifsc_code"].update(source="vibes"), "source"),
        (lambda p: p["pages"][0]["lines"][0].update(bbox=[0, 0, 1, 1]), "bbox"),
        (lambda p: p["pages"][0]["lines"][0].update(confidence=1.7), "confidence"),
        (lambda p: p["pages"][0]["lines"][0].pop("is_handwritten"), "is_handwritten"),
        (lambda p: p["quality"].update(verdict="maybe"), "quality.verdict"),
        (lambda p: p.update(doc_id="other"), "does not match"),
        (lambda p: p["fields"]["ifsc_code"].update(value_en=7), "value_en"),
    ],
)
def test_validate_prediction_rejects_contract_violations(harness, mutate, fragment):
    payload = _envelope(
        "d",
        [
            {
                "text": "x",
                "bbox": [0, 0, 1, 0, 1, 1, 0, 1],
                "confidence": 0.5,
                "script": "Latn",
                "language": "eng",
                "is_handwritten": False,
                "line_index": 0,
            }
        ],
        {"ifsc_code": _field("BARB0417293")},
        quality="accept",
        escalated=False,
        timings={"s4_ocr": 1.0},
    )
    mutate(payload)
    errors = harness.validate_prediction(payload, "d")
    assert errors, fragment
    assert any(fragment in e for e in errors), errors


# -------------------------------------------------- envelope vocabulary (Q6)


def _one_doc_report(harness, mini_corpus: Path, tmp_path: Path, doc_id: str, doc_type: str,
                    fields: dict) -> dict:
    """Score a predictions directory holding exactly one document."""
    predictions = tmp_path / "single"
    predictions.mkdir()
    (predictions / f"{doc_id}.json").write_text(
        json.dumps(
            _envelope(doc_id, None, fields, quality=None, escalated=None, timings=None),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return harness.evaluate(mini_corpus, predictions, split="dev", types=[doc_type])


def test_an_amount_claim_scores_against_the_corpus_disputed_amount(
    harness, mini_corpus: Path, tmp_path: Path
):
    """Q6's aggregate distortion: `amount` had 69 FP and 0 GT because of the name alone."""
    doc_id = f"{FORM}_v0_clean"
    gt = _read_gt(mini_corpus, doc_id, FORM)
    rendered = gt["fields_bilingual"]["disputed_amount"]["value"]
    report = _one_doc_report(harness, mini_corpus, tmp_path, doc_id, FORM,
                             {"amount": _field(rendered)})
    by_field = report["fields"]["by_field"]
    assert by_field["disputed_amount"]["tp"] == 1
    assert by_field["disputed_amount"]["fp"] == 0
    assert "amount" not in by_field, "reporting must key on the corpus name, not the envelope's"


def test_the_alias_table_travels_in_the_report(report):
    aliases = report["settings"]["envelope_field_aliases"]
    assert aliases["utr_rrn"] == "transaction_reference"
    assert aliases["card_fragment"] == "card_last_four"


# ---------------------------------------------------- bilingual slots (Q6)


def _bilingual(report: dict) -> dict:
    return report["fields"]["bilingual"]


def test_a_native_script_value_with_a_translation_is_scored_in_both_slots(
    harness, mini_corpus: Path, tmp_path: Path
):
    doc_id = f"{LETTER}_v0_clean"
    spec = _read_gt(mini_corpus, doc_id, LETTER)["fields_bilingual"]["complainant_name"]
    claim = _field(spec["value"], source="slm")
    claim["value_en"] = spec["value_en"]
    counts = _bilingual(_one_doc_report(harness, mini_corpus, tmp_path, doc_id, LETTER,
                                        {"complainant_name": claim}))
    assert counts["value_is_native"] == 1
    assert counts["value_is_translation"] == 0
    assert counts["value_en_correct"] == 1
    assert counts["gt_bilingual_fields"] >= 1


def test_a_translation_in_the_immutable_value_is_counted_separately(
    harness, mini_corpus: Path, tmp_path: Path
):
    """`value` is the page's own string; putting the English there is a contract breach.

    It still counts as finding the field — precision and recall are unchanged — but the
    breach has to be visible somewhere, and this is where.
    """
    doc_id = f"{LETTER}_v0_clean"
    spec = _read_gt(mini_corpus, doc_id, LETTER)["fields_bilingual"]["complainant_name"]
    report = _one_doc_report(harness, mini_corpus, tmp_path, doc_id, LETTER,
                             {"complainant_name": _field(spec["value_en"], source="slm")})
    counts = _bilingual(report)
    assert counts["value_is_translation"] == 1
    assert counts["value_is_native"] == 0
    assert counts["value_en_absent"] == 1
    # the headline metric is deliberately unmoved by this
    assert report["fields"]["by_field"]["complainant_name"]["tp"] == 1


def test_a_latin_only_document_contributes_nothing_to_the_bilingual_table(
    harness, mini_corpus: Path, tmp_path: Path
):
    doc_id = f"{FORM}_v0_clean"
    gt = _read_gt(mini_corpus, doc_id, FORM)
    rendered = gt["fields_bilingual"]["ifsc_code"]["value"]
    counts = _bilingual(_one_doc_report(harness, mini_corpus, tmp_path, doc_id, FORM,
                                        {"ifsc_code": _field(rendered)}))
    assert counts == dict.fromkeys(counts, 0)


# ------------------------------------------------------------------- escalation


def test_escalation_precision_and_recall(report):
    esc = report["escalation"]
    assert (esc["tp"], esc["fp"], esc["fn"]) == (1, 1, 1)
    assert esc["precision"] == pytest.approx(0.5)
    assert esc["recall"] == pytest.approx(0.5)
    assert esc["f1"] == pytest.approx(0.5)
    assert esc["reference_rule"] == "quality_or_hand"


def test_escalation_reference_rule_changes_the_numbers(harness, mini_corpus: Path, fixture_bundle):
    quality_only = harness.evaluate(
        mini_corpus, fixture_bundle["predictions"], split="dev", escalation_reference="quality"
    )["escalation"]
    # under `quality` alone nothing in a clean-profile corpus should escalate
    assert quality_only["tp"] == 0
    assert quality_only["fp"] == 2


# ----------------------------------------------------------------- quality gate


def test_quality_gate_confusion_matrix(report):
    quality = report["quality_gate"]
    assert quality["scored"] == 4
    assert quality["confusion"]["clean"] == {"accept": 2, "repair": 1, "bounce": 1}
    assert quality["confusion"]["degraded"] == {"accept": 0, "repair": 0, "bounce": 0}
    assert quality["expectation_satisfied_rate"] == pytest.approx(0.5)


# ------------------------------------------------------------------ language id


def test_language_id_is_line_level(report, fixture_bundle):
    lid = report["language_id"]["overall"]
    form_lines, letter_lines = fixture_bundle["form_lines"], fixture_bundle["letter_lines"]
    assert lid["gt_lines"] == 3 * form_lines + 2 * letter_lines
    assert lid["matched"] == 2 * form_lines + (form_lines - 3) + letter_lines
    assert lid["correct"] == lid["matched"] - 1  # exactly one injected LID miss
    assert lid["accuracy_matched"] == pytest.approx(lid["correct"] / lid["matched"])
    assert lid["accuracy_all_lines"] < lid["accuracy_matched"]


def test_language_id_is_cut_by_script_and_modality(report):
    cells = report["language_id"]["by_script_modality"]
    assert set(cells) == {"Latn|print", "Deva|hand"}
    assert cells["Deva|hand"]["accuracy_matched"] == pytest.approx(1.0)
    assert cells["Deva|hand"]["coverage"] == pytest.approx(0.5)


# ---------------------------------------------------------------------- latency


def test_latency_percentiles_per_stage(report):
    stages = report["latency_ms"]["by_stage"]
    assert set(stages) == {"s1_ingest", "s4_ocr"}
    assert stages["s1_ingest"]["n"] == 4
    assert stages["s1_ingest"]["p50"] == pytest.approx(25.0)
    assert stages["s1_ingest"]["p95"] == pytest.approx(38.5)
    assert stages["s4_ocr"]["p50"] == pytest.approx(250.0)


def test_latency_is_also_cut_by_profile(report):
    assert "s4_ocr|clean" in report["latency_ms"]["by_stage_profile"]


# -------------------------------------------------------------------- rendering


def test_human_readable_report_renders(harness, report):
    text = harness.render_report(report)
    for heading in (
        "Recognition — CER / WER by script x modality x profile",
        "Detection — IoU and line recall by profile",
        "Fields — precision / recall / F1 by field",
        "Fields — by source mechanism",
        "Fields — bilingual slots",
        "Schema validity",
        "Escalation vs reference rule",
        "Quality gate — confusion",
        "Language ID — LINE level",
        "Latency (ms) by stage",
    ):
        assert heading in text, heading
    assert "locked test split and were not scored" in text
    assert "iiit-indic-hw-words" in text


def test_cli_writes_json_and_table(harness, mini_corpus: Path, fixture_bundle, tmp_path: Path, capsys):
    out = tmp_path / "nested" / "report.json"
    code = harness.main(
        [
            "--corpus",
            str(mini_corpus),
            "--predictions",
            str(fixture_bundle["predictions"]),
            "--json-out",
            str(out),
        ]
    )
    assert code == 0
    captured = capsys.readouterr().out
    assert "RBI OCR corpus evaluation" in captured
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["split"]["name"] == "dev"
    assert written["recognition"]["overall"]["lines"] > 0


def test_cli_final_run_scores_everything(harness, mini_corpus: Path, fixture_bundle, capsys):
    code = harness.main(
        [
            "--corpus",
            str(mini_corpus),
            "--predictions",
            str(fixture_bundle["predictions"]),
            "--split",
            "all",
            "--final",
            "--quiet",
        ]
    )
    assert code == 0
    assert capsys.readouterr().out == ""


def test_filters_narrow_the_scored_set(harness, mini_corpus: Path, fixture_bundle):
    report = harness.evaluate(
        mini_corpus, fixture_bundle["predictions"], split="dev", types=[LETTER]
    )
    assert report["split"]["documents_scored"] == 2
    assert set(report["recognition"]["by_script"]) == {"Deva"}
