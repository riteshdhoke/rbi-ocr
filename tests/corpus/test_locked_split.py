"""The locked test split must be unreachable without --final.

TEST-DATA-SPEC section 7: two of the five variants per type are held out. Synthetic data
makes overfitting to your own generator embarrassingly easy, so the refusal has to be a
hard error rather than a warning nobody reads.

Does NOT handle: enforcing that anyone actually runs --final honestly. This is a guard
rail, not a lock.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _docs(harness, corpus_dir: Path):
    _, docs = harness.load_corpus(corpus_dir)
    return docs


def test_locked_variants_are_three_and_four(harness):
    assert harness.LOCKED_TEST_VARIANTS == frozenset({3, 4})


def test_dev_split_excludes_the_locked_variants(harness, corpus_dir: Path):
    selected = harness.select_documents(_docs(harness, corpus_dir), "dev", final=False)
    assert selected
    assert {d.variant for d in selected} == {0, 1, 2}


def test_test_split_without_final_raises(harness, corpus_dir: Path):
    with pytest.raises(harness.LockedSplitError) as exc:
        harness.select_documents(_docs(harness, corpus_dir), "test", final=False)
    assert "--final" in str(exc.value)
    assert "[3, 4]" in str(exc.value)


def test_all_split_without_final_raises(harness, corpus_dir: Path):
    with pytest.raises(harness.LockedSplitError):
        harness.select_documents(_docs(harness, corpus_dir), "all", final=False)


def test_final_unlocks_test_and_all(harness, corpus_dir: Path):
    docs = _docs(harness, corpus_dir)
    test_split = harness.select_documents(docs, "test", final=True)
    all_split = harness.select_documents(docs, "all", final=True)
    dev_split = harness.select_documents(docs, "dev", final=False)
    assert {d.variant for d in test_split} == {3, 4}
    assert len(all_split) == len(docs) == len(dev_split) + len(test_split)


def test_unknown_split_is_typed(harness, corpus_dir: Path):
    with pytest.raises(harness.UnknownSplitError):
        harness.select_documents(_docs(harness, corpus_dir), "holdout", final=True)


def test_every_type_contributes_to_both_splits(harness, corpus_dir: Path):
    docs = _docs(harness, corpus_dir)
    dev = harness.select_documents(docs, "dev", final=False)
    test = harness.select_documents(docs, "test", final=True)
    assert {d.doc_type for d in dev} == {d.doc_type for d in test}
    assert len({d.doc_type for d in dev}) == 10


def test_evaluate_refuses_the_locked_split(harness, mini_corpus: Path, tmp_path: Path):
    predictions = tmp_path / "preds"
    predictions.mkdir()
    with pytest.raises(harness.LockedSplitError):
        harness.evaluate(mini_corpus, predictions, split="test", final=False)


def test_cli_exit_code_for_locked_split(harness, mini_corpus: Path, tmp_path: Path, capsys):
    predictions = tmp_path / "preds"
    predictions.mkdir()
    code = harness.main(
        ["--corpus", str(mini_corpus), "--predictions", str(predictions), "--split", "all"]
    )
    assert code == 3
    assert "LockedSplitError" in capsys.readouterr().err


def test_report_records_what_was_withheld(harness, mini_corpus: Path, tmp_path: Path):
    predictions = tmp_path / "preds"
    predictions.mkdir()
    report = harness.evaluate(mini_corpus, predictions, split="dev")
    split = report["split"]
    assert split["name"] == "dev"
    assert split["final"] is False
    assert split["locked_variants"] == [3, 4]
    assert split["documents_selected"] == 6  # 2 types x variants 0,1,2
    assert split["documents_withheld"] == 4  # 2 types x variants 3,4
    assert len(split["missing_predictions"]) == 6


def test_missing_corpus_is_typed(harness, tmp_path: Path):
    with pytest.raises(harness.CorpusNotFoundError):
        harness.load_corpus(tmp_path / "nope")


def test_missing_predictions_dir_is_typed(harness, mini_corpus: Path, tmp_path: Path):
    with pytest.raises(harness.PredictionsNotFoundError):
        harness.evaluate(mini_corpus, tmp_path / "absent", split="dev")


def test_malformed_index_is_typed(harness, tmp_path: Path):
    corpus = tmp_path / "broken"
    corpus.mkdir()
    (corpus / "index.json").write_text(json.dumps({"seed": 1}), encoding="utf-8")
    with pytest.raises(harness.CorpusIndexInvalidError):
        harness.load_corpus(corpus)
