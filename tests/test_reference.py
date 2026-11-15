"""The tracked reference SHA-256 of the recorded outputs and the comparison script (no recorded run read)."""
import csv
import hashlib
import importlib.util
import json
import re
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
REF = REPO_DIR / "reference" / "expected_outputs_sha256.csv"
PAPER_EXPERIMENTS = {"E1", "E2", "E4", "E6", "E7", "E7b", "E10", "E9b"}


def _rows():
    with open(REF, newline="") as f:
        return list(csv.DictReader(f))


def _module():
    spec = importlib.util.spec_from_file_location("compare_outputs", REPO_DIR / "reference" / "compare_outputs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reference_rows_are_well_formed():
    rows = _rows()
    assert rows and all(re.fullmatch(r"[0-9a-f]{64}", r["sha256"]) for r in rows)
    assert {r["scope"] for r in rows} == {"run", "paper_figures"}
    assert {r["status"] for r in rows} <= {"expected", "superseded"}
    assert all(r["compare"] == "file" or r["compare"].startswith(("json_without:", "csv_columns:")) for r in rows)
    keys = [(r["scope"], r["path"]) for r in rows]
    assert len(keys) == len(set(keys))
    assert {r["experiment"] for r in rows if r["scope"] == "run"} == PAPER_EXPERIMENTS
    assert all(r["note"] for r in rows if r["status"] == "superseded" or r["compare"] != "file")
    # the four TeX tables whose comment lines were reworded are listed as such
    tex = {r["path"] for r in rows if "comment lines only" in r["note"]}
    assert tex == {"table_phase1_haghirad.tex", "table_phase1_highcap.tex", "table_phase2_indomain.tex",
                   "table_nn_disagreement.tex"}


def test_e6_predictions_of_the_paper_strategies_are_expected():
    """Only the strategies the release does not run, and the run record, are superseded."""
    e6 = [r for r in _rows() if r["experiment"] == "E6"]
    retained = ("RF_zero_shot", "XGB_zero_shot", "FTT_zero_shot", "RF_local", "XGB_local", "FTT_head", "FTT_full")
    removed = ("RF_pooled", "RF_pooled_w1", "XGB_pooled", "XGB_pooled_w1", "FTT_local")
    preds = [r for r in e6 if "/predictions/" in r["path"]]
    model = [re.search(r"__([A-Za-z]+_[a-z_0-9]+?)__seed", r["path"]).group(1) for r in preds]
    assert {m for m, r in zip(model, preds) if r["status"] == "expected"} == set(retained)
    assert {m for m, r in zip(model, preds) if r["status"] == "superseded"} == set(removed)
    assert sum(r["status"] == "expected" for r in preds) == 1698
    superseded_other = {r["path"] for r in e6 if r["status"] == "superseded" and "/predictions/" not in r["path"]
                        and not r["path"].endswith("__FTT_local.json")}
    assert superseded_other == {"E6_phase3/e6_run_record.json"}
    split = [r for r in e6 if r["path"].endswith("split_summary.csv")]
    assert split[0]["status"] == "expected" and split[0]["compare"].startswith("csv_columns:")
    assert not any(f"oov_test_votes__{m}" in split[0]["compare"] for m in removed)


def _write_reference(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "experiment", "path", "sha256", "status", "compare", "note"],
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({"scope": "run", "experiment": "EX", "note": "n", "compare": "file", **r})


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _case(tmp_path):
    """A run directory with three files and a reference covering them (plus one superseded row)."""
    mod = _module()
    run = tmp_path / "run"
    (run / "EX").mkdir(parents=True)
    (run / "EX" / "a.csv").write_text("x\n")
    detail = {"n": 3, "rate": 0.25, "analysis": "old wording"}
    (run / "EX" / "d.json").write_text(json.dumps(detail))
    (run / "EX" / "s.csv").write_text("keep,drop\n1,2\n")
    detail_mode, split_mode = "json_without:analysis", "csv_columns:keep"
    rows = [{"path": "EX/a.csv", "sha256": _sha(b"x\n"), "status": "expected"},
            {"path": "EX/d.json", "sha256": mod.digest(run / "EX" / "d.json", detail_mode), "status": "expected",
             "compare": detail_mode},
            {"path": "EX/s.csv", "sha256": mod.digest(run / "EX" / "s.csv", split_mode), "status": "expected",
             "compare": split_mode},
            {"path": "EX/old.csv", "sha256": "0" * 64, "status": "superseded"}]
    ref = tmp_path / "ref.csv"
    _write_reference(ref, rows)
    return mod, run, ref


def _status(mod, *argv):
    return mod.main([*argv])


def test_compare_complete_and_declared_semantic_differences(tmp_path, capsys):
    mod, run, ref = _case(tmp_path)
    assert _status(mod, "--run", str(run), "--reference", str(ref)) == mod.COMPLETE
    # declared differences: other wording in the excluded key, a removed column -> still complete
    (run / "EX" / "d.json").write_text(json.dumps({"rate": 0.25, "n": 3, "analysis": "new wording"}))
    (run / "EX" / "s.csv").write_text("keep\n1\n")
    assert _status(mod, "--run", str(run), "--reference", str(ref)) == mod.COMPLETE
    assert "RESULT: COMPLETE" in capsys.readouterr().out


def test_compare_fails_on_a_changed_number_or_byte(tmp_path, capsys):
    mod, run, ref = _case(tmp_path)
    (run / "EX" / "d.json").write_text(json.dumps({"n": 4, "rate": 0.25, "analysis": "old wording"}))
    assert _status(mod, "--run", str(run), "--reference", str(ref)) == mod.FAIL
    mod, run, ref = _case(tmp_path / "b")
    (run / "EX" / "a.csv").write_text("y\n")
    assert _status(mod, "--run", str(run), "--reference", str(ref)) == mod.FAIL
    assert "DIFFER   EX/a.csv" in capsys.readouterr().out


def test_compare_fails_on_absent_root_empty_selection_or_nothing_compared(tmp_path, capsys):
    mod, run, ref = _case(tmp_path)
    assert _status(mod, "--run", str(tmp_path / "absent"), "--reference", str(ref)) == mod.FAIL
    assert _status(mod, "--run", str(run), "--experiments", "EX", "--match", "nothing",
                   "--reference", str(ref)) == mod.FAIL
    empty = tmp_path / "empty"
    (empty / "EX").mkdir(parents=True)
    assert _status(mod, "--run", str(empty), "--reference", str(ref), "--partial") == mod.FAIL
    assert _status(mod, "--run", str(empty), "--experiments", "EX", "--reference", str(ref)) == mod.FAIL
    # an existing run directory without any experiment directory of the reference: empty selection
    (tmp_path / "no_experiment").mkdir()
    assert _status(mod, "--run", str(tmp_path / "no_experiment"), "--reference", str(ref)) == mod.FAIL
    out = capsys.readouterr().out
    assert out.count("RESULT: FAIL") == 5 and "COMPLETE" not in out


def test_compare_superseded_rows_never_pass_alone(tmp_path):
    mod, run, ref = _case(tmp_path)
    for name in ("a.csv", "d.json", "s.csv"):
        (run / "EX" / name).unlink()
    (run / "EX" / "old.csv").write_text("anything\n")
    assert _status(mod, "--run", str(run), "--reference", str(ref), "--partial") == mod.FAIL


def test_compare_partial_is_distinct_from_complete(tmp_path, capsys):
    mod, run, ref = _case(tmp_path)
    (run / "EX" / "s.csv").unlink()
    assert _status(mod, "--run", str(run), "--reference", str(ref)) == mod.FAIL
    assert _status(mod, "--run", str(run), "--reference", str(ref), "--partial") == mod.PARTIAL == 2
    out = capsys.readouterr().out
    assert "SKIPPED  EX/s.csv" in out and "coverage: 2 of 3" in out and "RESULT: PARTIAL" in out
