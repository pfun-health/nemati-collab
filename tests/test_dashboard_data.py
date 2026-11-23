import json
import runpy
import sys
from pathlib import Path

dashboard = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "examples/fantasia_dashboard_data.py")
)


def test_failed_subject_entry_is_not_used_for_dashboard_fit(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard["pc"], "list_subjects", lambda *_: ["f2y03"])
    monkeypatch.setattr(dashboard["pc"], "subject_features", lambda *_: object())

    def subject_entry(*_):
        raise KeyError("missing RSA summary")

    def example_fit(*_):
        raise AssertionError("failed subject must not be fitted")

    monkeypatch.setitem(dashboard, "subject_entry", subject_entry)
    monkeypatch.setitem(dashboard, "example_fit", example_fit)
    output = tmp_path / "dashboard.json"
    monkeypatch.setattr(sys, "argv", ["fantasia_dashboard_data.py", str(output)])

    dashboard["main"]()

    result = json.loads(output.read_text())
    assert result["subjects"] == []
    assert "f2y03" in result["failed"]
    assert result["fit"] is None
