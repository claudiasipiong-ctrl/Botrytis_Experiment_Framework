"""Incomplete audit groups must never claim that all checks passed."""
import audit


def test_group_with_missing_resources_is_skip():
    report = audit.Report()
    audit._group_result(report, "imports", 20, 12, [])
    assert report.rows == [("imports", None, "8/20 ejecutados correctamente; 12 omitidos; 0 fallidos")]


def test_group_failure_takes_priority_over_skips():
    report = audit.Report()
    audit._group_result(report, "imports", 20, 12, ["broken import"])
    assert report.rows[-1][1] is False and report.failed == 1


def test_help_reports_partial_execution(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(audit.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stderr="No module named 'torch'", stdout=""))
    report = audit.Report()
    audit.check_help(report)
    assert report.rows and all(status is None for _, status, _ in report.rows)
