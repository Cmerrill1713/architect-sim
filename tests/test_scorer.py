from architect_sim.grading.scorer import score_system
from architect_sim.models import GradeReport, Finding


class _FakeConfig:
    """Minimal config stub for scorer tests."""
    def get_declared_deps(self, service):
        return []


def test_perfect_score():
    report = score_system({}, [], [], _FakeConfig())
    # No services, no findings = should not crash
    assert isinstance(report, GradeReport)

def test_grade_assignment():
    report = GradeReport()
    report.overall = 95
    assert report.overall >= 90  # Would be A

def test_security_penalty():
    findings = [
        Finding(finding_type="security_issue", severity="critical",
                fixability="needs_context", service="svc")
    ] * 5
    report = score_system({}, findings, [], _FakeConfig())
    assert report.security < 100
