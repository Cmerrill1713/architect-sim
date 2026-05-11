"""Persistent score history -- track grades over time per project.

Stores scores in a JSON file at {project_root}/.architect-sim/history.json
so you can see if the system is improving or regressing.
"""
import json
import datetime
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class ScoreSnapshot:
    timestamp: str
    commit: str           # git commit hash
    overall: float
    grade: str
    completeness: float
    contract_fidelity: float
    resilience: float
    dependency_hygiene: float
    coverage: float
    security: float
    total_services: int
    total_endpoints: int
    total_calls: int
    findings_critical: int
    findings_warning: int
    findings_info: int
    fixes_applied: int


class ScoreHistory:
    def __init__(self, root: str):
        self.root = Path(root)
        self.history_dir = self.root / ".architect-sim"
        self.history_file = self.history_dir / "history.json"
        self.snapshots = []
        self._load()

    def _load(self):
        if self.history_file.exists():
            try:
                with open(self.history_file) as f:
                    data = json.load(f)
                self.snapshots = data.get("snapshots", [])
            except (json.JSONDecodeError, OSError):
                self.snapshots = []

    def record(self, grade_report, findings: list, blueprints: dict, fixes_applied: int = 0):
        """Record a new score snapshot."""
        import subprocess

        # Get current git commit
        commit = ""
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--format=%H"],
                cwd=str(self.root),
                capture_output=True, text=True, timeout=5,
            )
            commit = result.stdout.strip()[:12]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        snapshot = ScoreSnapshot(
            timestamp=datetime.datetime.now().isoformat(),
            commit=commit,
            overall=round(grade_report.overall, 2),
            grade=grade_report.grade,
            completeness=round(grade_report.completeness, 2),
            contract_fidelity=round(grade_report.contract_fidelity, 2),
            resilience=round(grade_report.resilience, 2),
            dependency_hygiene=round(grade_report.dependency_hygiene, 2),
            coverage=round(grade_report.coverage, 2),
            security=round(grade_report.security, 2),
            total_services=len(blueprints),
            total_endpoints=sum(len(bp.endpoints) for bp in blueprints.values()),
            total_calls=sum(len(bp.outbound_calls) for bp in blueprints.values()),
            findings_critical=sum(1 for f in findings if f.severity == "critical"),
            findings_warning=sum(1 for f in findings if f.severity == "warning"),
            findings_info=sum(1 for f in findings if f.severity == "info"),
            fixes_applied=fixes_applied,
        )

        self.snapshots.append(asdict(snapshot))
        self._save()
        return snapshot

    def _save(self):
        self.history_dir.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, "w") as f:
            json.dump({"snapshots": self.snapshots}, f, indent=2)

    def get_trend(self, last_n: int = 10) -> list:
        """Get the last N snapshots."""
        return self.snapshots[-last_n:]

    def format_trend(self, last_n: int = 10) -> str:
        """Format score trend as markdown."""
        snapshots = self.get_trend(last_n)
        if not snapshots:
            return "No history recorded yet.\n"

        lines = ["## Score History\n"]
        lines.append("| Date | Commit | Grade | Score | Critical | Fixes |")
        lines.append("|------|--------|-------|-------|----------|-------|")

        for s in snapshots:
            date = s["timestamp"][:10]
            commit = s.get("commit", "")[:8]
            lines.append(
                f"| {date} | {commit} | {s['grade']} | {s['overall']:.1f} | "
                f"{s.get('findings_critical', '?')} | {s.get('fixes_applied', 0)} |"
            )

        # Show trend
        if len(snapshots) >= 2:
            first = snapshots[0]["overall"]
            last = snapshots[-1]["overall"]
            delta = last - first
            direction = "improving" if delta > 0 else "declining" if delta < 0 else "stable"
            lines.append(f"\n**Trend:** {direction} ({'+' if delta > 0 else ''}{delta:.1f} over {len(snapshots)} runs)")

        return "\n".join(lines)
