"""Iteration ledger for tracking fix loop progress."""
import json
from pathlib import Path
from ..models import IterationRecord


class Ledger:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.output_dir / "iteration_ledger.jsonl"
        self.records = []

    def append(self, record: IterationRecord):
        """Append an iteration record to the ledger."""
        self.records.append(record)
        entry = {
            "iteration": record.iteration,
            "finding_type": record.finding.finding_type if record.finding else None,
            "finding_service": record.finding.service if record.finding else None,
            "fix_applied": record.fix_applied,
            "files_changed": record.files_changed,
            "score_before": record.score_before,
            "score_after": record.score_after,
            "accepted": record.accepted,
            "reason": record.reason,
        }
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_scores(self) -> list:
        """Get the score progression."""
        return [r.score_after.get("overall", 0) for r in self.records if r.score_after]

    def check_plateau(self, window: int = 5, min_delta: float = 0.005) -> bool:
        """Check if scores have plateaued."""
        scores = self.get_scores()
        if len(scores) < window + 1:
            return False
        recent_best = max(scores[-window:])
        prior_best = max(scores[:-window])
        return recent_best <= prior_best + min_delta
