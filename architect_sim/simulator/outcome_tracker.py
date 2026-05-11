"""Outcome learning and feedback loop for architect-sim.

Tracks what actually happened after recommendations were applied, learns which
predictions were accurate, and adjusts future confidence scores.

Every run records predictions. On subsequent runs (after changes), it compares
the new state to old predictions and learns:
- "Predicted adding retries would improve resilience to 78%. Actual: 82%. Accuracy: 95%."
- "Predicted removing dead ports would resolve 79 findings. Actual: 71. Accuracy: 90%."

Over time, probabilities get more accurate -- the tracker learns which types of
changes actually deliver results.
"""
import json
import uuid
import datetime
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Prediction:
    prediction_id: str
    timestamp: str
    commit: str
    scenario_name: str
    predicted_score_delta: float
    predicted_dimension_deltas: dict
    predicted_findings_resolved: int
    probability: float
    status: str  # "pending", "verified", "expired"


@dataclass
class Outcome:
    prediction_id: str
    timestamp: str
    commit: str
    actual_score_delta: float
    actual_dimension_deltas: dict
    actual_findings_resolved: int
    accuracy: float
    notes: str


@dataclass
class LearningState:
    total_predictions: int = 0
    verified_predictions: int = 0
    avg_accuracy: float = 0.0
    category_accuracy: dict = field(default_factory=dict)
    overconfident_categories: list = field(default_factory=list)
    underconfident_categories: list = field(default_factory=list)
    confidence_adjustment: float = 1.0


# -- Accuracy math --

def _calculate_accuracy(predicted_delta: float, actual_delta: float) -> float:
    """Calculate how accurate a prediction was.

    Perfect prediction: accuracy = 1.0
    Off by 50%: accuracy = 0.5
    Wrong direction (predicted improvement, got regression): accuracy = 0.0

    If same sign: 1.0 - |predicted - actual| / max(|predicted|, |actual|, 1)
    If different signs: 0.0
    """
    # Both zero -- perfect
    if predicted_delta == 0 and actual_delta == 0:
        return 1.0

    # Different signs -- wrong direction is a total miss
    if (predicted_delta > 0) != (actual_delta > 0) and predicted_delta != 0 and actual_delta != 0:
        return 0.0

    # One is zero, the other is not -- partial miss proportional to magnitude
    if predicted_delta == 0 or actual_delta == 0:
        nonzero = max(abs(predicted_delta), abs(actual_delta))
        return max(0.0, 1.0 - nonzero / max(nonzero, 1.0))

    denom = max(abs(predicted_delta), abs(actual_delta), 1.0)
    return max(0.0, 1.0 - abs(predicted_delta - actual_delta) / denom)


def _get_current_commit(root: str) -> str:
    """Get the current short git commit hash."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%H"],
            cwd=root, capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip()[:12]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return ""


def _categorize_scenario(scenario_name: str) -> str:
    """Extract a broad category from a scenario name for per-category learning.

    Maps scenario names like "Add retry wrappers" -> "resilience",
    "Fix 12 phantom calls" -> "completeness", etc.
    """
    lower = scenario_name.lower()
    if any(w in lower for w in ("retry", "circuit", "timeout", "fallback", "resilien")):
        return "resilience"
    if any(w in lower for w in ("phantom", "missing endpoint", "endpoint", "404")):
        return "completeness"
    if any(w in lower for w in ("mismatch", "contract", "schema", "field")):
        return "contract_fidelity"
    if any(w in lower for w in ("circular", "undeclared", "dependency", "dep")):
        return "dependency_hygiene"
    if any(w in lower for w in ("orphan", "dead", "unused", "remove", "cleanup")):
        return "coverage"
    if any(w in lower for w in ("security", "auth", "inject", "vuln")):
        return "security"
    if any(w in lower for w in ("llm", "model", "routing", "optimization")):
        return "llm_optimization"
    return "general"


class OutcomeTracker:
    """Tracks predictions, verifies outcomes, and learns confidence adjustments."""

    # Predictions older than this (days) without verification get expired
    EXPIRY_DAYS = 30

    def __init__(self, root: str):
        self.root = Path(root)
        self.data_dir = self.root / ".architect-sim"
        self.data_file = self.data_dir / "outcomes.json"
        self.predictions: list[dict] = []
        self.outcomes: list[dict] = []
        self.learning_state = LearningState()
        self._load()

    def _load(self):
        if self.data_file.exists():
            try:
                with open(self.data_file) as f:
                    data = json.load(f)
                self.predictions = data.get("predictions", [])
                self.outcomes = data.get("outcomes", [])
                ls = data.get("learning_state", {})
                self.learning_state = LearningState(
                    total_predictions=ls.get("total_predictions", 0),
                    verified_predictions=ls.get("verified_predictions", 0),
                    avg_accuracy=ls.get("avg_accuracy", 0.0),
                    category_accuracy=ls.get("category_accuracy", {}),
                    overconfident_categories=ls.get("overconfident_categories", []),
                    underconfident_categories=ls.get("underconfident_categories", []),
                    confidence_adjustment=ls.get("confidence_adjustment", 1.0),
                )
            except (json.JSONDecodeError, OSError):
                pass

    def _save(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "predictions": self.predictions,
            "outcomes": self.outcomes,
            "learning_state": asdict(self.learning_state),
        }
        with open(self.data_file, "w") as f:
            json.dump(data, f, indent=2)

    # -- Recording --

    def record_prediction(
        self,
        scenario_name: str,
        predicted_delta: float,
        predicted_dims: dict,
        predicted_resolved: int,
        probability: float,
        commit: str = "",
    ) -> str:
        """Record a new prediction for later verification.

        Returns the prediction_id.
        """
        if not commit:
            commit = _get_current_commit(str(self.root))

        pred = Prediction(
            prediction_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now().isoformat(),
            commit=commit,
            scenario_name=scenario_name,
            predicted_score_delta=predicted_delta,
            predicted_dimension_deltas=predicted_dims,
            predicted_findings_resolved=predicted_resolved,
            probability=probability,
            status="pending",
        )
        self.predictions.append(asdict(pred))
        self.learning_state.total_predictions += 1
        self._save()
        return pred.prediction_id

    # -- Verification --

    def verify_predictions(
        self,
        current_grade,
        current_findings: list,
        current_commit: str = "",
    ) -> list[dict]:
        """Compare current state against pending predictions.

        For each pending prediction:
        1. Check if the commit has changed (work was done)
        2. Compare predicted vs actual score deltas
        3. Calculate accuracy
        4. Update learning state
        5. Mark prediction as verified

        Args:
            current_grade: GradeReport from the latest run
            current_findings: Current list of Finding objects
            current_commit: Current git commit (auto-detected if empty)

        Returns:
            List of newly created outcome dicts
        """
        if not current_commit:
            current_commit = _get_current_commit(str(self.root))

        new_outcomes = []
        now = datetime.datetime.now()

        # Build a quick lookup for the previous run's baseline from the most recent
        # verified outcome or from the oldest pending prediction's implied baseline
        prev_score = self._get_baseline_score()

        current_finding_count = len(current_findings)
        current_overall = current_grade.overall

        for pred in self.predictions:
            if pred["status"] != "pending":
                continue

            # Expire old predictions
            pred_time = datetime.datetime.fromisoformat(pred["timestamp"])
            if (now - pred_time).days > self.EXPIRY_DAYS:
                pred["status"] = "expired"
                continue

            # Only verify if the commit changed (someone did work)
            if pred["commit"] == current_commit and current_commit:
                continue

            # Calculate actual deltas
            actual_score_delta = current_overall - prev_score

            # Per-dimension deltas
            actual_dims = {}
            for dim in ("completeness", "contract_fidelity", "resilience",
                        "dependency_hygiene", "coverage", "security"):
                actual_dims[dim] = getattr(current_grade, dim, 0.0)

            # Findings resolved: we compare predicted resolved vs actual change
            # We don't have the exact baseline count, so use predicted_findings_resolved
            # as the expectation and compare direction/magnitude
            predicted_resolved = pred["predicted_findings_resolved"]
            predicted_delta = pred["predicted_score_delta"]

            # Score accuracy
            score_accuracy = _calculate_accuracy(predicted_delta, actual_score_delta)

            # Dimension accuracy (average across predicted dimensions)
            dim_accuracies = []
            predicted_dims = pred.get("predicted_dimension_deltas", {})
            for dim, predicted_dim_delta in predicted_dims.items():
                actual_dim_val = actual_dims.get(dim, 0.0)
                # We need a baseline per dimension -- approximate from the predicted delta
                # by using the relationship: baseline + predicted_delta ≈ current
                # So actual_delta ≈ actual_val - (actual_val - actual_score_delta * weight)
                # Simplify: just compare predicted dim delta to proportional actual change
                dim_acc = _calculate_accuracy(predicted_dim_delta, actual_dim_val - (actual_dim_val - actual_score_delta))
                dim_accuracies.append(dim_acc)

            overall_accuracy = score_accuracy
            if dim_accuracies:
                # Weight score accuracy at 60%, dimension accuracy at 40%
                dim_avg = sum(dim_accuracies) / len(dim_accuracies)
                overall_accuracy = 0.6 * score_accuracy + 0.4 * dim_avg

            # Generate notes
            notes = self._generate_notes(pred, actual_score_delta, overall_accuracy)

            outcome = Outcome(
                prediction_id=pred["prediction_id"],
                timestamp=now.isoformat(),
                commit=current_commit,
                actual_score_delta=round(actual_score_delta, 2),
                actual_dimension_deltas={k: round(v, 2) for k, v in actual_dims.items()},
                actual_findings_resolved=max(0, predicted_resolved - current_finding_count)
                if predicted_resolved > 0 else 0,
                accuracy=round(overall_accuracy, 3),
                notes=notes,
            )

            self.outcomes.append(asdict(outcome))
            new_outcomes.append(asdict(outcome))
            pred["status"] = "verified"

        # Update learning state from all verified outcomes
        if new_outcomes:
            self._update_learning_state()
            self._update_confidence_adjustments()

        self._save()
        return new_outcomes

    def _get_baseline_score(self) -> float:
        """Get the implied baseline score from the oldest pending prediction.

        If a prediction says "score will improve by +5" and was made when score was 70,
        the baseline is 70. We infer this from score history if available.
        """
        # Try score history first
        history_file = self.data_dir / "history.json"
        if history_file.exists():
            try:
                with open(history_file) as f:
                    data = json.load(f)
                snapshots = data.get("snapshots", [])
                if snapshots:
                    return snapshots[-1].get("overall", 0.0)
            except (json.JSONDecodeError, OSError):
                pass

        # Fallback: use the most recent outcome's implied score
        if self.outcomes:
            last = self.outcomes[-1]
            return last.get("actual_score_delta", 0.0)

        return 0.0

    def _generate_notes(self, pred: dict, actual_delta: float, accuracy: float) -> str:
        """Auto-generate an explanation of prediction vs actual."""
        predicted_delta = pred["predicted_score_delta"]
        scenario = pred["scenario_name"]

        if accuracy >= 0.9:
            quality = "Excellent prediction"
        elif accuracy >= 0.7:
            quality = "Good prediction"
        elif accuracy >= 0.5:
            quality = "Fair prediction"
        elif accuracy > 0.0:
            quality = "Poor prediction"
        else:
            quality = "Wrong direction"

        parts = [f"{quality} for '{scenario}'."]
        parts.append(f"Predicted delta: {predicted_delta:+.1f}, actual: {actual_delta:+.1f}.")

        if predicted_delta > 0 and actual_delta > predicted_delta:
            parts.append("Underestimated the improvement.")
        elif predicted_delta > 0 and actual_delta < predicted_delta and actual_delta > 0:
            parts.append("Overestimated the improvement.")
        elif predicted_delta > 0 and actual_delta <= 0:
            parts.append("Predicted improvement but score regressed.")
        elif predicted_delta < 0 and actual_delta < predicted_delta:
            parts.append("Regression was worse than predicted.")

        return " ".join(parts)

    # -- Learning --

    def _update_learning_state(self):
        """Recompute learning state from all verified outcomes."""
        verified = [o for o in self.outcomes]
        if not verified:
            return

        self.learning_state.verified_predictions = len(verified)
        self.learning_state.avg_accuracy = (
            sum(o["accuracy"] for o in verified) / len(verified)
        )

        # Per-category accuracy
        category_outcomes: dict[str, list[float]] = {}
        for o in verified:
            # Find the matching prediction
            pred = self._find_prediction(o["prediction_id"])
            if not pred:
                continue
            cat = _categorize_scenario(pred["scenario_name"])
            category_outcomes.setdefault(cat, []).append(o["accuracy"])

        self.learning_state.category_accuracy = {
            cat: round(sum(accs) / len(accs), 3)
            for cat, accs in category_outcomes.items()
        }

    def _update_confidence_adjustments(self):
        """Learn from verified predictions to adjust future confidence.

        For each category, compare avg predicted improvement vs avg actual improvement.
        If we consistently overpredict by 20%: adjustment = 0.8
        If we consistently underpredict by 10%: adjustment = 1.1

        Global adjustment is the mean of all category adjustments.
        """
        category_predicted: dict[str, list[float]] = {}
        category_actual: dict[str, list[float]] = {}

        for o in self.outcomes:
            pred = self._find_prediction(o["prediction_id"])
            if not pred:
                continue
            cat = _categorize_scenario(pred["scenario_name"])
            category_predicted.setdefault(cat, []).append(pred["predicted_score_delta"])
            category_actual.setdefault(cat, []).append(o["actual_score_delta"])

        overconfident = []
        underconfident = []
        adjustments = []

        for cat in category_predicted:
            if cat not in category_actual:
                continue
            avg_pred = sum(category_predicted[cat]) / len(category_predicted[cat])
            avg_actual = sum(category_actual[cat]) / len(category_actual[cat])

            if abs(avg_pred) < 0.01:
                # Can't compute ratio from near-zero predictions
                adjustments.append(1.0)
                continue

            ratio = avg_actual / avg_pred if avg_pred != 0 else 1.0

            # Clamp to reasonable range [0.5, 1.5]
            adjustment = max(0.5, min(1.5, ratio))
            adjustments.append(adjustment)

            if ratio < 0.85:
                overconfident.append(cat)
            elif ratio > 1.15:
                underconfident.append(cat)

        self.learning_state.overconfident_categories = sorted(overconfident)
        self.learning_state.underconfident_categories = sorted(underconfident)

        if adjustments:
            self.learning_state.confidence_adjustment = round(
                sum(adjustments) / len(adjustments), 3
            )
        else:
            self.learning_state.confidence_adjustment = 1.0

    def _find_prediction(self, prediction_id: str) -> Optional[dict]:
        """Find a prediction by ID."""
        for p in self.predictions:
            if p["prediction_id"] == prediction_id:
                return p
        return None

    # -- Query --

    def get_confidence_adjustment(self, category: str) -> float:
        """Get the learned confidence adjustment for a category.

        If we've been overconfident on "resilience" predictions,
        returns a factor < 1.0 to reduce future confidence.
        If underconfident, returns > 1.0.

        Falls back to the global adjustment if no category-specific data.
        """
        cat_acc = self.learning_state.category_accuracy.get(category)
        if cat_acc is None:
            return self.learning_state.confidence_adjustment

        # Rebuild per-category adjustment from prediction/outcome pairs
        predicted_deltas = []
        actual_deltas = []
        for o in self.outcomes:
            pred = self._find_prediction(o["prediction_id"])
            if not pred:
                continue
            if _categorize_scenario(pred["scenario_name"]) != category:
                continue
            predicted_deltas.append(pred["predicted_score_delta"])
            actual_deltas.append(o["actual_score_delta"])

        if not predicted_deltas:
            return self.learning_state.confidence_adjustment

        avg_pred = sum(predicted_deltas) / len(predicted_deltas)
        avg_actual = sum(actual_deltas) / len(actual_deltas)

        if abs(avg_pred) < 0.01:
            return self.learning_state.confidence_adjustment

        ratio = avg_actual / avg_pred
        return max(0.5, min(1.5, round(ratio, 3)))

    def get_pending_count(self) -> int:
        """How many predictions are still awaiting verification."""
        return sum(1 for p in self.predictions if p["status"] == "pending")

    def get_verified_count(self) -> int:
        """How many predictions have been verified."""
        return sum(1 for p in self.predictions if p["status"] == "verified")

    # -- Reporting --

    def format_learning_report(self) -> str:
        """Show prediction accuracy history and learned adjustments."""
        ls = self.learning_state
        lines = ["## Outcome Learning Report\n"]

        # Summary stats
        pending = self.get_pending_count()
        lines.append(f"**Predictions:** {ls.total_predictions} total, "
                      f"{ls.verified_predictions} verified, {pending} pending")
        lines.append(f"**Average Accuracy:** {ls.avg_accuracy:.1%}")
        lines.append(f"**Global Confidence Adjustment:** {ls.confidence_adjustment:.3f}")
        lines.append("")

        # Category breakdown
        if ls.category_accuracy:
            lines.append("### Per-Category Accuracy\n")
            lines.append("| Category | Accuracy | Adjustment |")
            lines.append("|----------|----------|------------|")
            for cat, acc in sorted(ls.category_accuracy.items(), key=lambda x: -x[1]):
                adj = self.get_confidence_adjustment(cat)
                flag = ""
                if cat in ls.overconfident_categories:
                    flag = " (overconfident)"
                elif cat in ls.underconfident_categories:
                    flag = " (underconfident)"
                lines.append(f"| {cat} | {acc:.1%} | {adj:.3f}{flag} |")
            lines.append("")

        # Overconfident / underconfident warnings
        if ls.overconfident_categories:
            cats = ", ".join(ls.overconfident_categories)
            lines.append(f"**Overconfident in:** {cats} -- predictions consistently exceed actual results")
        if ls.underconfident_categories:
            cats = ", ".join(ls.underconfident_categories)
            lines.append(f"**Underconfident in:** {cats} -- actual results consistently exceed predictions")
        if ls.overconfident_categories or ls.underconfident_categories:
            lines.append("")

        # Recent outcomes
        recent = self.outcomes[-10:]
        if recent:
            lines.append("### Recent Outcomes\n")
            lines.append("| Scenario | Predicted | Actual | Accuracy |")
            lines.append("|----------|-----------|--------|----------|")
            for o in reversed(recent):
                pred = self._find_prediction(o["prediction_id"])
                name = pred["scenario_name"][:40] if pred else "?"
                lines.append(
                    f"| {name} | {pred['predicted_score_delta']:+.1f} | "
                    f"{o['actual_score_delta']:+.1f} | {o['accuracy']:.1%} |"
                )
            lines.append("")

        # Pending predictions
        pending_preds = [p for p in self.predictions if p["status"] == "pending"]
        if pending_preds:
            lines.append("### Pending Predictions\n")
            for p in pending_preds[-5:]:
                lines.append(
                    f"- **{p['scenario_name']}**: {p['predicted_score_delta']:+.1f} "
                    f"(p={p['probability']:.0%}, commit {p['commit'][:8]})"
                )
            if len(pending_preds) > 5:
                lines.append(f"  ... and {len(pending_preds) - 5} more")
            lines.append("")

        if not self.predictions:
            lines.append("No predictions recorded yet. Run architect-sim to start tracking.\n")

        return "\n".join(lines)
