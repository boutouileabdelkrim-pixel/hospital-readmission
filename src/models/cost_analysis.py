"""Cost-sensitive decision analysis for the readmission model.

Turns ML probabilities into business impact by modeling the asymmetric
cost of errors in a clinical setting:

  - False negative (miss a real readmission): costs FN_COST
  - False positive (deploy an unnecessary intervention): costs FP_COST
  - True positive: deploy intervention that prevents readmission -> costs FP_COST
  - True negative: no cost

With strong FN/FP asymmetry (30:1), the unconstrained cost minimum flags
almost every patient, which is operationally useless. We therefore add a
precision floor (MIN_PRECISION) so the chosen threshold is deployable.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import confusion_matrix

from src.data.clean import TARGET_BIN
from src.features.clinical import TEST_PARQUET, VAL_PARQUET
from src.models.explain import load_booster
from src.utils.paths import REPORTS_DIR

logger = logging.getLogger(__name__)

COST_CURVE_PATH = REPORTS_DIR / "cost_curve.png"
COST_JSON_PATH = REPORTS_DIR / "cost_analysis.json"

# Default clinical cost assumptions (in USD). Overridable via CLI.
FN_COST = 15_000.0
FP_COST = 500.0

# Operational precision floor: flagging > 85% of patients is useless, so we
# require precision >= 0.15 before accepting a threshold as "optimal".
MIN_PRECISION = 0.15

THRESHOLDS = np.linspace(0.01, 0.99, 99)


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------


def confusion_at_threshold(
    y_true: np.ndarray, y_proba: np.ndarray, t: float
) -> tuple[int, int, int, int]:
    """Return (tn, fp, fn, tp) at threshold t."""
    y_pred = (y_proba >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def precision_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, t: float) -> float:
    """Precision at threshold t (0.0 if no positive prediction)."""
    _, fp, _, tp = confusion_at_threshold(y_true, y_proba, t)
    denom = tp + fp
    return float(tp) / denom if denom > 0 else 0.0


def total_cost(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    t: float,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
) -> float:
    """Total cost in USD at threshold t.

    A TP means intervention was deployed and prevented the readmission, so
    we only pay the intervention cost (fp_cost) instead of the FN cost.
    """
    _, fp, fn, tp = confusion_at_threshold(y_true, y_proba, t)
    return float(fn * fn_cost + fp * fp_cost + tp * fp_cost)


def baseline_cost(y_true: np.ndarray, fn_cost: float = FN_COST) -> float:
    """Cost if we do nothing: every readmission happens, no intervention."""
    return float(int(y_true.sum()) * fn_cost)


def savings_vs_baseline(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    t: float,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
) -> float:
    """USD saved at threshold t, compared to the no-model baseline."""
    return baseline_cost(y_true, fn_cost) - total_cost(y_true, y_proba, t, fn_cost, fp_cost)


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------


def optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: np.ndarray = THRESHOLDS,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
    min_precision: float = MIN_PRECISION,
) -> dict:
    """Find the threshold minimizing total cost, subject to a precision floor.

    Rationale: with strong FN/FP asymmetry, the unconstrained minimum flags
    almost every patient, which is operationally useless. We require
    precision >= `min_precision` (default 0.15) so the tool is deployable.
    """
    costs = np.array([total_cost(y_true, y_proba, float(t), fn_cost, fp_cost) for t in thresholds])
    precisions = np.array([precision_at_threshold(y_true, y_proba, float(t)) for t in thresholds])

    feasible = precisions >= min_precision
    if not feasible.any():
        # No threshold meets the precision floor -> fall back to unconstrained
        best_idx = int(np.argmin(costs))
        constraint_applied = False
    else:
        feasible_costs = np.where(feasible, costs, np.inf)
        best_idx = int(np.argmin(feasible_costs))
        constraint_applied = True

    best_t = float(thresholds[best_idx])
    best_cost = float(costs[best_idx])
    base_cost = baseline_cost(y_true, fn_cost)
    savings = base_cost - best_cost
    n = len(y_true)
    savings_per_1000 = (savings / n) * 1000 if n > 0 else 0.0

    tn, fp, fn, tp = confusion_at_threshold(y_true, y_proba, best_t)
    return {
        "optimal_threshold": best_t,
        "min_total_cost_usd": best_cost,
        "baseline_total_cost_usd": base_cost,
        "savings_usd": savings,
        "savings_pct": 100.0 * savings / base_cost if base_cost > 0 else 0.0,
        "savings_per_1000_patients_usd": savings_per_1000,
        "n_samples": int(n),
        "n_positives": int(y_true.sum()),
        "at_optimal": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "cost_per_patient_at_optimal": best_cost / n if n > 0 else 0.0,
        "precision_at_optimal": float(precisions[best_idx]),
        "constraint_applied": constraint_applied,
        "min_precision_floor": float(min_precision),
    }


def cost_at_default_05(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
) -> dict:
    """Cost at the ML default threshold 0.5, for comparison."""
    t = 0.5
    c = total_cost(y_true, y_proba, t, fn_cost, fp_cost)
    base = baseline_cost(y_true, fn_cost)
    tn, fp, fn, tp = confusion_at_threshold(y_true, y_proba, t)
    return {
        "threshold": t,
        "total_cost_usd": c,
        "savings_usd": base - c,
        "savings_pct": 100.0 * (base - c) / base if base > 0 else 0.0,
        "at_threshold": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------


def save_cost_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    path: Path = COST_CURVE_PATH,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
    min_precision: float = MIN_PRECISION,
) -> Path:
    """Save a PNG of total cost vs threshold, with the constrained optimum."""
    thresholds = THRESHOLDS
    costs = np.array([total_cost(y_true, y_proba, float(t), fn_cost, fp_cost) for t in thresholds])
    precisions = np.array([precision_at_threshold(y_true, y_proba, float(t)) for t in thresholds])

    feasible = precisions >= min_precision
    best_t = None
    if feasible.any():
        feasible_costs = np.where(feasible, costs, np.inf)
        best_t = float(thresholds[int(np.argmin(feasible_costs))])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, costs / 1e6, color="#377eb8", label="Total cost (model)")
    ax.axhline(
        baseline_cost(y_true, fn_cost) / 1e6,
        color="black",
        linestyle="--",
        label="Baseline (no model)",
    )
    if best_t is not None:
        ax.axvline(
            best_t,
            color="#e41a1c",
            linestyle=":",
            label=f"Optimal t = {best_t:.2f} (precision ≥ {min_precision})",
        )
    ax.axvline(0.5, color="gray", linestyle=":", label="Default t = 0.50")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Total cost (millions USD)")
    ax.set_title("Cost-sensitive decision: total cost vs threshold")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved cost curve: %s", path)
    return path


def save_json(payload: dict, path: Path = COST_JSON_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    logger.info("Saved cost analysis: %s", path)
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Cost-sensitive threshold analysis")
    parser.add_argument(
        "--split",
        default="val",
        choices=["val", "test"],
        help="Split used to pick the optimal threshold",
    )
    parser.add_argument(
        "--eval-split",
        default="test",
        choices=["val", "test"],
        help="Split used to report savings (default: test)",
    )
    parser.add_argument("--fn-cost", type=float, default=FN_COST)
    parser.add_argument("--fp-cost", type=float, default=FP_COST)
    parser.add_argument("--min-precision", type=float, default=MIN_PRECISION)
    args = parser.parse_args()

    booster = load_booster()

    def _load(split: str) -> tuple[np.ndarray, np.ndarray]:
        p = VAL_PARQUET if split == "val" else TEST_PARQUET
        df = pd.read_parquet(p)
        y = df[TARGET_BIN].to_numpy(dtype=int)
        X = df.drop(columns=[TARGET_BIN])
        proba = booster.predict(xgb.DMatrix(X))
        return y, proba

    y_opt, p_opt = _load(args.split)
    y_eval, p_eval = _load(args.eval_split)

    print(f"Cost assumptions : FN=${args.fn_cost:,.0f}  FP=${args.fp_cost:,.0f}")
    print(f"Min precision    : {args.min_precision:.2f}")
    print(f"Optimal split    : {args.split}  ({len(y_opt):,} samples)")
    print(f"Evaluation split : {args.eval_split}  ({len(y_eval):,} samples)")
    print()

    result = optimal_threshold(
        y_opt,
        p_opt,
        fn_cost=args.fn_cost,
        fp_cost=args.fp_cost,
        min_precision=args.min_precision,
    )
    default = cost_at_default_05(y_eval, p_eval, fn_cost=args.fn_cost, fp_cost=args.fp_cost)

    print(f"Optimal threshold      : {result['optimal_threshold']:.3f}")
    print(f"  Precision @ optimal  : {result['precision_at_optimal']:.3f}")
    print(f"  Constraint applied   : {result['constraint_applied']}")
    print(f"  Savings vs baseline  : ${result['savings_usd']:,.0f}  ({result['savings_pct']:.1f}%)")
    print(f"  Savings / 1000 pts   : ${result['savings_per_1000_patients_usd']:,.0f}")
    print(
        f"  Confusion (opt)      : TN={result['at_optimal']['tn']}  "
        f"FP={result['at_optimal']['fp']}  FN={result['at_optimal']['fn']}  "
        f"TP={result['at_optimal']['tp']}"
    )
    print()
    print("Default threshold = 0.50")
    print(
        f"  Savings vs baseline  : ${default['savings_usd']:,.0f}  ({default['savings_pct']:.1f}%)"
    )
    print(
        f"  Confusion (0.50)     : TN={default['at_threshold']['tn']}  "
        f"FP={default['at_threshold']['fp']}  FN={default['at_threshold']['fn']}  "
        f"TP={default['at_threshold']['tp']}"
    )
    print()
    improvement = result["savings_usd"] - default["savings_usd"]
    print(f"👉 Improvement (optimal vs 0.5): ${improvement:,.0f} more saved")
    print()

    save_cost_curve(
        y_opt,
        p_opt,
        fn_cost=args.fn_cost,
        fp_cost=args.fp_cost,
        min_precision=args.min_precision,
    )
    save_json(
        {
            "computed_at": datetime.now(UTC).isoformat(),
            "cost_assumptions": {
                "fn_cost_usd": args.fn_cost,
                "fp_cost_usd": args.fp_cost,
            },
            "min_precision": args.min_precision,
            "optimal_split": args.split,
            "eval_split": args.eval_split,
            "optimal_analysis_on_optimal_split": result,
            "default_threshold_0p5_on_eval_split": default,
            "improvement_optimal_vs_default_usd": improvement,
        }
    )
    print("✅ Cost analysis complete.")


if __name__ == "__main__":
    main()
