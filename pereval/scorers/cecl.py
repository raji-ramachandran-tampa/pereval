"""Score lifetime mean loss and optional predictive intervals on held-out draws."""

from __future__ import annotations

import csv
import io
import math

import numpy as np

from pereval.scorers.interval import interval_score


def score_predictions(truth: list[dict], text: str | None) -> dict[str, float]:
    predictions, duplicates = {}, set()
    if text:
        for row in csv.DictReader(io.StringIO(text)):
            key = row.get("pool_id")
            if key in predictions:
                duplicates.add(key)
            try:
                predictions[key] = float(row["ecl"])
            except (KeyError, TypeError, ValueError):
                predictions[key] = float("nan")
    total_balance = sum(p["balance"] for p in truth)
    regret = mae = zero_regret = predicted_total = true_total = 0.0
    missing = 0
    for pool in truth:
        balance, expected = pool["balance"], pool["ecl"]
        value = predictions.get(pool["pool_id"], float("nan"))
        weight, true_rate = balance / total_balance, expected / balance
        valid = (
            pool["pool_id"] not in duplicates
            and math.isfinite(value)
            and 0 <= value <= balance
        )
        if valid:
            error = (value - expected) / balance
            regret += weight * error**2
            mae += weight * abs(error)
            predicted_total += value
        else:
            # Suite convention: max(degenerate score, 5 * oracle score).
            # The exact mean oracle has zero squared error, leaving the
            # zero-allowance anchor for this pool. Completion remains separate.
            missing += 1
            regret += weight * true_rate**2
            mae += weight * true_rate
        zero_regret += weight * true_rate**2
        true_total += expected
    result = {
        "ecl_regret": regret,
        "zero_regret": zero_regret,
        "loss_rate_mae_bps": mae * 10000,
        "completion": 1 - missing / len(truth),
        "runs": 1.0,
        "regret_worst": regret,
        "regret_spread": 0.0,
    }
    # Never present an incomplete sum as a portfolio estimate.
    if missing == 0:
        result.update(
            portfolio_bias=predicted_total - true_total,
            portfolio_ecl=predicted_total,
            true_portfolio_ecl=true_total,
        )
    if truth and "loss_samples" in truth[0]:
        result.update(score_loss_intervals(truth, text))
        result["regret_worst"] = result["winkler_regret"]
    return result


def score_loss_intervals(truth, text):
    """Dollar submissions, balance-weighted loss-rate scores on held-out draws."""
    rows, duplicates = {}, set()
    for row in csv.DictReader(io.StringIO(text or "")):
        key = row.get("pool_id")
        if key in rows:
            duplicates.add(key)
        rows[key] = row
    totals = {
        "winkler_agent": 0.0,
        "winkler_oracle": 0.0,
        "winkler_degenerate": 0.0,
        "coverage": 0.0,
        "mean_width": 0.0,
        "interval_completion": 0.0,
    }
    total_balance = sum(p["balance"] for p in truth)
    for pool in truth:
        balance = pool["balance"]
        weight = balance / total_balance
        samples = np.asarray(pool["loss_samples"]) / balance
        oracle = float(
            interval_score(
                pool["lower"] / balance, pool["upper"] / balance, samples
            ).mean()
        )
        degenerate = float(interval_score(0.0, 0.0, samples).mean())
        try:
            row = rows[pool["pool_id"]]
            point, lo, hi = [
                float(row[k]) / balance for k in ("ecl", "ecl_lower", "ecl_upper")
            ]
            valid = (
                pool["pool_id"] not in duplicates
                and all(math.isfinite(x) for x in (point, lo, hi))
                and 0 <= point <= 1
                and 0 <= lo <= hi <= 1
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if valid:
            agent = float(interval_score(lo, hi, samples).mean())
            totals["coverage"] += weight * float(
                ((samples >= lo) & (samples <= hi)).mean()
            )
            totals["mean_width"] += weight * (hi - lo)
            totals["interval_completion"] += 1 / len(truth)
        else:
            agent = max(degenerate, 5 * oracle)
        totals["winkler_agent"] += weight * agent
        totals["winkler_oracle"] += weight * oracle
        totals["winkler_degenerate"] += weight * degenerate
    totals["winkler_regret"] = totals["winkler_agent"] - totals["winkler_oracle"]
    totals["degenerate_regret"] = (
        totals["winkler_degenerate"] - totals["winkler_oracle"]
    )
    # Legacy completion is for the mean; overall completion requires both outputs.
    totals["point_completion"] = sum(
        1
        for p in truth
        if p["pool_id"] not in duplicates
        and _valid_mean(rows.get(p["pool_id"]), p["balance"])
    ) / len(truth)
    totals["completion"] = totals["interval_completion"]
    return totals


def _valid_mean(row, balance):
    try:
        value = float(row["ecl"])
        return math.isfinite(value) and 0 <= value <= balance
    except (KeyError, TypeError, ValueError):
        return False


def cecl_scorer():
    from inspect_ai.scorer import Score, mean, scorer, stderr
    from inspect_ai.util import sandbox

    @scorer(
        name="cecl",
        metrics={
            "ecl_regret": [mean(), stderr()],
            **{
                key: [mean()]
                for key in (
                    "winkler_regret",
                    "winkler_agent",
                    "winkler_oracle",
                    "winkler_degenerate",
                    "degenerate_regret",
                    "coverage",
                    "mean_width",
                    "interval_completion",
                    "point_completion",
                )
            },
            "zero_regret": [mean()],
            "loss_rate_mae_bps": [mean()],
            "completion": [mean()],
            "runs": [mean()],
            "regret_worst": [mean()],
            "regret_spread": [mean()],
        },
    )
    def factory():
        async def score(state, target):
            try:
                text = await sandbox().read_file("predictions.csv")
            except FileNotFoundError:
                text = None
            value = score_predictions(state.metadata["truth"], text)
            return Score(
                value=value,
                explanation=(
                    f"Lifetime ECL regret {value['ecl_regret']:.6g}; "
                    f"loss-rate MAE {value['loss_rate_mae_bps']:.2f} bps; "
                    f"completion {value['completion']:.0%}."
                    + (
                        f" Winkler regret {value['winkler_regret']:.6g}; "
                        f"coverage {value['coverage']:.1%}."
                        if "winkler_regret" in value
                        else ""
                    )
                ),
            )

        return score

    return factory()
