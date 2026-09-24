"""Score expected lifetime loss, not a tail quantile or a realized loss draw."""

from __future__ import annotations

import csv
import io
import math


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
    return result


def cecl_scorer():
    from inspect_ai.scorer import Score, mean, scorer, stderr
    from inspect_ai.util import sandbox

    @scorer(
        name="cecl",
        metrics={
            "ecl_regret": [mean(), stderr()],
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
                ),
            )

        return score

    return factory()
