"""References using only the same four input files available to the agent."""

from __future__ import annotations

import csv
import io
import json

import numpy as np
from scipy.special import expit, logit

from pereval.tasks.cecl.generator import (
    csv_text,
    default_paths,
    extend_rates,
    lifetime_loss,
    path_losses,
)


def predict(files: dict[str, str], method: str = "cohort") -> str:
    if method not in ("cohort", "naive"):
        raise ValueError("Unknown CECL baseline")
    history = list(csv.DictReader(io.StringIO(files["data/history.csv"])))
    forecast = list(csv.DictReader(io.StringIO(files["data/forecast.csv"])))
    pools = list(csv.DictReader(io.StringIO(files["data/pools.csv"])))
    policy = json.loads(files["data/policy.json"])

    def features(rows):
        return np.array(
            [[1, float(r["unemployment"]), float(r["hpi_growth"])] for r in rows]
        )

    fitted = {}
    for segment in {p["segment"] for p in pools}:
        rows = [r for r in history if r["segment"] == segment]
        rates = np.array([[float(r[k]) for k in ("pd", "lgd", "prepay")] for r in rows])
        beta = np.linalg.lstsq(
            features(rows), logit(np.clip(rates, 1e-5, 1 - 1e-5)), rcond=None
        )[0]
        fitted[segment] = (expit(features(forecast) @ beta), rates.mean(axis=0))
    predictions = []
    for pool in pools:
        forecast_rates, historical = fitted[pool["segment"]]
        term, balance = int(pool["remaining_quarters"]), float(pool["balance"])
        if method == "naive":
            # Deliberately ignores runoff, competing exits and the forecast.
            ecl = balance * min(1.0, historical[0] * historical[1] * term)
        else:
            path = extend_rates(
                forecast_rates, historical, term, policy["reversion_quarters"]
            )
            ecl = lifetime_loss(balance, term, path)
        row = {"pool_id": pool["pool_id"], "ecl": ecl}
        if policy.get("simulation"):
            if method == "naive":
                row.update(ecl_lower=ecl, ecl_upper=ecl)
            else:
                draws = default_paths(
                    path[:, 0],
                    20000,
                    np.random.default_rng(1729),
                    policy["rho"],
                    policy["persistence"],
                )
                losses = path_losses(balance, path, draws)
                row.update(
                    ecl=float(losses.mean()),
                    ecl_lower=float(np.quantile(losses, 0.025)),
                    ecl_upper=float(np.quantile(losses, 0.975)),
                )
        predictions.append(row)
    return csv_text(predictions)


def reference_solver(method: str):
    from inspect_ai.solver import solver
    from inspect_ai.util import sandbox

    @solver(name=f"cecl_{method}")
    def reference():
        async def solve(state, generate):
            files = {
                f"data/{name}": await sandbox().read_file(f"data/{name}")
                for name in ("history.csv", "forecast.csv", "pools.csv", "policy.json")
            }
            await sandbox().write_file("predictions.csv", predict(files, method))
            return state

        return solve

    return reference()
