"""Synthetic loan pools with analytical or simulated lifetime loss oracles.

Only public_files() enters the agent sandbox. Coefficients and future expected
losses remain host-side. Rates are quarterly; prepayment is conditional on no
default. Principal amortizes equally over the remaining contractual term.
"""

from __future__ import annotations

import csv
import io
import json

import numpy as np
from scipy.special import expit, ndtr, ndtri


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def lifetime_loss(balance: float, term: int, rates: np.ndarray) -> float:
    """Expected undiscounted net principal loss, no new lending or renewals."""
    rates = np.asarray(rates, dtype=float)
    if (
        balance < 0
        or not np.isfinite(balance)
        or term < 1
        or rates.shape != (term, 3)
        or not np.isfinite(rates).all()
        or np.any((rates < 0) | (rates > 1))
    ):
        raise ValueError(
            "Require finite balance >= 0, positive term and term x 3 rates in [0,1]"
        )
    survival, loss = 1.0, 0.0
    for q, (pd, lgd, prepay) in enumerate(rates):
        exposure = balance * (1 - q / term) * survival
        loss += exposure * pd * lgd
        survival *= (1 - pd) * (1 - prepay)
    return float(loss)


def extend_rates(
    forecast: np.ndarray, historical: np.ndarray, term: int, reversion_quarters: int
) -> np.ndarray:
    """Hold supplied forecast, then blend each rate to its historical mean."""
    forecast = np.asarray(forecast, dtype=float)
    historical = np.asarray(historical, dtype=float)
    if (
        term < 1
        or reversion_quarters < 0
        or forecast.ndim != 2
        or forecast.shape[1] != 3
        or len(forecast) < 1
        or historical.shape != (3,)
    ):
        raise ValueError("Invalid forecast, historical rates, term or reversion period")
    path = []
    for q in range(term):
        if q < len(forecast):
            path.append(forecast[q])
        else:
            weight = (
                min(1.0, (q - len(forecast) + 1) / reversion_quarters)
                if reversion_quarters
                else 1.0
            )
            path.append((1 - weight) * forecast[-1] + weight * historical)
    return np.asarray(path)


def _rates(x: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    return expit(np.column_stack([np.ones(len(x)), x]) @ coefficients.T)


def default_paths(means, n_paths, rng, rho=0.02, persistence=0.6):
    """Stationary Gaussian AR(1) shocks with E[PD_q] = means[q].

    Shocks start independently of the historical mean-rate measurement noise.
    Rates are aggregate large-pool fractions, not finite-loan default counts.
    """
    means = np.asarray(means, dtype=float)
    if (
        means.ndim != 1
        or not len(means)
        or not np.isfinite(means).all()
        or np.any((means < 0) | (means > 1))
        or n_paths < 1
        or not 0 <= rho < 1
        or not -1 < persistence < 1
    ):
        raise ValueError("Invalid means, path count, dispersion or persistence")
    z = rng.standard_normal(n_paths)
    paths = np.empty((n_paths, len(means)))
    for q, mean in enumerate(means):
        if q:
            z = persistence * z + np.sqrt(1 - persistence**2) * rng.standard_normal(
                n_paths
            )
        paths[:, q] = ndtr((ndtri(mean) + np.sqrt(rho) * z) / np.sqrt(1 - rho))
    return paths


def path_losses(balance, rates, pds):
    """Apply competing exits and amortization separately on each full path."""
    rates, pds = np.asarray(rates, dtype=float), np.asarray(pds, dtype=float)
    # Reuse accounting validation, including rates and balance boundaries.
    lifetime_loss(balance, len(rates), rates)
    if (
        pds.ndim != 2
        or pds.shape[1] != len(rates)
        or not len(pds)
        or not np.isfinite(pds).all()
        or np.any((pds < 0) | (pds > 1))
    ):
        raise ValueError("Invalid default paths")
    survival = np.ones(len(pds))
    losses = np.zeros(len(pds))
    for q, (_, lgd, prepay) in enumerate(rates):
        losses += balance * (1 - q / len(rates)) * survival * pds[:, q] * lgd
        survival *= (1 - pds[:, q]) * (1 - prepay)
    return losses


def generate(
    seed: int = 1,
    n_history: int = 80,
    forecast_quarters: int = 8,
    reversion_quarters: int = 4,
    scenario: str = "baseline",
    simulation: bool = False,
    oracle_n: int = 10000,
) -> dict:
    if n_history < 20 or forecast_quarters < 1 or reversion_quarters < 0:
        raise ValueError(
            "Need >=20 historical quarters, >=1 forecast quarter and nonnegative reversion"
        )
    if scenario not in ("baseline", "adverse", "benign"):
        raise ValueError("scenario must be baseline, adverse or benign")
    if oracle_n < 100:
        raise ValueError("oracle_n must be >= 100")
    rng = np.random.default_rng(seed)
    oracle_streams = np.random.SeedSequence([seed, 9341]).spawn(6)
    x = np.zeros((n_history, 2))
    for t in range(1, n_history):
        shock = rng.multivariate_normal([0, 0], [[1, -0.5], [-0.5, 1]])
        x[t] = 0.65 * x[t - 1] + 0.6 * shock
    target = {"baseline": [0.2, -0.1], "adverse": [2.5, -2], "benign": [-1, 1]}[
        scenario
    ]
    future = np.linspace(x[-1], target, forecast_quarters + 1)[1:]
    forecast = [
        {
            "quarter": q + 1,
            "unemployment": float(5 + a),
            "hpi_growth": float(0.03 + b * 0.04),
        }
        for q, (a, b) in enumerate(future)
    ]
    history, pools, truth = [], [], []
    parameters = {}
    for segment_index, segment in enumerate(("A", "B", "C")):
        coefficients = np.array(
            [
                [
                    rng.uniform(-5, -3.5),
                    rng.uniform(0.25, 0.65),
                    rng.uniform(-0.5, -0.15),
                ],
                [rng.uniform(-1, 0.3), rng.uniform(0.1, 0.3), rng.uniform(-0.5, -0.2)],
                [rng.uniform(-3.5, -2), rng.uniform(-0.3, -0.1), rng.uniform(0.1, 0.3)],
            ]
        )
        parameters[segment] = coefficients.tolist()
        expected = _rates(x, coefficients)
        for q, ((a, b), (pd, lgd, prepay)) in enumerate(zip(x, expected)):
            history.append(
                {
                    "segment": segment,
                    "quarter": q + 1,
                    "unemployment": float(5 + a),
                    "hpi_growth": float(0.03 + b * 0.04),
                    "pd": float(rng.binomial(10000, pd) / 10000),
                    "lgd": float(rng.beta(lgd * 400, (1 - lgd) * 400)),
                    "prepay": float(rng.binomial(10000, prepay) / 10000),
                }
            )
        if simulation:
            full_rates = extend_rates(
                _rates(future, coefficients),
                expected.mean(axis=0),
                40,
                reversion_quarters,
            )
            # Same-segment pools share shocks; independent streams across segments.
            draws = [
                default_paths(full_rates[:, 0], oracle_n, np.random.default_rng(stream))
                for stream in oracle_streams[2 * segment_index : 2 * segment_index + 2]
            ]
        for j, term in enumerate((4, 12, 24, 40)):
            balance = float(rng.integers(1_000_000, 10_000_001))
            pool = {
                "pool_id": f"{segment}{j + 1}",
                "segment": segment,
                "balance": balance,
                "remaining_quarters": term,
            }
            pools.append(pool)
            rates = extend_rates(
                _rates(future, coefficients),
                expected.mean(axis=0),
                term,
                reversion_quarters,
            )
            record = dict(**pool, ecl=lifetime_loss(balance, term, rates))
            if simulation:
                calibration, evaluation = [
                    path_losses(balance, rates, d[:, :term]) for d in draws
                ]
                record.update(
                    ecl=float(calibration.mean()),
                    ecl_mc_se=float(calibration.std(ddof=1) / np.sqrt(oracle_n)),
                    lower=float(np.quantile(calibration, 0.025)),
                    upper=float(np.quantile(calibration, 0.975)),
                    loss_samples=evaluation.tolist(),
                )
            truth.append(record)
    return {
        "history": history,
        "forecast": forecast,
        "pools": pools,
        "policy": {
            "forecast_quarters": forecast_quarters,
            "reversion_quarters": reversion_quarters,
            **(
                {
                    "simulation": "aggregate_probit_ar1_v1",
                    "rho": 0.02,
                    "persistence": 0.6,
                    "interval_level": 0.95,
                }
                if simulation
                else {}
            ),
        },
        "truth": truth,
        "parameters": parameters,
    }


def public_files(bundle: dict) -> dict[str, str]:
    return {
        "data/history.csv": csv_text(bundle["history"]),
        "data/forecast.csv": csv_text(bundle["forecast"]),
        "data/pools.csv": csv_text(bundle["pools"]),
        "data/policy.json": json.dumps(bundle["policy"]),
    }
