"""Distribution, lifetime accounting and held-out interval scoring checks."""

import numpy as np
import pytest
from scipy.special import ndtri

from pereval.scorers.cecl import score_predictions
from pereval.tasks.cecl.baselines import predict
from pereval.tasks.cecl.generator import (
    csv_text,
    default_paths,
    generate,
    lifetime_loss,
    path_losses,
    public_files,
)


def test_marginal_means_and_latent_persistence():
    means = np.array([0.01, 0.04, 0.1, 0.2])
    draws = default_paths(means, 150000, np.random.default_rng(4), persistence=0.7)
    se = draws.std(axis=0) / np.sqrt(len(draws))
    assert np.all(abs(draws.mean(axis=0) - means) < 5 * se)
    latent = ndtri(draws)
    assert np.corrcoef(latent[:, 0], latent[:, 1])[0, 1] == pytest.approx(0.7, abs=0.01)


def test_zero_noise_and_endpoint_probabilities():
    rates = np.tile([0.03, 0.4, 0.05], (12, 1))
    draws = default_paths(rates[:, 0], 100, np.random.default_rng(1), rho=0)
    np.testing.assert_allclose(
        path_losses(100, rates, draws), lifetime_loss(100, 12, rates)
    )
    edges = default_paths([0, 1], 10, np.random.default_rng(1))
    np.testing.assert_array_equal(edges, np.tile([0, 1], (10, 1)))


def test_path_accounting_and_survivor_selection():
    rates = np.array([[0.1, 0.5, 0.2], [0.2, 0.25, 0]])
    assert path_losses(100, rates, [[0.1, 0.2]])[0] == pytest.approx(6.8)
    rates = np.tile([0.08, 0.5, 0], (40, 1))
    losses = []
    for persistence in (0, 0.8):
        pd = default_paths(
            rates[:, 0],
            100000,
            np.random.default_rng(2),
            rho=0.1,
            persistence=persistence,
        )
        losses.append(path_losses(100, rates, pd))
    assert losses[0].mean() == pytest.approx(lifetime_loss(100, 40, rates), abs=0.03)
    assert losses[1].mean() < losses[0].mean() - 0.2
    assert losses[1].std() > losses[0].std()


def test_reproducible_independent_oracle_and_public_isolation():
    b = generate(8, simulation=True)
    assert b == generate(8, simulation=True)
    old = generate(8)
    assert b["history"] == old["history"]
    assert b["pools"] == old["pools"]
    text = "".join(public_files(b).values())
    assert "loss_samples" not in text and "ecl_mc_se" not in text
    assert "aggregate_probit_ar1_v1" in text
    for p in b["truth"]:
        draws = np.array(p["loss_samples"])
        assert 0 <= draws.min() <= draws.max() <= p["balance"]
        assert abs(draws.mean() - p["ecl"]) < 6 * p["ecl_mc_se"]
        assert 0.93 < np.mean((draws >= p["lower"]) & (draws <= p["upper"])) < 0.97
        assert p["lower"] != np.quantile(draws, 0.025)


def test_interval_oracle_missing_and_invalid_submissions():
    truth = generate(2, simulation=True)["truth"]
    rows = [
        {
            "pool_id": p["pool_id"],
            "ecl": p["ecl"],
            "ecl_lower": p["lower"],
            "ecl_upper": p["upper"],
        }
        for p in truth
    ]
    exact = score_predictions(truth, csv_text(rows))
    assert exact["ecl_regret"] == 0
    assert exact["winkler_regret"] == pytest.approx(0)
    assert exact["completion"] == pytest.approx(1)
    assert 0.93 < exact["coverage"] < 0.97
    missing = score_predictions(truth, None)
    assert missing["winkler_agent"] >= missing["winkler_degenerate"]
    assert missing["completion"] == 0
    for lo, hi in [(10, 1), (-1, 10), (0, float("inf")), (0, float("nan"))]:
        bad = [dict(r, ecl_lower=lo, ecl_upper=hi) for r in rows]
        assert score_predictions(truth, csv_text(bad))["completion"] == 0
    assert score_predictions(truth, csv_text(rows + rows))["completion"] == 0
    point_only = csv_text([{"pool_id": p["pool_id"], "ecl": p["ecl"]} for p in truth])
    score = score_predictions(truth, point_only)
    assert score["point_completion"] == 1 and score["completion"] == 0


@pytest.mark.parametrize("scenario", ["baseline", "adverse", "benign"])
def test_public_reference_and_task_wiring(scenario):
    from pereval.tasks.cecl.task import SIMULATION_INSTRUCTIONS, cecl

    b = generate(3, scenario=scenario, simulation=True)
    score = score_predictions(b["truth"], predict(public_files(b)))
    naive = score_predictions(b["truth"], predict(public_files(b), "naive"))
    assert score["completion"] == pytest.approx(1)
    assert score["winkler_regret"] < naive["winkler_regret"]
    assert score["ecl_regret"] < score["zero_regret"]
    assert 0.85 < score["coverage"] < 1
    task = cecl(
        n_instances=1, seed=3, simulation=True, baseline="cohort", oracle_n=1000
    )
    assert "loss_samples" in task.dataset[0].metadata["truth"][0]
    assert "pool_id,ecl,ecl_lower,ecl_upper" in SIMULATION_INSTRUCTIONS


@pytest.mark.parametrize("kwargs", [{"rho": 1}, {"persistence": 1}, {"n_paths": 0}])
def test_invalid_simulation_parameters(kwargs):
    args = {"n_paths": 100, "rng": np.random.default_rng(1)}
    args.update(kwargs)
    with pytest.raises(ValueError):
        default_paths([0.1], **args)
