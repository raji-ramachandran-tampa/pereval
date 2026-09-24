"""Lifetime accounting identities, reference quality and strict output scoring."""

import json

import numpy as np
import pytest

from pereval.scorers.cecl import score_predictions
from pereval.tasks.cecl.baselines import predict
from pereval.tasks.cecl.generator import (
    csv_text,
    extend_rates,
    generate,
    lifetime_loss,
    public_files,
)


def test_two_quarter_hand_calculation():
    # Q1 loss 100*.1*.5=5. Q2 surviving principal 50*.9*.8=36;
    # Q2 loss 36*.2*.25=1.8. Default and prepayment are competing exits.
    assert lifetime_loss(
        100, 2, np.array([[0.1, 0.5, 0.2], [0.2, 0.25, 0]])
    ) == pytest.approx(6.8)


@pytest.mark.parametrize(
    "pd,lgd,prepay,expected",
    [
        (0, 0.5, 0, 0),
        (1, 1, 0, 100),
        (0.1, 0, 0.1, 0),
        (0.1, 0.5, 1, 5),
    ],
)
def test_boundary_cases(pd, lgd, prepay, expected):
    assert lifetime_loss(100, 20, np.tile([pd, lgd, prepay], (20, 1))) == pytest.approx(
        expected
    )


def test_constant_hazard_matches_closed_form():
    term, balance, pd, lgd = 16, 1000, 0.03, 0.4
    expected = sum(
        balance * (1 - q / term) * (1 - pd) ** q * pd * lgd for q in range(term)
    )
    assert lifetime_loss(
        balance, term, np.tile([pd, lgd, 0], (term, 1))
    ) == pytest.approx(expected)
    assert lifetime_loss(balance, term, np.tile([pd, lgd, 0.15], (term, 1))) < expected


def test_reversion_boundaries_and_short_maturity():
    forecast = np.array([[0.1, 0.5, 0.2], [0.2, 0.6, 0.1]])
    historical = np.array([0.02, 0.3, 0.3])
    path = extend_rates(forecast, historical, 6, 2)
    np.testing.assert_allclose(path[2], (forecast[-1] + historical) / 2)
    np.testing.assert_allclose(path[3:], np.tile(historical, (3, 1)))
    np.testing.assert_allclose(extend_rates(forecast, historical, 1, 2), forecast[:1])
    np.testing.assert_allclose(extend_rates(forecast, historical, 3, 0)[2], historical)


def test_determinism_randomization_and_isolation():
    bundle = generate(seed=11)
    assert bundle == generate(seed=11)
    assert bundle["parameters"] != generate(seed=12)["parameters"]
    files = public_files(bundle)
    assert len(files) == 4
    assert "ecl" not in "".join(files.values())
    assert "parameters" not in "".join(files.values())
    assert json.loads(files["data/policy.json"])["forecast_quarters"] == 8
    assert all(0 <= p["ecl"] <= p["balance"] for p in bundle["truth"])


def test_exact_oracle_and_missing_penalty():
    truth = generate()["truth"]
    text = csv_text([{"pool_id": p["pool_id"], "ecl": p["ecl"]} for p in truth])
    perfect = score_predictions(truth, text)
    assert perfect["ecl_regret"] == 0
    assert perfect["portfolio_bias"] == 0
    missing = score_predictions(truth, None)
    assert missing["ecl_regret"] == pytest.approx(perfect["zero_regret"])
    assert missing["completion"] == 0
    assert "portfolio_ecl" not in missing
    zero = csv_text([{"pool_id": p["pool_id"], "ecl": 0} for p in truth])
    assert missing["loss_rate_mae_bps"] == pytest.approx(
        score_predictions(truth, zero)["loss_rate_mae_bps"]
    )
    assert score_predictions(truth, zero)["completion"] == 1


@pytest.mark.parametrize(
    "text",
    [
        "pool_id,ecl\nA,nan\n",
        "pool_id,ecl\nA,inf\n",
        "pool_id,ecl\nA,-1\n",
        "pool_id,ecl\nA,101\n",
        "pool_id,ecl\nA,10\nA,10\n",
        "pool_id,wrong\nA,10\n",
    ],
)
def test_invalid_output_cannot_gain_credit(text):
    assert score_predictions([{"pool_id": "A", "balance": 100, "ecl": 10}], text)[
        "ecl_regret"
    ] == pytest.approx(0.01)


def test_offsetting_pool_errors_do_not_cancel():
    truth = [{"pool_id": k, "balance": 100, "ecl": 10} for k in ("A", "B")]
    score = score_predictions(truth, "pool_id,ecl\nA,0\nB,20\n")
    assert score["portfolio_bias"] == 0
    assert score["ecl_regret"] == pytest.approx(0.01)


def test_partial_missing_uses_pool_weighted_zero_anchor():
    truth = [
        {"pool_id": "A", "balance": 100, "ecl": 10},
        {"pool_id": "B", "balance": 300, "ecl": 60},
    ]
    result = score_predictions(truth, "pool_id,ecl\nA,10\n")
    assert result["ecl_regret"] == pytest.approx(0.75 * 0.2**2)
    assert result["loss_rate_mae_bps"] == pytest.approx(0.75 * 0.2 * 10000)
    assert result["completion"] == 0.5
    assert "portfolio_ecl" not in result


def test_zero_loss_missing_still_reports_incompletion():
    result = score_predictions([{"pool_id": "A", "balance": 100, "ecl": 0}], None)
    assert result["ecl_regret"] == result["zero_regret"] == 0
    assert result["completion"] == 0


@pytest.mark.parametrize("scenario", ["baseline", "adverse", "benign"])
def test_fitted_reference_beats_naive_and_zero(scenario):
    for seed in range(5):
        bundle = generate(seed=seed, scenario=scenario)
        files = public_files(bundle)
        fitted = score_predictions(bundle["truth"], predict(files))
        naive = score_predictions(bundle["truth"], predict(files, "naive"))
        assert fitted["completion"] == 1
        assert fitted["ecl_regret"] < naive["ecl_regret"]
        assert fitted["ecl_regret"] < fitted["zero_regret"]


def test_scenario_and_reversion_sensitivity():
    benign = generate(seed=3, scenario="benign")
    adverse = generate(seed=3, scenario="adverse")
    assert benign["history"] == adverse["history"]
    assert sum(p["ecl"] for p in adverse["truth"]) > sum(
        p["ecl"] for p in benign["truth"]
    )
    immediate = generate(seed=3, scenario="adverse", reversion_quarters=0)
    assert (
        adverse["truth"][0]["ecl"] == immediate["truth"][0]["ecl"]
    )  # matures before reversion
    assert adverse["truth"][-1]["ecl"] > immediate["truth"][-1]["ecl"]


def test_task_wiring_and_stability():
    from inspect_ai.scorer import Score

    from pereval.scorers.stability import stability
    from pereval.tasks.cecl.task import cecl

    task = cecl(n_instances=3, seed=7, baseline="cohort", repeats=2)
    assert len(task.dataset) == 3
    assert [s.id.split("-")[2] for s in task.dataset] == [
        "baseline",
        "adverse",
        "benign",
    ]
    assert all(set(s.files) == set(public_files(generate())) for s in task.dataset)
    reduced = stability()([Score(value={"ecl_regret": x}) for x in (0.01, 0.04)])
    assert reduced.value["regret_worst"] == 0.04
    assert reduced.value["regret_spread"] == pytest.approx(0.03)


def test_reference_solver_to_scorer_round_trip(monkeypatch):
    """Exercise Inspect callables with a file sandbox double; no Docker required."""
    import asyncio
    from types import SimpleNamespace

    import inspect_ai.util

    from pereval.scorers.cecl import cecl_scorer
    from pereval.tasks.cecl.baselines import reference_solver

    bundle = generate(seed=9)
    files = public_files(bundle)

    class FileSandbox:
        async def read_file(self, name):
            if name not in files:
                raise FileNotFoundError(name)
            return files[name]

        async def write_file(self, name, text):
            files[name] = text

    monkeypatch.setattr(inspect_ai.util, "sandbox", lambda: FileSandbox())

    async def run():
        state = SimpleNamespace(metadata={"truth": bundle["truth"]})
        await reference_solver("cohort")(state, None)
        score = await cecl_scorer()(state, None)
        assert score.value["completion"] == 1
        assert score.value["ecl_regret"] < score.value["zero_regret"]
        del files["predictions.csv"]
        missing = await cecl_scorer()(state, None)
        assert missing.value["completion"] == 0

    asyncio.run(run())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_history": 2},
        {"forecast_quarters": 0},
        {"reversion_quarters": -1},
        {"scenario": "unknown"},
    ],
)
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError):
        generate(**kwargs)
