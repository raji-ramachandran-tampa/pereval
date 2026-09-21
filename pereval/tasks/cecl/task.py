"""CECL lifetime expected credit loss evaluation using the shared Docker sandbox."""

from __future__ import annotations

import numpy as np
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import basic_agent, system_message
from inspect_ai.tool import bash, python

from pereval.scorers.cecl import cecl_scorer
from pereval.scorers.stability import epochs
from pereval.tasks.ballistic.task import COMPOSE
from pereval.tasks.cecl.baselines import reference_solver
from pereval.tasks.cecl.generator import generate, public_files

INSTRUCTIONS = """Build lifetime expected credit loss estimates for closed-end loan pools.
Inputs in data/:
- history.csv: segment-level quarterly observed PD (default probability), LGD
  (net loss/defaulted principal after recoveries), prepay (full prepayment
  probability conditional on no default), unemployment (percent), hpi_growth
  (year-over-year fraction). Rates are quarterly, not annualized. Historical
  observations are noisy estimates of the segment's conditional mean rates.
- forecast.csv: one supplied economic path for the reasonable-and-supportable
  period. Treat this path as the expectation scenario for this exercise.
- pools.csv: pool_id, segment, current principal balance in dollars and remaining
  contractual quarters. Pools share their segment's rates; there is no age effect.
- policy.json: forecast and reversion lengths, in quarters.

Estimate quarterly PD, LGD and prepayment from the segment history and forecast.
Beyond the forecast period, linearly blend each rate from its last forecast
value to its segment historical mean over reversion_quarters. The first
post-forecast quarter uses weight 1/reversion_quarters on history; the last uses
weight 1. Zero means immediate reversion. Afterward hold historical rates fixed.
Estimate historical means from the supplied history.

For a pool with initial balance B and remaining term T, a surviving loan's
beginning principal in quarter q (1-based) is B*(1-(q-1)/T). Default occurs first
on beginning principal. Nondefaulting loans may then prepay fully; remaining
loans make their scheduled equal-principal payment. Neither defaulted nor
prepaid loans re-enter. Recoveries are fully reflected in LGD. Within a quarter,
LGD is the conditional mean severity of defaults. There is no discounting, new
lending, renewal, extension, accrued interest or fees in this exercise.

Output the conditional MEAN total net principal loss over each pool's remaining
life, not a stress percentile or one year's loss. Write predictions.csv with
exactly pool_id,ecl, one row per pool, dollars with 0 <= ecl <= balance. The
portfolio allowance is the sum of pool allowances. Primary score is balance-
weighted squared error in pool lifetime loss rates; zero is perfect. Pool errors
cannot cancel. Missing, duplicate, nonfinite or out-of-bounds answers are penalized.

Python has numpy, pandas, scipy, statsmodels and scikit-learn; no internet.
Executions use fresh interpreters: save self-contained scripts to files.
Write a complete predictions.csv early and refine it before finishing.
"""


@task
def cecl(
    n_instances: int = 3,
    seed: int | None = None,
    n_history: int = 80,
    forecast_quarters: int = 8,
    reversion_quarters: int = 4,
    scenario: str = "rotate",
    baseline: str = "",
    repeats: int = 1,
    message_limit: int = 500,
) -> Task:
    if n_instances < 1 or repeats < 1:
        raise ValueError("n_instances and repeats must be positive")
    if baseline not in ("", "naive", "cohort"):
        raise ValueError("baseline must be empty, naive or cohort")
    if scenario not in ("rotate", "baseline", "adverse", "benign"):
        raise ValueError("Unknown CECL scenario")
    samples = []
    for i, child in enumerate(np.random.SeedSequence(seed).generate_state(n_instances)):
        selected = (
            ("baseline", "adverse", "benign")[i % 3]
            if scenario == "rotate"
            else scenario
        )
        bundle = generate(
            int(child), n_history, forecast_quarters, reversion_quarters, selected
        )
        samples.append(
            Sample(
                id=f"instance-{i}-{selected}-seed-{int(child)}",
                input="Estimate lifetime expected credit losses and write predictions.csv.",
                files=public_files(bundle),
                metadata={"truth": bundle["truth"]},
            )
        )
    solver = (
        reference_solver(baseline)
        if baseline
        else basic_agent(
            init=system_message(INSTRUCTIONS),
            tools=[bash(timeout=240), python(timeout=240)],
            message_limit=message_limit,
        )
    )
    return Task(
        dataset=samples,
        solver=solver,
        scorer=cecl_scorer(),
        sandbox=("docker", COMPOSE),
        epochs=epochs(repeats),
    )
