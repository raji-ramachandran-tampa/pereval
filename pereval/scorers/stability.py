"""Within-instance stability: repeated agent runs on one fixed instance.

Every other measurement in this suite varies the instance between runs, so its
spread mixes two things: how hard the drawn instance was, and how much the agent's
method wanders. Those have different consequences and only one of them is the
agent's fault.

This reducer measures the second alone. Set `-T repeats=k` on any task and Inspect
runs the agent k times on each sample, on byte-identical inputs, and this reducer
collapses the k scores into one record that keeps the dispersion:

    regret_worst    the worst of the k runs, the suite's pessimistic statistic
    regret_spread   max - min across the k runs, pure method variance
    runs            k, so a table can show what the spread was measured over

Why it is a headline number rather than a diagnostic: it decides how far evidence
about the process carries to the artifact. A spread near zero means the model in
front of a validator is close to what the pipeline reliably produces, so evidence
about the pipeline is evidence about it. A large spread means it is a draw, and has
to be judged on its own terms.

The problem is reproducibility, not change control. Rerunning a development tool is
not a model change: a model change is a change to what runs in production, and
exercising a model during development or validation is not one. What instability
costs is developmental evidence, which SR 26-2 puts at the centre of conceptual
soundness when it asks for "the quality and extent of developmental evidence"
(Conceptual Soundness, page 8 of the guidance attached to
https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm). A specification
that cannot be re-derived cannot be replicated by an independent team either.

The suite's own reproducibility experiment (docs/limitations.md) found this spread
to be large for capable models and near zero only for models that had collapsed to
a fixed deterministic method, which is the worst behaviour the tasks exist to
expose. That inversion is the reason this is worth measuring on every task rather
than once as a side study.
"""

from __future__ import annotations

import numpy as np
from inspect_ai.scorer import Score, ScoreReducer, score_reducer

# The primary regret key differs by scorer; check each supported task family.
_REGRET_KEYS = ("winkler_regret", "pinball_regret", "ecl_regret")


def _primary(value: dict) -> str | None:
    return next((k for k in _REGRET_KEYS if k in value), None)


@score_reducer(name="stability")
def stability() -> ScoreReducer:
    """Reduce k same-instance runs to their mean, worst case, and spread."""

    def reduce(scores: list[Score]) -> Score:
        values = [s.value for s in scores if isinstance(s.value, dict)]
        if not values:
            return scores[0]
        key = _primary(values[0])
        keys = {k for v in values for k, x in v.items() if isinstance(x, int | float)}
        out: dict[str, float] = {}
        for k in keys:
            xs = [float(v[k]) for v in values if k in v and np.isfinite(float(v[k]))]
            out[k] = float(np.mean(xs)) if xs else float("nan")
        if key is not None:
            regrets = [float(v[key]) for v in values
                       if key in v and np.isfinite(float(v[key]))]
            if regrets:
                out["regret_worst"] = float(np.max(regrets))
                out["regret_spread"] = float(np.max(regrets) - np.min(regrets))
        out["runs"] = float(len(values))
        n = len(values)
        detail = ", ".join(f"{float(v[key]):.4g}" for v in values) if key else ""
        return Score(
            value=out,
            explanation=(
                f"{n} same-instance runs; mean {out.get(key, float('nan')):.4g}, "
                f"worst {out.get('regret_worst', float('nan')):.4g}, "
                f"spread {out.get('regret_spread', float('nan')):.4g}"
                + (f" [{detail}]" if detail else "")
            ),
            metadata={"per_run": [s.value for s in scores]},
        )

    return reduce


def epochs(repeats: int):
    """Inspect Epochs for `repeats` same-instance runs, or None for a single run."""
    if repeats is None or repeats <= 1:
        return None
    from inspect_ai import Epochs

    return Epochs(int(repeats), [stability()])
