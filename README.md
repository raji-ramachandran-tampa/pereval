# perEval[^1]

**An [Inspect](https://inspect.aisi.org.uk/)-based evaluation suite for quantitative model development tasks.**

> **Status: working prototype.** Example tasks, baselines, and tests are in place; the results shown here are for illustration only due to small sample size. Corners deliberately cut are documented in [docs/limitations.md](docs/limitations.md) rather than hidden.

## Summary

Generic coding and Q&A benchmarks don't test whether an LLM agent can *develop and estimate a statistical model*. perEval probes that corner with tasks drawn from diverse areas including credit risk and macroeconomic loss modeling.

It qualifies a development *process*, not a model. A nine-quarter stress projection cannot be validated against outcomes, because the actuals arrive nine quarters late on a path that is not the scenario path, so the testable object is the process that produced it. [docs/claim.md](docs/claim.md) states what a score supports, what it does not, and how the coverage maps onto SR 26-2.

### Design Principles

1. **Objective verification.** Tasks are designed to have an objectively verifiable target. See [docs/task-design.md](docs/task-design.md) for details.
2. **Scalar regret.** Each task scores a continuous regret against an oracle or reference. A scalar carries more information than a single bit (pass/fail). A task keeps discriminating even when every LLM agent does well on it, or every LLM agent struggles: the regret *magnitudes* separate them.
3. **Pessimistic assessment.** As of mid-2026, credit risk and stress testing models are well within the capabilities of most LLM agents. The question is not whether LLMs can do it at all or how well they do it on average, but whether the *worst case* result is still good enough. Maximum regret (worst result across several runs) is used for ranking.

Scores are anchored against three references rather than one, and a run that failed for reasons outside the agent's control is reported as unmeasured rather than scored. Both rules are set out in [docs/task-design.md](docs/task-design.md#anchoring-and-non-response).

## Quick Start

```bash
pip install -e .
inspect eval pereval/tasks/ccar/task.py --model <provider/model>        # needs Docker
```

Every task also ships reference solvers that run without a model and bracket the score from both ends:

```bash
inspect eval pereval/tasks/ccar/task.py -T baseline=vasicek --model mockllm/model   # competent
inspect eval pereval/tasks/ccar/task.py -T baseline=naive   --model mockllm/model   # floor
```

Use `-T n_instances=N` for many fresh instances with standard errors, and `-T repeats=K` to rerun the agent on the *same* instance, which separates method instability from instance difficulty ([why that is the number to watch](docs/limitations.md#same-instance-stability)). See [docs/setup.md](docs/setup.md) for the Python environment, Docker install (required only for the sandboxed evaluation), and model credentials.

## Tasks

| Task | Type | Challenge |
| --- | --- | --- |
| [CCAR stress loss](docs/tasks/ccar.md) | realistic domain | feature selection under collinearity, transform discovery, bounded functional form, stress extrapolation |
| [CECL lifetime losses](docs/tasks/cecl.md) | synthetic credit portfolio | lifetime allowance, amortization, defaults and prepayments, recoveries, forecast reversion |
| [Macro tail quantiles](docs/tasks/quantile.md) | realistic domain | population vs sample quantile, tail extrapolation from 10 observations |
| [Ballistic extrapolation](docs/tasks/ballistic.md) | controlled mechanism | out-of-range extrapolation against velocity-dependent drag |
| [Orbital: two-body, three-body, hyperbolic flyby](docs/tasks/orbital.md) | controlled mechanism | periodic signal recovery, coupled retrograde geometry, angles-only orbit determination |

CCAR and quantile are the original realistic domain tasks. CECL adds a synthetic lifetime-credit-loss task; it has no measured agent results yet and is not included in the score table below. The mechanism tasks calibrate the harness across a difficulty gradient with exactly known ground truth.

## Summary Scores

> **Not a leaderboard, and not comparable across columns.** Every column is Winkler regret except Quantile, which is pinball regret, and the scales differ by orders of magnitude, so a number in one column says nothing about another.
>
> **Within a column, only order-of-magnitude gaps mean anything.** Re-running the same model on byte-identical instances moved cells by 7.4x, and the reproducibility study found swings up to 30x, so any two cells within a factor of a few are the same cell.

Each cell is the **worst-case (maximum) regret** over at least three runs, CCAR over eight instances. Lower is better everywhere, and a cell is reported only if none of its runs failed for reasons outside the agent's control. The last column is the **mean rank** (a Borda count within each column), deliberately the only aggregate, since the cells are not comparable across columns. Per-task detail is in the docs linked above, and the caveats behind every number are in [docs/limitations.md](docs/limitations.md).

| Model | CCAR | Ballistic | Two-body | Three-body¹ | Flyby | Quantile | Mean rank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| *Oracle (true model)* | 0.031 | — | 0.005 | 0.039 | 0.31 | — | *floor* |
| Kimi K3⁴ | 0.12 | 14 | 0.094 | **3.3** | **50** | 0.077 | **2.00** |
| deepseek-v4-flash-0731⁵ | 0.30 | 14 | 0.30 | 269 | 522 | 0.067 | 3.17 |
| GLM-5.1 | 0.055 | 12 | 0.092 | 436 | 899 | 0.25 | 3.92 |
| mimo-v2.5-free | 0.57 | 12 | 0.19 | 2426 | 292 | 0.15 | 5.25 |
| ling-3.0-flash:free | 0.193 | 43 | 1556 | 3309 | 591 | 0.076³ | 5.83 |
| nemotron-3-ultra:free | 0.27 | 16 | 2429 | 817 | 1356 | 0.099 | 6.33 |
| nemotron-3-super:free | 0.45 | 60 | 106 | 370 | 918 | 0.12 | 6.58 |
| *Naive baseline* | 0.85 | 22 | 17 | 332 | 20037 | 0.14 | *6.67* |
| Claude Haiku 4.5 | 0.37 | 83 | 105 | 1850 | 1348 | 0.10 | 7.17 |
| laguna-m.1:free | 1.1 | 60 | 78 | 2067² | 1014 | 0.16 | 8.08 |
| *Degenerate answer* | *0.57* | *61* | *2861* | *3019* | *138* | *0.12* | *not ranked* |

**Two models separate from the field, and nothing below them is ordered.** Permuting ranks within each column reproduces the observed spread at p = 0.004, but removing Kimi K3 and deepseek-0731 takes it to p = 0.561, so the aggregate detects two genuinely different models rather than a resolved ranking ([detail](docs/limitations.md#what-completing-one-row-did-to-the-aggregate)).

**Three-body and the flyby are the tasks with headroom left.** On three-body, K3's 3.3 sits roughly two orders of magnitude clear of a next-best 269, and seven of the nine ranked models are worse than the naive harmonic fit. On the flyby, nine of the ten ranked rows are worse than the degenerate answer and K3 is the only one that beats it. Two-body and CCAR, by contrast, are solved by most of the cast, which is the difficulty gradient working as designed.

**The three reference rows are the point of the table.** The *oracle* is the true generating model, the floor a perfect answer reaches (ballistic and quantile have none by design). The *naive baseline* is the obvious wrong method, and two ranked models fall through it. The *degenerate answer* is a single constant with no interval, and it reads differently in every column on purpose: ling posts the lowest quantile score in the table and is still worse than a constant on three-body.

CCAR's response law is now drawn per instance rather than fixed and published. That changed the answer key, not the question, so archived runs stay in the column ([detail](docs/limitations.md#the-ccar-response-law-was-public)).

¹ Three-body was re-measured under the corrected circular scorer. Re-running moved cells hard and in both directions on identical instances, so the column inherits run-to-run instability along with the fix ([detail](docs/limitations.md#re-measuring-three-body)).

² laguna keeps an archived value on the superseded scorer, because `poolside/laguna-m.1` is no longer served and cannot be re-run. It is last on any reading.

³ ling's 0.076 is the lowest model score in that column but is not a better method: on two of three seeds it reproduces the moment-matched normal baseline to four decimal places ([detail](docs/tasks/quantile.md)).

⁴ Kimi K3 is the only frontier model here and is not the only paid one (Claude Haiku 4.5 is also paid). Its cells come from two serving endpoints, compared on one matched instance ([detail](docs/limitations.md#serving-provenance-is-not-controlled)).

⁵ deepseek-v4-flash-0731 replaces deepseek-v4-flash-free, a different and now-superseded model version whose row had three unmeasured cells. It beats the degenerate constant on every run of every task except two of three on the flyby ([detail](docs/limitations.md#budget-confounds)).

## Layout

```
pereval/            Python package: Inspect tasks and scorers
  tasks/ccar/       FRED-calibrated macro + Vasicek generator, task, OLS + Vasicek baselines
  tasks/cecl/       synthetic lifetime loss generator, task, cohort + naive baselines
  tasks/quantile/   screened FRED YoY snapshot, generator, task, six reference estimators
  tasks/ballistic/  generator, Inspect task, Docker sandbox, quadratic baseline
  tasks/orbit/      two-body, three-body, and hyperbolic-flyby generators, tasks, baselines
  scorers/          oracle-anchored interval scorer (linear and circular), pinball regret
scripts/            FRED enumeration and screening, transcript export, paired analysis
tests/              scorer validation suite + generator/scorer integration
runs/, logs/        archived transcripts and Inspect logs behind every published table
docs/               claim and scope, task design, limitations, per-task scores
```

## License

MIT for this repository's own code. The ballistic task depends on [py-ballisticcalc](https://github.com/o-murphy/py-ballisticcalc) (LGPL-3.0), used as an unmodified installed dependency and not redistributed here, so it imposes no obligations on this code.

[^1]: *pereval* (Russian: перевал, "mountain pass"): the hard route through, not around. It also happens to end in `eval`.
