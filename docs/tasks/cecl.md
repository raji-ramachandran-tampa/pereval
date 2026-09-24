# CECL Lifetime Expected Credit Losses

This task tests whether an agent can estimate the lifetime allowance for a synthetic portfolio of closed-end, amortizing loans. Unlike CCAR's nine-quarter default-rate projection, the output is expected net principal loss in dollars over each pool's remaining contractual life. It is a modelling benchmark, not a complete accounting compliance assessment. No agent results have been measured for this task yet; it is not part of the README's existing ranking.

The separate [CECL Gemini Pilot, September 2026](https://github.com/raji-ramachandran-tampa/pereval/blob/codex/cecl-gemini-pilot-september-2026/pilots/cecl-gemini-september-2026/REPORT.md) is maintained in the contributor fork. It uses hosted Python and inline inputs, so it is not evidence for the Inspect/Docker isolation protocol and is excluded from this repository's run archive and rankings.

## Run

```bash
inspect eval pereval/tasks/cecl/task.py --model <provider/model>
inspect eval pereval/tasks/cecl/task.py -T baseline=cohort --model mockllm/model
inspect eval pereval/tasks/cecl/task.py -T baseline=naive --model mockllm/model
inspect eval pereval/tasks/cecl/task.py -T n_instances=9 -T seed=123 -T repeats=3 --model <provider/model>
inspect eval pereval/tasks/cecl/task.py -T scenario=adverse -T forecast_quarters=4 -T reversion_quarters=0 --model <provider/model>
```

Docker is required for Inspect runs, including reference solvers. Pure generator, reference and scoring functions can run locally without Docker or model access. `n_instances=3` rotates baseline, adverse and benign paths. Other options are `n_history` (80), `forecast_quarters` (8), `reversion_quarters` (4), `repeats` (1) and `message_limit` (500). Pin `scenario` for paired sensitivity comparisons.

## The Task

- `history.csv`: 80 noisy quarterly observations per segment of default probability (PD), loss given default (LGD), conditional prepayment probability, unemployment and house-price growth.
- `forecast.csv`: one macroeconomic path over the forecast period.
- `pools.csv`: 12 pools across three risk segments, with randomized balances and remaining terms of 4, 12, 24 and 40 quarters.
- `policy.json`: forecast and reversion lengths.

The agent estimates three conditional rate models per segment and builds a lifetime loss calculation. Rates depend on the macro variables through logistic relationships with coefficients redrawn per instance. This is a synthetic mechanism; it does not inherit CCAR's FRED calibration or claim empirical fidelity. Historical rate observations are noisy, while ground truth uses their underlying expectations. Only the four input files enter the sandbox.

The benchmark prescribes a linear transition of PD, LGD and prepayment from the last forecast quarter to segment historical mean rates. The first subsequent quarter has historical weight `1/R`, reaching full reversion in quarter `R`. `R=0` means immediate reversion. After that the historical rates remain fixed. The oracle's historical target is the mean of the latent conditional rates over the supplied history; the agent estimates it from noisy observations. This policy is part of the exercise, not a claim that CECL mandates this particular method.

For a pool with balance B and T remaining quarters, beginning exposure at q is `B * (1 - (q-1)/T) * S(q-1)`. Survival starts at one and updates as `S(q) = S(q-1) * (1-PD(q)) * (1-prepay(q))`. Expected loss is the sum of `exposure(q) * PD(q) * LGD(q)` through maturity. Default occurs before prepayment and scheduled principal payment. Recoveries are included in LGD. No discounting is applied to this net-principal-loss method; it is not a discounted-cash-flow implementation.

This tests lifetime versus annual horizons, amortization, competing exits, segment differences, net recovery severity, forecast sensitivity and reversion. Short pools mature before reversion; long pools expose incorrect tail assumptions.

## Scoring And References

Submit `predictions.csv` containing `pool_id,ecl`, in dollars. Each estimate must be finite and between zero and current principal. The portfolio allowance is the sum of the pool estimates. No prediction interval is requested: the target is the conditional mean lifetime loss, not a percentile of realized losses.

Primary `ecl_regret` is the balance-weighted squared error of pool loss rates: `sum(B_i / sum(B) * ((predicted_i - true_i) / B_i)^2)`. The exact conditional-mean oracle scores zero, so this is excess squared loss over the oracle. Lower is better; it is not comparable to CCAR's Winkler regret. Opposite errors in different pools cannot cancel. Also reported: loss-rate MAE in basis points, completion, and, for complete outputs only, portfolio dollars and signed portfolio bias.

Missing, malformed, duplicate, nonfinite or out-of-bounds pool estimates follow the suite convention: `max(degenerate score, 5 * oracle score)` for each pool. The oracle squared-error score is exactly zero, so this reduces to the zero-allowance reference score, `true_loss_rate^2`, weighted by pool balance. The MAE diagnostic likewise uses the zero-answer error. Missing answers cannot outperform the zero anchor; a valid but poor answer can score worse than it, as in the other tasks. Completion is reported separately, and portfolio totals are omitted unless every pool has a valid prediction. For a genuinely zero-loss pool, a missing answer has zero error but still zero completion. Extra IDs earn no credit. Infrastructure failures remain unmeasured. Repeated runs reuse the shared worst-case and spread reducer.

Three references anchor interpretation:

- Exact hidden conditional mean: zero regret, available to tests and the scorer.
- `cohort`: fits segment logit-linear rates using public inputs, then applies survival, amortization and reversion. Its functional family matches the current generator, so its performance is an estimation reference, not evidence of broad method robustness.
- `naive`: historical PD times historical LGD times remaining term times current balance, capped at balance; ignores runoff and forecasts. `zero_regret` also reports the degenerate zero-allowance answer's score.

## Why This Task Uses A Point Estimate

This is an explicit scope choice: the scored object is the conditional mean of lifetime net principal loss given the supplied macro path and policy. Squared error elicits that mean. A 95% prediction interval would instead describe variation in a future realized portfolio loss, which is a different target. A confidence interval around the estimated mean would describe estimation uncertainty and would require a separate repeated-sampling coverage experiment. Neither is interchangeable with the current target.

The generator currently produces expected pool losses analytically; it does not specify loan counts, individual loan sizes, cross-loan dependence, or a joint distribution of future default and recovery events. Those choices would materially determine a realized-loss interval. Adding arbitrary bounds around the expected allowance, or applying Winkler scoring to a deterministic oracle value, would not test calibrated predictive uncertainty. The appropriate extension would first define and simulate that predictive distribution, then request and score its intervals separately from the allowance estimate.

Accordingly, this version measures expected-loss accuracy and repeated-run stability, but does not measure interval calibration, coverage or sharpness. It covers only part of the suite's stated capabilities. Its point-only score must not be presented as interchangeable with the suite's interval scores or included in an aggregate ranking without an explicit decision about that narrower scope. Adding interval calibration remains a separate task-design extension, not a claim made by this version.

## What The Task Does Not Test

The history has exactly two macroeconomic columns, and both are true drivers. There are no distractor features and no feature-selection-under-collinearity challenge comparable to CCAR. The prompt fully supplies the accounting recursion, event ordering, amortization convention and rate-reversion blend. Implementing them correctly is tested; discovering or choosing those conventions is not.

The hidden work is estimating segment response laws and their coefficients from noisy observations, using them under out-of-range macroeconomic conditions, and consistently applying the disclosed mechanics over different contractual terms. The functional family is fixed and logistic, so the task does not establish robustness across alternative response families. These deliberately disclosed mechanics define a controlled exercise; they should not be interpreted as a claim that ASC 326 prescribes this particular recursion or reversion policy.

## Scope And Accounting Context

The [interagency policy statement on allowances for credit losses](https://www.federalreserve.gov/frrs/guidance/interagency-policy-statement-on-allowances-for-credit-losses.htm) describes expected losses over contractual terms with expected prepayments and reversion to historical information beyond reasonable and supportable forecasts. Those concepts motivate this task. The supplied path, reversion policy and loan conventions are benchmark assumptions, not prescribed choices for an institution.

The first version covers funded closed-end principal only. It excludes revolving commitments, renewals and extensions, purchased credit-deteriorated assets, collateral-dependent measurement, accrued interest, qualitative overlays, discounted cash flow methods, scenario mixtures, and management's selection and documentation of reasonable and supportable forecasts. It does not validate financial reporting controls or certify an allowance under ASC 326. Parameters vary but the logistic response family is fixed; broader families and real portfolio validation remain future work.
