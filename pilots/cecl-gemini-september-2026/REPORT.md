# CECL Gemini Pilot, September 2026

Pilot date: 2026-09-22.

Planned: two Gemini models, three shared seeded instances (baseline/adverse/benign), two independent requests per instance. Each instance contains 12 loan pools. This is an exploratory comparison, not a statistically resolved ranking.

## Findings On Completed Requests

**gemini-3.1-flash-lite was worse than predicting zero for every pool on 2 of 5 measured runs**, on the primary squared-error metric.

- Baseline repeat 1: regret 0.164894, versus zero-answer regret 0.00606664 (27.18 times as large).
- Benign repeat 2: regret 0.000729034, versus zero-answer regret 0.000537742 (1.36 times as large).

On byte-identical baseline inputs, the two repeats differed by 27.60 times in mean absolute loss-rate error and 959.41 times in squared-error regret. These ratios describe different metrics and must not be conflated.

On byte-identical adverse inputs, the two repeats differed by 1.19 times in mean absolute loss-rate error and 1.24 times in squared-error regret. These ratios describe different metrics and must not be conflated.

gemini-3.1-flash-lite: 5 scoreable requests out of six planned; mean error 847.14 bps. On exactly those same instances and repeat weights, the fitted cohort reference averaged 16.74 bps and the naive reference 605.14 bps. The model used the hosted code tool in 2 requests.

gemini-3.8-flash: no usable model results; no accuracy comparison is supported.

Gemini 3.8 returned HTTP 503 on both baseline attempts, so its four remaining planned calls were not made. Flash-Lite returned HTTP 503 on the first benign attempt. A separate minimal Gemini 3.5 Flash-Lite hosted-code probe timed out (HTTP 504); it was not added to the comparison. These are provider failures, not scored model errors. The intended two-model evaluation remains incomplete.

| Model / reference | Measured runs | Mean loss-rate error (bps) | Worst squared-rate regret | Largest same-instance regret spread | Runs using code |
| --- | ---: | ---: | ---: | ---: | ---: |
| gemini-3.1-flash-lite | 5/6 | 847.14 | unmeasured | 0.164722 | 2/6 |
| gemini-3.8-flash | 0/6 | unmeasured | unmeasured | unmeasured | 0/6 |
| cohort reference | 3 deterministic | 14.39 | 9.98365e-06 | 0 | local reference |
| naive reference | 3 deterministic | 671.38 | 0.0158477 | 0 | local reference |

Lower errors are better. Model means cover measured requests only; the reference rows cover all three instances, so use the matched comparison below when model coverage is incomplete. One basis point is 0.01% of pool principal. The MAE is balance-weighted within each instance, then averaged over requests. Worst regret is the maximum balance-weighted squared loss-rate error across all six requests. The spread measures the difference between the two requests on byte-identical inputs.

## Per-Run Results

| Model | Scenario | Repeat | Status | MAE (bps) | Regret |
| --- | --- | ---: | --- | ---: | ---: |
| gemini-3.1-flash-lite | baseline | 1 | measured | 3022.59 | 0.164894 |
| gemini-3.1-flash-lite | baseline | 2 | measured | 109.52 | 0.000171871 |
| gemini-3.1-flash-lite | adverse | 1 | measured | 505.64 | 0.00371215 |
| gemini-3.1-flash-lite | adverse | 2 | measured | 426.11 | 0.00300416 |
| gemini-3.1-flash-lite | benign | 1 | unmeasured | unmeasured | unmeasured |
| gemini-3.1-flash-lite | benign | 2 | measured | 171.86 | 0.000729034 |
| gemini-3.8-flash | baseline | 1 | unmeasured | unmeasured | unmeasured |
| gemini-3.8-flash | baseline | 2 | unmeasured | unmeasured | unmeasured |

## Method And Limitations

Docker Desktop could not start because its dockerInference runtime socket was inaccessible. The reversible rename attempt also failed; no factory reset or data deletion was performed. This pilot instead supplied the public CSV/JSON inputs inline and enabled Google's hosted Python tool. The target data, economic assumptions and scorer are unchanged. Tool environment, input delivery, budget and final-answer extraction differ from Inspect/Docker, so these scores must not be inserted into the existing benchmark leaderboard.

Both models use temperature 1, LOW thinking level, one request per attempt, a 32,768 output-token ceiling, a 300-second request timeout and no automatic retries. A tool-enabled request does not guarantee tool use. Returned numeric answers are scored even when the model chooses not to execute code; that choice is model behavior. API failures and non-STOP finishes are unmeasured. No model is ranked unless all six requests are measured. Blank or invalid predictions on a normal completion receive the benchmark's missing-answer penalty.

The initial runner marked a normally completed answer without code execution as unmeasured. This report corrects that classification by rescoring every archived STOP response, regardless of tool use. Raw response files and original run records remain intact; summary.json and results.csv are the corrected analysis. No outputs were repaired or regenerated for scoring.

The exact oracle scores zero. The fitted cohort reference knows the generator's logistic family but estimates its coefficients from the same public data. Only three instances and two repeats are available, so the observed maximum is not a bound on future errors. Cases share a fixed response family and are synthetic.

Estimated token cost for recorded requests: $0.1054. Computed from returned prompt, tool-context, candidate and thinking-token counts, allowing for cached-input discounts, at [Google's published standard prices](https://ai.google.dev/gemini-api/docs/pricing), not a billing statement. Excludes three endpoint probes and any unreported failed-request usage. Gemini 2.5 Flash-Lite returned HTTP 404 in a preliminary attempt; its bulky inputs/truth archive was removed during review.

Reproduce analysis without further API calls: `python -m scripts.cecl_pilot_report`. Run/resume the pilot with `python -m scripts.cecl_gemini_pilot` after installing `google-genai` and setting GEMINI_API_KEY. Existing attempt records are skipped. Prompts, hidden targets, raw API responses, extracted predictions, and usage are stored beside this report.
