"""Recompute hosted-pilot scores from archived responses, without API calls."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from pereval.scorers.cecl import score_predictions
from scripts.cecl_gemini_pilot import estimated_cost, extract_csv


def summarize(folder: Path):
    manifest = json.loads((folder / "manifest.json").read_text())
    rows, groups = [], defaultdict(list)
    for model in manifest["models"]:
        for scenario in ("baseline", "adverse", "benign"):
            for repeat in (1, 2):
                stem = f"{model}-{scenario}-repeat{repeat}"
                path = folder / f"{stem}.json"
                if not path.exists():
                    continue
                record = json.loads(path.read_text())
                row = {k: record[k] for k in ("model", "scenario", "repeat", "seconds")}
                row.update(
                    status="unmeasured",
                    executions=record.get("code_executions", 0),
                    cost=estimated_cost(
                        record.get("usage", {}),
                        manifest["pricing_usd_per_million"][model],
                    ),
                )
                raw_path = folder / f"{stem}-response.json"
                if raw_path.exists():
                    raw = json.loads(raw_path.read_text())
                    candidate = (raw.get("candidates") or [{}])[0]
                    if candidate.get("finish_reason") == "STOP":
                        # Tool use is a model choice, not an infrastructure failure.
                        # Score all naturally completed answers, even fabricated ones.
                        text = "\n".join(
                            p["text"]
                            for p in candidate.get("content", {}).get("parts", [])
                            if p.get("text") and not p.get("thought")
                        )
                        truth = json.loads(
                            (folder / f"{scenario}-truth.json").read_text()
                        )
                        predictions = extract_csv(text)
                        row.update(
                            status="measured", **score_predictions(truth, predictions)
                        )
                        if predictions:
                            (folder / f"{stem}-predictions.csv").write_text(
                                predictions, encoding="utf-8"
                            )
                rows.append(row)
                groups[model].append(row)
    summary = []
    for model, records in groups.items():
        measured = [r for r in records if r["status"] == "measured"]
        complete = len(records) == 6 and len(measured) == 6
        spreads = []
        for scenario in ("baseline", "adverse", "benign"):
            values = [r["ecl_regret"] for r in measured if r["scenario"] == scenario]
            if len(values) == 2:
                spreads.append(max(values) - min(values))
        summary.append(
            {
                "model": model,
                "measured": len(measured),
                "planned": 6,
                "rankable": complete,
                "mean_mae_bps": mean(r["loss_rate_mae_bps"] for r in measured)
                if measured
                else None,
                "worst_regret": max(r["ecl_regret"] for r in measured)
                if complete
                else None,
                "max_repeat_spread": max(spreads) if spreads else None,
                "completion": mean(r["completion"] for r in measured)
                if measured
                else 0,
                "runs_using_code": sum(r["executions"] > 0 for r in records),
                "estimated_cost_usd": sum(r["cost"] for r in records),
            }
        )
    refs = {
        m: [
            json.loads((folder / f"{s}-references.json").read_text())[m]
            for s in ("baseline", "adverse", "benign")
        ]
        for m in ("cohort", "naive")
    }
    matched = {}
    for model, records in groups.items():
        measured = [r for r in records if r["status"] == "measured"]
        if measured:
            matched[model] = {
                method: mean(
                    json.loads(
                        (folder / f"{r['scenario']}-references.json").read_text()
                    )[method]["loss_rate_mae_bps"]
                    for r in measured
                )
                for method in ("cohort", "naive")
            }
    data = {
        "models": summary,
        "runs": rows,
        "references": refs,
        "matched_reference_mae_bps": matched,
    }
    (folder / "summary.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    stream = io.StringIO()
    fields = [
        "model",
        "scenario",
        "repeat",
        "status",
        "ecl_regret",
        "loss_rate_mae_bps",
        "completion",
        "portfolio_bias",
        "executions",
        "seconds",
        "cost",
    ]
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    (folder / "results.csv").write_text(stream.getvalue(), encoding="utf-8")
    lines = [
        "# CECL Gemini Pilot — September 2026",
        "",
        "Pilot date: 2026-09-22.",
        "",
        (
            "Planned: two Gemini models, three shared seeded instances (baseline/adverse/benign), "
            "two independent requests per instance. Each instance contains 12 loan pools. "
            "This is an exploratory comparison, not a statistically resolved ranking."
        ),
        "",
        "| Model / reference | Measured runs | Mean loss-rate error (bps) | Worst squared-rate regret | Largest same-instance regret spread | Runs using code |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    def number(value, fmt=".6g"):
        return format(value, fmt) if value is not None else "unmeasured"

    for result in summary:
        lines.append(
            f"| {result['model']} | {result['measured']}/6 | "
            f"{number(result['mean_mae_bps'], '.2f')} | {number(result['worst_regret'])} | "
            f"{number(result['max_repeat_spread'])} | {result['runs_using_code']}/6 |"
        )
    for method, values in refs.items():
        lines.append(
            f"| {method} reference | 3 deterministic | "
            f"{mean(v['loss_rate_mae_bps'] for v in values):.2f} | "
            f"{max(v['ecl_regret'] for v in values):.6g} | 0 | local reference |"
        )
    lines.extend(
        [
            "",
            (
                "Lower errors are better. Model means cover measured requests only; the reference rows cover all three instances, "
                "so use the matched comparison below when model coverage is incomplete. One basis point is 0.01% of pool principal. "
                "The MAE is balance-weighted within each instance, then averaged over requests. "
                "Worst regret is the maximum balance-weighted squared loss-rate error across all six requests. "
                "The spread measures the difference between the two requests on byte-identical inputs."
            ),
            "",
            "## Per-run results",
            "",
            "| Model | Scenario | Repeat | Status | MAE (bps) | Regret |",
            "| --- | --- | ---: | --- | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['scenario']} | {row['repeat']} | {row['status']} | "
            f"{number(row.get('loss_rate_mae_bps'), '.2f')} | {number(row.get('ecl_regret'))} |"
        )
    lines.extend(
        [
            "",
            "## Method and limitations",
            "",
            (
                "Docker Desktop could not start because its dockerInference runtime socket was inaccessible. "
                "The reversible rename attempt also failed; no factory reset or data deletion was performed. "
                "This pilot instead supplied the public CSV/JSON inputs inline and enabled Google's hosted Python tool. "
                "The target data, economic assumptions and scorer are unchanged. Tool environment, input delivery, "
                "budget and final-answer extraction differ from Inspect/Docker, so these scores must not be inserted "
                "into the existing benchmark leaderboard."
            ),
            "",
            (
                "Both models use temperature 1, LOW thinking level, one request per attempt, a 32,768 output-token "
                "ceiling, a 300-second request timeout and no automatic retries. A tool-enabled request does not "
                "guarantee tool use. Returned numeric answers are scored even when the model chooses not to execute "
                "code; that choice is model behavior. API failures and non-STOP finishes are unmeasured. "
                "No model is ranked unless all six requests are measured. Blank or invalid predictions on a normal "
                "completion receive the benchmark's missing-answer penalty."
            ),
            "",
            (
                "The initial runner marked a normally completed answer without code execution as unmeasured. "
                "This report corrects that classification by rescoring every archived STOP response, regardless "
                "of tool use. Raw response files and original run records remain intact; summary.json and results.csv "
                "are the corrected analysis. No outputs were repaired or regenerated for scoring."
            ),
            "",
            (
                "The exact oracle scores zero. The fitted cohort reference knows the generator's logistic family "
                "but estimates its coefficients from the same public data. Only three instances and two repeats "
                "are available, so the observed maximum is not a bound on future errors. Cases share a fixed "
                "response family and are synthetic."
            ),
            "",
            (
                f"Estimated token cost for recorded requests: ${sum(r['estimated_cost_usd'] for r in summary):.4f}. "
                "Computed from returned prompt, tool-context, candidate and thinking-token counts, allowing for cached-input discounts, at "
                "[Google's published standard prices](https://ai.google.dev/gemini-api/docs/pricing), "
                "not a billing statement. Excludes three endpoint probes and any unreported failed-request usage. "
                "Gemini 2.5 Flash-Lite returned HTTP 404 in a preliminary attempt, archived separately."
            ),
            "",
            (
                "Reproduce analysis without further API calls: `python -m scripts.cecl_pilot_report`. "
                "Run/resume the pilot with `python -m scripts.cecl_gemini_pilot` after installing `google-genai` "
                "and setting GEMINI_API_KEY. Existing attempt records are skipped. Prompts, hidden targets, raw "
                "API responses, extracted predictions, and usage are stored beside this report."
            ),
            "",
        ]
    )
    findings = ["## Findings on completed requests", ""]
    for result in summary:
        model = result["model"]
        if model in matched:
            findings.append(
                f"{model}: {result['measured']} scoreable requests out of six planned; "
                f"mean error {result['mean_mae_bps']:.2f} bps. On exactly those same instances "
                f"and repeat weights, the fitted cohort reference averaged {matched[model]['cohort']:.2f} bps "
                f"and the naive reference {matched[model]['naive']:.2f} bps. "
                f"The model used the hosted code tool in {result['runs_using_code']} requests."
            )
        else:
            findings.append(
                f"{model}: no usable model results; no accuracy comparison is supported."
            )
        findings.append("")
    findings.extend(
        [
            (
                "Gemini 3.8 returned HTTP 503 on both baseline attempts, so its four remaining planned calls "
                "were not made. Flash-Lite returned HTTP 503 on the first benign attempt. A separate minimal "
                "Gemini 3.5 Flash-Lite hosted-code probe timed out (HTTP 504); it was not added to the comparison. "
                "These are provider failures, not scored model errors. The intended two-model evaluation remains incomplete."
            ),
            "",
        ]
    )
    lines[6:6] = findings
    (folder / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder", nargs="?", default="runs/cecl-gemini3-hosted-20260922"
    )
    print(json.dumps(summarize(Path(parser.parse_args().folder)), indent=2))
