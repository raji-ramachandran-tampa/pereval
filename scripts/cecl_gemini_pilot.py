"""Small hosted-code CECL pilot; deliberately separate from Inspect/Docker results.

Run from the repository root with GEMINI_API_KEY set. Requires google-genai.
Each response, public prompt and hidden scoring target is archived for audit.
No model-generated code executes on the host. Existing run files are never rerun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from pereval.scorers.cecl import score_predictions
from pereval.tasks.cecl.baselines import predict
from pereval.tasks.cecl.generator import generate, public_files
from pereval.tasks.cecl.task import INSTRUCTIONS

MODELS = ("gemini-3.1-flash-lite", "gemini-3.8-flash")
SCENARIOS = ("baseline", "adverse", "benign")
ADAPTER = """
Hosted-execution adaptation: the input files are supplied verbatim below, rather
than preinstalled. Read the embedded contents into Python strings/data frames or
create local files in your hosted code environment. Use the supplied Python code
execution tool to do the modelling and arithmetic. Do not rely on local file
delivery to the evaluator: your FINAL text must contain the full predictions.csv
contents in one fenced csv block, with pool_id,ecl and all 12 rows. Use the actual
supplied observations, not a shortened example. No web/search tool is available.
"""


def extract_csv(text):
    blocks = re.findall(r"```(?:csv)?\s*\n(pool_id,ecl\s*\n.*?)```", text, re.DOTALL)
    if blocks:
        return blocks[-1].strip() + "\n"
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "pool_id,ecl"]
    if not starts:
        return None
    rows = ["pool_id,ecl"]
    for line in lines[starts[-1] + 1 :]:
        if not line.strip() or "," not in line:
            break
        rows.append(line.strip())
    return "\n".join(rows) + "\n"


def save(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def estimated_cost(usage, price):
    inputs = usage.get("prompt_token_count", 0) + usage.get(
        "tool_use_prompt_token_count", 0
    )
    cached = min(inputs, usage.get("cached_content_token_count", 0))
    outputs = usage.get("candidates_token_count", 0) + usage.get(
        "thoughts_token_count", 0
    )
    return (
        (inputs - cached) * price[0] + cached * price[0] * 0.1 + outputs * price[1]
    ) / 1e6


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="runs/cecl-gemini3-hosted-20260922")
    parser.add_argument("--max-new-runs", type=int, default=12)
    parser.add_argument("--only-model", choices=MODELS)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    from google import genai
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=INSTRUCTIONS + ADAPTER,
        tools=[types.Tool(code_execution=types.ToolCodeExecution())],
        temperature=1.0,
        max_output_tokens=32768,
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    manifest = {
        "protocol": "hosted-code-pilot-v1",
        "seed": 123,
        "models": MODELS,
        "repeats": 2,
        "config": config.model_dump(mode="json", exclude_none=True),
        "pricing_source": "https://ai.google.dev/gemini-api/docs/pricing",
        "pricing_usd_per_million": {MODELS[0]: [0.25, 1.50], MODELS[1]: [0.75, 3.75]},
        "limitations": "Hosted Python and inline inputs, not Inspect/Docker. Three instances, two repeats; exploratory only.",
    }
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != json.loads(
        json.dumps(manifest)
    ):
        raise SystemExit(
            "Existing pilot configuration differs; use another output directory"
        )
    save(manifest_path, manifest)
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=types.HttpOptions(
            timeout=300000, retry_options=types.HttpRetryOptions(attempts=1)
        ),
    )
    new_runs = 0
    seeds = np.random.SeedSequence(123).generate_state(3)
    for i, scenario in enumerate(SCENARIOS):
        bundle = generate(seed=int(seeds[i]), scenario=scenario)
        files = public_files(bundle)
        prompt = "\n\n".join(
            f"FILE {name}\n```\n{content}\n```" for name, content in files.items()
        )
        (out / f"{scenario}-inputs.txt").write_text(prompt, encoding="utf-8")
        save(out / f"{scenario}-truth.json", bundle["truth"])
        save(
            out / f"{scenario}-references.json",
            {
                method: score_predictions(bundle["truth"], predict(files, method))
                for method in ("cohort", "naive")
            },
        )
        for model in MODELS:
            if args.only_model and model != args.only_model:
                continue
            for repeat in (1, 2):
                stem = f"{model}-{scenario}-repeat{repeat}"
                result_path = out / f"{stem}.json"
                if result_path.exists() or new_runs >= args.max_new_runs:
                    continue
                new_runs += 1
                start = time.monotonic()
                record = {
                    "model": model,
                    "scenario": scenario,
                    "repeat": repeat,
                    "instance_seed": int(seeds[i]),
                    "input_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                }
                print(f"Starting {stem}", flush=True)
                try:
                    response = client.models.generate_content(
                        model=model, contents=prompt, config=config
                    )
                    raw = response.model_dump(
                        mode="json", exclude_none=True, exclude={"sdk_http_response"}
                    )
                    save(out / f"{stem}-response.json", raw)
                    candidate = response.candidates[0] if response.candidates else None
                    parts = (
                        candidate.content.parts
                        if candidate and candidate.content
                        else []
                    )
                    text = "\n".join(p.text for p in parts if p.text and not p.thought)
                    (out / f"{stem}.md").write_text(text, encoding="utf-8")
                    record["finish_reason"] = (
                        str(candidate.finish_reason) if candidate else "no_candidate"
                    )
                    record["code_executions"] = sum(
                        p.code_execution_result is not None for p in parts
                    )
                    record["successful_code_executions"] = sum(
                        p.code_execution_result is not None
                        and p.code_execution_result.outcome == types.Outcome.OUTCOME_OK
                        for p in parts
                    )
                    record["usage"] = raw.get("usage_metadata", {})
                    usage = record["usage"]
                    price = manifest["pricing_usd_per_million"][model]
                    record["estimated_cost_usd"] = estimated_cost(usage, price)
                    if (
                        not candidate
                        or candidate.finish_reason != types.FinishReason.STOP
                    ):
                        record["status"] = "unmeasured_finish"
                    else:
                        csv = extract_csv(text)
                        if csv:
                            (out / f"{stem}-predictions.csv").write_text(
                                csv, encoding="utf-8"
                            )
                        record["scores"] = score_predictions(bundle["truth"], csv)
                        record["status"] = "measured"
                except Exception as error:  # noqa: BLE001 -- archive failed attempts without leaking credentials
                    record.update(
                        status="unmeasured_error",
                        error_type=type(error).__name__,
                        error_code=getattr(error, "code", None),
                    )
                    # Never archive exception bodies that could contain credentials.
                record["seconds"] = time.monotonic() - start
                save(result_path, record)
                print(json.dumps(record), flush=True)
                if record["status"] == "unmeasured_error":
                    print(
                        "Stopping pilot after API error; no automatic paid retries.",
                        flush=True,
                    )
                    client.close()
                    return
    client.close()


if __name__ == "__main__":
    main()
