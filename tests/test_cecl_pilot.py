"""Archive-only pilot analysis tests; never call a model provider."""

import json
import shutil
from pathlib import Path

import pytest

from scripts.cecl_gemini_pilot import estimated_cost, extract_csv
from scripts.cecl_pilot_report import summarize


def test_final_csv_preserves_bad_rows_for_scorer():
    text = "```csv\npool_id,ecl\nA,1\n```\nFinal:\n```csv\npool_id,ecl\nA,nan\nA,2\n```"
    assert extract_csv(text) == "pool_id,ecl\nA,nan\nA,2\n"
    assert extract_csv("No answer") is None


def test_cost_includes_tool_context_and_cached_discount():
    usage = {
        "prompt_token_count": 100,
        "tool_use_prompt_token_count": 200,
        "cached_content_token_count": 50,
        "candidates_token_count": 20,
        "thoughts_token_count": 10,
    }
    assert estimated_cost(usage, [0.25, 1.5]) == pytest.approx(
        (250 * 0.25 + 50 * 0.025 + 30 * 1.5) / 1e6
    )


def test_no_execution_answer_scored_but_failed_model_not_ranked(tmp_path):
    def save(name, data):
        (tmp_path / name).write_text(json.dumps(data))

    save(
        "manifest.json",
        {"models": ["test"], "pricing_usd_per_million": {"test": [0.25, 1.5]}},
    )
    for scenario in ("baseline", "adverse", "benign"):
        save(f"{scenario}-truth.json", [{"pool_id": "A", "balance": 100, "ecl": 10}])
        save(
            f"{scenario}-references.json",
            {m: {"loss_rate_mae_bps": 0, "ecl_regret": 0} for m in ("cohort", "naive")},
        )
    save(
        "test-baseline-repeat1.json",
        {
            "model": "test",
            "scenario": "baseline",
            "repeat": 1,
            "seconds": 1,
            "status": "unmeasured_no_successful_execution",
        },
    )
    save(
        "test-baseline-repeat1-response.json",
        {
            "candidates": [
                {
                    "finish_reason": "STOP",
                    "content": {"parts": [{"text": "```csv\npool_id,ecl\nA,10\n```"}]},
                }
            ]
        },
    )
    result = summarize(tmp_path)[0]
    assert result["measured"] == 1
    assert result["mean_mae_bps"] == 0
    assert result["worst_regret"] is None
    assert not result["rankable"]


def test_archived_review_findings_reproduce(tmp_path):
    archive = (
        Path(__file__).resolve().parents[1] / "pilots" / "cecl-gemini-september-2026"
    )
    shutil.copytree(archive, tmp_path, dirs_exist_ok=True)
    result = summarize(tmp_path)[0]
    assert result["measured"] == 5
    assert result["worse_than_zero_runs"] == 2
    baseline = next(
        p for p in result["repeat_comparisons"] if p["scenario"] == "baseline"
    )
    assert baseline["mae_ratio"] == pytest.approx(27.5997382)
    assert baseline["regret_ratio"] == pytest.approx(959.406944)
    assert result["mean_mae_bps"] == pytest.approx(847.1427513)
