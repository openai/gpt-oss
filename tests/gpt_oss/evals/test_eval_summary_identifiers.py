import json

from gpt_oss.evals.__main__ import _collect_merge_metrics


def test_collect_merge_metrics_preserves_eval_names_with_underscores(tmp_path) -> None:
    hard_result = tmp_path / "hard.json"
    hard_result.write_text(json.dumps({"score": 0.75}))
    consensus_result = tmp_path / "consensus.json"
    consensus_result.write_text(json.dumps({"score": 0.5}))

    result_paths = {
        ("healthbench_hard", "gpt-oss-120b-low"): str(hard_result),
        ("healthbench_consensus", "gpt-oss-20b-high"): str(consensus_result),
    }

    assert _collect_merge_metrics(result_paths) == [
        {
            "eval_name": "healthbench_hard",
            "model_name": "gpt-oss-120b-low",
            "metric": 0.75,
        },
        {
            "eval_name": "healthbench_consensus",
            "model_name": "gpt-oss-20b-high",
            "metric": 0.5,
        },
    ]


def test_collect_merge_metrics_keeps_f1_score_precedence(tmp_path) -> None:
    result_file = tmp_path / "result.json"
    result_file.write_text(json.dumps({"f1_score": 0.8, "score": 0.2}))

    assert _collect_merge_metrics({("gpqa", "model"): str(result_file)}) == [
        {"eval_name": "gpqa", "model_name": "model", "metric": 0.8}
    ]
