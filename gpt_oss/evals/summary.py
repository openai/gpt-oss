import json


def collect_merge_metrics(
    result_paths: dict[tuple[str, str], str],
) -> list[dict[str, str | float | None]]:
    """Collect merged eval metrics while preserving structured identifiers."""
    merge_metrics = []
    for (eval_name, model_name), result_filename in result_paths.items():
        try:
            with open(result_filename) as f:
                result = json.load(f)
        except Exception as e:
            print(e, result_filename)
            continue
        metric = result.get("f1_score", result.get("score", None))
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": metric}
        )
    return merge_metrics
