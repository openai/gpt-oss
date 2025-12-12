import argparse
import json
from datetime import datetime

from . import report
from .basic_eval import BasicEval
from .gpqa_eval import GPQAEval
from .aime_eval import AIME25Eval
from .healthbench_eval import HealthBenchEval
from .livecodebench_eval import LiveCodeBenchEval
from .chat_completions_sampler import (
    OPENAI_SYSTEM_MESSAGE_API,
    ChatCompletionsSampler,
)
from .responses_sampler import ResponsesSampler
from .harmony_sampler import HarmonySampler


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-oss-120b,gpt-oss-20b",
        help="Select a model by name. Accepts a comma-separated list.",
    )
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="low,medium,high",
        help="Reasoning effort (low, medium, high). Accepts a comma-separated list.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["responses", "chat_completions", "harmony"],
        default="responses",
        help="Sampler backend to use for models. 'harmony' uses openai_harmony tokenization with SGLang /generate endpoint.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the API.",
    )
    parser.add_argument(
        "--eval",
        type=str,
        default="gpqa,healthbench,healthbench_hard,healthbench_consensus,aime25",
        help="Select an eval by name. Accepts a comma-separated list. Options: basic, gpqa, healthbench, healthbench_hard, healthbench_consensus, aime25, livecodebench",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p (nucleus) sampling parameter",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter (sglang/vLLM specific)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Maximum number of output tokens",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=1584,
        help="Number of threads to run.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Run in debug mode"
    )
    parser.add_argument(
        "--examples", type=int, help="Number of examples to use (overrides default)"
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help="Number of repeats per example (default: 1 in debug mode, 8 otherwise)",
    )
    parser.add_argument(
        "--dump-inputs",
        type=str,
        default=None,
        help="Path to JSONL file to dump input tokens (harmony sampler only)",
    )
    parser.add_argument(
        "--decode-output-tokens",
        action="store_true",
        help="Decode output tokens using our tokenizer instead of using server's decoded text (harmony sampler only)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Request timeout in seconds (default: 1800)",
    )
    parser.add_argument(
        "--lcb-workers",
        type=int,
        default=64,
        help="Number of parallel workers for LiveCodeBench code evaluation (default: 64)",
    )
    parser.add_argument(
        "--lcb-version",
        type=str,
        default="release_v6",
        help="LiveCodeBench version tag (default: release_v6). Options: release_v5, release_v6",
    )

    args = parser.parse_args()

    if args.sampler == "responses":
        sampler_cls = ResponsesSampler
    elif args.sampler == "chat_completions":
        sampler_cls = ChatCompletionsSampler
    else:  # harmony
        sampler_cls = HarmonySampler

    models = {}
    for model_name in args.model.split(","):
        for reasoning_effort in args.reasoning_effort.split(","):
            sampler_kwargs = dict(
                model=model_name,
                reasoning_model=True,
                reasoning_effort=reasoning_effort,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                base_url=args.base_url,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            # Add harmony sampler specific options
            if args.sampler == "harmony":
                if args.dump_inputs:
                    sampler_kwargs["dump_inputs_dir"] = args.dump_inputs
                if args.decode_output_tokens:
                    sampler_kwargs["decode_output_tokens"] = True
            models[f"{model_name}-{reasoning_effort}"] = sampler_cls(**sampler_kwargs)

    print(f"Running with args {args}")

    grading_sampler = ChatCompletionsSampler(
        model="gpt-4.1-2025-04-14",
        system_message=OPENAI_SYSTEM_MESSAGE_API,
        max_tokens=2048,
        base_url="https://api.openai.com/v1",
    )

    def get_evals(eval_name, debug_mode):
        num_examples = (
            args.examples if args.examples is not None else (5 if debug_mode else None)
        )
        # Determine n_repeats: use --n-repeats if provided, else 1 for debug, else 8
        if args.n_repeats is not None:
            n_repeats = args.n_repeats
        else:
            n_repeats = 1 if debug_mode else 8
        # Set num_examples = None to reproduce full evals
        match eval_name:
            case "basic":
                return BasicEval()
            case "gpqa":
                return GPQAEval(
                    n_repeats=n_repeats,
                    num_examples=num_examples,
                    debug=debug_mode,
                    n_threads=args.n_threads or 1,
                )
            case "healthbench":
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=n_repeats,
                    n_threads=args.n_threads or 1,
                    subset_name=None,
                )
            case "healthbench_hard":
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=n_repeats,
                    n_threads=args.n_threads or 1,
                    subset_name="hard",
                )
            case "healthbench_consensus":
                return HealthBenchEval(
                    grader_model=grading_sampler,
                    num_examples=10 if debug_mode else num_examples,
                    n_repeats=n_repeats,
                    n_threads=args.n_threads or 1,
                    subset_name="consensus",
                )
            case "aime25":
                return AIME25Eval(
                    n_repeats=n_repeats,
                    num_examples=num_examples,
                    n_threads=args.n_threads or 1,
                )
            case "livecodebench":
                return LiveCodeBenchEval(
                    n_repeats=n_repeats,
                    num_examples=num_examples,
                    n_threads=args.n_threads or 1,
                    lcb_workers=args.lcb_workers,
                    lcb_version=args.lcb_version,
                )
            case _:
                raise Exception(f"Unrecognized eval type: {eval_name}")

    evals = {}
    for eval_name in args.eval.split(","):
        evals[eval_name] = get_evals(eval_name, args.debug)

    debug_suffix = "_DEBUG" if args.debug else ""
    print(debug_suffix)
    mergekey2resultpath = {}
    print(f"Running the following evals: {evals}")
    print(f"Running evals for the following models: {models}")

    now = datetime.now()
    date_str = now.strftime("%Y%m%d_%H%M%S")
    for model_name, sampler in models.items():
        model_name = model_name.replace("/", "__")
        for eval_name, eval_obj in evals.items():
            result = eval_obj(sampler)
            # ^^^ how to use a sampler
            file_stem = f"{eval_name}_{model_name}_temp{args.temperature}"
            # file stem should also include the year, month, day, and time in hours and minutes
            file_stem += f"_{date_str}"
            report_filename = f"/tmp/{file_stem}{debug_suffix}.html"
            print(f"Writing report to {report_filename}")
            with open(report_filename, "w") as fh:
                fh.write(report.make_report(result))
            assert result.metrics is not None
            metrics = result.metrics | {"score": result.score}
            # Sort metrics by key
            metrics = dict(sorted(metrics.items()))
            print(metrics)
            result_filename = f"/tmp/{file_stem}{debug_suffix}.json"
            with open(result_filename, "w") as f:
                f.write(json.dumps(metrics, indent=2))
            print(f"Writing results to {result_filename}")

            full_result_filename = f"/tmp/{file_stem}{debug_suffix}_allresults.json"
            with open(full_result_filename, "w") as f:
                result_dict = {
                    "score": result.score,
                    "metrics": result.metrics,
                    "htmls": result.htmls,
                    "convos": result.convos,
                    "metadata": result.metadata,
                }
                f.write(json.dumps(result_dict, indent=2))
                print(f"Writing all results to {full_result_filename}")

            mergekey2resultpath[f"{file_stem}"] = result_filename

    merge_metrics = []
    for eval_model_name, result_filename in mergekey2resultpath.items():
        try:
            result = json.load(open(result_filename, "r+"))
        except Exception as e:
            print(e, result_filename)
            continue
        result = result.get("f1_score", result.get("score", None))
        eval_name = eval_model_name[: eval_model_name.find("_")]
        model_name = eval_model_name[eval_model_name.find("_") + 1 :]
        merge_metrics.append(
            {"eval_name": eval_name, "model_name": model_name, "metric": result}
        )
    print(merge_metrics)
    return merge_metrics


if __name__ == "__main__":
    main()
