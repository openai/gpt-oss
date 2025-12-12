"""
LiveCodeBench: https://huggingface.co/datasets/livecodebench/code_generation_lite

Two-phase evaluation:
1. Phase 1: Collect all model responses and extract code
2. Phase 2: Batch evaluate code execution in parallel using ProcessPoolExecutor
"""
import argparse
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from contextlib import redirect_stdout, redirect_stderr
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset
from tqdm import tqdm

from . import report
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult


# HuggingFace dataset configuration
LCB_HF_DATASET = "livecodebench/code_generation_lite"
LCB_DEFAULT_VERSION = "release_v6"

LIVECODEBENCH_INSTRUCTIONS = """
You are a python coding expert that solves problems step-by-step.
You must provide the reasoning to arriving at your solution and the code to solve the problem.
Do not try simulating the code execution. The code must be enclosed within ```python delimiters.
"""


def parse_code(text: str) -> Optional[str]:
    """Parse code from ```python or plain ``` code block.

    Priority:
    1. Last ```python block
    2. Last plain ``` block
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()
    if not text:
        return None

    # Try ```python blocks first (most specific)
    python_matches = list(re.finditer(r"```python(.*?)```", text, re.DOTALL))
    if python_matches:
        return python_matches[-1].group(1).strip()

    # Fall back to plain ``` blocks
    plain_matches = list(re.finditer(r"```(.*?)```", text, re.DOTALL))
    if plain_matches:
        # Get the last match
        code = plain_matches[-1].group(1).strip()
        # Remove language tag if present (e.g., ```python\n or ```py\n)
        code = re.sub(r'^(?:python|py)\s*\n', '', code, flags=re.IGNORECASE)
        return code

    return None


def get_lcb_dir() -> str:
    """Get the LiveCodeBench submodule directory path."""
    return os.path.abspath(os.path.join(
        os.path.dirname(__file__), "submodules", "LiveCodeBench"))


@lru_cache(maxsize=4)
def load_lcb_from_huggingface(version_tag: str = LCB_DEFAULT_VERSION) -> List[Dict[str, Any]]:
    """Load LiveCodeBench questions from HuggingFace.

    Args:
        version_tag: Version tag for the dataset (e.g., "release_v5", "release_v6")

    Returns:
        List of examples with question_id, question_content (prompt), and starter_code.
    """
    print(f"Loading LiveCodeBench from HuggingFace: {LCB_HF_DATASET} ({version_tag})...")
    ds = load_dataset(LCB_HF_DATASET, version_tag=version_tag, split="test")

    examples = []
    for row in ds:
        examples.append({
            "question_id": row["question_id"],
            "prompt": row["question_content"],  # The problem description
            "starter_code": row.get("starter_code", ""),  # Starter code if available
        })

    print(f"Loaded {len(examples)} problems from HuggingFace")
    return examples


def format_prompt_with_starter_code(prompt: str, starter_code: str = "") -> str:
    """Append the format section with starter code to the prompt.

    This matches the format used in the working harmonize_inputs.py pipeline.
    """
    format_section = "\n### Format: You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n```python\n"
    if starter_code:
        format_section += starter_code + "\n"
    format_section += "```\n"
    return prompt + format_section


@lru_cache(maxsize=4)
def load_lcb_benchmark_for_eval(version_tag: str = LCB_DEFAULT_VERSION) -> Dict[str, Any]:
    """Load LiveCodeBench benchmark from submodule for test execution.

    Args:
        version_tag: Version tag for the dataset (e.g., "release_v5", "release_v6")

    This is needed because test execution requires the LCB library's
    instance objects which contain test cases.
    """
    lcb_dir = get_lcb_dir()

    if not os.path.isdir(lcb_dir):
        raise FileNotFoundError(
            f"LiveCodeBench submodule required at: {lcb_dir}")

    original_cwd = os.getcwd()
    os.chdir(lcb_dir)

    if lcb_dir not in sys.path:
        sys.path.insert(0, lcb_dir)

    try:
        os.environ['TQDM_DISABLE'] = '1'

        from lcb_runner.utils.scenarios import Scenario
        from lcb_runner.runner.scenario_router import build_prompt_benchmark

        mock_args = argparse.Namespace(
            scenario=Scenario.codegeneration, release_version=version_tag,
            subset="code_generation", language="python", not_fast=False,
            start_date=None, end_date=None, k=[1], num_samples=1,
            timeout=60, num_workers=1, num_process_evaluate=1,
            model_name="standalone_eval", output_dir="/tmp",
            prompt_type="custom", continue_existing=False, evaluate=True
        )

        full_benchmark, _ = build_prompt_benchmark(mock_args)
        return {inst.question_id: inst for inst in full_benchmark}

    finally:
        os.chdir(original_cwd)
        os.environ.pop('TQDM_DISABLE', None)


def evaluate_livecodebench_detailed(
        code: Optional[str], question_id: str,
        version_tag: str = LCB_DEFAULT_VERSION) -> Tuple[bool, str]:
    """Evaluate LiveCodeBench code generation with detailed results.

    Args:
        code: The code to evaluate
        question_id: The question ID to look up test cases
        version_tag: Version tag for the dataset (e.g., "release_v5", "release_v6")

    Returns:
        Tuple[bool, str]: (passed, detailed_reason)
    """
    if not code or not question_id:
        return False, "No code or question_id provided"

    lcb_dir = get_lcb_dir()

    try:
        benchmark_map = load_lcb_benchmark_for_eval(version_tag)
    except Exception as e:
        return False, f"Failed to load benchmark: {type(e).__name__}: {e}"

    instance = benchmark_map.get(question_id)
    if not instance:
        return False, f"Question ID '{question_id}' not found in benchmark"

    original_cwd = os.getcwd()
    temp_dir = f"/tmp/temp_lcb_eval_{question_id}_{int(time.time())}"
    os.makedirs(temp_dir, exist_ok=True)

    try:
        os.chdir(lcb_dir)
        os.environ['TQDM_DISABLE'] = '1'

        from lcb_runner.utils.scenarios import Scenario
        from lcb_runner.evaluation import extract_instance_results
        from lcb_runner.runner.scenario_router import sort_and_extract_save_results, get_metrics

        mock_args = argparse.Namespace(
            scenario=Scenario.codegeneration, release_version=version_tag,
            subset="code_generation", language="python", not_fast=False,
            start_date=None, end_date=None, k=[1], num_samples=1,
            timeout=60, num_workers=1, num_process_evaluate=1,
            model_name="standalone_eval", output_dir=temp_dir,
            prompt_type="custom", continue_existing=False, evaluate=True,
        )

        batch_benchmark = [instance]
        batch_custom_outputs = [[code]]

        save_results = [inst.insert_output(output, output)
                        for inst, output in zip(batch_benchmark, batch_custom_outputs)]

        _, combined_results = sort_and_extract_save_results(
            mock_args.scenario, save_results)
        _, instance_results, _ = get_metrics(
            mock_args.scenario, mock_args, batch_benchmark, combined_results
        )

        graded = extract_instance_results(instance_results)
        passed = graded and graded[0] and graded[0][0]

        # Try to extract detailed results
        detailed_reason = ""
        try:
            if combined_results and len(combined_results) > 0:
                result_info = combined_results[0]
                if hasattr(result_info, 'result') and result_info.result:
                    test_results = result_info.result
                    if isinstance(test_results, dict):
                        detailed_reason = f"Test results: {test_results}"
                    elif isinstance(test_results, list):
                        num_passed = sum(1 for r in test_results if r)
                        num_total = len(test_results)
                        detailed_reason = f"Passed {num_passed}/{num_total} test cases"
                    else:
                        detailed_reason = f"Result: {test_results}"
                elif hasattr(result_info, 'status'):
                    detailed_reason = f"Status: {result_info.status}"
        except Exception:
            pass

        if not detailed_reason:
            if passed:
                detailed_reason = "All tests passed"
            else:
                detailed_reason = "Failed one or more test cases"

        return passed, detailed_reason

    except Exception as e:
        return False, f"Evaluation error: {type(e).__name__}: {str(e)[:200]}"
    finally:
        os.chdir(original_cwd)
        shutil.rmtree(temp_dir, ignore_errors=True)
        os.environ.pop('TQDM_DISABLE', None)


def evaluate_livecodebench_worker(args: Tuple[int, str, str, str]) -> Tuple[int, bool, str]:
    """Worker function for parallel LiveCodeBench evaluation.

    Args:
        args: (index, code, question_id, version_tag)

    Returns:
        Tuple[int, bool, str]: (index, passed, detailed_reason)
    """
    idx, code, question_id, version_tag = args

    # Suppress all stdout/stderr from worker processes to prevent pollution
    try:
        with open(os.devnull, 'w') as devnull:
            with redirect_stdout(devnull), redirect_stderr(devnull):
                os.environ['TQDM_DISABLE'] = '1'
                passed, reason = evaluate_livecodebench_detailed(code, question_id, version_tag)
                return idx, passed, reason
    except Exception as e:
        return idx, False, f"Error: {type(e).__name__}: {e}"


class LiveCodeBenchEval(Eval):
    """
    LiveCodeBench evaluation with two-phase approach:
    1. Collect all model responses
    2. Batch evaluate code execution in parallel
    """

    def __init__(
        self,
        n_repeats: int = 1,
        num_examples: int | None = None,
        n_threads: int = 1,
        lcb_workers: int = 64,
        test_timeout: int = 60,
        lcb_version: str = LCB_DEFAULT_VERSION,
    ):
        """
        Initialize LiveCodeBench evaluation.

        Args:
            n_repeats: Number of times to repeat each example
            num_examples: Limit number of examples (for debugging)
            n_threads: Number of threads for collecting model responses
            lcb_workers: Number of parallel workers for code evaluation
            test_timeout: Timeout for each test execution in seconds
            lcb_version: LiveCodeBench version tag (e.g., "release_v5", "release_v6")
        """
        self.n_repeats = n_repeats
        self.n_threads = n_threads
        self.lcb_workers = lcb_workers
        self.test_timeout = test_timeout
        self.lcb_version = lcb_version

        # Load questions from HuggingFace
        examples = load_lcb_from_huggingface(lcb_version)

        # Limit examples if specified
        if num_examples:
            examples = examples[:num_examples]

        # Repeat examples
        examples = examples * n_repeats

        self.examples = examples
        print(f"Total examples to evaluate: {len(self.examples)}")

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        """
        Two-phase evaluation:
        1. Collect all model responses (using n_threads for parallelism)
        2. Batch evaluate all code in parallel using ProcessPoolExecutor
        """
        # Phase 1: Collect all model responses
        collected_results: List[Dict[str, Any]] = []

        def collect_response(row: dict) -> Dict[str, Any]:
            """Collect a single model response."""
            question_id = row["question_id"]

            # Construct prompt with starter code format section
            user_prompt = format_prompt_with_starter_code(
                row["prompt"],
                row.get("starter_code", "")
            )
            # Combine instructions and user prompt into a single user message
            full_prompt = f"{LIVECODEBENCH_INSTRUCTIONS}\n\n{user_prompt}"
            prompt_messages = [
                sampler._pack_message(
                    content=full_prompt,
                    role="user"
                ),
            ]
            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list

            # Extract code from response
            extracted_code = parse_code(response_text)

            return {
                "question_id": question_id,
                "prompt": row["prompt"],
                "response_text": response_text,
                "extracted_code": extracted_code,
                "actual_queried_prompt_messages": actual_queried_prompt_messages,
            }

        # Collect responses (can be parallelized with n_threads)
        collected_results = report.map_with_progress(
            collect_response, self.examples, num_threads=self.n_threads
        )

        # Phase 2: Batch evaluate all code in parallel
        print(f"\nEvaluating {len(collected_results)} code samples with {self.lcb_workers} workers...")

        # Pre-load benchmark in main process before forking (for test execution)
        try:
            _ = load_lcb_benchmark_for_eval(self.lcb_version)
        except Exception as e:
            print(f"Warning: Failed to pre-load benchmark for evaluation: {e}")

        # Prepare work items
        work_items = []
        for idx, result in enumerate(collected_results):
            if result["extracted_code"]:
                work_items.append((idx, result["extracted_code"], result["question_id"], self.lcb_version))

        print(f"Extracted code from {len(work_items)} / {len(collected_results)} responses")
        # Initialize scores to 0
        scores = [0.0] * len(collected_results)
        eval_details = ["No code extracted"] * len(collected_results)

        if work_items:
            max_workers = min(self.lcb_workers, len(work_items))
            print(f"Submitting {len(work_items)} code samples for evaluation...")

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_idx = {
                    executor.submit(evaluate_livecodebench_worker, item): item[0]
                    for item in work_items
                }

                future_timeout = self.test_timeout * 1.2
                for future in tqdm(as_completed(future_to_idx, timeout=future_timeout * len(work_items)),
                                   total=len(future_to_idx),
                                   desc="Evaluating code"):
                    idx = future_to_idx[future]
                    try:
                        result_idx, passed, reason = future.result(timeout=future_timeout)
                        scores[result_idx] = 1.0 if passed else 0.0
                        eval_details[result_idx] = reason
                    except TimeoutError:
                        scores[idx] = 0.0
                        eval_details[idx] = "Timeout: Test execution exceeded time limit"
                    except Exception as e:
                        scores[idx] = 0.0
                        eval_details[idx] = f"Error: {type(e).__name__}: {e}"

        # Generate results
        print("\nGenerating results...")
        single_results = []

        for idx, result in enumerate(collected_results):
            score = scores[idx]
            detail = eval_details[idx]

            # Generate HTML report
            html = report.jinja_env.from_string(report.HTML_JINJA).render(
                prompt_messages=result["actual_queried_prompt_messages"],
                next_message=dict(content=result["response_text"], role="assistant"),
                score=score,
                correct_answer=f"question_id: {result['question_id']}",
                extracted_answer=f"Code extracted: {'Yes' if result['extracted_code'] else 'No'}, {detail}",
            )

            convo = result["actual_queried_prompt_messages"] + [
                dict(content=result["response_text"], role="assistant")
            ]

            single_results.append(SingleEvalResult(
                html=html,
                score=score,
                convo=convo,
                metrics={
                    "chars": len(result["response_text"]),
                    "code_extracted": 1.0 if result["extracted_code"] else 0.0,
                }
            ))

        # Calculate summary stats
        total = len(single_results)
        passed = sum(1 for r in single_results if r.score > 0)
        code_extracted = sum(1 for r in collected_results if r["extracted_code"])

        print(f"\nLiveCodeBench Results:")
        print(f"  Total samples: {total}")
        print(f"  Code extracted: {code_extracted}/{total} ({100*code_extracted/total:.1f}%)")
        print(f"  Tests passed: {passed}/{total} ({100*passed/total:.1f}%)")

        return report.aggregate_results(single_results)

