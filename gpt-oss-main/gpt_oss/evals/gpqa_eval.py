"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark
David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022
"""

import random

import pandas

from . import report
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult
from .abcd_grader import extract_abcd


QUERY_TEMPLATE_MULTICHOICE = """
{Question}

(A) {A}
(B) {B}
(C) {C}
(D) {D}

Express your final answer as the corresponding option 'A', 'B', 'C', or 'D'.
""".strip()


def format_multichoice_question(row):
    return QUERY_TEMPLATE_MULTICHOICE.format(**row)


class GPQAEval(Eval):
    def __init__(
        self,
        n_repeats: int = 8,
        variant: str = "diamond",
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
        debug: bool = False,
        n_threads: int = 1,
    ):
        df = pandas.read_csv(
            f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
        )
        rng = random.Random(0)

        if debug:
            examples = [row.to_dict() for _, row in df.iterrows() if "ESPRESSO spectrograph, please" in row["Question"]]
        else:
            examples = [row.to_dict() for _, row in df.iterrows()]
            if num_examples:
                assert n_repeats == 1, "n_repeats only supported for num_examples = None"
                examples = rng.sample(examples, num_examples)

        examples = examples * n_repeats
        examples = [example | {"permutation": rng.sample(range(4), 4)} for example in examples]
        self.examples = examples
        self.n_repeats = n_repeats
        self.n_threads = n_threads

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            choices = [
                row["Correct Answer"],
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            choices = [choices[i] for i in row["permutation"]]
            correct_index = choices.index(row["Correct Answer"])
            correct_answer = "ABCD"[correct_index]
            choices_dict = dict(
                A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
            )
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(choices_dict), role="user"
                )
            ]
            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list
            extracted_answer = extract_abcd(response_text)
            score = 1.0 if extracted_answer == correct_answer else 0.0
            html = report.jinja_env.from_string(report.HTML_JINJA).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )
            convo = actual_queried_prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html, score=score, convo=convo, metrics={"chars": len(response_text)}
            )

        results = report.map_with_progress(fn, self.examples, num_threads=self.n_threads)
        return report.aggregate_results(results)


if __name__ == "__main__":
    import json
    import sys

    with open(sys.argv[1], "r") as f:
        results = json.load(f)

    passes = 0
    for convo, html in zip(results["convos"], results["htmls"]):
        message = convo[-1]["content"]
        import re

        # the ground truth is in <p>Correct Answer: A</p> in the html
        ground_truth = re.search(r"<p>Correct Answer: (A|B|C|D)</p>", html)
        ground_truth = ground_truth.group(1)
        extracted_answer = extract_abcd(message)
        if extracted_answer == ground_truth:
            passes += 1
        elif len(message) > 15:
            print("no match:", message)
            print("ground truth:", ground_truth)
            print("extracted answer:", extracted_answer)
            print("--------------------------------")

    pass_rate = passes / len(results["convos"])
    print(f"pass@1: {pass_rate}")