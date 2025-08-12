"""
Humanity's Last Exam: version 8ad33fcec9b6d58f437798bd6a3414ed78208fa3
"""
import random
from . import report

from .eq_grader import EqualityGrader
from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

from datasets import load_dataset

INSTRUCTION_PREAMBLE = (
    "Answer the following exam question. Give just your answer. If it is multiple choice, "
    "give only and exactly the letter of the correct answer. If it is not multiple choice, "
    "your answer entire answer must exact-string-match what is in the answer key to be marked correct."
)

INSTRUCTION_POSTAMBLE = (
    "Your response should be in the following format:\n"
    "Explanation: {your explanation for your answer choice}\n"
    "Answer: {your chosen answer}\n"
    "Confidence: {your confidence score between 0% and 100% for your answer}"
)

def format_hle_question(row):
    return f"{INSTRUCTION_PREAMBLE}\n\n{row['question']}\n\n{INSTRUCTION_POSTAMBLE}"


class HLEEval(Eval):
    def __init__(
        self,
        grader_model: SamplerBase,
        n_repeats: int | None = None,  # default to 1
        n_examples: int | None = None,  # restrict to a subset of the data for debugging
        n_threads: int = 1,
    ):
        n_repeats = n_repeats or 1

        dataset = load_dataset(
            "cais/hle",
            split="test",
            revision="8ad33fcec9b6d58f437798bd6a3414ed78208fa3"
        )

        examples = [{
            "id": row["id"],
            "question": row["question"],
            "answer": row["answer"],
            "image": row["image"],
        } for row in dataset]
        rng = random.Random(0)
        if n_examples:
            examples = rng.sample(examples, n_examples)
        examples = examples * n_repeats
        self.examples = examples
        self.n_repeats = n_repeats
        self.n_threads = n_threads
        self.grader = EqualityGrader(grader_model)

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=format_hle_question(row), role="user"
                )
            ]
            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            actual_queried_prompt_messages = sampler_response.messages
            score, metadata = self.grader.grade_sample(row["question"], response_text, row["answer"])
            html = report.jinja_env.from_string(report.HTML_JINJA).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
                extracted_answer=response_text,
            )
            convo = actual_queried_prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                response=sampler_response,
                score=score,
                html=html,
                convo=convo,
                example_level_metadata=metadata,
            )

        results = report.map_with_progress(fn, self.examples, num_threads=self.n_threads)
        return report.aggregate_results(results)

