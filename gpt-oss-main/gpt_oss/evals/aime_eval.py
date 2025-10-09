"""
AIME 2025: https://huggingface.co/datasets/opencompass/AIME2025
"""
import random
import re
import pandas
from . import report

from .types import Eval, EvalResult, SamplerBase, SingleEvalResult


AIME_TEMPLATE = """
{question}
Please reason step by step, and put your final answer within \\boxed{{}}.
"""

def format_aime_question(row):
    return AIME_TEMPLATE.format(question=row["question"])

def extract_boxed_text(text):
    pattern = r'boxed{(.*?)}|framebox{(.*?)}'
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        for match in matches[::-1]:
            for group in match:
                if group != "":
                    return group.split(',')[-1].strip()
    pattern = r'\d+'  # get the last integer if no pattern found
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return matches[-1]
    return ""

def normalize_number(s):
    match = re.match(r"\d+", s)  # match digits from the start
    if not match:
        return None
    return match.group(0)

class AIME25Eval(Eval):
    def __init__(
        self,
        n_repeats: int = 4,
        num_examples: int | None = None,  # restrict to a subset of the data for debugging
        n_threads: int = 1,
    ):
        path1 = f"https://huggingface.co/datasets/opencompass/AIME2025/raw/main/aime2025-I.jsonl"
        df1 = pandas.read_json(path1, lines=True)
        path2 = f"https://huggingface.co/datasets/opencompass/AIME2025/raw/main/aime2025-II.jsonl"
        df2 = pandas.read_json(path2, lines=True)
        examples = [row.to_dict() for _, row in df1.iterrows()] + [row.to_dict() for _, row in df2.iterrows()]
        examples = [{
            "question": row["question"],
            "answer": normalize_number(row["answer"]) if isinstance(row["answer"], str) else row["answer"],
        } for row in examples]
        rng = random.Random(0)
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
            prompt_messages = [
                sampler._pack_message(
                    content=format_aime_question(row), role="user"
                )
            ]
            sampler_response = sampler(prompt_messages)
            response_text = sampler_response.response_text
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list
            extracted_answer = extract_boxed_text(response_text)
            correct_answer = int(row["answer"])
            try: # All AIME answers are integers, so we convert the extracted answer to an integer
                extracted_answer = int(extracted_answer)
            except (ValueError, TypeError):
                extracted_answer = None
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

