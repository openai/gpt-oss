"""
Basic eval
"""
from . import report

from .types import Eval, EvalResult, SamplerBase, SingleEvalResult

class BasicEval(Eval):
    def __init__(self,):
        self.examples = [{
            "question": "hi",
            "answer": "hi, how can i help?",
        }]

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        def fn(row: dict):
            sampler_response = sampler([
                sampler._pack_message(content=row["question"], role="user")
            ])
            response_text = sampler_response.response_text
            extracted_answer = response_text
            actual_queried_prompt_messages = sampler_response.actual_queried_message_list
            score = 1.0 if len(extracted_answer) > 0 else 0.0
            html = report.jinja_env.from_string(report.HTML_JINJA).render(
                prompt_messages=actual_queried_prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=row["answer"],
                extracted_answer=extracted_answer,
            )
            convo = actual_queried_prompt_messages + [dict(content=response_text, role="assistant")]
            return SingleEvalResult(
                html=html, score=score, convo=convo, metrics={"chars": len(response_text)}
            )

        results = report.map_with_progress(fn, self.examples, num_threads=1)
        return report.aggregate_results(results)

