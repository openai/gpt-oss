import json
from threading import Event

from gpt_oss.evals.healthbench_eval import HealthBenchEval, RubricItem
from gpt_oss.evals.types import SamplerResponse


def test_concurrent_grades_stay_aligned_with_rubrics() -> None:
    lower_value_grade_ready = Event()

    class OutOfOrderGrader:
        def __call__(self, messages):
            prompt = messages[0]["content"]
            if "higher-value criterion" in prompt:
                assert lower_value_grade_ready.wait(timeout=5)
                grade = {
                    "criteria_met": True,
                    "explanation": "higher-value result",
                }
            else:
                grade = {
                    "criteria_met": False,
                    "explanation": "lower-value result",
                }
                lower_value_grade_ready.set()
            return SamplerResponse(
                response_text=json.dumps(grade),
                actual_queried_message_list=messages,
                response_metadata={},
            )

    evaluation = HealthBenchEval.__new__(HealthBenchEval)
    evaluation.grader_model = OutOfOrderGrader()
    rubrics = [
        RubricItem("higher-value criterion", 2, ["higher_value"]),
        RubricItem("lower-value criterion", 1, ["lower_value"]),
    ]

    metrics, _, rubric_grades = evaluation.grade_sample(
        prompt=[{"role": "user", "content": "question"}],
        response_text="answer",
        example_tags=[],
        rubric_items=rubrics,
    )

    assert metrics == {
        "overall_score": 2 / 3,
        "higher_value": 1.0,
        "lower_value": 0.0,
    }
    assert [
        (grade["criterion"], grade["criteria_met"], grade["explanation"])
        for grade in rubric_grades
    ] == [
        ("higher-value criterion", True, "higher-value result"),
        ("lower-value criterion", False, "lower-value result"),
    ]
