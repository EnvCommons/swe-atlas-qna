import json
import os

import pytest

from swe_atlas_qna import (
    SWEAtlasQnA,
    RubricItem,
    AnswerInput,
    calculate_score,
    TASKS,
    FULL_TASKS,
)


# ---------------------------------------------------------------------------
# Unit tests for calculate_score (binary: 1.0 if all pass, 0.0 otherwise)
# ---------------------------------------------------------------------------

def test_calculate_score_all_positive_met():
    items = [
        RubricItem("1", "criterion A", "positive hli verifier", "must have"),
        RubricItem("2", "criterion B", "positive hli verifier", "should have"),
    ]
    grades = [{"criteria_met": True}, {"criteria_met": True}]
    assert calculate_score(items, grades) == pytest.approx(1.0)


def test_calculate_score_one_positive_not_met():
    items = [
        RubricItem("1", "criterion A", "positive hli verifier", "must have"),
        RubricItem("2", "criterion B", "positive hli verifier", "should have"),
    ]
    grades = [{"criteria_met": True}, {"criteria_met": False}]
    assert calculate_score(items, grades) == pytest.approx(0.0)


def test_calculate_score_none_met():
    items = [
        RubricItem("1", "criterion A", "positive hli verifier", "must have"),
        RubricItem("2", "criterion B", "positive hli verifier", "should have"),
    ]
    grades = [{"criteria_met": False}, {"criteria_met": False}]
    assert calculate_score(items, grades) == pytest.approx(0.0)


def test_calculate_score_negative_not_met_is_good():
    """Negative verifier not met (good) + positive met = resolved."""
    items = [
        RubricItem("1", "good thing", "positive hli verifier", "must have"),
        RubricItem("2", "bad thing", "negative hli verifier", "must have"),
    ]
    # Positive met, negative not met → all pass → 1.0
    grades = [{"criteria_met": True}, {"criteria_met": False}]
    assert calculate_score(items, grades) == pytest.approx(1.0)


def test_calculate_score_negative_met_is_bad():
    """Negative verifier met (bad) → not resolved even if positive is met."""
    items = [
        RubricItem("1", "good thing", "positive hli verifier", "must have"),
        RubricItem("2", "bad thing", "negative hli verifier", "must have"),
    ]
    # Positive met, negative also met (bad) → fail → 0.0
    grades = [{"criteria_met": True}, {"criteria_met": True}]
    assert calculate_score(items, grades) == pytest.approx(0.0)


def test_calculate_score_only_negative_all_avoided():
    """Only negative verifiers, none met → resolved."""
    items = [
        RubricItem("1", "bad thing A", "negative hli verifier", "must have"),
        RubricItem("2", "bad thing B", "negative hli verifier", "should have"),
    ]
    grades = [{"criteria_met": False}, {"criteria_met": False}]
    assert calculate_score(items, grades) == pytest.approx(1.0)


def test_calculate_score_only_negative_one_triggered():
    """Only negative verifiers, one triggered → not resolved."""
    items = [
        RubricItem("1", "bad thing A", "negative hli verifier", "must have"),
        RubricItem("2", "bad thing B", "negative hli verifier", "should have"),
    ]
    grades = [{"criteria_met": False}, {"criteria_met": True}]
    assert calculate_score(items, grades) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Data integrity tests
# ---------------------------------------------------------------------------

def test_task_count():
    assert len(TASKS) == 124


def test_all_tasks_have_rubrics():
    for task in TASKS:
        assert task["task_id"] in FULL_TASKS
        rubric_raw = FULL_TASKS[task["task_id"]]["rubric"]
        rubric = json.loads(rubric_raw)
        assert len(rubric) > 0, f"Task {task['task_id']} has empty rubric"


def test_list_splits():
    assert SWEAtlasQnA.list_splits() == ["test"]


def test_list_tasks_count():
    tasks = SWEAtlasQnA.list_tasks("test")
    assert len(tasks) == 124


def test_list_tasks_invalid_split():
    with pytest.raises(ValueError):
        SWEAtlasQnA.list_tasks("train")


# ---------------------------------------------------------------------------
# Grading integration test (requires ANTHROPIC_API_KEY)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_grading_good_vs_bad():
    """Reference answer should score higher than a nonsense answer."""
    task = TASKS[0]
    task_id = task["task_id"]
    full_task = FULL_TASKS[task_id]

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not set")

    env = SWEAtlasQnA(
        task_spec=task,
        secrets={"anthropic_api_key": api_key, "api_key": "dummy"},
    )

    # Grade the reference answer (should resolve = 1.0)
    good_result = await env._grade_sample(full_task["reference_answer"])
    good_score = good_result["overall_score"]

    # Grade a bad answer (should not resolve = 0.0)
    bad_result = await env._grade_sample("I don't know. The code does something with files.")
    bad_score = bad_result["overall_score"]

    assert good_score > bad_score, (
        f"Reference answer ({good_score:.2%}) should score higher than bad answer ({bad_score:.2%})"
    )
