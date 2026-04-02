import json
import re
from typing import List

import anthropic
from datasets import load_dataset
from pydantic import BaseModel

from openreward import AsyncOpenReward, SandboxSettings
from openreward.environments import JSONObject, TextBlock, ToolOutput, tool

from cli_environment import CLIEnvironment
from prompts import GRADER_TEMPLATE


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

DATASET = load_dataset("ScaleAI/SWE-Atlas-QnA", split="test")

TASKS: list[dict] = []          # Public task specs (exposed via list_tasks)
FULL_TASKS: dict[str, dict] = {}  # Private data keyed by task_id (rubric + reference_answer)

for row in DATASET:
    task_id = row["task_id"]

    TASKS.append({
        "task_id": task_id,
        "prompt": row["prompt"],
        "repository_url": row["repository_url"],
        "repository_base_commit": row["repository_base_commit"],
        "language": row["language"],
        "category": row["category"],
        "docker_image": row["docker_image"],
    })

    FULL_TASKS[task_id] = {
        "rubric": row["rubric"],
        "reference_answer": row["reference_answer"],
    }


# ---------------------------------------------------------------------------
# Rubric helpers
# ---------------------------------------------------------------------------

class RubricItem:
    def __init__(self, id: str, title: str, verifier_type: str, importance: str):
        self.id = id
        self.title = title
        self.verifier_type = verifier_type  # "positive hli verifier" or "negative hli verifier"
        self.importance = importance          # "must have" or "should have"

    def __str__(self):
        return self.title

    @classmethod
    def from_dict(cls, d: dict) -> "RubricItem":
        annotations = d.get("annotations", {})
        return cls(
            id=d["id"],
            title=d["title"],
            verifier_type=annotations.get("type", "positive hli verifier"),
            importance=annotations.get("importance", "should have"),
        )


def parse_json_to_dict(json_string: str) -> dict:
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())
    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON decoding failed: {e}")
        return {}


def calculate_score(rubric_items: list[RubricItem], grading_responses: list[dict]) -> float:
    """Binary scoring matching SWE-Atlas Task Resolve Rate.

    Returns 1.0 if ALL rubric items are correctly resolved:
    - Positive verifiers: criteria_met must be True
    - Negative verifiers: criteria_met must be False
    Otherwise returns 0.0.
    """
    for item, grade in zip(rubric_items, grading_responses, strict=True):
        met = grade["criteria_met"]
        if item.verifier_type == "positive hli verifier" and not met:
            return 0.0
        if item.verifier_type == "negative hli verifier" and met:
            return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class TaskSpec(BaseModel):
    task_id: str
    prompt: str
    repository_url: str
    repository_base_commit: str
    language: str
    category: str
    docker_image: str


class AnswerInput(BaseModel, extra="forbid"):
    answer: str


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class SWEAtlasQnA(CLIEnvironment):

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec, secrets=secrets)

        self.validated = TaskSpec.model_validate(task_spec)

        # Load private grading data
        full_task = FULL_TASKS[self.validated.task_id]
        self.rubric_items = [
            RubricItem.from_dict(r) for r in json.loads(full_task["rubric"])
        ]
        self.reference_answer = full_task["reference_answer"]

        # Anthropic client for grading (Claude Opus 4.5, matching original SWE-Atlas evaluation)
        anthropic_api_key = secrets.get("anthropic_api_key")
        if not anthropic_api_key:
            raise ValueError("Anthropic API key must be provided via secrets['anthropic_api_key']")
        self.client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)

        # Sandbox settings – each task has its own docker image
        self.sandbox_settings = SandboxSettings(
            environment="GeneralReasoning/SWE-Atlas-QnA",
            image=self.validated.docker_image,
            machine_size="2:2",
        )
        or_client = AsyncOpenReward(api_key=secrets.get("api_key"))
        self.sandbox = or_client.sandbox(self.sandbox_settings)

    async def setup(self) -> None:
        await self.sandbox.start()
        # Pin the repository to the exact commit
        commit = self.validated.repository_base_commit
        await self.sandbox.run("git config --global --add safe.directory /app")
        await self.sandbox.run(f"cd /app && git restore . && git reset --hard {commit} && git clean -fdq")

    async def teardown(self) -> None:
        await self.sandbox.stop()

    async def get_prompt(self) -> List[TextBlock]:
        prompt = f"""You are a code comprehension agent. Your task is to investigate a codebase and answer a question about it.

## Question
{self.validated.prompt}

## Environment
- The repository is available at `/app`
- Language: {self.validated.language}
- Repository: {self.validated.repository_url}

## Available Tools
You have access to CLI tools for exploring the codebase:
- `bash`: Execute shell commands
- `read`: Read file contents
- `write`: Write to files
- `edit`: Edit files with exact string replacement
- `grep`: Search for regex patterns in files
- `glob`: Find files matching glob patterns
- `todo_write`: Track your investigation progress

## Instructions
1. Explore the codebase thoroughly to answer the question
2. Use runtime verification where possible (run code, check logs, trace execution)
3. Provide evidence-based reasoning in your answer
4. When ready, call the `answer` tool with your complete response

Important: Focus on runtime evidence over static code reading. Run the code, check logs, trace execution paths, and collect artifacts to support your answer."""
        return [TextBlock(text=prompt)]

    # -------------------------------------------------------------------
    # Grading
    # -------------------------------------------------------------------

    async def _grade_sample(self, response_text: str) -> dict:
        async def grade_rubric_item(item: RubricItem) -> dict:
            grader_prompt = (
                GRADER_TEMPLATE
                .replace("<<task_prompt>>", self.validated.prompt)
                .replace("<<reference_answer>>", self.reference_answer)
                .replace("<<agent_response>>", response_text)
                .replace("<<rubric_criterion>>", item.title)
                .replace("<<criterion_type>>", item.verifier_type)
            )

            while True:
                res = await self.client.messages.create(
                    model="claude-opus-4-5-20251101",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": grader_prompt}],
                )
                content = res.content[0].text if res.content else ""
                parsed = parse_json_to_dict(content)
                if "criteria_met" in parsed and isinstance(parsed["criteria_met"], bool):
                    break
                print("Grading failed due to bad JSON output, retrying...")

            return parsed

        grading_responses = []
        for item in self.rubric_items:
            grade = await grade_rubric_item(item)
            grading_responses.append(grade)

        score = calculate_score(self.rubric_items, grading_responses)

        return {
            "overall_score": score,
            "resolved": score == 1.0,
            "grading_details": [
                {
                    "rubric_id": item.id,
                    "title": item.title,
                    "type": item.verifier_type,
                    "importance": item.importance,
                    "criteria_met": grade["criteria_met"],
                    "explanation": grade.get("explanation", ""),
                }
                for item, grade in zip(self.rubric_items, grading_responses)
            ],
        }

    @tool
    async def answer(self, params: AnswerInput) -> ToolOutput:
        """Submit your answer to the code comprehension question. This triggers grading and ends the task."""
        grader_output = await self._grade_sample(params.answer)
        reward = grader_output["overall_score"]

        return ToolOutput(
            metadata=grader_output,
            blocks=[TextBlock(text=f"Overall Score: {reward:.2%}")],
            reward=reward,
            finished=True,
        )

    # -------------------------------------------------------------------
    # Task listing
    # -------------------------------------------------------------------

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split == "test":
            return TASKS  # type: ignore
        raise ValueError(f"Unknown split: {split}")

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["test"]
