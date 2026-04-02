# SWE-Atlas-QnA

[![OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/SWE-Atlas-QnA) [![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-Dataset-orange)](https://huggingface.co/datasets/ScaleAI/SWE-Atlas-QnA)

## Description

**SWE-Atlas-QnA** is an environment for evaluating agents on deep code comprehension across production-grade software systems. It is based on the [Codebase QnA benchmark](https://labs.scale.com/leaderboard/sweatlas-qna) from Scale AI, part of the SWE-Atlas suite. Agents must investigate real open-source codebases — tracing execution paths, explaining architectural decisions, and answering deeply technical questions — using runtime verification rather than static code reading alone.

The benchmark spans 11 open-source repositories across 4 languages (Go, Python, C, TypeScript), covering systems including mail servers, terminal emulators, distributed object storage, observability platforms, and secret scanners.

## Capabilities

- Deep code comprehension across multiple programming languages
- Runtime verification and execution tracing
- Architectural understanding of production-grade systems
- Evidence-based reasoning with artifact collection (logs, traces, API responses)
- Multi-step codebase exploration and investigation

## Compute Requirements

Agents in SWE-Atlas-QnA are given a sandbox with 2GB of RAM and 2 CPUs. Each task uses a dedicated Docker image with the target repository pre-installed at `/app`.

## Tasks

There is one split: test (124 tasks). Each task presents a question about a specific open-source codebase. The agent must explore the repository, run code, collect evidence, and provide a comprehensive answer. Tasks span 5 categories across 11 repositories and 4 programming languages.

Each task includes a rubric with multiple criteria of two types:
- **Positive verifiers** ("must have" / "should have"): factual claims the answer should contain
- **Negative verifiers** ("must have" / "should have"): incorrect claims the answer should avoid

## Reward Structure

This is a sparse, binary reward environment matching the original SWE-Atlas Task Resolve Rate metric. The agent explores the codebase using CLI tools, then calls the `answer` tool with its response. An LLM grader (Claude Opus 4.5, matching the original evaluation) scores each rubric criterion independently as met or not met.

The reward is binary — all-or-nothing:
- **1.0** if ALL positive verifier criteria are met AND no negative verifier criteria are triggered
- **0.0** otherwise

There is no partial credit. The "must have" and "should have" annotations are informational and do not affect the score.

## Data

Tasks are sourced from the [SWE-Atlas Codebase QnA benchmark](https://huggingface.co/datasets/ScaleAI/SWE-Atlas-QnA) by Scale AI. Each task includes a question, expert reference answer, evaluation rubric, and a Docker image containing the target repository at a pinned commit.

## Tools

Agents are given access to 7 CLI tools mirroring the Claude Code toolset, plus an environment-specific tool:

- `bash`: Execute shell commands
- `read`: Read file contents
- `write`: Write to files
- `edit`: Edit files with exact string replacement
- `grep`: Search for regex patterns in files
- `glob`: Find files matching glob patterns
- `todo_write`: Track investigation progress
- `answer`: Submit the final answer for grading (ends the task)

## Time Horizon

SWE-Atlas-QnA is a multi-turn environment. Agents typically make many tool calls to explore the codebase (reading files, running commands, tracing execution) before submitting a final answer.

## Environment Difficulty

Even frontier models scoring >80% on SWE-Bench achieve less than 30% on SWE-Atlas Codebase QnA, highlighting significant gaps in deep code comprehension capabilities.

## Other Environment Requirements

SWE-Atlas-QnA requires an Anthropic API key (`anthropic_api_key` secret) for LLM-based grading of answers using Claude Opus 4.5, matching the original SWE-Atlas evaluation methodology.

## Safety

Agents in SWE-Atlas-QnA explore open-source codebases within isolated Docker containers. The environment does not present direct safety risks — agents interact only with the sandboxed repository and have no access to external systems beyond the sandbox. Agents are instructed not to permanently modify source code.

## Citations

```bibtex
@dataset{ScaleAISWEAtlasQnA,
  author    = {Scale AI},
  title     = {SWE-Atlas Codebase QnA},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/ScaleAI/SWE-Atlas-QnA}
}
```
