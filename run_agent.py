"""Run an agent locally against the environment and save interaction logs to JSONL."""

import json
import asyncio
import os
import traceback
from datetime import datetime

from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from swe_atlas_qna import SWEAtlasQnA, TASKS


async def main():
    openai_api_key = os.environ["OPENAI_API_KEY"]
    or_api_key = os.environ["OPENREWARD_API_KEY"]

    oai_client = AsyncOpenAI(api_key=openai_api_key)

    MODEL_NAME = "gpt-4.1"
    MAX_TURNS = 50

    # Pick first task
    task = TASKS[0]
    task_id = task["task_id"]
    print(f"Task: {task_id} ({task['language']}, {task['category']})")
    print(f"Docker image: {task['docker_image']}")

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    secrets = {"anthropic_api_key": anthropic_api_key, "api_key": or_api_key}

    # Allow image override for testing
    IMAGE_OVERRIDE = os.environ.get("IMAGE_OVERRIDE")
    if IMAGE_OVERRIDE:
        task = dict(task)
        task["docker_image"] = IMAGE_OVERRIDE
        print(f"Using image override: {IMAGE_OVERRIDE}")

    env = SWEAtlasQnA(task_spec=task, secrets=secrets)

    logs = []

    def log_entry(role, content, **extra):
        entry = {"role": role, "content": content, "timestamp": datetime.utcnow().isoformat(), **extra}
        logs.append(entry)
        return entry

    try:
        print("Starting sandbox...")
        await env.setup()
        print("Sandbox started.")

        prompt_blocks = await env.get_prompt()
        prompt_text = prompt_blocks[0].text
        log_entry("system", prompt_text)

        tool_schemas = [
            {"type": "function", "function": {"name": "bash", "description": "Execute a bash command.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The bash command to execute"}, "description": {"type": "string", "description": "Brief description of what the command does"}, "timeout": {"type": "number", "description": "Timeout in seconds (default 30)"}}, "required": ["command"]}}},
            {"type": "function", "function": {"name": "read", "description": "Read file contents with optional offset and limit.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string", "description": "Path to the file"}, "offset": {"type": "integer", "description": "Start reading from this line number"}, "limit": {"type": "integer", "description": "Maximum number of lines to read"}}, "required": ["file_path"]}}},
            {"type": "function", "function": {"name": "grep", "description": "Search for a regex pattern in files recursively.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Regex pattern to search for"}, "path": {"type": "string", "description": "Directory or file to search in"}, "glob": {"type": "string", "description": "Optional filename glob filter (e.g. '*.py', '*.c')"}}, "required": ["pattern"]}}},
            {"type": "function", "function": {"name": "glob", "description": "Find files matching a glob pattern.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "Filename glob pattern (e.g. '*.py', 'Makefile')"}, "path": {"type": "string", "description": "Directory to search in"}}, "required": ["pattern"]}}},
            {"type": "function", "function": {"name": "write", "description": "Write content to a file.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "content": {"type": "string"}}, "required": ["file_path", "content"]}}},
            {"type": "function", "function": {"name": "edit", "description": "Perform exact string replacement in a file.", "parameters": {"type": "object", "properties": {"file_path": {"type": "string"}, "old_string": {"type": "string"}, "new_string": {"type": "string"}, "replace_all": {"type": "boolean"}}, "required": ["file_path", "old_string", "new_string"]}}},
            {"type": "function", "function": {"name": "answer", "description": "Submit your final answer to the code comprehension question. This triggers LLM grading and ends the task.", "parameters": {"type": "object", "properties": {"answer": {"type": "string", "description": "Your complete answer with evidence"}}, "required": ["answer"]}}},
        ]

        messages = [{"role": "user", "content": prompt_text}]
        finished = False
        turn = 0

        while not finished and turn < MAX_TURNS:
            turn += 1
            print(f"\n--- Turn {turn} ---")

            response = await oai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tool_schemas,
                tool_choice="auto",
            )

            choice = response.choices[0]
            assistant_msg = choice.message

            log_entry("assistant", assistant_msg.content or "", tool_calls=[
                {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                for tc in (assistant_msg.tool_calls or [])
            ])

            messages.append(assistant_msg)

            if not assistant_msg.tool_calls:
                # Model responded with text instead of calling answer tool — nudge it
                text = assistant_msg.content or ""
                print(f"Assistant (no tool calls): {text[:200]}")
                if text and turn < MAX_TURNS:
                    messages.append({
                        "role": "user",
                        "content": "Please submit your answer using the `answer` tool now."
                    })
                    continue
                break

            for tc in assistant_msg.tool_calls:
                tool_name = tc.function.name
                tool_args = json.loads(tc.function.arguments)
                print(f"Tool: {tool_name}({json.dumps(tool_args)[:120]})")

                tool_method = getattr(env, tool_name, None)
                if tool_method is None:
                    tool_output_text = f"Unknown tool: {tool_name}"
                    reward = 0.0
                    finished = False
                else:
                    try:
                        from cli_environment import BashParams, GlobParams, GrepParams, ReadParams, WriteParams, EditParams, TodoWriteParams
                        from swe_atlas_qna import AnswerInput

                        param_map = {
                            "bash": BashParams,
                            "glob": GlobParams,
                            "grep": GrepParams,
                            "read": ReadParams,
                            "write": WriteParams,
                            "edit": EditParams,
                            "todo_write": TodoWriteParams,
                            "answer": AnswerInput,
                        }

                        params_cls = param_map.get(tool_name)
                        if params_cls:
                            params = params_cls(**tool_args)
                            result = await tool_method(params)
                        else:
                            result = await tool_method()

                        tool_output_text = result.blocks[0].text if result.blocks else ""
                        reward = result.reward if result.reward is not None else 0.0
                        finished = result.finished if result.finished is not None else False
                    except Exception as e:
                        tool_output_text = f"Error: {str(e)}"
                        reward = 0.0
                        finished = False
                        print(f"  Tool error: {e}")

                log_entry("tool", tool_output_text[:2000], tool_call_id=tc.id, tool_name=tool_name, reward=reward, finished=finished)
                print(f"  -> reward={reward:.3f}, finished={finished}, output={tool_output_text[:150]}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_output_text[:8000],
                })

                if finished:
                    print(f"\n=== FINISHED === reward={reward:.3f}")
                    break

        if not finished:
            print(f"\nForcing answer submission after {MAX_TURNS} turns...")
            # Ask the model to summarize what it found and submit
            messages.append({
                "role": "user",
                "content": "You have used all available turns. Please call the `answer` tool NOW with your best answer based on everything you've found so far."
            })
            response = await oai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tool_schemas,
                tool_choice={"type": "function", "function": {"name": "answer"}},
            )
            choice = response.choices[0]
            assistant_msg = choice.message
            if assistant_msg.tool_calls:
                tc = assistant_msg.tool_calls[0]
                tool_args = json.loads(tc.function.arguments)
                print(f"Forced answer call: {json.dumps(tool_args)[:200]}")
                from swe_atlas_qna import AnswerInput
                params = AnswerInput(**tool_args)
                result = await env.answer(params)
                reward = result.reward if result.reward is not None else 0.0
                finished = result.finished or False
                tool_output_text = result.blocks[0].text if result.blocks else ""
                log_entry("tool", tool_output_text[:2000], tool_call_id=tc.id, tool_name="answer", reward=reward, finished=finished)
                print(f"  -> reward={reward:.3f}, finished={finished}")
            else:
                print("Model did not call answer even when forced")

    finally:
        print("\nTearing down sandbox...")
        await env.teardown()
        print("Sandbox stopped.")

    # Save logs
    out_path = os.path.join(os.path.dirname(__file__), "agent_run.jsonl")
    with open(out_path, "w") as f:
        for entry in logs:
            f.write(json.dumps(entry) + "\n")
    print(f"\nSaved {len(logs)} log entries to {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
