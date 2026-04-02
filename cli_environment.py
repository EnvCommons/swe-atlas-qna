import os
from typing import Dict, List, Any, Optional
from pydantic import BaseModel

from openreward.environments import (Environment, JSONObject, TextBlock,
                                     ToolOutput, tool)

from utils import download_text, upload_text


# -- Pydantic models matching Claude Code tool signatures --

class BashParams(BaseModel, extra="forbid"):
    command: str
    description: str = ""
    timeout: Optional[float] = 30.0


class GlobParams(BaseModel, extra="forbid"):
    pattern: str
    path: Optional[str] = None


class GrepParams(BaseModel, extra="forbid"):
    pattern: str
    path: Optional[str] = None
    glob: Optional[str] = None


class ReadParams(BaseModel, extra="forbid"):
    file_path: str
    offset: Optional[int] = None
    limit: Optional[int] = None


class WriteParams(BaseModel, extra="forbid"):
    file_path: str
    content: str


class EditParams(BaseModel, extra="forbid"):
    file_path: str
    old_string: str
    new_string: str
    replace_all: bool = False


class TodoWriteParams(BaseModel, extra="forbid"):
    todos: List[Dict[str, Any]] = []


class CLIEnvironment(Environment):
    """
    CLI Environment providing 7 tools that mirror the Claude Code toolset:
    bash, glob, grep, read, write, edit, todo_write.
    """

    def __init__(self, task_spec: JSONObject = {}, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.todos: List[Dict[str, Any]] = []

    async def setup(self) -> None:
        await self.sandbox.start()

    async def teardown(self) -> None:
        await self.sandbox.stop()

    # -- bash --

    @tool
    async def bash(self, params: BashParams) -> ToolOutput:
        """Execute a bash command."""
        try:
            output, code = await self.sandbox.run(params.command.strip())
            return ToolOutput(
                blocks=[TextBlock(text=f"{output}\n\n(exit {code})")],
                metadata={"output": output, "exit_code": code},
                reward=0.0,
                finished=False,
            )
        except Exception as e:
            return ToolOutput(
                metadata={"error": str(e)},
                blocks=[TextBlock(text=f"Error executing command: {str(e)}")],
                finished=False,
            )

    # -- glob --

    @tool
    async def glob(self, params: GlobParams) -> ToolOutput:
        """Find files matching a glob pattern."""
        try:
            search_path = params.path or "."
            cmd = f"find {search_path} -name '{params.pattern}' -type f | sort"
            output, code = await self.sandbox.run(cmd)
            return ToolOutput(
                metadata={"output": output, "exit_code": code},
                blocks=[TextBlock(text=f"{output}\n\n(exit {code})")],
                reward=0.0,
                finished=False,
            )
        except Exception as e:
            return ToolOutput(
                metadata={"error": str(e)},
                blocks=[TextBlock(text=f"Error in glob search: {str(e)}")],
                finished=False,
            )

    # -- grep --

    @tool
    async def grep(self, params: GrepParams) -> ToolOutput:
        """Search for a regex pattern in files."""
        try:
            search_path = params.path or "."
            if params.glob:
                cmd = f"find {search_path} -name '{params.glob}' -type f -exec grep -Hn '{params.pattern}' {{}} \\;"
            else:
                cmd = f"grep -r -n '{params.pattern}' {search_path}"
            output, code = await self.sandbox.run(cmd)
            return ToolOutput(
                metadata={"output": output, "exit_code": code},
                blocks=[TextBlock(text=f"{output}\n\n(exit {code})")],
                reward=0.0,
                finished=False,
            )
        except Exception as e:
            return ToolOutput(
                metadata={"error": str(e)},
                blocks=[TextBlock(text=f"Error in grep search: {str(e)}")],
                finished=False,
            )

    # -- read --

    @tool
    async def read(self, params: ReadParams) -> ToolOutput:
        """Read file contents."""
        try:
            if params.offset and params.limit:
                end_line = params.offset + params.limit
                cmd = f"sed -n '{params.offset},{end_line}p' {params.file_path} | cat -n"
                output, code = await self.sandbox.run(cmd)
            elif params.offset:
                cmd = f"tail -n +{params.offset} {params.file_path} | cat -n"
                output, code = await self.sandbox.run(cmd)
            elif params.limit:
                cmd = f"head -n {params.limit} {params.file_path} | cat -n"
                output, code = await self.sandbox.run(cmd)
            else:
                content = await download_text(self.sandbox, params.file_path)
                lines = content.splitlines()
                output = "\n".join(f"{idx + 1}\t{line}" for idx, line in enumerate(lines))
                if content.endswith("\n") and output:
                    output += "\n"
                code = 0
            return ToolOutput(
                metadata={"output": output, "exit_code": code},
                blocks=[TextBlock(text=f"{output}\n\n(exit {code})")],
                reward=0.0,
                finished=False,
            )
        except Exception as e:
            return ToolOutput(
                metadata={"error": str(e)},
                blocks=[TextBlock(text=f"Error reading file: {str(e)}")],
                finished=False,
            )

    # -- write --

    @tool
    async def write(self, params: WriteParams) -> ToolOutput:
        """Write content to a file."""
        try:
            dir_name = os.path.dirname(params.file_path)
            if dir_name:
                await self.sandbox.run(f"mkdir -p {dir_name}")
            await upload_text(
                self.sandbox,
                params.file_path,
                params.content,
                ensure_trailing_newline=True,
            )
            return ToolOutput(
                metadata={"output": "", "exit_code": 0},
                blocks=[TextBlock(text=f"Successfully wrote to {params.file_path}\n\n(exit 0)")],
                reward=0.0,
                finished=False,
            )
        except Exception as e:
            return ToolOutput(
                metadata={"error": str(e)},
                blocks=[TextBlock(text=f"Error writing file: {str(e)}")],
                finished=False,
            )

    # -- edit --

    @tool
    async def edit(self, params: EditParams) -> ToolOutput:
        """Perform exact string replacement in a file."""
        try:
            content = await download_text(self.sandbox, params.file_path)

            count = content.count(params.old_string)
            if count == 0:
                return ToolOutput(
                    metadata={"error": f"String not found in {params.file_path}"},
                    blocks=[TextBlock(text=f"Error: old_string not found in {params.file_path}")],
                    finished=False,
                )
            if not params.replace_all and count > 1:
                return ToolOutput(
                    metadata={"error": f"old_string appears {count} times; use replace_all or provide more context"},
                    blocks=[TextBlock(text=f"Error: old_string appears {count} times in {params.file_path}. Must be unique unless replace_all=true.")],
                    finished=False,
                )

            if params.replace_all:
                new_content = content.replace(params.old_string, params.new_string)
            else:
                new_content = content.replace(params.old_string, params.new_string, 1)

            await upload_text(self.sandbox, params.file_path, new_content, ensure_trailing_newline=True)

            return ToolOutput(
                metadata={"output": "", "exit_code": 0},
                blocks=[TextBlock(text=f"Successfully edited {params.file_path}\n\n(exit 0)")],
                reward=0.0,
                finished=False,
            )
        except Exception as e:
            return ToolOutput(
                metadata={"error": str(e)},
                blocks=[TextBlock(text=f"Error editing file: {str(e)}")],
                finished=False,
            )

    # -- todo_write --

    @tool
    def todo_write(self, params: TodoWriteParams) -> ToolOutput:
        """
        Manage todo list for task planning and progress tracking.

        Each todo item should have: id, content, status, priority.
        Status options: "pending", "in_progress", "completed"
        Priority options: "high", "medium", "low"
        """
        try:
            self.todos = params.todos

            output_lines = ["=== TODO LIST ==="]
            for todo in self.todos:
                status_icon = {"pending": "⏳", "in_progress": "🔄", "completed": "✅"}.get(todo.get("status", "pending"), "❓")
                priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(todo.get("priority", "medium"), "⚪")
                output_lines.append(f"{status_icon} {priority_icon} {todo.get('content', 'No description')}")

            text = "\n".join(output_lines)
            return ToolOutput(
                metadata={"todos": self.todos, "count": len(self.todos)},
                blocks=[TextBlock(text=text)],
                finished=False,
                reward=0.0,
            )
        except Exception as e:
            return ToolOutput(
                metadata={"error": str(e)},
                blocks=[TextBlock(text=f"Error managing todos: {str(e)}")],
                finished=False,
            )
