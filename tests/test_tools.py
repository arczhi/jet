"""Built-in tool behavior, including the failure modes that matter."""

from __future__ import annotations

from pathlib import Path

import pytest

from jet.errors import ToolExecutionError
from jet.tools.base import ToolContext
from jet.tools.builtin.files import EditFileTool, ListFilesTool, ReadFileTool, WriteFileTool
from jet.tools.builtin.search import GlobTool, GrepTool
from jet.tools.builtin.shell import RunCommandTool


async def test_read_file_with_range(tool_context: ToolContext) -> None:
    result = await ReadFileTool().run({"path": "a.txt", "offset": 2, "limit": 1}, tool_context)
    assert "beta" in result.output
    assert "lines 2-2 of 3" in result.output


async def test_read_file_missing_is_explicit(tool_context: ToolContext) -> None:
    with pytest.raises(ToolExecutionError, match="not a file"):
        await ReadFileTool().run({"path": "nope.txt"}, tool_context)


async def test_path_escape_is_refused(tool_context: ToolContext) -> None:
    with pytest.raises(ToolExecutionError, match="outside the workspace"):
        await ReadFileTool().run({"path": "../../etc/passwd"}, tool_context)


async def test_absolute_path_outside_workspace_is_refused(tool_context: ToolContext) -> None:
    with pytest.raises(ToolExecutionError, match="outside the workspace"):
        await ReadFileTool().run({"path": "/etc/hosts"}, tool_context)


async def test_write_then_read_roundtrip(tool_context: ToolContext) -> None:
    result = await WriteFileTool().run({"path": "new/note.txt", "content": "hello"}, tool_context)
    assert "created" in result.output
    read = await ReadFileTool().run({"path": "new/note.txt"}, tool_context)
    assert "hello" in read.output


async def test_edit_requires_unique_match(tool_context: ToolContext) -> None:
    await WriteFileTool().run({"path": "dup.txt", "content": "same\nsame\n"}, tool_context)
    with pytest.raises(ToolExecutionError, match="occurs 2 times"):
        await EditFileTool().run(
            {"path": "dup.txt", "old_string": "same", "new_string": "different"}, tool_context
        )
    result = await EditFileTool().run(
        {"path": "dup.txt", "old_string": "same", "new_string": "different", "replace_all": True},
        tool_context,
    )
    assert "2 replacements" in result.output


async def test_edit_missing_match_is_explicit(tool_context: ToolContext) -> None:
    with pytest.raises(ToolExecutionError, match="not found"):
        await EditFileTool().run({"path": "a.txt", "old_string": "zzz", "new_string": "yyy"}, tool_context)


async def test_edit_identical_strings_is_refused(tool_context: ToolContext) -> None:
    with pytest.raises(ToolExecutionError, match="identical"):
        await EditFileTool().run(
            {"path": "a.txt", "old_string": "alpha", "new_string": "alpha"}, tool_context
        )


async def test_list_files_skips_noise_dirs(tool_context: ToolContext, workspace: Path) -> None:
    (workspace / ".git").mkdir()
    (workspace / ".git" / "HEAD").write_text("ref")
    (workspace / "node_modules").mkdir()
    (workspace / "node_modules" / "pkg.js").write_text("x")
    result = await ListFilesTool().run({"path": ".", "depth": 2}, tool_context)
    assert "a.txt" in result.output
    assert "src/mod.py" in result.output
    assert "HEAD" not in result.output
    assert "pkg.js" not in result.output


async def test_grep_python_path(tool_context: ToolContext) -> None:
    tool_context.prefer_rg = False
    result = await GrepTool().run({"pattern": r"def \w+", "glob": "*.py"}, tool_context)
    assert "src/mod.py:1:def add" in result.output
    assert result.meta["engine"] == "python"


async def test_grep_ignore_case_and_limit(tool_context: ToolContext) -> None:
    tool_context.prefer_rg = False
    result = await GrepTool().run({"pattern": "ALPHA", "ignore_case": True}, tool_context)
    assert "a.txt:1:alpha" in result.output


async def test_grep_rg_path_when_available(tool_context: ToolContext) -> None:
    if not __import__("shutil").which("rg"):
        pytest.skip("ripgrep not installed")
    result = await GrepTool().run({"pattern": "alpha"}, tool_context)
    assert "alpha" in result.output
    assert result.meta["engine"] == "rg"


async def test_grep_invalid_regex(tool_context: ToolContext) -> None:
    with pytest.raises(ToolExecutionError, match="invalid regex"):
        await GrepTool().run({"pattern": "("}, tool_context)


async def test_glob(tool_context: ToolContext) -> None:
    result = await GlobTool().run({"pattern": "**/*.py"}, tool_context)
    assert "src/mod.py" in result.output


async def test_run_command_success_and_failure(tool_context: ToolContext) -> None:
    ok = await RunCommandTool().run({"command": "echo hello"}, tool_context)
    assert ok.ok and "hello" in ok.output and "exit 0" in ok.output
    bad = await RunCommandTool().run({"command": "exit 3"}, tool_context)
    assert bad.ok is False and "exit 3" in bad.output


async def test_run_command_timeout_is_killed(tool_context: ToolContext) -> None:
    with pytest.raises(ToolExecutionError, match="exceeded"):
        await RunCommandTool().run({"command": "sleep 5", "timeout_s": 1}, tool_context)


async def test_run_command_rejects_bad_timeout(tool_context: ToolContext) -> None:
    with pytest.raises(ToolExecutionError, match="timeout_s"):
        await RunCommandTool().run({"command": "echo hi", "timeout_s": 0}, tool_context)
