# AVA v3 — Tool catalog

This file is the **human source of truth** for tools available to AVA v3 via
MCP. JSON schemas for the model are auto-generated from FastMCP type hints by
`schemas/generate_schemas.py` — never edit `schemas/generated/` by hand.

Format per tool:

```markdown
## tool: <name>
**When to use**: <short trigger description>
**When NOT to use**: <negative scope; prevents over-calling>
**Arguments**:
- `<arg>` (<type>, required|optional): <description, units, allowed values>
**Returns**: <type and shape>
**Example**: <one input → output line>
```

The "When NOT to use" line is mandatory. The discrimination training set
(300 examples, P6) trains the student to obey it.

---

## tool: calculator
**When to use**: arithmetic, unit conversion, financial math, geometry, mod/div checks.
**When NOT to use**: questions answerable from world knowledge (capitals, dates, definitions); when the question contains no numbers; when a programmatic answer in `python_exec` is more natural.
**Arguments**:
- `expression` (string, required): math expression. Supports +, −, ×, ÷, **, sqrt(), log(), e, pi, factorial.
**Returns**: string with numeric result.
**Example**: expression="(72 - 32) * 5 / 9" → "22.22"

---

## tool: python_exec
**When to use**: short data manipulation, exact math beyond `calculator`, list / dict transforms, parsing numeric tables, regex over user-supplied strings.
**When NOT to use**: file system access, network calls, long-running compute (> 5 s), anything reading or writing the user's machine outside the sandbox.
**Arguments**:
- `code` (string, required): Python source. Use `print()` to emit output. RestrictedPython sandbox: no imports beyond `math`, `statistics`, `re`, `json`, `itertools`, `functools`.
**Returns**: stdout from execution, capped at 4 KB.
**Example**: code="print(sum(int(x) for x in '1,2,3,4'.split(',')))" → "10"

---

## tool: code_exec
**When to use**: full Python including third-party scientific libraries, simulations, training scripts, parsing arbitrary files, when execution time may exceed `python_exec`'s 5 s cap.
**When NOT to use**: when `python_exec` already suffices; when the task involves no actual code; when the user has not asked for code to run.
**Arguments**:
- `code` (string, required): Python source.
- `timeout_s` (int, optional, default 30, max 120): wall-time cap.
- `libraries` (list[string], optional): pip packages to ensure available before run.
**Returns**: dict with keys stdout, stderr, exit_code, elapsed_s.
**Example**: code="import numpy as np; print(np.linalg.norm([3,4]))" → {stdout:"5.0", stderr:"", exit_code:0, elapsed_s:0.4}

---

## tool: web_fetch
**When to use**: user explicitly references a URL; news / docs / API reference lookup that is too recent for training data.
**When NOT to use**: questions answerable from training data; when the user has not provided a URL and the model is uncertain which to fetch (ask first).
**Arguments**:
- `url` (string, required): full http(s) URL. Domain must be on the allowlist (configured server-side).
- `format` (enum["markdown","raw"], optional, default "markdown"): output format.
- `max_chars` (int, optional, default 8192): truncation cap.
**Returns**: string content of the URL, possibly truncated.
**Example**: url="https://example.com" → "# Example domain\\n..."

---

## tool: file_read
**When to use**: user references a local path; reading code or configs to answer about them; checking existence of an artifact under `root_paths`.
**When NOT to use**: paths outside the allowlisted roots (server will reject anyway, but model should not request them).
**Arguments**:
- `path` (string, required): absolute or relative path under an allowlisted root.
- `start_line` (int, optional): 1-indexed first line.
- `end_line` (int, optional): 1-indexed last line.
**Returns**: string contents.
**Example**: path="README.md", start_line=1, end_line=5 → "# AVA\\n..."

---

## tool: file_write
**When to use**: explicitly asked to create or modify a file the user controls.
**When NOT to use**: speculative changes; never used as a side effect of a different task.
**Arguments**:
- `path` (string, required): destination path under an allowlisted root.
- `content` (string, required): full file content.
- `append` (bool, optional, default false): if true, append to existing file instead of overwriting.
**Returns**: dict with keys path, bytes_written.
**Example**: path="notes.txt", content="hello" → {path:"notes.txt", bytes_written:5}

---

## tool: memory_get / memory_set
**When to use**: persistent KV across turns or sessions (user preferences, partial state of a long task, intermediate computations).
**When NOT to use**: short-lived turn-scoped state (use the assistant's own context); secrets or credentials.
**Arguments (memory_set)**:
- `key` (string, required): identifier.
- `value` (any JSON-serializable, required): payload.
**Arguments (memory_get)**:
- `key` (string, required): identifier.
**Returns**: stored value or null.
**Example**: memory_set(key="user_name", value="Afsah") → {ok:true}; memory_get(key="user_name") → "Afsah"

---

## tool: shell_exec
**When to use**: a single safe shell command from the deny-list-vetted set when the user explicitly asks for it.
**When NOT to use**: untrusted commands; commands that mutate global state outside `root_paths`; chained pipelines a user did not ask for.
**Arguments**:
- `command` (string, required): single command line. The server enforces a deny-list (rm, dd, mkfs, format, shutdown, reboot, etc.) and forbids network access.
- `timeout_s` (int, optional, default 10, max 60): wall-time cap.
**Returns**: dict with stdout, stderr, exit_code.
**Example**: command="ls -la /tmp" → {stdout:"...", stderr:"", exit_code:0}
