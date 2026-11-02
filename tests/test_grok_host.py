"""Grok Build as a host. Offline: detection, hook output, the updates.jsonl reader, tool names.

A Grok payload that looks like another agent's must stay that agent's. Grok's own SessionStart
reads only `systemMessage`, so model context is held and sent on the first non-subagent
PostToolUse, which does deliver `hookSpecificOutput.additionalContext`.
"""
import io
import json
import os

from tokeneconomizer import trace
from tokeneconomizer.loopguard.state import SpiralState, canon_tool
from tokeneconomizer.plugin import denylist, hook, hosts, sidecall
from tokeneconomizer.plugin.paths import repo_identity, repo_state_dir
from tokeneconomizer.plugin.store import Store


def _call(env, payload, host=None):
    out = io.StringIO()
    hook.main(["--host", host] if host else [], io.StringIO(json.dumps(payload)), out, env)
    return json.loads(out.getvalue().splitlines()[-1])


def _env(tmp_path, **extra):
    env = {"TOKENECONOMIZER_HOME": str(tmp_path / "home"), "TE_FOLD_SYNC": "1",
           "TE_BACKFILL": "off", "TE_REPOMAP": "off", "TE_LOOPGUARD": "off",
           "TE_MICROSKILLS": "envfacts", "GIT_CEILING_DIRECTORIES": str(tmp_path),
           "PATH": os.environ.get("PATH", "")}
    env.update(extra)
    return env


def _repo(tmp_path):
    r = tmp_path / "repo"
    (r / ".git").mkdir(parents=True)
    (r / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (r / ".git" / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/acme/widget.git\n')
    return str(r)


def _store(env, repo, host):
    return Store(repo_state_dir(repo_identity(repo, env), host, env))


def _seed(env, repo, host):
    st = _store(env, repo, host)
    st.write("injection.json", {"rendered": "Facts:\n- run tests with python -m pytest",
                                "rendered_ids": ["pytest"]})
    return st


def _rpc(update):
    return {"jsonrpc": "2.0", "method": "session/update", "params": {"update": update}}


def _write_jsonl(path, updates):
    with open(path, "w", encoding="utf-8") as fh:
        for upd in updates:
            fh.write(json.dumps(_rpc(upd)) + "\n")
    return path


# ---- detection: other hosts stay where they were -----------------------------------------
def test_other_hosts_are_not_classified_as_grok():
    assert hosts.detect({"turn_id": "t"}).name == "codex"
    assert hosts.detect({"transcript_path": "/h/.codex/sessions/2026/rollout-a.jsonl"}).name == "codex"
    assert hosts.detect({"transcript_path": "/h/.claude/projects/x/s.jsonl"}).name == "claude"
    assert hosts.detect({"session_id": "s", "prompt_id": "p", "promptId": "p"}).name == "claude"
    assert hosts.detect({"transcript_path": "/h/.copilot/session-state/s/events.jsonl"}).name == "copilot"
    copilot_env = {"COPILOT_PLUGIN_ROOT": "/p", "COPILOT_CLI": "1", "GROK_HOOK_EVENT": "stop"}
    assert hosts.detect({"session_id": "s"}, copilot_env).name == "copilot"
    assert hosts.detect({"session_id": "s", "transcript_path": "/h/.claude/projects/x/s.jsonl"},
                        {"GROK_HOOK_EVENT": "post_tool_use", "GROK_SESSION_ID": "g"}).name == "claude"
    assert hosts.detect({"turn_id": "t", "transcript_path": "/h/.grok/sessions/x/updates.jsonl"}
                        ).name == "codex"
    assert hosts.detect({"transcript_path": r"C:\Users\a\.codex\sessions\rollout-a.jsonl"}).name == "codex"


def test_grok_is_detected_from_its_transcript_or_its_hook_environment():
    assert hosts.detect({"transcript_path": "/h/.grok/sessions/%2Frepo/abc/updates.jsonl"}).name == "grok"
    assert hosts.detect({"transcript_path": r"C:\Users\a\.grok\sessions\abc\updates.jsonl"}).name == "grok"
    assert hosts.detect({"session_id": "s"}, {"GROK_HOOK_EVENT": "post_tool_use"}).name == "grok"
    assert hosts.detect({"session_id": "s", "transcript_path": "/opt/grok-home/sessions/a/updates.jsonl"},
                        {"GROK_SESSION_ID": "s"}).name == "grok"
    assert hosts.detect({"turn_id": "t"}, hint="grok").name == "grok"
    assert hosts.default_host({}) == "claude"
    assert hosts.default_host({"GROK_SESSION_ID": "s"}) == "grok"
    assert hosts.default_host({"GROK_HOOK_EVENT": "stop"}) == "grok"
    assert hosts.default_host({"CLAUDECODE": "1", "GROK_SESSION_ID": "s"}) == "claude"
    assert hosts.default_host({"COPILOT_CLI": "1", "GROK_SESSION_ID": "s"}) == "copilot"
    assert hosts.default_host({}, recent="pi") == "pi"
    assert hosts.default_host({"GROK_SESSION_ID": "s"}, recent="pi") == "grok"
    assert denylist.supports("grok") is False
    assert denylist.supports("claude") is True


def test_grok_turn_identity_subagent_and_tool_result():
    h = hosts.HOSTS["grok"]()
    assert h.turn_key({"promptId": "abc"}) == "abc"
    assert h.turn_key({"prompt_id": "from-alias"}) == "from-alias"
    assert h.turn_key({}) == ""
    assert h.session_id({"sessionId": "sid"}) == "sid"
    assert h.is_subagent({"subagentType": "egm-design"}) is True
    assert h.is_subagent({"session_id": "s"}) is False
    assert h.prompt_context is False and h.keeps_start_context is False
    assert h.defers_start_context is True
    assert hosts.ClaudeCode().defers_start_context is False
    name, args, text, code = h.tool_call({
        "toolName": "run_terminal_command",
        "toolInput": {"command": "python -m pytest -q", "description": "tests"},
        "toolResult": {"type": "Bash", "exit_code": 2, "output": ["nope"],
                       "output_for_prompt": "nope: failed"},
    })
    assert name == "run_terminal_command" and args["command"].startswith("python -m pytest")
    assert text == "nope: failed" and code == 2
    ctx = h.context("PostToolUse", "facts")
    assert ctx["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert ctx["hookSpecificOutput"]["additionalContext"] == "facts"
    assert "additionalContext" not in ctx


def test_claude_still_reads_prompt_id_and_does_not_defer(tmp_path):
    env, repo = _env(tmp_path), _repo(tmp_path)
    st = _seed(env, repo, "claude")
    base = {"session_id": "c1", "cwd": repo, "transcript_path": str(tmp_path / "c.jsonl")}
    start = _call(env, dict(base, hook_event_name="SessionStart", source="startup",
                            prompt_id="p"))
    assert start["hookSpecificOutput"]["additionalContext"].startswith("Facts:")
    sess = st.read("sessions/c1.json")
    assert "deferred_context" not in sess
    post = _call(env, dict(base, hook_event_name="PostToolUse", tool_name="Bash",
                           tool_input={"command": "true"}, tool_response="ok"))
    assert post == {}


def test_grok_holds_session_start_context_until_the_first_tool_result(tmp_path):
    env, repo = _env(tmp_path), _repo(tmp_path)
    st = _seed(env, repo, "grok")
    base = {"session_id": "g1", "sessionId": "g1", "cwd": repo,
            "transcript_path": "/h/.grok/sessions/x/g1/updates.jsonl"}
    start = _call(env, dict(base, hook_event_name="SessionStart", source="new"))
    # SessionStart still builds the Claude-shaped context; Grok shows only systemMessage,
    # so the same text is kept for the first PostToolUse.
    held = start["hookSpecificOutput"]["additionalContext"]
    assert held.startswith("Facts:")
    assert st.read("sessions/g1.json")["deferred_context"] == held
    child = _call(env, dict(base, hook_event_name="PostToolUse", subagentType="egm-design",
                            tool_name="read_file", tool_input={"target_file": "a.py"},
                            tool_response="x", promptId="t1"))
    assert child == {}
    assert st.read("sessions/g1.json")["deferred_context"] == held
    post = _call(env, dict(base, hook_event_name="PostToolUse", promptId="t1",
                           tool_name="read_file", tool_input={"target_file": "a.py"},
                           tool_response="x"))
    assert post["hookSpecificOutput"]["hookEventName"] == "PostToolUse"
    assert post["hookSpecificOutput"]["additionalContext"] == held
    assert "deferred_context" not in st.read("sessions/g1.json")
    again = _call(env, dict(base, hook_event_name="PostToolUse", promptId="t1",
                            tool_name="grep", tool_input={"pattern": "x"}, tool_response=""))
    assert again == {}


def test_grok_still_delivers_held_context_when_the_guard_is_on(tmp_path):
    env, repo = _env(tmp_path, TE_LOOPGUARD="nudge"), _repo(tmp_path)
    st = _seed(env, repo, "grok")
    base = {"session_id": "g3", "cwd": repo,
            "transcript_path": "/h/.grok/sessions/x/g3/updates.jsonl"}
    _call(env, dict(base, hook_event_name="SessionStart", source="new"))
    post = _call(env, dict(base, hook_event_name="PostToolUse", promptId="t",
                           tool_name="read_file", tool_input={"target_file": "a.py"},
                           tool_response="x"))
    assert "python -m pytest" in post["hookSpecificOutput"]["additionalContext"]
    assert st.read("sessions/g3.json")["turn"]["cum_turns"] == 1
    assert "deferred_context" not in st.read("sessions/g3.json")


def test_grok_prompt_id_is_the_turn_and_an_empty_tool_event_stays_on_it(tmp_path):
    env, repo = _env(tmp_path, TE_LOOPGUARD="nudge", TE_MICROSKILLS="off"), _repo(tmp_path)
    st = _store(env, repo, "grok")
    base = {"session_id": "g2", "cwd": repo,
            "transcript_path": str(tmp_path / "missing-updates.jsonl")}
    _call(env, dict(base, hook_event_name="SessionStart", source="new"), "grok")
    _call(env, dict(base, hook_event_name="UserPromptSubmit", promptId="turn-a"), "grok")
    assert st.read("sessions/g2.json")["turn"]["turn_key"] == "turn-a"
    _call(env, dict(base, hook_event_name="PostToolUse", tool_name="read_file",
                    tool_input={"target_file": "a.py"}, tool_response="file"), "grok")
    assert st.read("sessions/g2.json")["turn"]["turn_key"] == "turn-a"
    assert st.read("sessions/g2.json")["turn"]["cum_turns"] == 1


def test_grok_tool_names_join_the_existing_canon_without_moving_it():
    assert canon_tool("Bash") == "bash"
    assert canon_tool("Edit") == "edit"
    assert canon_tool("Write") == "create"
    assert canon_tool("run_terminal_command") == "bash"
    assert canon_tool("run_terminal_cmd") == "bash"
    assert canon_tool("read_file") == "view"
    assert canon_tool("search_replace") == "edit"
    assert canon_tool("list_dir") == "glob"
    assert canon_tool("grep") == "grep"
    st = SpiralState()
    st.update("search_replace", {"target_file": "pkg/mod.py", "old_string": "a", "new_string": "b"},
              "ok", False)
    assert st.edit_counts == {"mod.py": 1}
    st.update("read_file", {"target_file": "pkg/mod.py"}, "assert False", False)
    assert st.error_outputs == 0          # a successful read is content, not a failure


def test_updates_jsonl_usage_prompts_and_calls(tmp_path):
    path = _write_jsonl(str(tmp_path / "updates.jsonl"), [
        {"sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": "fix "},
         "_meta": {"promptIndex": 0, "modelId": "grok-4.7"}},
        {"sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": "the bug"},
         "_meta": {"promptIndex": 0, "modelId": "grok-4.7"}},
        {"sessionUpdate": "tool_call", "toolCallId": "c1", "title": "run_terminal_command",
         "rawInput": {"command": "python -m pytest -q", "description": "tests"}},
        {"sessionUpdate": "tool_call_update", "toolCallId": "c1", "status": "completed",
         "rawOutput": {"type": "Bash", "exit_code": 1, "output": ["fail"],
                       "output_for_prompt": "1 failed", "current_dir": "/repo"}},
        {"sessionUpdate": "tool_call", "toolCallId": "c2", "title": "read_file",
         "rawInput": {"target_file": "pkg/mod.py"}},
        {"sessionUpdate": "tool_call_update", "toolCallId": "c2", "status": "failed",
         "rawOutput": {"type": "ReadFile", "FileContent": {"content": "missing"}},
         "content": [{"type": "content", "content": {"type": "text", "text": "not found"}}]},
        {"sessionUpdate": "turn_completed", "prompt_id": "p0", "stop_reason": "end_turn",
         "usage": {"inputTokens": 1000, "cachedReadTokens": 800, "cacheCreationTokens": 50,
                   "outputTokens": 10, "reasoningTokens": 5, "modelCalls": 2, "totalTokens": 1010,
                   "modelUsage": {"grok-4.7": {"modelCalls": 2}}}},
        {"sessionUpdate": "user_message_chunk", "content": {"type": "text", "text": "again"},
         "_meta": {"promptIndex": 1}},
        {"sessionUpdate": "turn_completed", "prompt_id": "p1", "stop_reason": "end_turn",
         "usage": {"inputTokens": 200, "cachedReadTokens": 0, "cacheCreationTokens": 0,
                   "outputTokens": 20, "reasoningTokens": 0, "modelCalls": 1, "totalTokens": 220,
                   "modelUsage": {"grok-4.7": {"modelCalls": 1}}}},
    ])
    assert trace.grok_prompt_count(path) == 2
    usage = trace.grok_session_usage(path)
    # inputTokens includes the cache read; the ledger wants the uncached split.
    assert usage["input_tokens"] == (1000 - 800 - 50) + 200
    assert usage["cache_read_input_tokens"] == 800
    assert usage["cache_creation_input_tokens"] == 50
    assert usage["output_tokens"] == 10 + 5 + 20
    assert usage["requests"] == 3 and usage["model"] == "grok-4.7"
    out = str(tmp_path / "claude.jsonl")
    assert trace.grok_session_to_claude(path, out) == 2
    calls = trace.load_trajectory(out)
    # The converted log keeps Grok's titles. The guard canonicalizes them when it folds.
    assert calls[0][0] == "run_terminal_command" and canon_tool(calls[0][0]) == "bash"
    assert calls[0][1] == {"command": "python -m pytest -q", "description": "tests"}
    assert calls[0][2] == "1 failed" and calls[0][3] is False
    assert calls[1][0] == "read_file" and canon_tool(calls[1][0]) == "view"
    assert calls[1][3] is False and "not found" in calls[1][2]
    host = hosts.HOSTS["grok"]()
    work = tmp_path / "work"
    work.mkdir()
    traced = host.mechanism_trace(path, str(work))
    assert trace.load_trajectory(traced)[0][0] == "run_terminal_command"
    assert host.session_usage(path)["input_tokens"] == usage["input_tokens"]
    assert host.prompt_count(path) == 2


def test_grok_history_reads_encoded_session_directories(tmp_path):
    home = tmp_path / "grok-home"
    cwd = tmp_path / "repo"
    cwd.mkdir()
    encoded = cwd.as_posix().replace("/", "%2F")
    sess = home / "sessions" / encoded / "abc"
    sess.mkdir(parents=True)
    (sess / "updates.jsonl").write_text("{}\n")
    (home / "sessions" / "not-a-path" / "nope").mkdir(parents=True)
    (home / "sessions" / "not-a-path" / "nope" / "updates.jsonl").write_text("{}\n")
    found = hosts.HOSTS["grok"]().history({"GROK_HOME": str(home)})
    assert found == [("abc", str(sess / "updates.jsonl"), str(cwd))]


def test_a_grok_side_call_does_not_go_through_claude():
    argv = sidecall.grok_argv("say hi", model="grok-4")
    assert argv[0] == "grok" and "-p" in argv and "say hi" in argv
    assert "--output-format" in argv and "json" in argv
    assert "--max-turns" in argv and "--permission-mode" in argv and "plan" in argv
    assert "--no-subagents" in argv
    assert "--tools" not in argv
    deny = argv[argv.index("--disallowed-tools") + 1]
    for name in ("run_terminal_command", "run_terminal_cmd", "write", "search_replace",
                 "web_search", "web_fetch", "spawn_subagent"):
        assert name in deny.split(",")
    cfg = type("C", (), {"paid_model": ""})()
    runner = sidecall.runner_for("grok", cfg)
    assert "grok_call" in runner.__code__.co_names
    assert "claude_call" not in runner.__code__.co_names
    assert "claude_call" in sidecall.runner_for("claude", cfg).__code__.co_names


def test_trace_exports_the_grok_readers():
    for name in ("grok_prompt_count", "grok_session_usage", "grok_session_to_claude"):
        assert name in trace.__all__
