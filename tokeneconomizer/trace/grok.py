"""Grok Build session logs (`updates.jsonl`).

Each line is a JSON-RPC `session/update`. The conversation is `params.update`:
`user_message_chunk` (a prompt, possibly several chunks sharing `_meta.promptIndex`),
`tool_call` / `tool_call_update` paired by `toolCallId`, and `turn_completed` with that
turn's usage. `grok_session_to_claude` writes Claude Code envelopes so the other readers
stay agent-agnostic, the same way a Pi session is converted.

`turn_completed.usage.inputTokens` is the full prompt, cache included (`totalTokens` is
that plus `outputTokens`). The ledger's `input_tokens` is the uncached remainder.
`reasoningTokens` is not inside `outputTokens`, so it is added to the output count.
`_meta.totalTokens` on other events is cumulative and is not summed.
"""
import json
import os

from .jsonl import records

_CAP = 20000
_TERMINAL = ("completed", "failed", "error")


def _as_text(value, cap=_CAP):
    """Text out of a Grok tool field: a string, a list of strings or text blocks, or a
    content object."""
    if isinstance(value, str):
        return value[:cap]
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                else:
                    inner = item.get("content")
                    if isinstance(inner, str):
                        parts.append(inner)
                    elif isinstance(inner, dict) and isinstance(inner.get("text"), str):
                        parts.append(inner["text"])
        return "\n".join(parts)[:cap]
    if isinstance(value, dict):
        for key in ("content_concise", "content", "text", "raw_output", "output_for_prompt"):
            if key in value:
                text = _as_text(value.get(key), cap)
                if text:
                    return text
    return ""


def grok_result_text(obj, cap=_CAP):
    """Model-facing text of a tool result: a hook `tool_response`, or a `tool_call_update`."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj[:cap]
    if isinstance(obj, list):
        return _as_text(obj, cap)
    if not isinstance(obj, dict):
        return str(obj)[:cap]
    if "rawOutput" in obj or obj.get("sessionUpdate") == "tool_call_update":
        faced = _as_text(obj.get("content"), cap)
        if faced:
            return faced
    raw = obj.get("rawOutput") if isinstance(obj.get("rawOutput"), dict) else obj
    for key in ("output_for_prompt", "output", "stdout", "stderr", "error", "message",
                "failure"):
        text = _as_text(raw.get(key), cap)
        if text:
            return text
    for key in ("FileContent", "Content"):
        text = _as_text(raw.get(key), cap)
        if text:
            return text
    return _as_text(raw.get("content"), cap)


def grok_exit_code(obj):
    """An integer exit code on a tool result, or None when it names none."""
    if not isinstance(obj, dict):
        return None
    raw = obj.get("rawOutput") if isinstance(obj.get("rawOutput"), dict) else obj
    for key in ("exit_code", "exitCode"):
        value = raw.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _updates(path):
    for ev in records(path):
        params = ev.get("params")
        if not isinstance(params, dict):
            continue
        upd = params.get("update")
        if isinstance(upd, dict) and upd.get("sessionUpdate"):
            yield upd


def _chunk_text(upd):
    return _as_text(upd.get("content"))


def grok_prompt_count(path):
    """How many prompts the person gave: distinct `promptIndex`, plus chunks that name none."""
    seen, bare = set(), 0
    for upd in _updates(path):
        if upd.get("sessionUpdate") != "user_message_chunk":
            continue
        meta = upd.get("_meta") if isinstance(upd.get("_meta"), dict) else {}
        index = meta.get("promptIndex")
        if isinstance(index, int) and not isinstance(index, bool):
            seen.add(index)
        else:
            bare += 1
    return len(seen) + bare


def grok_session_usage(path):
    """Token breakdown of a Grok session: the sum of each `turn_completed` usage."""
    totals = {"input_tokens": 0, "cache_read_input_tokens": 0,
              "cache_creation_input_tokens": 0, "output_tokens": 0}
    requests, models = 0, {}
    for upd in _updates(path):
        if upd.get("sessionUpdate") != "turn_completed":
            continue
        usage = upd.get("usage")
        if not isinstance(usage, dict):
            continue
        incoming = int(usage.get("inputTokens") or 0)
        cached = int(usage.get("cachedReadTokens") or 0)
        created = int(usage.get("cacheCreationTokens") or 0)
        totals["input_tokens"] += max(0, incoming - cached - created)
        totals["cache_read_input_tokens"] += cached
        totals["cache_creation_input_tokens"] += created
        totals["output_tokens"] += (int(usage.get("outputTokens") or 0)
                                    + int(usage.get("reasoningTokens") or 0))
        requests += int(usage.get("modelCalls") or 1)
        by_model = usage.get("modelUsage")
        if isinstance(by_model, dict):
            for name, info in by_model.items():
                calls = 1
                if isinstance(info, dict) and info.get("modelCalls"):
                    calls = int(info["modelCalls"])
                models[name] = models.get(name, 0) + calls
    totals["requests"] = requests
    totals["model"] = max(models, key=models.get) if models else None
    return totals


def grok_session_to_claude(path, out_path):
    """Write a Grok `updates.jsonl` as Claude Code transcript lines.

    Tool names are Grok's titles (`run_terminal_command`, `read_file`, ...). Returns the
    number of tool calls written, or None when the file is missing."""
    if not path or not os.path.exists(path):
        return None
    pending, emitted, n = {}, set(), 0
    open_index, buf = None, []

    def flush_user(out):
        nonlocal open_index, buf
        text = "".join(buf)
        open_index, buf = None, []
        if text:
            out.write(json.dumps({"type": "user", "message": {
                "role": "user", "content": text}}) + "\n")

    with open(out_path, "w", encoding="utf-8") as out:
        for upd in _updates(path):
            kind = upd.get("sessionUpdate")
            if kind == "user_message_chunk":
                meta = upd.get("_meta") if isinstance(upd.get("_meta"), dict) else {}
                index = meta.get("promptIndex")
                if open_index is not None and index != open_index:
                    flush_user(out)
                open_index = index
                buf.append(_chunk_text(upd))
            elif kind == "tool_call":
                flush_user(out)
                cid = upd.get("toolCallId")
                if cid:
                    pending[cid] = {"name": upd.get("title") or "?",
                                    "args": upd.get("rawInput") or {}}
            elif kind == "tool_call_update":
                status = str(upd.get("status") or "")
                if status not in _TERMINAL:
                    continue
                cid = upd.get("toolCallId")
                if not cid or cid in emitted:
                    continue
                info = pending.get(cid) or {"name": "?", "args": {}}
                code = grok_exit_code(upd)
                ok = status == "completed" and (code is None or code == 0)
                raw = upd.get("rawOutput") if isinstance(upd.get("rawOutput"), dict) else {}
                cwd = raw.get("current_dir") if isinstance(raw.get("current_dir"), str) else None
                flush_user(out)
                message = {"role": "assistant", "content": [{
                    "type": "tool_use", "id": cid, "name": info["name"],
                    "input": info["args"] if isinstance(info["args"], dict) else {}}]}
                line = {"type": "assistant", "message": message}
                if cwd:
                    line["cwd"] = cwd
                out.write(json.dumps(line) + "\n")
                out.write(json.dumps({"type": "user", "message": {"role": "user", "content": [{
                    "type": "tool_result", "tool_use_id": cid,
                    "content": grok_result_text(upd), "is_error": not ok}]}}) + "\n")
                emitted.add(cid)
                n += 1
        flush_user(out)
    return n
