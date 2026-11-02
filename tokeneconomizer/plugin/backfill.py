"""Learning from the sessions a repository ran before the plugin was installed.

The agent keeps its own history: Claude Code writes every session's log under
`~/.claude/projects/` (or `$CLAUDE_CONFIG_DIR/projects/`). The first time the plugin starts a
session in a repository, it finds that history's sessions of the same repository -- by the
directory each log says it started in, resolved to a repository the way a live session is -- and
folds the most recent `backfill_sessions` of them, oldest first, exactly as if each had just
ended. Later sessions then keep learning from there.

It runs once per repository and agent (`backfill.json`), in a detached `te backfill`, under its
own lock, so two sessions starting together do not both read the history. Each session is folded
at most once (`folded.json`), so a historical session resumed later is not counted twice, and a
backfill cut short picks up where it stopped at the next session start.

The history is folded whatever micro-skills is set to, and a session folded while it was off
taught it nothing. So when micro-skills is on and some folded sessions -- the history's or live
ones -- were never fed to it, the same detached `te backfill` catches it up (`catch_up`):
bounded like the history, and only micro-skills reads them, since the ledger and tool pruning
have counted them already. Turning micro-skills on then starts it from what the repository's
sessions already showed, not from nothing.
"""
import os
import time

from . import fold
from .paths import repo_identity

STATE = "backfill.json"


def needed(ctx):
    """Is there anything for `run` to read: the history (once), or folded sessions micro-skills
    has not learned from?"""
    if ctx.config.backfill == "off":
        return False
    return not ctx.store.read(STATE) or _unread(ctx)


def _unread(ctx):
    if ctx.config.microskills == "off":
        return False
    folded = ctx.store.read("folded.json") or {}
    return any(e.get("microskills") in (None, fold.OFF) for e in folded.values())


def history(ctx, exclude=(), stale_s=fold.STALE_S, now=None):
    """[(session id, log path)] of this agent's earlier sessions of this repository, oldest
    first: the most recent `backfill_sessions` not yet folded. A log written in the last
    `stale_s` may belong to a session still running, which folds at its own end."""
    now = now or time.time()
    folded = ctx.store.read("folded.json") or {}
    keys = {}
    found = []
    for sid, path, cwd in ctx.host.history(ctx.env):
        if sid in folded or sid in exclude:
            continue
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if now - mtime < stale_s:
            continue
        if cwd not in keys:
            # A directory since deleted resolves to itself, and so to no repository's key.
            keys[cwd] = repo_identity(cwd, ctx.env).key
        if keys[cwd] == ctx.repo.key:
            found.append((mtime, sid, path))
    found.sort()
    return [(sid, path) for _, sid, path in found[-ctx.config.backfill_sessions:]]


def run(ctx, exclude=()):
    """Fold the repository's history once, then catch micro-skills up. Returns what
    `backfill.json` records, or None when another backfill holds the lock."""
    with ctx.store.locked("backfill", timeout=0) as got:
        if not got:
            return None
        state = ctx.store.read(STATE)
        if not state:
            todo = history(ctx, exclude)
            folded = [sid for sid, path in todo
                      if fold.fold_session(ctx, sid, path, backfill=True) is not None]
            state = {"at": time.time(), "found": len(todo), "folded": len(folded)}
            ctx.store.write(STATE, state)
        fed = catch_up(ctx, exclude)
        if fed:
            prior = (state.get("caught_up") or {}).get("microskills", 0)
            state["caught_up"] = {"microskills": prior + fed, "at": time.time()}
            ctx.store.write(STATE, state)
        return state


def _marks_from_ledger(ctx):
    """For entries folded before `folded.json` kept the mark: a session's ledger record has a
    non-null "injection" exactly when micro-skills read it."""
    return {r.get("session_id"): fold.FED if r.get("injection") is not None else fold.OFF
            for r in ctx.store.read_lines("ledger.jsonl") if r.get("kind") == "session"}


def catch_up(ctx, exclude=()):
    """Feed micro-skills the folded sessions it has not learned from: the most recent
    `backfill_sessions` of them, oldest first. Older ones are marked SKIPPED and those whose log
    is gone UNREADABLE, so no later start tries them again. Returns how many it fed."""
    if ctx.config.microskills == "off":
        return 0
    with ctx.store.locked("fold", timeout=fold.FOLD_LOCK_WAIT_S) as got:
        if not got:
            return 0
        folded = ctx.store.read("folded.json") or {}
        if any("microskills" not in e for e in folded.values()):
            ledger = _marks_from_ledger(ctx)
            for sid, e in folded.items():
                e.setdefault("microskills", ledger.get(sid, fold.OFF))
        waiting = []
        for sid, e in folded.items():
            if sid in exclude or e["microskills"] != fold.OFF:
                continue
            try:
                waiting.append((os.path.getmtime(e.get("transcript") or ""), sid))
            except OSError:
                e["microskills"] = fold.UNREADABLE
        waiting.sort()
        reach = max(0, ctx.config.backfill_sessions)
        todo = waiting[len(waiting) - reach:] if reach else []
        for _, sid in waiting[:len(waiting) - len(todo)]:
            folded[sid]["microskills"] = fold.SKIPPED
        ctx.store.write("folded.json", folded)
    return sum(1 for _, sid in todo if fold.feed_microskills(ctx, sid) == fold.FED)


def spawn(ctx, exclude=()):
    """Run the backfill in a detached process (inline when TE_FOLD_SYNC=1, for tests)."""
    if ctx.env.get("TE_FOLD_SYNC") == "1":
        return run(ctx, exclude)
    return ctx.spawn(["backfill"] + [a for sid in exclude for a in ("--exclude", sid)],
                     "fold.log")
