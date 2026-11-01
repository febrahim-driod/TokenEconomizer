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
"""
import os
import time

from . import fold
from .paths import repo_identity

STATE = "backfill.json"


def needed(ctx):
    return ctx.config.backfill != "off" and not ctx.store.read(STATE)


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
    """Fold the repository's history once. Returns what `backfill.json` records, or None when
    another backfill holds the lock or this one already ran."""
    with ctx.store.locked("backfill", timeout=0) as got:
        if not got or ctx.store.read(STATE):
            return None
        todo = history(ctx, exclude)
        folded = [sid for sid, path in todo
                  if fold.fold_session(ctx, sid, path, backfill=True) is not None]
        state = {"at": time.time(), "found": len(todo), "folded": len(folded)}
        ctx.store.write(STATE, state)
        return state


def spawn(ctx, exclude=()):
    """Run the backfill in a detached process (inline when TE_FOLD_SYNC=1, for tests)."""
    if ctx.env.get("TE_FOLD_SYNC") == "1":
        return run(ctx, exclude)
    return ctx.spawn(["backfill"] + [a for sid in exclude for a in ("--exclude", sid)],
                     "fold.log")
