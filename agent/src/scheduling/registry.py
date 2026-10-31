"""Allowlist registry for internal scheduled callables (scheduler unification).

The ONLY name→callable mapping available to ``executor_kind == "internal"``
jobs. Security contract — this is the safety bedrock, "no arbitrary shell,
no dynamic import":

* Keys are literal strings registered in THIS module. The production surface
  is the explicit ``_INTERNAL_CALLABLES`` dict at the bottom, populated by
  module edit only. Nothing in a request path (HTTP body, job file free
  text, environment variable) may ever register a callable;
  :func:`register_internal` exists for module edits and tests.
* Resolution is an exact-key dict lookup. There is NO dynamic resolution of
  any kind: names are never parsed as import paths, never turned into code,
  and never walked through attribute chains. A name that happens to be a
  real import path (a stdlib process-spawning function, say) resolves to
  nothing unless that exact string is a registered key — the registry's
  name space and Python's import space are fully isolated.
* Malformed names are rejected before any lookup: the shape allowlist
  (dotted identifiers only) fails every separator, ``..``, whitespace and
  shell-metacharacter form by construction.
* Values are callables taking no required arguments, sync or async. The run
  contract lives on the executor side (see
  :mod:`src.scheduled_research.internal_dispatch`): sync callables are
  offloaded through ``asyncio.to_thread``, async callables are awaited
  directly, always in a background task — never on the serialized tick loop.
* Domain imports are DEFERRED: each production entry is a literal zero-arg
  wrapper whose body imports the domain callable at first firing. This
  module is imported by the generic :mod:`src.scheduling` package (and,
  through it, by ``scheduled_research.models``), so a top-level domain
  import here would drag the whole ``br_data``/``memory`` stack into every
  process that merely touches the scheduling vocabulary — and invert the
  dependency direction of the generic kernel. The wrappers keep the dict
  literal-keyed and exact-match-only; the deferred import is a fixed module
  statement inside a function body, never a name-driven resolution.
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable

#: A registered internal callable: invoked with no arguments, either sync
#: (return value discarded) or async. ``...`` tolerates defaulted/optional
#: parameters; the run contract above is what the executor guarantees.
InternalRunnable = Callable[..., Awaitable[None] | None]

#: Dotted-identifier shape allowlist, e.g. ``memory.consolidate``. Being an
#: allowlist (not a denylist) is the point: path separators, ``..``, ``:``,
#: whitespace and shell metacharacters (``; | & $ < > ( )`` and backtick)
#: fail the shape check by construction and never reach the dict.
_VALID_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


class InternalCallableError(KeyError):
    """A name cannot be resolved to a registered internal callable.

    Subclasses :class:`KeyError` so consumers may catch either; ``__str__``
    is overridden because :class:`KeyError` would otherwise ``repr()`` the
    message and quote it.
    """

    def __str__(self) -> str:
        return str(self.args[0]) if self.args else ""


def _check_name(name: object) -> str:
    """Return *name* if it has the dotted-identifier shape, else reject it."""
    if not isinstance(name, str) or not _VALID_NAME_RE.fullmatch(name):
        raise InternalCallableError(
            f"internal callable name {name!r} is rejected: only dotted identifiers "
            "(letters, digits, '_', single '.' separators) are allowed"
        )
    return name


def register_internal(name: str, fn: InternalRunnable, *, replace: bool = False) -> None:
    """Register *fn* under the literal key *name*.

    MODULE/TEST-ONLY: the production surface is the explicit dict entries at
    the bottom of this module, added by module edit. No runtime input may
    reach this function — the invariant is documented here and guarded by
    ``test_registry.py``.

    Args:
        name: Literal registry key (dotted-identifier shape).
        fn: Zero-required-arg callable, sync or async.
        replace: Allow overwriting an existing key. Off by default so a
            production entry is never silently clobbered.

    Raises:
        InternalCallableError: *name* is malformed, or already registered
            without ``replace=True``.
        TypeError: *fn* is not callable.
    """
    key = _check_name(name)
    if not callable(fn):
        raise TypeError(f"internal callable {key!r} requires a callable, got {type(fn).__name__}")
    if not replace and key in _INTERNAL_CALLABLES:
        raise InternalCallableError(f"internal callable {key!r} is already registered")
    _INTERNAL_CALLABLES[key] = fn


def resolve_internal(name: str) -> InternalRunnable:
    """Return the callable registered under the EXACT key *name*.

    Exact-key dict lookup only — never a prefix match, never a parse, never
    a fallback to Python's import machinery.

    Args:
        name: Registry key from a job's ``internal_callable`` field.

    Returns:
        The registered callable object (identity preserved).

    Raises:
        InternalCallableError: *name* is malformed (rejected before lookup)
            or is not a registered key; the message for an unregistered
            well-formed name lists every registered name.
    """
    key = _check_name(name)
    try:
        return _INTERNAL_CALLABLES[key]
    except KeyError:
        raise InternalCallableError(
            f"internal callable {key!r} is not registered; registered names: {list_internal()}"
        ) from None


def list_internal() -> tuple[str, ...]:
    """Return every registered key, sorted."""
    return tuple(sorted(_INTERNAL_CALLABLES))


def is_registered(name: object) -> bool:
    """Whether *name* is a well-formed AND registered key. Never raises."""
    return isinstance(name, str) and bool(_VALID_NAME_RE.fullmatch(name)) and name in _INTERNAL_CALLABLES


# ---------------------------------------------------------------------------
# Production entries — literal zero-arg wrappers with a deferred domain
# import inside the function body (module docstring, "Domain imports are
# DEFERRED"). The wrapper is the registered callable object; the import it
# performs is a fixed statement, never derived from the registry key.
# ---------------------------------------------------------------------------


def _memory_consolidate() -> None:
    """Run one scheduled memory-consolidation pass (deferred domain import).

    Backs the ``memory.consolidate`` key: daily archive-only consolidation,
    opt-in via ``VT_MEMORY_CONSOLIDATION`` and an honest no-op unless
    ``VT_MEMORY_GC`` is on too (see :mod:`src.memory.consolidation`).
    """
    from src.memory.consolidation import consolidate

    consolidate()


def _br_data_run_sync() -> None:
    """Run one Brazilian official-dataset sync (deferred domain import).

    Backs the ``br_data.run_sync`` key: daily sync, opt-in via
    ``VIBE_BR_SYNC_ENABLED``; replaces the retired ``src.br_data`` daemon —
    Per-source errors become a failed scheduled attempt, so the executor
    records the failure and applies its existing retry budget.
    """
    from src.br_data.sync import run_sync

    report = run_sync()
    failed = sorted(
        name for name, outcome in report.items()
        if isinstance(outcome, dict) and "error" in outcome
    )
    if failed:
        # Report source names only; upstream diagnostics can carry local
        # paths or credentials and must not be copied into the job store.
        raise RuntimeError(f"Brazilian dataset sync failed for: {', '.join(failed)}")


def _paper_trade_run_pass() -> None:
    """Run one deterministic SMA-crossover pass on the paper account.

    Backs the ``paper_trade.run`` key (fork addition): a no-LLM autonomous
    trading pass for local models that strain on the full eager tool schema.
    Watchlist, notional cap, and profile are fixed in
    :mod:`src.scheduling.paper_run`; orders go through the same gated
    ``src.trading.service`` surface the agent tools use, so paper placement
    stays on the broker sandbox and a live profile would still hit the
    mandate gate.
    """
    from src.scheduling.paper_run import run_paper_pass

    run_paper_pass()


# ---------------------------------------------------------------------------
# Production allowlist — the ONLY name→callable surface (module docstring).
#
# A new entry is ONE literal wrapper function above plus ONE literal key
# here. Rules: literal keys only; values are the wrapper callable objects,
# never import-path strings; nothing populates this dict from runtime input.
# An internal firing whose key is missing fail-closes into the executor's
# dispatch-failure path (``failure_kind="dispatch"``).
# ---------------------------------------------------------------------------
_INTERNAL_CALLABLES: dict[str, InternalRunnable] = {
    "memory.consolidate": _memory_consolidate,
    "br_data.run_sync": _br_data_run_sync,
    "paper_trade.run": _paper_trade_run_pass,
}
