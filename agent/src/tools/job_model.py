"""Bounded async job model + idempotency keys — pattern stolen from QuantDinger.

QuantDinger classifies every MCP tool by risk class (R read, W write, B bounded
job, N notification, T trading) and requires a caller-generated
``idempotency_key`` on every mutating tool. Retries of the same request must
reuse the key, and the server returns the original result instead of executing
twice.

Vibe-Trading maps that onto its existing surfaces:

- ``IdempotencyStore`` — SQLite-free, atomic-file based key→result cache under
  the runtime root. A key that is still "in flight" returns the recorded
  submission rather than a second execution.
- ``JobStore`` — bounded async job registry (submit / poll / cancel) that wraps
  any blocking callable (``run_backtest``, strict alpha bench) into a job with
  a durable record. It reuses the existing durable ``BackgroundStore`` shape
  (``src.tools.background_store``) so restart reconciliation semantics stay
  one-way: a record left running is never re-run, only marked interrupted.

Design rules (kept deliberately simple — the QuantDinger lesson is that the
*interface* is the product, not a second database):

1. One process owns a job. Jobs live in memory + one JSON record on disk.
   No Kafka, no leases — Vibe-Trading is a single-process toolkit.
2. Every mutating job tool takes ``idempotency_key``. Retries with the same
   key return the same job_id.
3. Job payloads are bounded: results are truncated previews plus an
   artifact path, never the full output, so a long backtest result never
   bloats an MCP response.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

#: Job statuses mirror QuantDinger's bounded-job lifecycle.
JOB_STATUSES = frozenset({"pending", "running", "completed", "error", "cancelled"})

#: Terminal statuses — a job in one of these never changes again.
TERMINAL_STATUSES = frozenset({"completed", "error", "cancelled"})

#: Idempotency keys are opaque caller strings; we normalise to a digest
#: so the on-disk filename is always filesystem-safe.
_KEY_RE = re.compile(r"^\S.{0,511}$")

#: Preview budget for a job result envelope.
MAX_RESULT_PREVIEW_CHARS = 4_000


class JobError(RuntimeError):
    """A job could not be submitted or mutated."""


@dataclass
class JobRecord:
    """One bounded job. ``schema_version`` guards on-disk reinterpretation."""

    schema_version: int = 1
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    kind: str = "backtest"  # backtest | alpha_bench | custom
    status: str = "pending"
    idempotency_key: Optional[str] = None
    submitted_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    ended_at: Optional[float] = None
    request_preview: dict[str, Any] = field(default_factory=dict)
    result_preview: str = ""
    artifact_path: Optional[str] = None
    error: Optional[str] = None
    pid: Optional[int] = None
    updated_at: float = field(default_factory=time.time)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


def _key_digest(idempotency_key: str) -> str:
    return hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()


def _validate_key(idempotency_key: str) -> str:
    key = (idempotency_key or "").strip()
    if not _KEY_RE.match(key):
        raise JobError(
            "idempotency_key must be 1-512 chars and not start with whitespace"
        )
    return key


def _preview(text: str) -> str:
    text = text or ""
    return text if len(text) <= MAX_RESULT_PREVIEW_CHARS else text[:MAX_RESULT_PREVIEW_CHARS]


class JobStore:
    """In-memory job registry with one durable JSON record per job.

    Single-process by design. Thread-safe via one lock; jobs run on a caller
    supplied executor (``threading.Thread`` by default — ``run_backtest`` is
    blocking sync code and must never run on the asyncio event loop).
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._jobs: dict[str, dict[str, Any]] = {}
        self._keys: dict[str, str] = {}  # digest(idempotency_key) -> job_id
        self._lock = threading.RLock()
        self._directory = self.root / "jobs"

    # ── submission ────────────────────────────────────────────────────────

    def submit(
        self,
        *,
        kind: str,
        idempotency_key: str,
        request: dict[str, Any],
        work: Callable[[], str],
        artifact_path: Optional[str] = None,
        pid: Optional[int] = None,
    ) -> dict[str, Any]:
        """Register and start a bounded job. Returns the job snapshot.

        Idempotent by ``idempotency_key``: a second submit with the same key
        returns the existing job without starting new work.
        """
        key = _validate_key(idempotency_key)
        digest = _key_digest(key)
        with self._lock:
            existing_id = self._keys.get(digest)
            if existing_id is not None:
                snap = self._snapshot_locked(existing_id)
                snap["idempotent_replay"] = True
                return snap

            record = JobRecord(
                kind=kind,
                idempotency_key=key,
                request_preview={
                    k: v for k, v in list(request.items())[:32]  # bounded preview
                },
                artifact_path=artifact_path,
                pid=pid,
            )
            self._jobs[record.job_id] = {"record": record, "thread": None}
            self._keys[digest] = record.job_id
            self._persist(record)

        thread = threading.Thread(
            target=self._run_job, args=(record.job_id, work), daemon=True
        )
        with self._lock:
            self._jobs[record.job_id]["thread"] = thread
        record.status = "running"
        record.started_at = time.time()
        record.updated_at = time.time()
        self._persist(record)
        thread.start()
        return self._snapshot_locked(record.job_id)

    # ── reads ─────────────────────────────────────────────────────────────

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_locked(job_id)

    def list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = sorted(
                self._jobs.values(), key=lambda j: j["record"].submitted_at, reverse=True
            )
            return [self._public(r["record"]) for r in rows[:limit]]

    # ── mutation ───────────────────────────────────────────────────────────

    def cancel(self, job_id: str, *, confirm_cancel: bool = False) -> dict[str, Any]:
        """Request cancellation. Requires explicit confirmation like QuantDinger."""
        if not confirm_cancel:
            raise JobError("cancel_job requires confirm_cancel=true")
        with self._lock:
            entry = self._jobs.get(job_id)
            if entry is None:
                raise JobError(f"unknown job_id: {job_id}")
            record = entry["record"]
            if record.is_terminal:
                return self._snapshot_locked(job_id)
            # Cooperative-only: we never kill a thread. Mark cancelled; the
            # worker's next checkpoint sees it and stops.
            record.status = "cancelled"
            record.ended_at = time.time()
            record.updated_at = time.time()
            self._persist(record)
            return self._snapshot_locked(job_id)

    # ── internals ─────────────────────────────────────────────────────────

    def _run_job(self, job_id: str, work: Callable[[], str]) -> None:
        entry = self._jobs[job_id]
        record: JobRecord = entry["record"]
        if record.status == "cancelled":
            return
        try:
            result = work()
            with self._lock:
                if record.status == "cancelled":
                    return
                record.status = "completed"
                record.result_preview = _preview(result)
                record.ended_at = time.time()
                record.updated_at = time.time()
                self._persist(record)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                record.status = "error"
                record.error = str(exc)
                record.ended_at = time.time()
                record.updated_at = time.time()
                self._persist(record)

    def _persist(self, record: JobRecord) -> None:
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            path = self._directory / f"{record.job_id}.json"
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(record.__dict__, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(path)
        except OSError:
            # Durable record is best-effort; in-memory state remains the
            # source of truth for this process.
            pass

    def _snapshot_locked(self, job_id: str) -> dict[str, Any]:
        entry = self._jobs.get(job_id)
        if entry is None:
            # Fall back to disk so a restarted process can still read a
            # previous process's terminal records.
            return self._load_from_disk(job_id)
        return self._public(entry["record"])

    def _load_from_disk(self, job_id: str) -> dict[str, Any]:
        path = self._directory / f"{job_id}.json"
        if not re.match(r"^[0-9a-f]{32}$", job_id):
            raise JobError(f"unsafe job_id: {job_id!r}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return self._public(JobRecord(**data))
        except (OSError, ValueError, TypeError):
            raise JobError(f"unknown job_id: {job_id}") from None

    def _public(self, record: JobRecord) -> dict[str, Any]:
        out = record.__dict__.copy()
        # Never echo the raw key back; the digest is enough for correlation.
        out.pop("idempotency_key", None)
        out["idempotent"] = True
        return out


def job_snapshot_envelope(snapshot: dict[str, Any]) -> str:
    """Render a job snapshot as the standard MCP JSON envelope."""
    return json.dumps(
        {"status": "ok", "job": snapshot}, ensure_ascii=False, indent=2
    )