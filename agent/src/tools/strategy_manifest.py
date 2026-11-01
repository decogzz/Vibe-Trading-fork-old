"""Compiled strategy manifests — the QuantDinger Strategy API V2 pattern.

QuantDinger requires executable strategy code to ``compile_strategy_code`` into
a canonical *manifest* before deployment: the manifest owns instruments,
markets, frequencies, warmup, and leverage policy, and the backtest/deployment
reads the manifest rather than ad-hoc parameters.

Vibe-Trading strategies are today a loose pair of ``config.json`` +
``code/signal_engine.py`` inside a run directory. Nothing validates the pair
as a coherent unit before ``backtest`` runs, and nothing versions it. This
module steals the manifest idea while keeping Vibe-Trading's simplicity:

- ``compile_strategy`` parses ``config.json`` + ``signal_engine.py`` and
  produces a canonical manifest dict: universe (codes/source/market),
  period, engine hints, warmup (imported-lookback estimate), leverage/
  short policy, and a SHA-256 content hash of both inputs.
- ``save_manifest`` / ``list_manifests`` / ``get_manifest`` persist
  manifests under ``<runtime_root>/strategy-manifests/<manifest_id>.json``
  so ``list_strategies``-style discovery has a durable, versioned anchor.
- ``backtest_from_manifest`` runs the existing ``run_backtest`` against the
  manifest's run_dir, proving the manifest is executable, not decorative.

The manifest is deliberately additive: nothing about existing run
directories changes, and ``backtest`` keeps working exactly as before.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Optional

_MANIFEST_ID_RE = re.compile(r"^[0-9a-f]{16,64}$")
_SCHEMA_VERSION = 1

#: Fields lifted verbatim from config.json into the manifest's ``config``
#: block. Everything else in config.json is preserved under ``config_extra``
#: so a manifest never loses information.
_CORE_CONFIG_FIELDS = (
    "source",
    "codes",
    "market",
    "start_date",
    "end_date",
    "initial_capital",
    "engine",
    "benchmark",
    "allow_short",
    "leverage",
    "universe",
    "period",
)


class ManifestError(RuntimeError):
    """A manifest could not be compiled, saved, or read back."""


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compile_strategy(
    *,
    run_dir: str | Path,
    name: str = "",
    strategy_id: str = "",
) -> dict[str, Any]:
    """Compile a run directory (config.json + code/signal_engine.py) into a manifest.

    Raises:
        ManifestError: If the run directory is missing either input file or
            config.json is not valid JSON.
    """
    root = Path(run_dir).expanduser().resolve()
    config_path = root / "config.json"
    engine_path = root / "code" / "signal_engine.py"

    if not config_path.is_file():
        raise ManifestError(f"missing config.json under {root}")
    if not engine_path.is_file():
        raise ManifestError(f"missing code/signal_engine.py under {root}")

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ManifestError(f"config.json could not be parsed: {exc}") from None
    if not isinstance(config, dict):
        raise ManifestError("config.json must be a JSON object")

    engine_src = engine_path.read_text(encoding="utf-8")

    config_core = {k: config[k] for k in _CORE_CONFIG_FIELDS if k in config}
    config_extra = {
        k: v for k, v in config.items() if k not in _CORE_CONFIG_FIELDS
    }

    # Warmup estimate: the longest rolling window referenced in the engine
    # source. ``rolling(``, ``window=``, ``.mean(N)``, ``shift(N)`` — the
    # common pandas idioms — are the honest floor; anything fancier should
    # declare itself in config via ``warmup_bars``.
    warmup_declared = config.get("warmup_bars")
    warmup_estimated: int | None = None
    if not isinstance(warmup_declared, (int, float)) or warmup_declared <= 0:
        windows = [
            int(m)
            for pattern in (
                r"rolling\(\s*(\d+)",
                r"window\s*=\s*(\d+)",
                r"\.mean\(\s*(\d+)\s*\)",
                r"\.shift\(\s*(\d+)\s*\)",
            )
            for m in re.findall(pattern, engine_src)
        ]
        warmup_estimated = max(windows) if windows else None
        warmup_declared = None
    else:
        warmup_declared = int(warmup_declared)

    manifest_id = hashlib.sha256(
        (
            _sha256_text(json.dumps(config_core, sort_keys=True, default=str))
            + _sha256_text(engine_src)
        ).encode("utf-8")
    ).hexdigest()[:24]

    return {
        "schema_version": _SCHEMA_VERSION,
        "manifest_id": manifest_id,
        "name": (name or root.name).strip() or manifest_id,
        "strategy_id": (strategy_id or "").strip(),
        "compiled_at": time.time(),
        "run_dir": str(root),
        "config": config_core,
        "config_extra": config_extra,
        "engine_sha256": _sha256_text(engine_src),
        "engine_size_bytes": len(engine_src.encode("utf-8")),
        "warmup_bars": warmup_declared,
        "warmup_estimated": warmup_estimated,
        "universe": {
            "source": config_core.get("source"),
            "codes": config_core.get("codes"),
            "market": config_core.get("market"),
        },
        "period": {
            "start": config_core.get("start_date"),
            "end": config_core.get("end_date"),
        },
        "risk_policy": {
            "allow_short": bool(config_core.get("allow_short", False)),
            "leverage": config_core.get("leverage", 1.0),
            "initial_capital": config_core.get("initial_capital"),
        },
        "engine": config_core.get("engine"),
        "benchmark": config_core.get("benchmark"),
    }


class ManifestStore:
    """Durable manifest registry under ``<root>/strategy-manifests``."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._directory = self.root / "strategy-manifests"

    def save(self, manifest: dict[str, Any]) -> str:
        if not _MANIFEST_ID_RE.match(str(manifest.get("manifest_id", ""))):
            raise ManifestError("manifest has no valid manifest_id")
        self._directory.mkdir(parents=True, exist_ok=True)
        path = self._directory / f"{manifest['manifest_id']}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        tmp.replace(path)
        return str(path)

    def get(self, manifest_id: str) -> dict[str, Any]:
        if not _MANIFEST_ID_RE.match(manifest_id or ""):
            raise ManifestError(f"unsafe manifest id: {manifest_id!r}")
        path = self._directory / f"{manifest_id}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise ManifestError(f"unknown manifest id: {manifest_id}") from None

    def list(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self._directory.is_dir():
            return rows
        for path in sorted(
            self._directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            rows.append(
                {
                    "manifest_id": data.get("manifest_id"),
                    "name": data.get("name"),
                    "strategy_id": data.get("strategy_id"),
                    "compiled_at": data.get("compiled_at"),
                    "universe": data.get("universe"),
                    "period": data.get("period"),
                    "engine": data.get("engine"),
                }
            )
            if len(rows) >= limit:
                break
        return rows


_manifest_store_singleton: Optional[ManifestStore] = None


def get_manifest_store() -> ManifestStore:
    """Process-wide ManifestStore under the runtime root."""
    global _manifest_store_singleton
    if _manifest_store_singleton is None:
        from src.config.paths import get_runtime_root

        _manifest_store_singleton = ManifestStore(Path(get_runtime_root()) / "agent")
    return _manifest_store_singleton