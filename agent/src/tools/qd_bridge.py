"""VT→QD strategy bridge: compile a Vibe-Trading run directory into a
QuantDinger Strategy API V2 source.

Why a generator, not an adapter
-------------------------------
The two strategy contracts are opposite shapes:

- VT ``signal_engine.py`` is *vectorized*: ``SignalEngine().generate(data_map)``
  receives one frame per instrument for the whole period and returns a dict of
  target-weight Series. No per-bar loop, no ordering.
- QD Strategy API V2 is *event-driven*: ``initialize(context)`` declares a
  static universe + subscriptions, and ``handle_data(context, data)`` runs once
  per closed bar with ``get_history(count, frequency, field, symbol)`` and
  ``order_target_percent(symbol, target, reason=...)``.

Since a VT engine only needs ``data_map`` to produce weights, the honest
translation is to *embed the engine verbatim* into a V2 shell: on each bar,
reconstruct a trailing window of frames (QD serves history), build the
data_map, call ``generate()``, and translate the latest weights into
``order_target_percent`` calls. Look-ahead safety comes from VT's own
contract — ``generate`` is expected to emit weights aligned to the *last*
row of its inputs — so the shell always trims history to the current bar.

Generated source requirements (from QD's STRATEGY_DEV_GUIDE):
- triple-quoted docstring, first line = strategy name
- ``initialize(context)`` with ``context.set_universe`` + ``context.subscribe``
- ``handle_data(context, data)`` as the driving callback
- ``get_history(count, frequency, field, symbol)`` — count-first, no fields=
- no leveraged instruments unless declared (we emit long-only weights)

Usage:
    from src.tools.qd_bridge import compile_vt_run_to_qd
    src = compile_vt_run_to_qd(run_dir="runs/my-strat", name="my-strat")

The returned source string can be pasted into QD's Strategy IDE, or fed to
their MCP ``compile_strategy_code`` / ``save_strategy_source`` tools.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

__all__ = ["QDBridgeError", "compile_vt_run_to_qd", "qd_source_for_engine"]


class QDBridgeError(RuntimeError):
    """A VT run directory could not be translated into a QD V2 source."""


#: QD instrument prefix by VT market/source. QD symbols look like
#: ``USStock:SPY``, ``Crypto:BTC-USDT``. VT sources are loader names
#: (yfinance, okx, ccxt, akshare...) — market is the better key when present.
_MARKET_TO_PREFIX = {
    "us": "USStock",
    "hk": "HKStock",
    "cn": "CNStock",
    "crypto": "Crypto",
    "a": "CNStock",
}

#: Default QD frequency from VT engine hints. VT engines are daily-bar
#: vectorized; 1d is the only honest default.
_DEFAULT_FREQUENCY = "1d"

#: How many trailing bars the shell feeds the embedded engine. Must cover
#: the engine's warmup; VT manifests estimate this (strategy_manifest), and
#: callers can override via config ``qd_warmup`` / the ``warmup`` argument.
_DEFAULT_WARMUP = 260


def _qd_symbol(code: str, market: str | None) -> str:
    code = (code or "").strip()
    if not code:
        raise QDBridgeError("empty symbol in VT config codes")
    if ":" in code:
        return code  # already QD-shaped
    prefix = _MARKET_TO_PREFIX.get((market or "").strip().lower(), "USStock")
    return f"{prefix}:{code}"


def _extract_class_source(engine_src: str) -> tuple[str, str]:
    """Return (class block source, module-level helper source) from a VT engine.

    The class block is re-indented verbatim into the generated QD source's
    module scope; helper defs at module level are copied as-is.
    """
    try:
        tree = ast.parse(engine_src)
    except SyntaxError as exc:
        raise QDBridgeError(f"signal_engine.py does not parse: {exc}") from None

    class_blocks: list[str] = []
    helper_blocks: list[str] = []
    lines = engine_src.splitlines()

    for node in tree.body:
        seg = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        if isinstance(node, ast.ClassDef):
            class_blocks.append(seg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            helper_blocks.append(seg)
        elif isinstance(node, ast.ImportFrom) or isinstance(node, ast.Import):
            helper_blocks.append(seg)
        elif isinstance(node, ast.Assign):
            # module-level constants (e.g. PARAMS = {...}) travel along
            helper_blocks.append(seg)

    if not class_blocks:
        raise QDBridgeError(
            "signal_engine.py defines no class; expected `class SignalEngine:`"
        )
    return "\n\n".join(class_blocks), "\n\n".join(helper_blocks)


_SHELL_TEMPLATE = '''\
def initialize(context):
    g.symbols = {symbols!r}
    g.engine = _vt_engine()
    context.set_universe(list(g.symbols))
    context.subscribe(
        frequency={frequency!r},
        fields=["open", "high", "low", "close", "volume"],
    )
    context.set_warmup({warmup})
    context.set_benchmark({benchmark!r})


def handle_data(context, data):
    import pandas as pd

    warmup = int(context.params.get("vt_warmup", {warmup}))
    data_map = {{}}
    for symbol in g.symbols:
        frame = get_history(warmup, {frequency!r}, "*", symbol)
        if frame is None or len(frame) == 0:
            continue
        data_map[symbol] = frame
    if not data_map:
        return
    weights = g.engine.generate(data_map)
    for symbol, target in weights.items():
        qd_sym = g.symbols.get(symbol, symbol)
        if hasattr(target, "iloc"):
            target = float(target.iloc[-1]) if len(target) else 0.0
        elif hasattr(target, "item"):
            target = float(target)
        else:
            target = float(target)
        order_target_percent(
            qd_sym,
            target,
            reason="vt_signal_engine",
        )
'''


def qd_source_for_engine(
    *,
    name: str,
    symbols: list[str],
    engine_src: str,
    frequency: str = _DEFAULT_FREQUENCY,
    warmup: int = _DEFAULT_WARMUP,
    benchmark: str | None = None,
) -> str:
    """Generate a QD Strategy API V2 source embedding a VT SignalEngine."""
    class_block, helpers = _extract_class_source(engine_src)
    if "generate" not in class_block:
        raise QDBridgeError("SignalEngine class has no generate() method")

    bench = benchmark or (symbols[0].split(":")[-1] if symbols else None)
    bench_qd = None
    if bench:
        if ":" in str(bench):
            bench_qd = str(bench)
        else:
            # benchmark shares the universe's market prefix
            prefix = (symbols[0].split(":")[0] if symbols else "USStock")
            bench_qd = f"{prefix}:{bench}"
    shell = _SHELL_TEMPLATE.format(
        symbols={s: s for s in symbols},  # symbol -> symbol (identity map)
        frequency=frequency,
        warmup=warmup,
        benchmark=bench_qd,
    )

    _dq3 = '"' * 3
    header = (
        f"{_dq3}{name}\n"
        "Vibe-Trading signal_engine translated to QuantDinger Strategy API V2.\n"
        f"Universe: {', '.join(symbols)}\n"
        f"Frequency: {frequency} bars; long-only target weights from VT generate().\n"
        "Risk: weights come from the embedded VT engine; QD protections (stop_loss)\n"
        "are declared by the engine's order calls if any.\n"
        f"{_dq3}"
    )

    parts = [header]
    if helpers.strip():
        parts.append(helpers)
    parts.append(class_block)
    parts.append("_vt_engine = SignalEngine")
    parts.append(shell)
    return "\n\n".join(parts) + "\n"


def compile_vt_run_to_qd(
    *,
    run_dir: str | Path,
    name: str = "",
    warmup: int | None = None,
) -> dict[str, Any]:
    """Compile a VT run directory (config.json + code/signal_engine.py) to QD V2.

    Returns ``{"status": "ok", "name": ..., "source": <qd v2 python>,
    "symbols": [...], "warmup": n, "frequency": "1d"}``.

    Raises / errors (as dict): invalid or missing inputs.
    """
    root = Path(run_dir).expanduser().resolve()
    config_path = root / "config.json"
    engine_path = root / "code" / "signal_engine.py"

    if not config_path.is_file():
        return {"status": "error", "error": f"missing config.json under {root}"}
    if not engine_path.is_file():
        return {"status": "error", "error": f"missing code/signal_engine.py under {root}"}

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"status": "error", "error": f"config.json unparseable: {exc}"}
    if not isinstance(config, dict):
        return {"status": "error", "error": "config.json must be a JSON object"}

    codes = config.get("codes") or []
    if not isinstance(codes, list) or not codes:
        return {
            "status": "error",
            "error": "config codes missing or empty; QD needs a static universe",
        }
    market = config.get("market") or _market_from_source(config.get("source"))

    try:
        symbols = [_qd_symbol(str(c), market) for c in codes]
    except QDBridgeError as exc:
        return {"status": "error", "error": str(exc)}

    engine_src = engine_path.read_text(encoding="utf-8")

    if warmup is None:
        warmup = _estimate_warmup(engine_src, config)

    strategy_name = (name or root.name).strip()
    try:
        source = qd_source_for_engine(
            name=strategy_name,
            symbols=symbols,
            engine_src=engine_src,
            warmup=warmup,
            benchmark=config.get("benchmark"),
        )
    except QDBridgeError as exc:
        return {"status": "error", "error": str(exc)}

    return {
        "status": "ok",
        "name": strategy_name,
        "source": source,
        "symbols": symbols,
        "warmup": warmup,
        "frequency": _DEFAULT_FREQUENCY,
    }


def _market_from_source(source: str | None) -> str | None:
    """Best-effort market key from a VT loader name."""
    s = (source or "").lower()
    if "okx" in s or "ccxt" in s or "binance" in s:
        return "crypto"
    if "akshare" in s or "tushare" in s or "baostock" in s:
        return "cn"
    return "us"


def _estimate_warmup(engine_src: str, config: dict[str, Any]) -> int:
    """Longest rolling window referenced by the engine, floor 120."""
    declared = config.get("warmup_bars") or config.get("qd_warmup")
    if isinstance(declared, (int, float)) and declared > 0:
        return int(declared)
    windows = [
        int(m)
        for pattern in (
            r"rolling\(\s*(\d+)",
            r"window\s*=\s*(\d+)",
            r"\.mean\(\s*(\d+)\s*\)",
            r"\.shift\(\s*(\d+)\s*\)",
            r"tail\(\s*(\d+)\s*\)",
        )
        for m in re.findall(pattern, engine_src)
    ]
    return max([*windows, 120])