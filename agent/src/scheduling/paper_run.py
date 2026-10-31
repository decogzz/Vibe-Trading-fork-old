"""Deterministic scheduled paper-trading pass (SMA crossover).

Backs the ``paper_trade.run`` registry key: a no-LLM alternative to
prompt-dispatched autonomous trading for local models that strain on a
large eager tool schema. The pass is a fixed pipeline:

1. Read the selected paper connector's account and open positions.
2. Pull daily closes per watchlist symbol through the repository loader
   layer (``get_market_data`` tool — Yahoo for US equities, OKX for
   crypto), NOT the broker's own bars feed.
3. Compute SMA(20)/SMA(50) on completed daily bars.
4. Buy a symbol on a bullish crossover when it is not already held; sell a
   held symbol on a bearish crossover. At most one order per symbol per
   firing, notional-capped.
5. Append a one-line audit record to the runtime log.

Symbol mapping: loaders serve ``BTC-USDT``; Alpaca trades crypto as
``BTC/USD``. ``ALPACA_SYMBOL`` maps a watchlist entry to the order symbol.

Safety: the service layer routes paper profiles straight to the broker
sandbox; live profiles would still pass the mandate gate. The watchlist and
notional cap are fixed here so a misbehaving caller cannot widen them.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

PROFILE_ID = "alpaca-paper-trade"
# (loader symbol, alpaca order symbol, is_crypto)
WATCHLIST = (
    ("BTC-USDT", "BTC/USD", True),
    ("ETH-USDT", "ETH/USD", True),
    ("SOL-USDT", "SOL/USD", True),
    ("AAPL.US", "AAPL", False),
    ("NVDA.US", "NVDA", False),
    ("SPY.US", "SPY", False),
    ("MSFT.US", "MSFT", False),
)
MAX_NOTIONAL_USD = 5000.0
BARS = 120
# Alpaca crypto rejects time_in_force="day" ("invalid crypto time_in_force");
# stocks reject nothing we use, but GTC is safe for both.
_CRYPTO_TIF = "gtc"


def _tif(is_crypto: bool) -> str:
    return _CRYPTO_TIF if is_crypto else "day"


def _closes_from_market_data(payload: dict, loader_symbol: str) -> list[float]:
    """Extract ascending daily closes for one symbol from a get_market_data reply.

    The tool returns rows keyed directly by the requested symbol:
    ``{"AAPL.US": [{...close...}], "_provenance": {...}}``. On any error the
    top level is ``{"ok": false, "error": ...}`` — the status check in
    :func:`_fetch_closes` catches that first.
    """
    if payload.get("ok") is False or payload.get("error"):
        raise RuntimeError(str(payload.get("error") or payload)[:200])
    rows = payload.get(loader_symbol) or payload.get(loader_symbol.upper())
    if rows is None:
        # tolerate a nested {"data": {...}} wrapper if one ever appears
        nested = payload.get("data")
        if isinstance(nested, dict):
            rows = nested.get(loader_symbol)
    if not isinstance(rows, list):
        return []
    closes: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        value = row.get("close")
        if value is None:
            value = row.get("Close")
        if isinstance(value, (int, float)):
            closes.append(float(value))
    return closes


def _sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / period


def _fetch_closes(loader_symbol: str) -> list[float]:
    """Fetch ~120 daily closes for one symbol through the loader layer."""
    from src.tools.market_data_tool import MarketDataTool

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=BARS + 40)  # buffer for weekends/holidays
    reply = MarketDataTool().execute(
        codes=[loader_symbol],
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        source="auto",
        interval="1D",
        max_rows=0,
    )
    payload = json.loads(reply)
    return _closes_from_market_data(payload, loader_symbol)


def _account_equity() -> float | None:
    from src.trading.service import get_account

    account = get_account(PROFILE_ID)
    inner = account.get("account") or {}
    raw = inner.get("equity")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _positions_reply() -> dict:
    from src.trading.service import get_positions

    reply = get_positions(PROFILE_ID)
    if isinstance(reply.get("positions"), list):
        return reply
    return {"positions": reply.get("positions") or reply.get("data") or []}


def _norm(symbol: str) -> str:
    """Normalize an Alpaca symbol so ``BTC/USD`` == ``BTCUSD`` == ``btcusd``.

    Alpaca returns crypto positions without the slash (``BTCUSD``) while
    orders take the slashed form (``BTC/USD``).
    """
    return str(symbol or "").strip().upper().replace("/", "")


def _held_map(positions_reply: dict) -> dict[str, float]:
    """Map normalized uppercase symbol -> held quantity."""
    held: dict[str, float] = {}
    for row in positions_reply.get("positions") or []:
        if not isinstance(row, dict):
            continue
        symbol = _norm(row.get("symbol"))
        qty = row.get("quantity")
        if qty is None:
            qty = row.get("qty")
        try:
            qty = float(qty or 0.0)
        except (TypeError, ValueError):
            qty = 0.0
        if symbol and qty:
            held[symbol] = qty
    return held


def _pending_symbols() -> set[str]:
    from src.trading.service import get_open_orders

    reply = get_open_orders(PROFILE_ID)
    pending: set[str] = set()
    for row in reply.get("open_orders") or []:
        if isinstance(row, dict):
            symbol = _norm(row.get("symbol"))
            if symbol:
                pending.add(symbol)
    return pending


def run_paper_pass() -> None:
    """Execute one scheduled SMA-crossover pass on the paper account."""
    from src.trading.service import place_order

    equity_before = _account_equity()
    positions_reply = _positions_reply()
    held = _held_map(positions_reply)
    pending = _pending_symbols()

    actions: list[dict] = []
    for loader_symbol, order_symbol, is_crypto in WATCHLIST:
        try:
            closes = _fetch_closes(loader_symbol)
        except Exception as exc:  # noqa: BLE001 - one symbol must not abort the pass
            actions.append({"symbol": loader_symbol, "action": "skip", "reason": f"history error: {exc}"})
            continue
        fast = _sma(closes, 20)
        slow = _sma(closes, 50)
        if fast is None or slow is None:
            actions.append({"symbol": loader_symbol, "action": "skip", "reason": "insufficient bars"})
            continue
        bullish = fast > slow
        key = _norm(order_symbol)
        is_held = held.get(key, 0.0) > 0
        is_pending = key in pending
        tif = _tif(is_crypto)
        if bullish and not is_held and not is_pending:
            result = place_order(
                order_symbol,
                PROFILE_ID,
                side="buy",
                notional=MAX_NOTIONAL_USD,
                order_type="market",
                time_in_force=tif,
                session_id="scheduled-paper-pass",
            )
            actions.append({"symbol": order_symbol, "action": "buy", "fast": fast, "slow": slow, "result": result})
        elif not bullish and is_held and not is_pending:
            qty = held.get(key, 0.0)
            if qty > 0:
                result = place_order(
                    order_symbol,
                    PROFILE_ID,
                    side="sell",
                    quantity=qty,
                    order_type="market",
                    time_in_force=tif,
                    session_id="scheduled-paper-pass",
                )
                actions.append({"symbol": order_symbol, "action": "sell", "fast": fast, "slow": slow, "result": result})
            else:
                actions.append({"symbol": order_symbol, "action": "skip", "reason": "held but no quantity"})
        else:
            actions.append({"symbol": order_symbol, "action": "hold", "fast": fast, "slow": slow, "crypto": is_crypto})

    record = {
        "fired_at": datetime.now(timezone.utc).isoformat(),
        "profile": PROFILE_ID,
        "equity_before": equity_before,
        "equity_after": _account_equity(),
        "actions": actions,
    }
    _append_audit(record)


def _append_audit(record: dict) -> None:
    from pathlib import Path

    log_path = Path.home() / ".vibe-trading" / "paper_pass_log.jsonl"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        # The audit log is best-effort; the orders already happened.
        pass
