"""Deterministic scheduled paper-trading pass (SMA crossover + crypto RSI).

Backs the ``paper_trade.run`` registry key: a no-LLM alternative to
prompt-dispatched autonomous trading for local models that strain on a
large eager tool schema.

Per-asset rules (fixed here; callers cannot widen them):

* Stocks/ETFs (market-hours assets): daily SMA(20)/SMA(50) crossover —
  buy a bullish cross when unheld, sell a held symbol on a bearish cross.
  Orders stay ``time_in_force="day"``.
* Crypto (24/7 assets): 4H bars, six evaluation points per day instead of
  one, so crosses are acted on within hours rather than at the next daily
  fire. Two signals, one order per pass per symbol:
  - SMA crossover: 4H SMA(30)/SMA(50) (~5d/8.3d) — same trend-following
    rule, on a faster clock.
  - RSI(14) mean-reversion on 4H closes: oversold (<=30) is a BUY even
    inside a bearish trend — it buys short-term panic the SMA rule sits
    out. There is no RSI SELL entry (no shorting); instead the bearish-SMA
    sell is gated on RSI having normalized above 45, so an oversold entry
    is never instantly churned back out while it is still washed out.
* Crypto orders use ``time_in_force="gtc"`` (Alpaca rejects ``day`` for
  crypto; run 2026-11-01: ``invalid crypto time_in_force``).

Bars come from the repository loader layer (``get_market_data`` tool —
Yahoo for US equities, OKX for crypto), NOT the broker's bars feed.
``BTC-USDT`` loader data maps to the ``BTC/USD`` Alpaca order symbol, and
held/pending lookups normalize slashes away (``BTC/USD`` == ``BTCUSD``).

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
# Stock daily SMA windows
_SMA_FAST_STOCK = 20
_SMA_SLOW_STOCK = 50
# Crypto 4H SMA windows (~30 bars = 5 days, ~50 bars = 8.3 days)
_SMA_FAST_CRYPTO = 30
_SMA_SLOW_CRYPTO = 50
_RSI_PERIOD = 14
_RSI_BUY = 30.0   # oversold → buy trigger
_RSI_EXIT = 45.0  # sell gate: a bearish SMA only sells once RSI is above this
MAX_NOTIONAL_USD = 5000.0
BARS = 120
# Alpaca crypto rejects time_in_force="day" ("invalid crypto time_in_force");
# stocks reject nothing we use, but GTC is safe for both.
_CRYPTO_TIF = "gtc"
# 4H bars needed: SMA(50) + RSI(14) warmup + headroom for trend checks
_CRYPTO_LOOKBACK_DAYS = 30


def _tif(is_crypto: bool) -> str:
    return _CRYPTO_TIF if is_crypto else "day"


def _norm(symbol: str) -> str:
    """Normalize an Alpaca symbol so ``BTC/USD`` == ``BTCUSD`` == ``btcusd``.

    Alpaca returns crypto positions without the slash (``BTCUSD``) while
    orders take the slashed form (``BTC/USD``).
    """
    return str(symbol or "").strip().upper().replace("/", "")


def _sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    window = values[-period:]
    return sum(window) / period


def _rsi(closes: list[float], period: int = _RSI_PERIOD) -> float | None:
    """Wilder RSI on a plain close list (causal, no future bars)."""
    if len(closes) < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [max(d, 0.0) for d in deltas]
    losses = [max(-d, 0.0) for d in deltas]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for gain, loss in zip(gains[period:], losses[period:]):
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


def _load_rows(loader_symbol: str, interval: str, lookback_days: int) -> list[dict]:
    """Fetch chronological OHLCV rows through the repository loader layer."""
    from src.tools.market_data_tool import MarketDataTool

    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=lookback_days)
    reply = MarketDataTool().execute(
        codes=[loader_symbol],
        start_date=start.isoformat(),
        end_date=end.isoformat(),
        source="auto",
        interval=interval,
        max_rows=0,
    )
    payload = json.loads(reply)
    if payload.get("ok") is False or payload.get("error"):
        raise RuntimeError(str(payload.get("error") or payload)[:200])
    rows = payload.get(loader_symbol) or payload.get(loader_symbol.upper())
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and isinstance(row.get("close"), (int, float))]


def _fetch_closes(loader_symbol: str, interval: str = "1D", lookback_days: int = BARS + 40) -> list[float]:
    """Fetch chronological daily closes for one symbol through the loader layer."""
    rows = _load_rows(loader_symbol, interval, lookback_days)
    return [float(row["close"]) for row in rows]


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


def _decide_stock(closes: list[float]) -> dict:
    """Daily SMA(20)/SMA(50) crossover decision for a market-hours symbol."""
    fast = _sma(closes, _SMA_FAST_STOCK)
    slow = _sma(closes, _SMA_SLOW_STOCK)
    if fast is None or slow is None:
        return {"action": "skip", "reason": "insufficient bars"}
    bullish = fast > slow
    return {"action": "hold", "bullish": bullish, "fast": fast, "slow": slow}


def _decide_crypto(closes: list[float]) -> dict:
    """4H dual-signal decision for a 24/7 symbol.

    SMA(30)/SMA(50) on 4H bars plus RSI(14) mean-reversion. Returns the
    decision dict; ``signal`` names the rule that fired, so the audit log
    shows which path acted.
    """
    fast = _sma(closes, _SMA_FAST_CRYPTO)
    slow = _sma(closes, _SMA_SLOW_CRYPTO)
    if fast is None or slow is None:
        return {"action": "skip", "reason": "insufficient bars"}
    rsi = _rsi(closes)
    if rsi is None:
        return {"action": "skip", "reason": "insufficient bars for RSI"}
    bullish = fast > slow
    # RSI oversold is the only in-downtrend entry: it buys against short-term
    # panic. Overbought is NOT an entry (shorting is out of scope); a held
    # position is only sold by the bearish-SMA rule once RSI has normalized
    # above _RSI_EXIT, so an oversold entry is never instantly churned back
    # out while it is still oversold.
    if rsi <= _RSI_BUY:
        return {"action": "hold", "bullish": True, "signal": "rsi_oversold", "rsi": rsi, "fast": fast, "slow": slow}
    if bullish:
        return {"action": "hold", "bullish": True, "signal": "sma_bullish", "rsi": rsi, "fast": fast, "slow": slow}
    if rsi < _RSI_EXIT:
        # Still washed-out momentum: stand aside whether held or not.
        return {"action": "hold", "bullish": True, "signal": "rsi_below_exit", "rsi": rsi, "fast": fast, "slow": slow}
    return {"action": "hold", "bullish": False, "signal": "sma_bearish", "rsi": rsi, "fast": fast, "slow": slow}


def run_paper_pass() -> None:
    """Execute one scheduled pass on the paper account."""
    from src.trading.service import place_order

    equity_before = _account_equity()
    positions_reply = _positions_reply()
    held = _held_map(positions_reply)
    pending = _pending_symbols()

    actions: list[dict] = []
    for loader_symbol, order_symbol, is_crypto in WATCHLIST:
        key = _norm(order_symbol)
        tif = _tif(is_crypto)
        try:
            if is_crypto:
                closes = _fetch_closes(loader_symbol, interval="4H", lookback_days=_CRYPTO_LOOKBACK_DAYS)
                decision = _decide_crypto(closes)
            else:
                closes = _fetch_closes(loader_symbol)
                decision = _decide_stock(closes)
        except Exception as exc:  # noqa: BLE001 - one symbol must not abort the pass
            actions.append({"symbol": order_symbol, "action": "skip", "reason": f"history error: {exc}"})
            continue
        if decision.get("action") == "skip":
            actions.append({"symbol": order_symbol, "crypto": is_crypto, **decision})
            continue

        bullish = bool(decision.get("bullish"))
        is_held = held.get(key, 0.0) > 0
        is_pending = key in pending
        entry = {"symbol": order_symbol, "crypto": is_crypto, **decision}

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
            actions.append({**entry, "action": "buy", "result": result})
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
                actions.append({**entry, "action": "sell", "result": result})
            else:
                actions.append({**entry, "action": "skip", "reason": "held but no quantity"})
        else:
            actions.append({**entry, "action": "hold"})

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
