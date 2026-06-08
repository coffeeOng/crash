"""Core crash-detection engine.

Computes drawdowns from a rolling all-time high (ATH) and detects the first
crossing of each of five drawdown tiers (-10%, -15%, -20%, -25%, -30%). Fired
tiers reset whenever a new ATH is reached, so each drawdown cycle is scored
independently. Historical prices are fetched from yfinance (free, no API key)
and cached in a local SQLite database to avoid redundant network calls.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

# Five crash tiers as fractional drawdowns from the all-time high.
TIERS: tuple[float, ...] = (-0.10, -0.15, -0.20, -0.25, -0.30)

# SQLite cache lives alongside this module under data/.
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DB_PATH = os.path.join(_DATA_DIR, "prices.db")

# Refetch from yfinance only if the cached series is older than this.
_MAX_CACHE_AGE = timedelta(days=1)


@dataclass(frozen=True)
class TierEvent:
    """A single tier-crossing: the first day a drawdown tier was breached."""

    ticker: str
    date: pd.Timestamp
    tier: float          # e.g. -0.20 for the -20% tier
    close: float         # closing price on the crossing day
    ath: float           # all-time high in force when the tier fired


def _ensure_db() -> sqlite3.Connection:
    os.makedirs(_DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT NOT NULL,
            date   TEXT NOT NULL,
            close  REAL NOT NULL,
            PRIMARY KEY (ticker, date)
        )
        """
    )
    conn.commit()
    return conn


def _load_cached(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    rows = conn.execute(
        "SELECT date, close FROM prices WHERE ticker = ? ORDER BY date",
        (ticker,),
    ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["close"])
    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def _cache_is_fresh(conn: sqlite3.Connection, ticker: str) -> bool:
    latest = conn.execute(
        "SELECT MAX(date) FROM prices WHERE ticker = ?", (ticker,)
    ).fetchone()[0]
    if not latest:
        return False
    latest_dt = pd.to_datetime(latest)
    return (datetime.now() - latest_dt.to_pydatetime()) < _MAX_CACHE_AGE


def _store(conn: sqlite3.Connection, ticker: str, df: pd.DataFrame) -> None:
    conn.execute("DELETE FROM prices WHERE ticker = ?", (ticker,))
    conn.executemany(
        "INSERT OR REPLACE INTO prices (ticker, date, close) VALUES (?, ?, ?)",
        [
            (ticker, idx.strftime("%Y-%m-%d"), float(close))
            for idx, close in df["close"].items()
        ],
    )
    conn.commit()


def fetch_history(ticker: str, force_refresh: bool = False) -> pd.DataFrame:
    """Return a DataFrame indexed by date with a single ``close`` column.

    Uses the SQLite cache when it is less than a day old; otherwise fetches the
    full available history from yfinance and refreshes the cache.
    """
    conn = _ensure_db()
    try:
        if not force_refresh and _cache_is_fresh(conn, ticker):
            cached = _load_cached(conn, ticker)
            if not cached.empty:
                return cached

        raw = yf.Ticker(ticker).history(period="max", auto_adjust=True)
        if raw.empty or "Close" not in raw.columns:
            # Fall back to whatever is cached rather than failing hard.
            cached = _load_cached(conn, ticker)
            if not cached.empty:
                return cached
            raise ValueError(f"No price data available for '{ticker}'.")

        df = pd.DataFrame({"close": raw["Close"].astype(float)})
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        _store(conn, ticker, df)
        return df
    finally:
        conn.close()


def with_drawdown(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``ath`` (rolling all-time high) and ``drawdown`` columns."""
    out = df.copy()
    out["ath"] = out["close"].cummax()
    out["drawdown"] = (out["close"] - out["ath"]) / out["ath"]
    return out


def tier_events(df: pd.DataFrame, ticker: str) -> list[TierEvent]:
    """Detect the first crossing of each tier within every drawdown cycle.

    A cycle begins at each new all-time high (which clears the set of already
    fired tiers) and runs until the next new high.
    """
    data = with_drawdown(df)
    events: list[TierEvent] = []
    fired: set[float] = set()
    running_ath = float("-inf")

    for idx, row in data.iterrows():
        close = float(row["close"])
        ath = float(row["ath"])
        drawdown = float(row["drawdown"])

        # New all-time high -> reset the cycle.
        if ath > running_ath:
            running_ath = ath
            fired.clear()
            continue

        # Fire each not-yet-fired tier the current drawdown has reached,
        # from shallowest to deepest.
        for tier in TIERS:
            if tier not in fired and drawdown <= tier:
                fired.add(tier)
                events.append(
                    TierEvent(
                        ticker=ticker,
                        date=idx,
                        tier=tier,
                        close=close,
                        ath=ath,
                    )
                )

    return events


def current_status(df: pd.DataFrame, ticker: str) -> dict:
    """Snapshot of the latest bar: price, drawdown and the next tier trigger."""
    data = with_drawdown(df)
    last = data.iloc[-1]
    close = float(last["close"])
    ath = float(last["ath"])
    drawdown = float(last["drawdown"])

    # Tiers already breached in the current (most recent) drawdown cycle are
    # those deeper-or-equal to the current drawdown.
    next_tier = None
    for tier in TIERS:
        if drawdown > tier:  # not yet reached this tier (drawdown is negative)
            next_tier = tier
            break

    if next_tier is None:
        trigger_price = None
        drop_to_trigger = None
    else:
        trigger_price = ath * (1 + next_tier)
        drop_to_trigger = (trigger_price - close) / close

    return {
        "ticker": ticker,
        "date": data.index[-1],
        "close": close,
        "ath": ath,
        "drawdown": drawdown,
        "next_tier": next_tier,
        "trigger_price": trigger_price,
        "drop_to_trigger": drop_to_trigger,
    }
