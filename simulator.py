"""Backtesting engine for the tiered crash-buying strategy.

Compares a *crash ladder* — deploying slices of a cash pool as the market
breaches successive drawdown tiers — against a *lump-sum* baseline that invests
the whole pool on day one of the chosen window.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from crash import TIERS, tier_events, with_drawdown


@dataclass(frozen=True)
class Trade:
    """A single ladder deployment when a tier fired inside the window."""

    date: pd.Timestamp
    tier: float
    price: float
    cash_deployed: float
    shares: float


@dataclass
class BacktestResult:
    trades: list[Trade]
    cash_pool: float
    cash_deployed: float
    cash_remaining: float
    ladder_shares: float
    ladder_avg_cost: float        # average cost basis of ladder shares (0 if none)
    lumpsum_shares: float
    lumpsum_price: float
    final_price: float
    ladder_final_value: float
    lumpsum_final_value: float
    ladder_return_pct: float
    lumpsum_return_pct: float
    diff_return_pct: float         # ladder minus lump-sum, in percentage points
    diff_return_dollars: float
    equity_curve: pd.DataFrame     # index=date, cols: ladder, lumpsum


def _slice_window(df: pd.DataFrame, start, end) -> pd.DataFrame:
    window = df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
    return window


def backtest(
    df: pd.DataFrame,
    ticker: str,
    cash_pool: float,
    ladder: dict[float, float],
    start,
    end,
) -> BacktestResult:
    """Run the crash-ladder vs lump-sum comparison over ``[start, end]``.

    ``ladder`` maps each tier (e.g. -0.10) to the fraction of the pool (0..1)
    deployed when that tier first fires inside the window. Allocations that
    never trigger remain as uninvested cash.
    """
    window = _slice_window(df, start, end)
    if window.empty:
        raise ValueError("Selected date range contains no trading days.")

    final_price = float(window["close"].iloc[-1])

    # --- Lump-sum baseline: buy everything on the first day in the window. ---
    lumpsum_price = float(window["close"].iloc[0])
    lumpsum_shares = cash_pool / lumpsum_price
    lumpsum_final_value = lumpsum_shares * final_price

    # --- Crash ladder: deploy a slice as each tier first fires in-window. ---
    # Re-run tier detection on the full series so drawdown cycles are correct,
    # then keep only the events that land inside the window.
    events = [e for e in tier_events(df, ticker) if pd.Timestamp(start) <= e.date <= pd.Timestamp(end)]

    trades: list[Trade] = []
    deployed_per_tier: set[float] = set()
    for ev in events:
        frac = ladder.get(ev.tier, 0.0)
        if frac <= 0 or ev.tier in deployed_per_tier:
            continue
        deployed_per_tier.add(ev.tier)
        cash = cash_pool * frac
        shares = cash / ev.close
        trades.append(
            Trade(date=ev.date, tier=ev.tier, price=ev.close, cash_deployed=cash, shares=shares)
        )

    cash_deployed = sum(t.cash_deployed for t in trades)
    cash_remaining = cash_pool - cash_deployed
    ladder_shares = sum(t.shares for t in trades)
    ladder_avg_cost = (cash_deployed / ladder_shares) if ladder_shares > 0 else 0.0
    ladder_final_value = ladder_shares * final_price + cash_remaining

    ladder_return_pct = (ladder_final_value - cash_pool) / cash_pool * 100
    lumpsum_return_pct = (lumpsum_final_value - cash_pool) / cash_pool * 100
    diff_return_pct = ladder_return_pct - lumpsum_return_pct
    diff_return_dollars = ladder_final_value - lumpsum_final_value

    # --- Equity curves over the window. ---
    equity = pd.DataFrame(index=window.index)
    equity["lumpsum"] = lumpsum_shares * window["close"]

    shares_held = 0.0
    cash_held = cash_pool
    trade_by_date: dict[pd.Timestamp, list[Trade]] = {}
    for t in trades:
        trade_by_date.setdefault(t.date, []).append(t)

    ladder_values = []
    for idx, row in window.iterrows():
        for t in trade_by_date.get(idx, []):
            shares_held += t.shares
            cash_held -= t.cash_deployed
        ladder_values.append(shares_held * float(row["close"]) + cash_held)
    equity["ladder"] = ladder_values

    return BacktestResult(
        trades=trades,
        cash_pool=cash_pool,
        cash_deployed=cash_deployed,
        cash_remaining=cash_remaining,
        ladder_shares=ladder_shares,
        ladder_avg_cost=ladder_avg_cost,
        lumpsum_shares=lumpsum_shares,
        lumpsum_price=lumpsum_price,
        final_price=final_price,
        ladder_final_value=ladder_final_value,
        lumpsum_final_value=lumpsum_final_value,
        ladder_return_pct=ladder_return_pct,
        lumpsum_return_pct=lumpsum_return_pct,
        diff_return_pct=diff_return_pct,
        diff_return_dollars=diff_return_dollars,
        equity_curve=equity,
    )


def default_ladder() -> dict[float, float]:
    """Even 20% allocation across all five tiers."""
    return {tier: 0.20 for tier in TIERS}
