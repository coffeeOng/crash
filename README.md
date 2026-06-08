# Crash Buying Simulator

A local Streamlit tool for monitoring market drawdowns and backtesting a
*tiered crash-buying* strategy: deploy slices of a cash pool as an index
breaches successive drawdown tiers (-10%, -15%, -20%, -25%, -30%) measured from
its rolling all-time high.

Rebuilt 2026-06-05 from the original `~/Desktop/Fin` build (15 May 2026).

## Stack

- **Streamlit** — web UI (Dashboard / Chart / Simulator tabs)
- **yfinance** — free historical prices (no API key)
- **pandas / numpy** — drawdown maths
- **plotly** — interactive charts
- **SQLite** — local price cache at `data/prices.db`

## Modules

| File | Role |
|------|------|
| `crash.py` | Engine: `fetch_history`, `with_drawdown`, `tier_events`, `current_status`, `TierEvent` |
| `simulator.py` | Backtest: crash-ladder vs lump-sum (`backtest`, `BacktestResult`) |
| `app.py` | Streamlit 3-tab UI |
| `smoke_test.py` | Validates tier detection against 2008 / 2020 / 2022 crashes |

## Run

```bash
cd "/Volumes/SANDISK/crash-buying-simulator"
source .venv/bin/activate          # venv built with --copies (exFAT-safe)
streamlit run app.py --server.port 8511
```

Then open http://localhost:8511.

First launch downloads full price history per ticker and caches it; later
launches read from `data/prices.db` (refetched only when >1 day stale, or via
the sidebar **Refresh** button).

## Smoke test

```bash
source .venv/bin/activate
python smoke_test.py
```

## Notes

- Drawdowns use the **rolling all-time high** as baseline (institutional
  crash-investing convention), not a 52-week or fixed window.
- Default watchlist: `SPY`, `QQQ`, `VWRA.L`, `^GSPC`.
- Local-only by design — no cloud deployment.
