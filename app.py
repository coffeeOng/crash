"""Crash Buying Simulator — Streamlit UI.

Three tabs:
  * Dashboard  — at-a-glance drawdown status for a watchlist.
  * Chart      — dual-panel price/ATH + drawdown with tier markers.
  * Simulator  — backtest a tiered crash-buying ladder vs lump-sum.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots

import crash
from crash import TIERS, current_status, fetch_history, tier_events, with_drawdown
from simulator import backtest

# Curated watchlist groups. Tickers are US-listed (or US ADRs) so yfinance
# resolves them; overlaps across groups are de-duplicated at load time.
WATCHLISTS: dict[str, list[str]] = {
    "Magnificent 7": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA"],
    "AI Upstream (chips / compute / infra)": [
        "NVDA", "TSM", "ASML", "AVGO", "AMD", "MU",
        "AMAT", "LRCX", "KLAC", "ARM", "ANET", "VRT",
    ],
    "AI Downstream (cloud / software / apps)": [
        "MSFT", "GOOGL", "AMZN", "META", "PLTR",
        "CRM", "NOW", "ADBE", "ORCL", "SNOW",
    ],
    "Thematic ETFs (sector / AI)": [
        "SMH", "SOXX", "IGV", "SKYY", "XLK", "VGT", "AIQ", "BOTZ",
    ],
    "Diversifiers (broad / international)": [
        "VOO", "VTI", "IWM", "VXUS", "VEA", "VWO",
    ],
    "Benchmarks": ["SPY", "QQQ", "VWRA.L", "^GSPC"],
}
# Show every group by default (all tickers are cached, so first load is fast).
DEFAULT_GROUPS = list(WATCHLISTS)
FALLBACK_WATCHLIST = WATCHLISTS["Magnificent 7"]

# Status levels: key -> (emoji, label, accent colour).
LEVELS: dict[str, tuple[str, str, str]] = {
    "HEALTHY": ("🟢", "Healthy", "#2f9e44"),
    "WATCH": ("🟡", "Watch", "#f08c00"),
    "CRASH": ("🔴", "Crash", "#e03131"),
}

# One-line descriptions shown on the Dashboard cards. Tickers not listed here
# (e.g. user-added extras) fall back to a best-effort name lookup via yfinance.
DESCRIPTIONS: dict[str, str] = {
    # Magnificent 7
    "AAPL": "Apple — iPhone/Mac hardware, services, on-device AI.",
    "MSFT": "Microsoft — Windows, Office, Azure cloud, Copilot (OpenAI-backed).",
    "GOOGL": "Alphabet — Google Search, YouTube, Cloud, Gemini AI.",
    "AMZN": "Amazon — e-commerce + AWS, the largest cloud platform.",
    "NVDA": "NVIDIA — dominant AI/data-centre GPUs and CUDA software.",
    "META": "Meta — Facebook/Instagram/WhatsApp, Llama models, AI ads.",
    "TSLA": "Tesla — EVs, energy storage, autonomy and robotics.",
    # AI Upstream
    "TSM": "TSMC — world's largest chip foundry; fabricates most AI silicon.",
    "ASML": "ASML — sole maker of EUV lithography for advanced chips.",
    "AVGO": "Broadcom — custom AI accelerators (ASICs) and networking silicon.",
    "AMD": "AMD — CPUs and MI-series AI GPUs; NVIDIA's main rival.",
    "MU": "Micron — memory maker; HBM is critical for AI accelerators.",
    "AMAT": "Applied Materials — largest semiconductor fab-equipment maker.",
    "LRCX": "Lam Research — wafer etch and deposition equipment.",
    "KLAC": "KLA — chip process control, metrology and inspection tools.",
    "ARM": "Arm Holdings — CPU architecture/IP licensed across the industry.",
    "ANET": "Arista Networks — high-speed switching for AI data centres.",
    "VRT": "Vertiv — data-centre power and thermal/cooling infrastructure.",
    # AI Downstream
    "PLTR": "Palantir — AI/data analytics platforms (Foundry, AIP).",
    "CRM": "Salesforce — CRM software with Agentforce AI agents.",
    "NOW": "ServiceNow — enterprise workflow automation with built-in AI.",
    "ADBE": "Adobe — creative software with Firefly generative AI.",
    "ORCL": "Oracle — databases and fast-growing AI/OCI cloud capacity.",
    "SNOW": "Snowflake — cloud data platform powering AI/ML workloads.",
    # Thematic ETFs
    "SMH": "VanEck Semiconductor ETF — top chipmakers (NVDA, TSM, AVGO).",
    "SOXX": "iShares Semiconductor ETF — broad US semiconductor basket.",
    "IGV": "iShares Expanded Tech-Software ETF — enterprise/cloud software.",
    "SKYY": "First Trust Cloud Computing ETF — cloud infrastructure & SaaS.",
    "XLK": "Technology Select Sector SPDR — large-cap US tech.",
    "VGT": "Vanguard Information Technology ETF — broad US tech sector.",
    "AIQ": "Global X Artificial Intelligence & Technology ETF.",
    "BOTZ": "Global X Robotics & Artificial Intelligence ETF.",
    # Diversifiers
    "VOO": "Vanguard S&P 500 ETF — low-fee S&P 500 exposure.",
    "VTI": "Vanguard Total Stock Market ETF — the entire US market.",
    "IWM": "iShares Russell 2000 ETF — US small-cap benchmark.",
    "VXUS": "Vanguard Total International Stock ETF — global ex-US equities.",
    "VEA": "Vanguard FTSE Developed Markets ETF — developed markets ex-US.",
    "VWO": "Vanguard FTSE Emerging Markets ETF — emerging-market equities.",
    # Benchmarks
    "SPY": "SPDR S&P 500 ETF — broad US large-cap benchmark.",
    "QQQ": "Invesco QQQ — Nasdaq-100, tech-heavy benchmark.",
    "VWRA.L": "Vanguard FTSE All-World (acc, LSE) — global equity benchmark.",
    "^GSPC": "S&P 500 index — the underlying index itself.",
}

# Tier colours: shallow (yellow) -> deep (dark red).
TIER_COLORS = {
    -0.10: "#f4c20d",
    -0.15: "#f39200",
    -0.20: "#e8590c",
    -0.25: "#c92a2a",
    -0.30: "#7a0c0c",
}

st.set_page_config(page_title="Crash Buying Simulator", page_icon="📉", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load(ticker: str, force: bool = False) -> pd.DataFrame:
    return fetch_history(ticker, force_refresh=force)


def fmt_money(x: float) -> str:
    return f"${x:,.2f}"


def fmt_pct(x: float) -> str:
    return f"{x * 100:+.1f}%"


@st.cache_data(ttl=86400, show_spinner=False)
def company_name(ticker: str) -> str:
    """Best-effort company name for tickers without a curated description."""
    try:
        info = yf.Ticker(ticker).info or {}
        return info.get("longName") or info.get("shortName") or ""
    except Exception:  # noqa: BLE001 — name is cosmetic; never block a card
        return ""


def describe(ticker: str) -> str:
    return DESCRIPTIONS.get(ticker) or company_name(ticker)


def classify(dd: float) -> str:
    """Map a drawdown to a status level."""
    if dd > -0.10:
        return "HEALTHY"
    if dd > -0.20:
        return "WATCH"
    return "CRASH"


def card_data(ticker: str, force: bool) -> dict:
    """Compute everything a card needs, so filtering can happen before render."""
    try:
        status = current_status(load(ticker, force), ticker)
        return {
            "ticker": ticker,
            "level": classify(status["drawdown"]),
            "status": status,
            "desc": describe(ticker),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — keep one bad ticker from breaking the board
        return {"ticker": ticker, "level": None, "status": None, "desc": "", "error": str(exc)}


CARD_CSS = """
<style>
.cbs-card {
    border: 1px solid rgba(128,128,128,0.22);
    border-left: 5px solid var(--accent);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 14px;
    background: linear-gradient(180deg, rgba(128,128,128,0.06), rgba(128,128,128,0.02));
    min-height: 168px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.18);
}
.cbs-head { display:flex; align-items:center; justify-content:space-between; }
.cbs-ticker { font-size: 1.12rem; font-weight: 800; letter-spacing: 0.3px; }
.cbs-badge {
    font-size: 0.68rem; font-weight: 700; color: #fff;
    padding: 2px 9px; border-radius: 999px; background: var(--accent);
    text-transform: uppercase; letter-spacing: 0.4px; white-space: nowrap;
}
.cbs-desc { font-size: 0.76rem; opacity: 0.62; line-height: 1.28; margin: 6px 0 10px; min-height: 2.6em; }
.cbs-price { font-size: 1.5rem; font-weight: 800; line-height: 1.1; }
.cbs-dd { font-size: 0.92rem; font-weight: 700; margin-left: 6px; }
.cbs-meta { font-size: 0.72rem; opacity: 0.6; margin-top: 4px; line-height: 1.3; }
.cbs-bar { height: 6px; border-radius: 999px; background: rgba(128,128,128,0.18); margin-top: 10px; overflow: hidden; }
.cbs-bar-fill { height: 100%; border-radius: 999px; background: var(--accent); }
.cbs-chip {
    display:inline-block; font-size:0.82rem; font-weight:700; color:#fff;
    padding:4px 12px; border-radius:999px; margin-right:8px;
}
</style>
"""


def render_card(cd: dict) -> None:
    """Render one pre-computed card."""
    ticker = cd["ticker"]
    if cd["error"]:
        st.markdown(
            f"<div class='cbs-card' style='--accent:#868e96'>"
            f"<div class='cbs-head'><span class='cbs-ticker'>{ticker}</span>"
            f"<span class='cbs-badge' style='background:#868e96'>N/A</span></div>"
            f"<div class='cbs-meta'>Could not load: {cd['error']}</div></div>",
            unsafe_allow_html=True,
        )
        return

    status, level = cd["status"], cd["level"]
    _, label, color = LEVELS[level]
    dd = status["drawdown"]
    dd_color = "#e03131" if dd < 0 else "#2f9e44"
    depth_pct = min(abs(dd) / 0.30, 1.0) * 100  # crash-depth meter: 0% .. -30%

    if status["next_tier"] is not None:
        meta = (f"ATH {fmt_money(status['ath'])} · next {int(status['next_tier'] * 100)}% "
                f"@ {fmt_money(status['trigger_price'])} ({fmt_pct(status['drop_to_trigger'])})")
    else:
        meta = f"ATH {fmt_money(status['ath'])} · all tiers breached this cycle"

    st.markdown(
        f"<div class='cbs-card' style='--accent:{color}'>"
        f"<div class='cbs-head'>"
        f"<span class='cbs-ticker'>{ticker}</span>"
        f"<span class='cbs-badge'>{label}</span></div>"
        f"<div class='cbs-desc'>{cd['desc']}</div>"
        f"<div class='cbs-price'>{fmt_money(status['close'])}"
        f"<span class='cbs-dd' style='color:{dd_color}'>{fmt_pct(dd)}</span></div>"
        f"<div class='cbs-meta'>{meta}</div>"
        f"<div class='cbs-bar'><div class='cbs-bar-fill' style='width:{depth_pct:.0f}%'></div></div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar — watchlist editor + refresh                                        #
# --------------------------------------------------------------------------- #
st.sidebar.title("📉 Crash Buying Simulator")
st.sidebar.caption("Deploy cash into market drawdowns, tier by tier.")

groups = st.sidebar.multiselect(
    "Watchlist groups",
    options=list(WATCHLISTS),
    default=DEFAULT_GROUPS,
    help="Which categories to show on the Dashboard. All are shown by default.",
)
extra_raw = st.sidebar.text_area(
    "Extra tickers (one per line)", value="", height=80,
)

# Extra tickers the user typed that aren't already in a selected group.
group_tickers = {t for g in groups for t in WATCHLISTS[g]}
extra_tickers: list[str] = []
for line in extra_raw.splitlines():
    t = line.strip()
    if t and t not in group_tickers and t not in extra_tickers:
        extra_tickers.append(t)

# Flat, de-duplicated union used by the Chart and Simulator selectboxes.
watchlist: list[str] = []
for g in groups:
    for t in WATCHLISTS[g]:
        if t not in watchlist:
            watchlist.append(t)
for t in extra_tickers:
    if t not in watchlist:
        watchlist.append(t)

force_refresh = st.sidebar.button("🔄 Refresh prices (bypass cache)")
if force_refresh:
    load.clear()

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tiers: " + "  ".join(f"{int(t * 100)}%" for t in TIERS)
    + "\n\nData: yfinance · cached in data/prices.db"
)

dashboard_tab, chart_tab, sim_tab = st.tabs(["📊 Dashboard", "📈 Chart", "🧪 Simulator"])


# --------------------------------------------------------------------------- #
# Dashboard                                                                   #
# --------------------------------------------------------------------------- #
with dashboard_tab:
    st.markdown(CARD_CSS, unsafe_allow_html=True)

    # Build the ordered list of (section title, tickers) to display.
    sections: list[tuple[str, list[str]]] = [(g, WATCHLISTS[g]) for g in groups]
    if extra_tickers:
        sections.append(("Custom", extra_tickers))

    if not sections:
        st.info("Select at least one watchlist group (or add an extra ticker) in the sidebar.")
    else:
        # Compute each unique ticker's status once, then reuse for summary + cards.
        all_tickers = list(dict.fromkeys(t for _, ts in sections for t in ts))
        status_map = {t: card_data(t, force_refresh) for t in all_tickers}

        # --- Summary header: counts per status across all unique tickers. ---
        counts = {lvl: 0 for lvl in LEVELS}
        for t in all_tickers:
            lvl = status_map[t]["level"]
            if lvl:
                counts[lvl] += 1

        top = st.columns([3, 2])
        with top[0]:
            st.subheader("Current drawdown status")
            chips = "".join(
                f"<span class='cbs-chip' style='background:{LEVELS[lvl][2]}'>"
                f"{LEVELS[lvl][0]} {counts[lvl]} {LEVELS[lvl][1]}</span>"
                for lvl in LEVELS
            )
            st.markdown(
                f"<div style='margin:2px 0 6px'>{chips}"
                f"<span style='opacity:0.6;font-size:0.82rem'>"
                f"&nbsp;· {len(all_tickers)} stocks tracked</span></div>",
                unsafe_allow_html=True,
            )
        with top[1]:
            options = [f"{e} {lbl}" for e, lbl, _ in LEVELS.values()]
            opt_to_level = {f"{e} {lbl}": lvl for lvl, (e, lbl, _) in LEVELS.items()}
            picked = st.pills(
                "Filter by status", options, selection_mode="multi",
                default=options, key="status_filter",
            )
            selected_levels = {opt_to_level[o] for o in picked} or set(LEVELS)

        st.caption("🟢 Healthy > -10%  ·  🟡 Watch -10% to -20%  ·  🔴 Crash < -20% from all-time high")

        # --- Categorised, filtered sections. Empty sections are hidden. ---
        any_shown = False
        for title, tickers in sections:
            visible = [t for t in tickers if status_map[t]["level"] in selected_levels
                       or status_map[t]["error"]]
            if not visible:
                continue
            any_shown = True
            st.markdown(f"#### {title}")
            cols = st.columns(4)
            for i, ticker in enumerate(visible):
                with cols[i % 4]:
                    render_card(status_map[ticker])

        if not any_shown:
            st.info("No stocks match the selected status filter.")


# --------------------------------------------------------------------------- #
# Chart                                                                       #
# --------------------------------------------------------------------------- #
with chart_tab:
    st.subheader("Price, all-time high & drawdown")
    ticker = st.selectbox("Ticker", watchlist or FALLBACK_WATCHLIST, key="chart_ticker")
    if ticker:
        try:
            df = with_drawdown(load(ticker, force_refresh))
            events = tier_events(load(ticker, force_refresh), ticker)

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                row_heights=[0.65, 0.35],
                subplot_titles=("Price vs all-time high", "Drawdown from ATH"),
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(color="#1c7ed6")),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df["ath"], name="ATH",
                           line=dict(color="#adb5bd", dash="dot")),
                row=1, col=1,
            )

            # Tier-crossing markers on the price panel.
            for tier in TIERS:
                pts = [e for e in events if e.tier == tier]
                if pts:
                    fig.add_trace(
                        go.Scatter(
                            x=[e.date for e in pts], y=[e.close for e in pts],
                            mode="markers", name=f"{int(tier * 100)}% tier",
                            marker=dict(symbol="triangle-down", size=11,
                                        color=TIER_COLORS[tier]),
                        ),
                        row=1, col=1,
                    )

            fig.add_trace(
                go.Scatter(x=df.index, y=df["drawdown"] * 100, name="Drawdown %",
                           line=dict(color="#e8590c"), fill="tozeroy"),
                row=2, col=1,
            )
            for tier in TIERS:
                fig.add_hline(y=tier * 100, line=dict(color=TIER_COLORS[tier], dash="dash"),
                              row=2, col=1)

            fig.update_layout(height=640, hovermode="x unified",
                              legend=dict(orientation="h", y=1.08))
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Drawdown %", row=2, col=1)
            st.plotly_chart(fig, width="stretch")

            st.caption(f"{len(events)} tier-crossing events across the full history.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load {ticker}: {exc}")


# --------------------------------------------------------------------------- #
# Simulator                                                                   #
# --------------------------------------------------------------------------- #
with sim_tab:
    st.subheader("Backtest: crash ladder vs lump-sum")
    ticker = st.selectbox("Ticker", watchlist or FALLBACK_WATCHLIST, key="sim_ticker")

    try:
        df = load(ticker, force_refresh)
        min_d, max_d = df.index.min().date(), df.index.max().date()

        left, right = st.columns([1, 1])
        with left:
            cash_pool = st.number_input("Cash pool ($)", min_value=100.0,
                                        value=100_000.0, step=1_000.0)
            default_start = max(min_d, date(2007, 1, 1))
            date_range = st.date_input(
                "Backtest window", value=(default_start, max_d),
                min_value=min_d, max_value=max_d,
            )
        with right:
            st.caption("Ladder — % of pool deployed per tier")
            ladder = {}
            for tier in TIERS:
                ladder[tier] = st.slider(
                    f"{int(tier * 100)}% tier", min_value=0, max_value=100,
                    value=20, step=5, key=f"ladder_{tier}",
                ) / 100.0

        total_alloc = sum(ladder.values())
        if total_alloc > 1.0001:
            st.warning(f"Ladder allocates {total_alloc * 100:.0f}% of the pool "
                       "(>100%). Reduce some tiers.")

        if st.button("▶ Run backtest", type="primary") and isinstance(date_range, tuple) \
                and len(date_range) == 2:
            start, end = date_range
            result = backtest(df, ticker, cash_pool, ladder, start, end)

            m1, m2, m3 = st.columns(3)
            m1.metric("Crash ladder", fmt_money(result.ladder_final_value),
                      f"{result.ladder_return_pct:+.1f}%")
            m2.metric("Lump-sum", fmt_money(result.lumpsum_final_value),
                      f"{result.lumpsum_return_pct:+.1f}%")
            m3.metric("Ladder − Lump-sum", fmt_money(result.diff_return_dollars),
                      f"{result.diff_return_pct:+.1f} pts")

            c1, c2, c3 = st.columns(3)
            c1.caption(f"Cash deployed: {fmt_money(result.cash_deployed)} "
                       f"of {fmt_money(result.cash_pool)}")
            c2.caption(f"Ladder shares: {result.ladder_shares:,.2f} @ avg "
                       f"{fmt_money(result.ladder_avg_cost)}")
            c3.caption(f"Idle cash at end: {fmt_money(result.cash_remaining)}")

            eq = go.Figure()
            eq.add_trace(go.Scatter(x=result.equity_curve.index,
                                    y=result.equity_curve["ladder"],
                                    name="Crash ladder", line=dict(color="#2f9e44")))
            eq.add_trace(go.Scatter(x=result.equity_curve.index,
                                    y=result.equity_curve["lumpsum"],
                                    name="Lump-sum", line=dict(color="#1c7ed6")))
            eq.add_hline(y=cash_pool, line=dict(color="#adb5bd", dash="dot"),
                         annotation_text="Initial pool")
            eq.update_layout(height=420, hovermode="x unified",
                             legend=dict(orientation="h", y=1.1),
                             yaxis_title="Portfolio value ($)")
            st.plotly_chart(eq, width="stretch")

            if result.trades:
                trades_df = pd.DataFrame([
                    {
                        "Date": t.date.date(),
                        "Tier": f"{int(t.tier * 100)}%",
                        "Price": round(t.price, 2),
                        "Cash deployed": round(t.cash_deployed, 2),
                        "Shares": round(t.shares, 4),
                    }
                    for t in result.trades
                ])
                with st.expander(f"Trade log ({len(result.trades)} deployments)", expanded=True):
                    st.dataframe(trades_df, width="stretch", hide_index=True)
            else:
                st.info("No tiers fired inside this window — ladder stayed in cash.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Simulator error for {ticker}: {exc}")
