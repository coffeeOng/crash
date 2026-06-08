"""Validate the crash engine against three known SPY drawdown events.

Mirrors the original validation: 2008 GFC fired all five tiers, the 2020 COVID
crash fired all five in weeks, and the 2022 selloff reached -20%.
"""

from __future__ import annotations

from crash import TIERS, current_status, fetch_history, tier_events


def main() -> None:
    ticker = "SPY"
    print(f"Fetching {ticker} history...")
    df = fetch_history(ticker)
    print(f"  {len(df):,} rows, {df.index.min().date()} -> {df.index.max().date()}")

    events = tier_events(df, ticker)
    print(f"  {len(events)} tier-crossing events across full history\n")

    def tiers_in(year: int) -> set[float]:
        return {e.tier for e in events if e.date.year == year}

    for year, label in [(2008, "GFC"), (2020, "COVID"), (2022, "rate hikes")]:
        fired = sorted(tiers_in(year))
        pretty = ", ".join(f"{int(t * 100)}%" for t in fired) or "none"
        print(f"  {year} ({label}): tiers fired -> {pretty}")

    status = current_status(df, ticker)
    print(
        f"\n  Current {ticker}: {status['close']:.2f} "
        f"({status['drawdown'] * 100:+.1f}% from ATH {status['ath']:.2f})"
    )
    if status["next_tier"] is not None:
        print(
            f"  Next tier {int(status['next_tier'] * 100)}% at "
            f"{status['trigger_price']:.2f} "
            f"({status['drop_to_trigger'] * 100:+.1f}% from here)"
        )

    # Basic assertions: 2008 and 2020 should each fire all five tiers.
    assert len(TIERS) == 5
    assert tiers_in(2008) == set(TIERS), "Expected all 5 tiers in 2008"
    assert tiers_in(2020) == set(TIERS), "Expected all 5 tiers in 2020"
    print("\n✅ Smoke test passed.")


if __name__ == "__main__":
    main()
