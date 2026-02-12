"""
KiteConnect strategy lab
------------------------
Purpose:
- Pull historical OHLCV from Kite Connect
- Run multiple candidate strategies (including 20 DMA)
- Generate entry/exit signals for a chosen stock
- Compare strategy performance over recent data

You can run this directly from PyCharm.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from kiteconnect import KiteConnect


# ==============================
# 1) USER CONFIGURATION
# ==============================
API_KEY = "YOUR_API_KEY"
ACCESS_TOKEN = "YOUR_ACCESS_TOKEN"

# Example instrument token for NSE:INFY. Replace with your stock token.
INSTRUMENT_TOKEN = 408065  # INFY sample; update this for your stock.

# Historical settings
INTERVAL = "day"  # "day", "15minute", etc.
LOOKBACK_DAYS = 180

# If you want only the most recent week report, keep this True.
REPORT_LAST_WEEK = True


@dataclass
class Trade:
    strategy: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    return_pct: float
    reason: str


def get_kite_client(api_key: str, access_token: str) -> KiteConnect:
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)
    return kite


def fetch_ohlcv(
    kite: KiteConnect,
    instrument_token: int,
    interval: str,
    lookback_days: int,
) -> pd.DataFrame:
    to_date = datetime.now()
    from_date = to_date - timedelta(days=lookback_days)

    rows = kite.historical_data(
        instrument_token=instrument_token,
        from_date=from_date,
        to_date=to_date,
        interval=interval,
        oi=False,
    )
    if not rows:
        raise ValueError("No historical data returned. Check token/interval/access.")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Moving averages
    out["dma20"] = out["close"].rolling(20).mean()
    out["ema9"] = out["close"].ewm(span=9, adjust=False).mean()
    out["ema21"] = out["close"].ewm(span=21, adjust=False).mean()

    # ATR(14)
    hl = out["high"] - out["low"]
    hc = (out["high"] - out["close"].shift(1)).abs()
    lc = (out["low"] - out["close"].shift(1)).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    out["atr14"] = tr.rolling(14).mean()

    # RSI(14)
    delta = out["close"].diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(14).mean()
    avg_loss = pd.Series(loss).rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out["rsi14"] = 100 - (100 / (1 + rs))

    # Volume average
    out["vol20"] = out["volume"].rolling(20).mean()

    return out


def backtest_long_only(
    df: pd.DataFrame,
    strategy_name: str,
    entry_col: str,
    exit_col: str,
    stop_atr_mult: float = 1.5,
    target_rr: float = 2.0,
) -> Tuple[pd.DataFrame, List[Trade], Dict[str, float]]:
    data = df.copy()
    data["position"] = 0

    in_trade = False
    entry_price = None
    entry_time = None
    stop_price = None
    target_price = None

    trades: List[Trade] = []

    for i in range(len(data)):
        row = data.iloc[i]
        if np.isnan(row.get("atr14", np.nan)):
            continue

        if not in_trade and bool(row[entry_col]):
            in_trade = True
            entry_price = float(row["close"])
            entry_time = row["date"]
            risk = row["atr14"] * stop_atr_mult
            stop_price = entry_price - risk
            target_price = entry_price + (risk * target_rr)
            data.at[i, "position"] = 1
            continue

        if in_trade:
            data.at[i, "position"] = 1
            exit_signal = bool(row[exit_col])
            hit_stop = row["low"] <= stop_price
            hit_target = row["high"] >= target_price

            reason = None
            exit_price = None

            if hit_stop:
                reason = "stop_loss"
                exit_price = stop_price
            elif hit_target:
                reason = "target"
                exit_price = target_price
            elif exit_signal:
                reason = "strategy_exit"
                exit_price = float(row["close"])

            if reason is not None:
                ret = (exit_price - entry_price) / entry_price * 100
                trades.append(
                    Trade(
                        strategy=strategy_name,
                        entry_time=pd.Timestamp(entry_time),
                        entry_price=float(entry_price),
                        exit_time=pd.Timestamp(row["date"]),
                        exit_price=float(exit_price),
                        return_pct=float(ret),
                        reason=reason,
                    )
                )
                in_trade = False
                entry_price = entry_time = stop_price = target_price = None

    stats = compute_stats(trades)
    return data, trades, stats


def compute_stats(trades: List[Trade]) -> Dict[str, float]:
    if not trades:
        return {
            "trades": 0,
            "win_rate_pct": 0.0,
            "avg_return_pct": 0.0,
            "total_return_pct": 0.0,
            "expectancy_pct": 0.0,
        }

    rets = np.array([t.return_pct for t in trades], dtype=float)
    wins = rets[rets > 0]
    losses = rets[rets <= 0]

    win_rate = (len(wins) / len(rets)) * 100
    avg_return = np.mean(rets)
    total_return = np.sum(rets)

    avg_win = np.mean(wins) if len(wins) else 0.0
    avg_loss = np.mean(losses) if len(losses) else 0.0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    return {
        "trades": int(len(rets)),
        "win_rate_pct": float(win_rate),
        "avg_return_pct": float(avg_return),
        "total_return_pct": float(total_return),
        "expectancy_pct": float(expectancy),
    }


def build_strategy_signals(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    data = df.copy()

    # Strategy A: 20 DMA trend-follow breakout + RSI + volume confirmation
    a = data.copy()
    a["entry"] = (
        (a["close"] > a["dma20"])
        & (a["close"].shift(1) <= a["dma20"].shift(1))
        & (a["rsi14"] > 55)
        & (a["volume"] > a["vol20"])
    )
    a["exit"] = (a["close"] < a["dma20"]) | (a["rsi14"] < 45)

    # Strategy B: EMA9/EMA21 crossover
    b = data.copy()
    b["entry"] = (b["ema9"] > b["ema21"]) & (b["ema9"].shift(1) <= b["ema21"].shift(1))
    b["exit"] = (b["ema9"] < b["ema21"]) & (b["ema9"].shift(1) >= b["ema21"].shift(1))

    # Strategy C: Mean-reversion on pullback above 20DMA
    c = data.copy()
    c["entry"] = (
        (c["close"] > c["dma20"])
        & (c["rsi14"] < 40)
        & (c["close"] < c["ema9"])
    )
    c["exit"] = (c["rsi14"] > 55) | (c["close"] > c["ema9"])

    return {
        "DMA20_Breakout": a,
        "EMA_Crossover": b,
        "Pullback_Reversion": c,
    }


def summarize_week(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=["week", "trades", "avg_return_pct", "total_return_pct"])

    tdf = pd.DataFrame([t.__dict__ for t in trades])
    tdf["week"] = tdf["entry_time"].dt.to_period("W").astype(str)

    weekly = (
        tdf.groupby("week", as_index=False)
        .agg(trades=("return_pct", "count"), avg_return_pct=("return_pct", "mean"), total_return_pct=("return_pct", "sum"))
        .sort_values("week")
    )
    return weekly


def main() -> None:
    kite = get_kite_client(API_KEY, ACCESS_TOKEN)
    raw = fetch_ohlcv(kite, INSTRUMENT_TOKEN, INTERVAL, LOOKBACK_DAYS)
    data = add_indicators(raw)

    strategy_data = build_strategy_signals(data)

    all_stats = []
    all_trades = []

    for name, sdf in strategy_data.items():
        _, trades, stats = backtest_long_only(
            sdf,
            strategy_name=name,
            entry_col="entry",
            exit_col="exit",
            stop_atr_mult=1.5,
            target_rr=2.0,
        )

        stats_row = {"strategy": name, **stats}
        all_stats.append(stats_row)
        all_trades.extend(trades)

    stats_df = pd.DataFrame(all_stats).sort_values("expectancy_pct", ascending=False)
    trades_df = pd.DataFrame([t.__dict__ for t in all_trades]) if all_trades else pd.DataFrame()

    if REPORT_LAST_WEEK and not trades_df.empty:
        max_entry = trades_df["entry_time"].max()
        cutoff = max_entry - pd.Timedelta(days=7)
        trades_df = trades_df[trades_df["entry_time"] >= cutoff].copy()

    print("\n=== Strategy Ranking (best first by expectancy) ===")
    print(stats_df.to_string(index=False))

    if not trades_df.empty:
        print("\n=== Recent Trades ===")
        print(trades_df.sort_values(["strategy", "entry_time"]).to_string(index=False))

        week_summary = summarize_week([Trade(**row) for row in trades_df.to_dict("records")])
        print("\n=== Weekly Summary ===")
        print(week_summary.to_string(index=False))

        trades_df.to_csv("strategy_trades.csv", index=False)
        print("\nSaved: strategy_trades.csv")
    else:
        print("\nNo trades generated in selected period.")

    stats_df.to_csv("strategy_stats.csv", index=False)
    print("Saved: strategy_stats.csv")


if __name__ == "__main__":
    main()
