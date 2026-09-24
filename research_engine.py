#!/usr/bin/env python3
"""Research-only SPY intraday backtesting engine.

No trading/execution code is present here. All entries are simulated at the
next bar open after a completed-bar signal.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TZ = "America/New_York"
SYMBOL = "SPY"
DEFAULT_START = "2020-01-01"
DEFAULT_END = None


@dataclass
class Config:
    initial_capital: float = 10_000.0
    risk_per_trade: float = 0.005
    max_position_value: float = 2_500.0
    slippage_bps: float = 5.0
    commission_per_share: float = 0.0
    max_trades_per_day: int = 6
    max_daily_loss_pct: float = 0.01
    atr_period: int = 14
    risk_atr_multiple: float = 1.5
    reward_risk: float = 2.0
    session_start: str = "09:30"
    session_end: str = "16:00"
    flat_time: str = "15:55"


@dataclass
class Signal:
    side: int  # +1 long, -1 short
    stop_atr: float
    reward_risk: float
    max_hold_bars: int
    exit_on_opposite: bool = True
    target_to_vwap: bool = False


@dataclass
class Position:
    side: int
    qty: int
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    target_price: float
    strategy: str
    entry_signal_time: pd.Timestamp
    entry_reason: str
    max_hold_bars: int
    bars_held: int = 0


@dataclass
class Trade:
    strategy: str
    side: str
    signal_time: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    qty: int
    pnl: float
    pnl_pct: float
    exit_reason: str
    slippage_cost: float
    commission: float
    holding_bars: int
    entry_regime: str
    entry_time_of_day: str


# ----------------------------- data ---------------------------------

def _timeframe(freq: str):
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    f = freq.lower().replace("min", "m")
    if f.endswith("m"):
        return TimeFrame(int(f[:-1]), TimeFrameUnit.Minute)
    if f.endswith("h"):
        return TimeFrame(int(f[:-1]), TimeFrameUnit.Hour)
    if f in ("1d", "day"):
        return TimeFrame.Day
    raise ValueError(f"Unsupported timeframe: {freq}")


def _clean_bars(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if isinstance(df.index, pd.MultiIndex):
        names = list(df.index.names)
        if "symbol" in names:
            df = df.xs(SYMBOL, level="symbol")
        else:
            df = df.reset_index().set_index("timestamp")
    df = df.copy()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(TZ)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    keep = [c for c in ["open", "high", "low", "close", "volume", "trade_count", "vwap"] if c in df.columns]
    return df[keep].astype(float)


def fetch_alpaca_bars(
    start: str,
    end: str,
    timeframe: str = "5m",
    feed: str = "iex",
    cache_dir: str = "data_cache",
    no_cache: bool = False,
) -> pd.DataFrame:
    """Fetch in <=90-day chunks, keeping requests comfortably below 10k bars."""
    api_key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not api_key or not secret:
        raise RuntimeError("ALPACA_API_KEY and ALPACA_SECRET_KEY are required")

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    tf = timeframe.lower()
    safe_start = start.replace(":", "").replace("-", "")
    safe_end = end.replace(":", "").replace("-", "")
    feed_name = feed.value if hasattr(feed, "value") else str(feed)
    cache_file = cache / f"{SYMBOL}_{tf}_{safe_start}_{safe_end}_{feed_name}.csv"
    if cache_file.exists() and not no_cache:
        return pd.read_csv(cache_file, index_col=0, parse_dates=True).pipe(_clean_bars)

    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    feed_enum = DataFeed.IEX if isinstance(feed, str) and feed.lower() == "iex" else DataFeed.SIP if isinstance(feed, str) and feed.lower() == "sip" else feed
    client = StockHistoricalDataClient(api_key, secret)
    start_dt = pd.Timestamp(start, tz="UTC")
    end_dt = pd.Timestamp(end, tz="UTC")
    frames: List[pd.DataFrame] = []
    chunk = timedelta(days=90) if "m" in tf else timedelta(days=365)
    cur = start_dt
    while cur < end_dt:
        nxt = min(cur + chunk, end_dt)
        req = StockBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=_timeframe(tf),
            start=cur.to_pydatetime(),
            end=nxt.to_pydatetime(),
            feed=feed_enum,
            limit=10_000,
        )
        print(f"Fetching {tf}: {cur.date()} -> {nxt.date()}")
        df = client.get_stock_bars(req).df
        if not df.empty:
            frames.append(_clean_bars(df))
        cur = nxt
    if not frames:
        raise RuntimeError("No bars returned for requested range")
    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    out.to_csv(cache_file)
    return out


def regular_session(df: pd.DataFrame, start: str = "09:30", end: str = "16:00") -> pd.DataFrame:
    idx = df.index
    if idx.tz is None:
        idx = idx.tz_localize(TZ)
    else:
        idx = idx.tz_convert(TZ)
    out = df.copy()
    out.index = idx
    return out.between_time(start, end, inclusive="left")


def resample_ohlcv(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    rule = f"{minutes}min"
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    if "trade_count" in df.columns:
        agg["trade_count"] = "sum"
    out = df.resample(rule, origin="start_day", offset="30min").agg(agg).dropna(subset=["open", "high", "low", "close"])
    return out


# ----------------------------- features ---------------------------------

def add_features(df: pd.DataFrame, base_5m: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    x = df.copy()
    prev_close = x["close"].shift(1)
    tr = pd.concat([(x["high"] - x["low"]), (x["high"] - prev_close).abs(), (x["low"] - prev_close).abs()], axis=1).max(axis=1)
    x["tr"] = tr
    x["atr"] = tr.rolling(14, min_periods=14).mean()
    x["ema20"] = x["close"].ewm(span=20, adjust=False).mean()
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    x["sma200"] = x["close"].rolling(200, min_periods=200).mean()
    delta = x["close"].diff()
    up = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    x["rsi"] = 100 - (100 / (1 + rs))
    x["avg_volume20"] = x["volume"].rolling(20, min_periods=20).mean()
    x["rvol"] = x["volume"] / x["avg_volume20"].replace(0, np.nan)

    session = x.index.date
    typical = (x["high"] + x["low"] + x["close"]) / 3
    pv = typical * x["volume"]
    x["session_vwap"] = pv.groupby(session).cumsum() / x["volume"].groupby(session).cumsum().replace(0, np.nan)
    x["vwap_dist_atr"] = (x["close"] - x["session_vwap"]) / x["atr"].replace(0, np.nan)

    # Opening range: first 30 minutes (6 x 5-minute bars).
    minute_of_day = x.index.hour * 60 + x.index.minute
    x["or_high"] = np.nan
    x["or_low"] = np.nan
    for d, g in x.groupby(x.index.date):
        first = g.between_time("09:30", "09:59", inclusive="both")
        if first.empty:
            continue
        x.loc[g.index, "or_high"] = first["high"].max()
        x.loc[g.index, "or_low"] = first["low"].min()

    daily_close = x["close"].groupby(x.index.date).last().shift(1)
    session_open = x["open"].groupby(x.index.date).first()
    gaps = (session_open / daily_close - 1).replace([np.inf, -np.inf], np.nan)
    x["overnight_gap"] = pd.Series(x.index.date, index=x.index).map(gaps)

    # Simple causal regime classification.
    ema_spread = (x["ema20"] - x["ema50"]).abs() / x["atr"].replace(0, np.nan)
    x["regime"] = np.where(ema_spread > 0.8, np.where(x["ema20"] > x["ema50"], "trend_up", "trend_down"), "range")
    x["tod"] = pd.cut(minute_of_day, bins=[0, 600, 690, 780, 900, 1440], labels=["pre_open", "open", "midmorning", "midday", "late"], right=False)
    x["hour_minute"] = minute_of_day
    return x


def add_higher_tf_features(base5: pd.DataFrame, df30: pd.DataFrame) -> pd.DataFrame:
    h = df30[["ema20", "ema50", "atr", "close"]].copy().rename(columns={"ema20": "htf_ema20", "ema50": "htf_ema50", "atr": "htf_atr", "close": "htf_close"})
    # Shift one completed higher-TF bar so the 5m decision cannot use a bar still forming.
    h = h.shift(1)
    out = base5.join(h.reindex(base5.index, method="ffill"))
    out["htf_trend"] = np.where(out["htf_ema20"] > out["htf_ema50"], 1, np.where(out["htf_ema20"] < out["htf_ema50"], -1, 0))
    return out


# ----------------------------- strategies ---------------------------------

def signal_for(name: str, row: pd.Series, prev: Optional[pd.Series]) -> Optional[Signal]:
    close, atr = row["close"], row["atr"]
    if pd.isna(atr) or atr <= 0:
        return None
    t = int(row["hour_minute"])
    # Do not initiate in the final 5 minutes.
    if t >= 955:
        return None
    if name == "orb_rvol":
        if t < 600 or t > 690 or pd.isna(row["or_high"]) or pd.isna(row["rvol"]):
            return None
        if row["rvol"] < 1.2:
            return None
        if close > row["or_high"] and close > row["session_vwap"]:
            return Signal(+1, 1.5, 2.0, 36, True)
        if close < row["or_low"] and close < row["session_vwap"]:
            return Signal(-1, 1.5, 2.0, 36, True)
    elif name == "vwap_momentum":
        if t < 600 or t > 900 or pd.isna(row.get("htf_trend")) or pd.isna(row["rvol"]):
            return None
        if row["rvol"] < 1.0:
            return None
        if prev is not None:
            if prev["close"] <= prev["session_vwap"] < close and row["htf_trend"] >= 0 and row["ema20"] > row["ema50"]:
                return Signal(+1, 1.5, 2.0, 48, True)
            if prev["close"] >= prev["session_vwap"] > close and row["htf_trend"] <= 0 and row["ema20"] < row["ema50"]:
                return Signal(-1, 1.5, 2.0, 48, True)
    elif name == "trend_pullback":
        if t < 610 or t > 900 or pd.isna(row.get("htf_trend")):
            return None
        if row["htf_trend"] == 1 and close > row["ema20"] and row["low"] <= row["ema20"] and row["close"] > row["open"]:
            return Signal(+1, 1.25, 1.8, 48, True)
        if row["htf_trend"] == -1 and close < row["ema20"] and row["high"] >= row["ema20"] and row["close"] < row["open"]:
            return Signal(-1, 1.25, 1.8, 48, True)
    elif name == "vol_breakout":
        if t < 630 or t > 900 or pd.isna(row["rvol"]):
            return None
        lookback_high = row.get("rolling_high_20")
        lookback_low = row.get("rolling_low_20")
        if pd.isna(lookback_high) or pd.isna(lookback_low) or row["rvol"] < 1.5:
            return None
        if close > lookback_high and row["tr"] > 1.2 * atr:
            return Signal(+1, 1.5, 2.2, 36, True)
        if close < lookback_low and row["tr"] > 1.2 * atr:
            return Signal(-1, 1.5, 2.2, 36, True)
    elif name == "vwap_mean_revert":
        if t < 610 or t > 900 or row["regime"] != "range":
            return None
        if prev is None or pd.isna(row["vwap_dist_atr"]) or pd.isna(prev["vwap_dist_atr"]):
            return None
        if prev["vwap_dist_atr"] < -1.5 <= row["vwap_dist_atr"]:
            return Signal(+1, 1.0, 1.2, 24, False, True)
        if prev["vwap_dist_atr"] > 1.5 >= row["vwap_dist_atr"]:
            return Signal(-1, 1.0, 1.2, 24, False, True)
    return None


def prepare_intraday(df: pd.DataFrame) -> pd.DataFrame:
    x = regular_session(df)
    x = add_features(x)
    x["rolling_high_20"] = x["high"].rolling(20, min_periods=20).max().shift(1)
    x["rolling_low_20"] = x["low"].rolling(20, min_periods=20).min().shift(1)
    base30 = add_features(resample_ohlcv(x[["open","high","low","close","volume"]], 30))
    x = add_higher_tf_features(x, base30)
    return x.dropna(subset=["atr"]).copy()


# ----------------------------- execution helpers -----------------------------

def resolve_intrabar_exit(side: int, high: float, low: float, stop: float, target: float):
    """Return (price, reason) using conservative stop-first collision handling."""
    stop_hit = low <= stop if side > 0 else high >= stop
    target_hit = high >= target if side > 0 else low <= target
    if stop_hit:
        return stop, "stop"
    if target_hit:
        return target, "target"
    return None, None

def next_bar_index(signal_index: int, n_rows: int):
    return signal_index + 1 if signal_index + 1 < n_rows else None


# ----------------------------- simulator ---------------------------------

def _fill_price(price: float, side: int, is_entry: bool, slippage_bps: float) -> float:
    s = slippage_bps / 10_000.0
    # side = direction of traded position. Long entry buys; long exit sells.
    if is_entry:
        return price * (1 + s) if side > 0 else price * (1 - s)
    return price * (1 - s) if side > 0 else price * (1 + s)


def _qty_for_risk(equity: float, entry: float, stop_distance: float, cfg: Config) -> int:
    risk_budget = equity * cfg.risk_per_trade
    by_risk = math.floor(risk_budget / max(stop_distance, 0.01))
    by_value = math.floor(cfg.max_position_value / max(entry, 0.01))
    by_cash = math.floor(equity / max(entry, 0.01))
    return max(0, min(by_risk, by_value, by_cash))


def _close_trade(pos: Position, exit_time: pd.Timestamp, raw_exit: float, reason: str, cfg: Config, row: pd.Series, equity_before: float) -> Trade:
    exit_px = _fill_price(raw_exit, pos.side, False, cfg.slippage_bps)
    direction_pnl = (exit_px - pos.entry_price) * pos.qty * pos.side
    slip_cost = abs(exit_px - raw_exit) * pos.qty + abs(pos.entry_price - _fill_price(pos.entry_price, pos.side, True, 0.0)) * pos.qty
    commission = cfg.commission_per_share * pos.qty * 2
    pnl = direction_pnl - commission
    return Trade(
        strategy=pos.strategy,
        side="LONG" if pos.side > 0 else "SHORT",
        signal_time=str(pos.entry_signal_time),
        entry_time=str(pos.entry_time),
        exit_time=str(exit_time),
        entry_price=pos.entry_price,
        exit_price=exit_px,
        qty=pos.qty,
        pnl=float(pnl),
        pnl_pct=float(pnl / max(abs(pos.entry_price * pos.qty), 1e-9)),
        exit_reason=reason,
        slippage_cost=float(slip_cost),
        commission=float(commission),
        holding_bars=pos.bars_held,
        entry_regime=str(row.get("regime", "unknown")),
        entry_time_of_day=str(row.get("tod", "unknown")),
    )


def backtest_intraday(df: pd.DataFrame, strategy: str, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    equity = cfg.initial_capital
    cash = cfg.initial_capital
    position: Optional[Position] = None
    trades: List[Trade] = []
    eq_rows = []
    daily_start_equity = equity
    trade_day = None
    trades_today = 0

    rows = list(df.iterrows())
    pending_signal: Optional[Tuple[pd.Timestamp, Signal, pd.Series]] = None

    for i, (ts, row) in enumerate(rows):
        day = ts.date()
        if day != trade_day:
            trade_day = day
            daily_start_equity = equity
            trades_today = 0

        # Execute pending signal at this bar's open (next-bar entry).
        if position is None and pending_signal is not None:
            sig_time, sig, sig_row = pending_signal
            if ts.date() == sig_time.date() and int(row["hour_minute"]) < 955 and trades_today < cfg.max_trades_per_day:
                raw_entry = float(row["open"])
                stop_distance = max(float(sig.stop_atr * sig_row["atr"]), 0.01)
                entry_px = _fill_price(raw_entry, sig.side, True, cfg.slippage_bps)
                qty = _qty_for_risk(equity, entry_px, stop_distance, cfg)
                daily_loss_limit_hit = equity <= daily_start_equity * (1 - cfg.max_daily_loss_pct)
                if qty > 0 and not daily_loss_limit_hit:
                    if sig.side > 0:
                        stop = entry_px - stop_distance
                        target = float(sig_row["session_vwap"]) if sig.target_to_vwap else entry_px + sig.reward_risk * stop_distance
                        target = max(target, entry_px + 0.05 * stop_distance)
                    else:
                        stop = entry_px + stop_distance
                        target = float(sig_row["session_vwap"]) if sig.target_to_vwap else entry_px - sig.reward_risk * stop_distance
                        target = min(target, entry_px - 0.05 * stop_distance)
                    position = Position(sig.side, qty, ts, entry_px, stop, target, strategy, sig_time, strategy, sig.max_hold_bars)
                    trades_today += 1
            pending_signal = None

        # Manage open position using current bar's high/low.
        if position is not None:
            position.bars_held += 1
            hi, lo = float(row["high"]), float(row["low"])
            raw_exit, reason = resolve_intrabar_exit(position.side, hi, lo, position.stop_price, position.target_price)
            if raw_exit is not None:
                tr = _close_trade(position, ts, raw_exit, reason, cfg, row, equity)
                equity += tr.pnl
                trades.append(tr)
                position = None
            elif position.bars_held >= position.max_hold_bars:
                tr = _close_trade(position, ts, float(row["close"]), "time", cfg, row, equity)
                equity += tr.pnl
                trades.append(tr)
                position = None
            elif int(row["hour_minute"]) >= 955:
                tr = _close_trade(position, ts, float(row["close"]), "flat", cfg, row, equity)
                equity += tr.pnl
                trades.append(tr)
                position = None
            elif strategy != "vwap_mean_revert" and position is not None and i > 0:
                # Causal opposite-signal exit; no entry is created from it until next bar.
                prev = rows[i - 1][1]
                sig_now = signal_for(strategy, row, prev)
                if position.side < 0 and sig_now is not None and sig_now.side > 0 or position.side > 0 and sig_now is not None and sig_now.side < 0:
                    tr = _close_trade(position, ts, float(row["close"]), "opposite", cfg, row, equity)
                    equity += tr.pnl
                    trades.append(tr)
                    position = None

        if position is None and i > 0 and not (int(row["hour_minute"]) >= 955):
            prev = rows[i - 1][1]
            sig = signal_for(strategy, row, prev)
            if sig is not None and trades_today < cfg.max_trades_per_day:
                # Signal is at close of current bar; schedule next-bar open.
                pending_signal = (ts, sig, row.copy())

        mark_equity = equity
        if position is not None:
            mark_equity = equity + (float(row["close"]) - position.entry_price) * position.qty * position.side
        eq_rows.append({"timestamp": ts, "equity": mark_equity, "cash": cash, "position": 0 if position is None else position.side * position.qty})

    if position is not None:
        ts, row = rows[-1]
        tr = _close_trade(position, ts, float(row["close"]), "end", cfg, row, equity)
        equity += tr.pnl
        trades.append(tr)

    trades_df = pd.DataFrame([asdict(t) for t in trades])
    equity_df = pd.DataFrame(eq_rows).set_index("timestamp")
    return trades_df, equity_df


# ----------------------------- V1 baseline ---------------------------------

def backtest_v1_daily(daily: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x = daily.copy()
    x = x.sort_index()
    x["sma20"] = x.close.rolling(20).mean()
    x["sma50"] = x.close.rolling(50).mean()
    x["atr"] = (pd.concat([(x.high-x.low), (x.high-x.close.shift()).abs(), (x.low-x.close.shift()).abs()], axis=1).max(axis=1)).rolling(14).mean()
    equity = cfg.initial_capital
    trades = []
    eq = []
    pos = None
    for i in range(50, len(x)-1):
        day, row, nxt = x.index[i], x.iloc[i], x.iloc[i+1]
        if pos is None and row.sma20 > row.sma50 and not pd.isna(row.atr):
            raw_entry = float(nxt.open)
            entry = _fill_price(raw_entry, 1, True, cfg.slippage_bps)
            stop_dist = max(2.0 * float(row.atr), 0.01)
            qty = _qty_for_risk(equity, entry, stop_dist, cfg)
            if qty < 1:
                continue
            pos = {"entry": entry, "qty": qty, "stop": entry-stop_dist, "target": entry+2*stop_dist, "time": x.index[i+1], "signal": day}
        if pos is not None:
            hi, lo = float(nxt.high), float(nxt.low)
            stop_hit, target_hit = lo <= pos["stop"], hi >= pos["target"]
            if stop_hit or target_hit:
                raw = pos["stop"] if stop_hit else pos["target"]
                exit_px = _fill_price(raw, 1, False, cfg.slippage_bps)
                pnl = (exit_px-pos["entry"])*pos["qty"]
                reason = "stop" if stop_hit else "target"
                equity += pnl
                trades.append({"strategy":"v1_daily_sma","side":"LONG","signal_time":str(pos["signal"]),"entry_time":str(pos["time"]),"exit_time":str(nxt.name),"entry_price":pos["entry"],"exit_price":exit_px,"qty":pos["qty"],"pnl":pnl,"pnl_pct":pnl/(pos["entry"]*pos["qty"]),"exit_reason":reason,"slippage_cost":abs(raw-exit_px)*pos["qty"],"commission":0.0,"holding_bars":1,"entry_regime":"daily_sma20_gt_sma50","entry_time_of_day":"daily"})
                pos=None
        eq.append({"timestamp":x.index[i],"equity":equity,"cash":equity,"position":0 if pos is None else pos["qty"]})
    if pos is not None:
        exit_px = _fill_price(float(x.iloc[-1].close), 1, False, cfg.slippage_bps)
        pnl=(exit_px-pos["entry"])*pos["qty"]
        equity += pnl
        trades.append({"strategy":"v1_daily_sma","side":"LONG","signal_time":str(pos["signal"]),"entry_time":str(pos["time"]),"exit_time":str(x.index[-1]),"entry_price":pos["entry"],"exit_price":exit_px,"qty":pos["qty"],"pnl":pnl,"pnl_pct":pnl/(pos["entry"]*pos["qty"]),"exit_reason":"end","slippage_cost":abs(float(x.iloc[-1].close)-exit_px)*pos["qty"],"commission":0.0,"holding_bars":len(x),"entry_regime":"daily_sma20_gt_sma50","entry_time_of_day":"daily"})
    return pd.DataFrame(trades), pd.DataFrame(eq).set_index("timestamp")


# ----------------------------- metrics ---------------------------------

def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1).min()) if not equity.empty else 0.0


def sharpe_daily(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    daily = equity.resample("1D").last().ffill().pct_change().dropna()
    if daily.std() == 0 or daily.empty:
        return 0.0
    return float(np.sqrt(252) * daily.mean() / daily.std())


def cagr(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    days = max((equity.index[-1] - equity.index[0]).total_seconds() / 86400, 1.0)
    years = days / 365.25
    if equity.iloc[0] <= 0 or equity.iloc[-1] <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1/years) - 1)


def summarize(trades: pd.DataFrame, equity: pd.DataFrame, initial: float) -> Dict[str, float]:
    final_eq = float(equity.equity.iloc[-1]) if not equity.empty else initial
    pnl = final_eq - initial
    wins = trades[trades.pnl > 0] if not trades.empty else pd.DataFrame()
    losses = trades[trades.pnl < 0] if not trades.empty else pd.DataFrame()
    gross_profit = wins.pnl.sum() if not wins.empty else 0.0
    gross_loss = abs(losses.pnl.sum()) if not losses.empty else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0.0
    avg_win = wins.pnl.mean() if not wins.empty else 0.0
    avg_loss = losses.pnl.mean() if not losses.empty else 0.0
    win_rate = len(wins)/len(trades) if len(trades) else 0.0
    # Losing streak.
    streak = max_streak = 0
    for pnlv in trades.pnl.tolist() if not trades.empty else []:
        if pnlv < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    exposure = float((equity["position"].abs() > 0).mean()) if (not equity.empty and "position" in equity.columns) else 0.0
    days = equity.index.normalize().nunique() if not equity.empty else 1
    out = {
        "initial_capital": initial,
        "final_equity": final_eq,
        "total_pnl": pnl,
        "total_return": pnl/initial if initial else 0.0,
        "cagr": cagr(equity.equity) if not equity.empty else 0.0,
        "max_drawdown": max_drawdown(equity.equity) if not equity.empty else 0.0,
        "sharpe": sharpe_daily(equity.equity) if not equity.empty else 0.0,
        "profit_factor": pf,
        "win_rate": win_rate,
        "avg_trade": trades.pnl.mean() if not trades.empty else 0.0,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": win_rate*avg_win + (1-win_rate)*avg_loss,
        "trades": int(len(trades)),
        "trades_per_day": float(len(trades)/max(days,1)),
        "exposure": exposure,
        "max_losing_streak": int(max_streak),
        "slippage_cost": float(trades.slippage_cost.sum()) if not trades.empty else 0.0,
        "commission": float(trades.commission.sum()) if not trades.empty else 0.0,
    }
    if not trades.empty:
        out["long_trades"] = int((trades.side == "LONG").sum())
        out["short_trades"] = int((trades.side == "SHORT").sum())
        out["long_pnl"] = float(trades.loc[trades.side == "LONG", "pnl"].sum())
        out["short_pnl"] = float(trades.loc[trades.side == "SHORT", "pnl"].sum())
    else:
        out.update({"long_trades":0,"short_trades":0,"long_pnl":0.0,"short_pnl":0.0})
    return out


def grouped_report(trades: pd.DataFrame, col: str) -> pd.DataFrame:
    if trades.empty or col not in trades.columns:
        return pd.DataFrame()
    g = trades.groupby(col).agg(trades=("pnl","size"), pnl=("pnl","sum"), avg_trade=("pnl","mean"), win_rate=("pnl", lambda s: float((s>0).mean())))
    return g.reset_index()


# ----------------------------- CLI ---------------------------------

def run(args: argparse.Namespace) -> None:
    cfg = Config(slippage_bps=args.slippage_bps, risk_per_trade=args.risk_per_trade, max_position_value=args.max_position_value)
    end = args.end or datetime.now(timezone.utc).date().isoformat()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    names = ["v1_daily_sma", "orb_rvol", "vwap_momentum", "trend_pullback", "vol_breakout", "vwap_mean_revert"] if args.strategy == "all" else [args.strategy]
    results = []

    if args.smoke:
        start = (datetime.now(timezone.utc) - timedelta(days=14)).date().isoformat()
        end = datetime.now(timezone.utc).date().isoformat()
    intraday = fetch_alpaca_bars(start=args.start if not args.smoke else start, end=end, timeframe="5m", cache_dir=args.cache_dir, no_cache=args.no_cache)
    if args.timeframe == "15Min":
        intraday = resample_ohlcv(regular_session(intraday), 15)
    elif args.timeframe == "30Min":
        intraday = resample_ohlcv(regular_session(intraday), 30)
    intraday = prepare_intraday(intraday)

    # Daily baseline uses separate daily data.
    daily = None
    if "v1_daily_sma" in names:
        daily = fetch_alpaca_bars(start=args.start, end=end, timeframe="1d", cache_dir=args.cache_dir, no_cache=args.no_cache)

    for name in names:
        if name == "v1_daily_sma":
            t, e = backtest_v1_daily(daily, cfg)
        else:
            t, e = backtest_intraday(intraday, name, cfg)
        summary = summarize(t, e, cfg.initial_capital)
        summary["strategy"] = name
        results.append(summary)
        t.to_csv(outdir / f"{name}_trades.csv", index=False)
        e.to_csv(outdir / f"{name}_equity.csv")
        grouped_report(t, "entry_regime").to_csv(outdir / f"{name}_by_regime.csv", index=False)
        grouped_report(t, "entry_time_of_day").to_csv(outdir / f"{name}_by_time.csv", index=False)
        if not t.empty:
            tt = t.copy()
            tt["exit_time"] = pd.to_datetime(tt.exit_time, utc=True)
            tt["year"] = tt.exit_time.dt.year
            tt["month"] = tt.exit_time.dt.to_period("M").astype(str)
            tt.groupby("year").pnl.sum().rename("pnl").to_csv(outdir / f"{name}_yearly_pnl.csv")
            tt.groupby("month").pnl.sum().rename("pnl").to_csv(outdir / f"{name}_monthly_pnl.csv")

    summary_df = pd.DataFrame(results).set_index("strategy")
    summary_df.to_csv(outdir / "summary.csv")
    print("\nRESEARCH SUMMARY")
    cols = ["total_return","max_drawdown","sharpe","profit_factor","win_rate","trades","trades_per_day","exposure"]
    print(summary_df[cols].round(4).to_string())
    print("\nNo strategy is designated a winner; compare results across OOS periods and costs before any deployment decision.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", default="all", choices=["all","v1_daily_sma","orb_rvol","vwap_momentum","trend_pullback","vol_breakout","vwap_mean_revert"])
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--timeframe", default="5Min", choices=["5Min","15Min","30Min"])
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--cache-dir", default="data_cache")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--slippage-bps", type=float, default=5.0)
    p.add_argument("--risk-per-trade", type=float, default=0.005)
    p.add_argument("--max-position-value", type=float, default=2500.0)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
