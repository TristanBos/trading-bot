import yfinance as yf
import pandas as pd
import numpy as np

# ============================================================
# CONFIG
# ============================================================

SYMBOL = "SPY"
START_DATE = "2010-01-01"
END_DATE = None

INITIAL_CAPITAL = 10_000.0

MAX_POSITION_VALUE = 2_500.0
RISK_PER_TRADE = 0.01

ATR_STOP_MULTIPLIER = 2.0
REWARD_RISK = 2.0

EMA_FAST = 20
EMA_SLOW = 50
SMA_LONG = 200

RSI_PERIOD = 14
ATR_PERIOD = 14
VOLUME_PERIOD = 20

RSI_MIN = 50
RSI_MAX = 70

# Realistic execution assumptions
SLIPPAGE = 0.0005       # 0.05%
COMMISSION_PER_TRADE = 1.00

# ============================================================
# DOWNLOAD DATA
# ============================================================

print()
print("========================================")
print("       SPY V2 IMPROVED BACKTEST")
print("========================================")
print()

print("Downloading SPY historical data...")

data = yf.download(
    SYMBOL,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=False,
    progress=False
)

if data.empty:
    raise RuntimeError("No market data received.")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.dropna().copy()

# ============================================================
# INDICATORS
# ============================================================

data["EMA20"] = (
    data["Close"]
    .ewm(span=EMA_FAST, adjust=False)
    .mean()
)

data["EMA50"] = (
    data["Close"]
    .ewm(span=EMA_SLOW, adjust=False)
    .mean()
)

data["SMA200"] = (
    data["Close"]
    .rolling(SMA_LONG)
    .mean()
)

# RSI
delta = data["Close"].diff()

gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)

avg_gain = (
    gain
    .ewm(alpha=1 / RSI_PERIOD, adjust=False)
    .mean()
)

avg_loss = (
    loss
    .ewm(alpha=1 / RSI_PERIOD, adjust=False)
    .mean()
)

rs = avg_gain / avg_loss

data["RSI"] = 100 - (
    100 / (1 + rs)
)

# ATR
previous_close = data["Close"].shift(1)

true_range = pd.concat(
    [
        data["High"] - data["Low"],
        (data["High"] - previous_close).abs(),
        (data["Low"] - previous_close).abs()
    ],
    axis=1
).max(axis=1)

data["ATR"] = (
    true_range
    .ewm(alpha=1 / ATR_PERIOD, adjust=False)
    .mean()
)

# Volume
data["AverageVolume"] = (
    data["Volume"]
    .rolling(VOLUME_PERIOD)
    .mean()
)

# ============================================================
# BACKTEST STATE
# ============================================================

cash = INITIAL_CAPITAL

shares = 0
entry_price = 0
stop_price = 0
target_price = 0

entry_date = None

trades = []

equity_curve = []

# ============================================================
# MAIN BACKTEST LOOP
# ============================================================

for i in range(len(data)):

    row = data.iloc[i]
    date = data.index[i]

    close = float(row["Close"])
    high = float(row["High"])
    low = float(row["Low"])
    open_price = float(row["Open"])

    ema20 = row["EMA20"]
    ema50 = row["EMA50"]
    sma200 = row["SMA200"]
    rsi = row["RSI"]
    atr = row["ATR"]
    average_volume = row["AverageVolume"]
    volume = row["Volume"]

    # --------------------------------------------------------
    # Skip until indicators exist
    # --------------------------------------------------------

    if any(
        pd.isna(x)
        for x in [
            ema20,
            ema50,
            sma200,
            rsi,
            atr,
            average_volume
        ]
    ):
        equity_curve.append(
            cash + shares * close
        )
        continue

    # --------------------------------------------------------
    # EXIT EXISTING POSITION
    # --------------------------------------------------------

    if shares > 0:

        exit_price = None
        exit_reason = None

        # Stop and target are evaluated using the day's range.
        #
        # If BOTH are hit on the same day, we assume the STOP
        # happened first. This is deliberately conservative.

        if low <= stop_price and high >= target_price:

            exit_price = stop_price
            exit_reason = "STOP_AND_TARGET_SAME_DAY"

        elif low <= stop_price:

            exit_price = stop_price
            exit_reason = "STOP"

        elif high >= target_price:

            exit_price = target_price
            exit_reason = "TARGET"

        # Trend exit at close
        elif ema20 < ema50:

            exit_price = close
            exit_reason = "TREND_EXIT"

        # ----------------------------------------------------
        # EXECUTE EXIT
        # ----------------------------------------------------

        if exit_price is not None:

            # Apply slippage against us
            if exit_reason.startswith("STOP"):
                execution_price = exit_price * (
                    1 - SLIPPAGE
                )
            else:
                execution_price = exit_price * (
                    1 - SLIPPAGE
                )

            proceeds = (
                shares * execution_price
            )

            cash += proceeds
            cash -= COMMISSION_PER_TRADE

            profit = (
                execution_price - entry_price
            ) * shares

            trades.append(
                {
                    "entry_date": entry_date,
                    "exit_date": date,
                    "entry_price": entry_price,
                    "exit_price": execution_price,
                    "shares": shares,
                    "profit": profit,
                    "return_pct":
                        (
                            profit
                            /
                            (entry_price * shares)
                        ) * 100,
                    "reason": exit_reason
                }
            )

            shares = 0
            entry_price = 0
            stop_price = 0
            target_price = 0
            entry_date = None

    # --------------------------------------------------------
    # CHECK FOR NEW SIGNAL
    #
    # IMPORTANT:
    # Signal is based ONLY on today's completed candle.
    # Entry occurs TOMORROW at the open.
    # --------------------------------------------------------

    if shares == 0 and i < len(data) - 1:

        trend_condition = (
            ema20 > ema50
            and ema50 > sma200
        )

        price_condition = (
            close > ema20
        )

        momentum_condition = (
            RSI_MIN <= rsi <= RSI_MAX
        )

        volume_condition = (
            volume >= average_volume
        )

        buy_signal = (
            trend_condition
            and price_condition
            and momentum_condition
            and volume_condition
        )

        if buy_signal:

            next_row = data.iloc[i + 1]

            next_open = float(
                next_row["Open"]
            )

            # Apply entry slippage
            execution_entry = (
                next_open
                * (1 + SLIPPAGE)
            )

            stop_distance = (
                atr * ATR_STOP_MULTIPLIER
            )

            potential_stop = (
                execution_entry
                - stop_distance
            )

            if potential_stop > 0:

                risk_per_share = (
                    execution_entry
                    - potential_stop
                )

                # Use TOTAL portfolio equity
                current_equity = (
                    cash
                    + shares * close
                )

                max_risk_dollars = (
                    current_equity
                    * RISK_PER_TRADE
                )

                risk_quantity = int(
                    max_risk_dollars
                    /
                    risk_per_share
                )

                value_quantity = int(
                    MAX_POSITION_VALUE
                    /
                    execution_entry
                )

                cash_quantity = int(
                    (
                        cash
                        -
                        COMMISSION_PER_TRADE
                    )
                    /
                    execution_entry
                )

                quantity = min(
                    risk_quantity,
                    value_quantity,
                    cash_quantity
                )

                if quantity >= 1:

                    position_value = (
                        quantity
                        * execution_entry
                    )

                    if position_value <= cash:

                        shares = quantity

                        entry_price = (
                            execution_entry
                        )

                        entry_date = (
                            data.index[i + 1]
                        )

                        stop_price = (
                            execution_entry
                            - stop_distance
                        )

                        target_price = (
                            execution_entry
                            +
                            (
                                stop_distance
                                * REWARD_RISK
                            )
                        )

                        cash -= (
                            shares
                            * entry_price
                        )

                        cash -= (
                            COMMISSION_PER_TRADE
                        )

    # --------------------------------------------------------
    # PORTFOLIO EQUITY
    # --------------------------------------------------------

    if shares > 0:

        current_equity = (
            cash
            + shares * close
        )

    else:

        current_equity = cash

    equity_curve.append(
        current_equity
    )

# ============================================================
# CLOSE FINAL POSITION
# ============================================================

if shares > 0:

    final_date = data.index[-1]

    final_price = float(
        data.iloc[-1]["Close"]
    )

    execution_price = (
        final_price
        * (1 - SLIPPAGE)
    )

    cash += (
        shares
        * execution_price
    )

    cash -= COMMISSION_PER_TRADE

    profit = (
        execution_price
        - entry_price
    ) * shares

    trades.append(
        {
            "entry_date": entry_date,
            "exit_date": final_date,
            "entry_price": entry_price,
            "exit_price": execution_price,
            "shares": shares,
            "profit": profit,
            "return_pct":
                (
                    profit
                    /
                    (entry_price * shares)
                ) * 100,
            "reason": "END"
        }
    )

    shares = 0

# ============================================================
# RESULTS
# ============================================================

trades_df = pd.DataFrame(trades)

final_equity = cash

total_return = (
    final_equity
    /
    INITIAL_CAPITAL
    - 1
) * 100

if len(trades_df) > 0:

    winning_trades = (
        trades_df["profit"] > 0
    ).sum()

    losing_trades = (
        trades_df["profit"] <= 0
    ).sum()

    win_rate = (
        winning_trades
        /
        len(trades_df)
    ) * 100

    gross_profit = (
        trades_df.loc[
            trades_df["profit"] > 0,
            "profit"
        ].sum()
    )

    gross_loss = abs(
        trades_df.loc[
            trades_df["profit"] < 0,
            "profit"
        ].sum()
    )

    if gross_loss > 0:

        profit_factor = (
            gross_profit
            /
            gross_loss
        )

    else:

        profit_factor = float("inf")

else:

    winning_trades = 0
    losing_trades = 0
    win_rate = 0
    profit_factor = 0

# ============================================================
# DRAWDOWN
# ============================================================

equity_series = pd.Series(
    equity_curve,
    index=data.index[:len(equity_curve)]
)

running_max = (
    equity_series
    .cummax()
)

drawdown = (
    equity_series
    /
    running_max
    - 1
)

max_drawdown = (
    drawdown.min()
    * 100
)

# ============================================================
# BUY & HOLD
# ============================================================

first_price = float(
    data.iloc[0]["Close"]
)

last_price = float(
    data.iloc[-1]["Close"]
)

buy_hold_return = (
    last_price
    /
    first_price
    - 1
) * 100

# ============================================================
# YEARLY PERFORMANCE
# ============================================================

yearly_results = []

for year in sorted(
    set(
        pd.to_datetime(
            trades_df["exit_date"]
        ).dt.year
    )
) if len(trades_df) > 0 else []:

    year_trades = trades_df[
        pd.to_datetime(
            trades_df["exit_date"]
        ).dt.year == year
    ]

    year_profit = (
        year_trades["profit"].sum()
    )

    yearly_results.append(
        {
            "year": year,
            "trades": len(year_trades),
            "profit": year_profit,
            "win_rate":
                (
                    (
                        year_trades["profit"] > 0
                    ).sum()
                    /
                    len(year_trades)
                ) * 100
        }
    )

# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("========================================")
print("              RESULTS")
print("========================================")
print()

print(
    f"Initial capital:     "
    f"${INITIAL_CAPITAL:,.2f}"
)

print(
    f"Final capital:       "
    f"${final_equity:,.2f}"
)

print(
    f"Total return:        "
    f"{total_return:.2f}%"
)

print()

print(
    f"Total trades:        "
    f"{len(trades_df)}"
)

print(
    f"Winning trades:      "
    f"{winning_trades}"
)

print(
    f"Losing trades:       "
    f"{losing_trades}"
)

print(
    f"Win rate:            "
    f"{win_rate:.2f}%"
)

print(
    f"Profit factor:       "
    f"{profit_factor:.2f}"
)

print()

print(
    f"Maximum drawdown:    "
    f"{max_drawdown:.2f}%"
)

print()

print(
    f"SPY buy & hold:      "
    f"{buy_hold_return:.2f}%"
)

print()

print("========================================")
print("          YEARLY PERFORMANCE")
print("========================================")

for result in yearly_results:

    print(
        f"{result['year']}: "
        f"{result['trades']} trades | "
        f"Profit ${result['profit']:.2f} | "
        f"Win rate {result['win_rate']:.1f}%"
    )

print()

print("========================================")
print("            LAST 10 TRADES")
print("========================================")

if len(trades_df) > 0:

    print(
        trades_df
        .tail(10)
        .to_string(index=False)
    )

# ============================================================
# SAVE TRADE HISTORY
# ============================================================

trades_df.to_csv(
    "v2_trades.csv",
    index=False
)

print()
print(
    "Trade history saved to "
    "v2_trades.csv"
)

print()
print("BACKTEST COMPLETE.")
