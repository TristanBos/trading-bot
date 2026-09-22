from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    TakeProfitRequest,
    StopLossRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import (
    OrderSide,
    TimeInForce,
    OrderClass,
    QueryOrderStatus,
)
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed

from datetime import datetime, timedelta
import os
import math


# ============================================================
# CONFIGURATION
# ============================================================

API_KEY = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

# PAPER TRADING ONLY
trading_client = TradingClient(
    API_KEY,
    SECRET_KEY,
    paper=True
)

data_client = StockHistoricalDataClient(
    API_KEY,
    SECRET_KEY
)


SYMBOL = "SPY"

# Maximum amount of account equity that can be allocated
MAX_POSITION_VALUE = 2500.0

# Maximum percentage of account equity we are willing
# to risk if the initial stop is hit.
RISK_PER_TRADE = 0.01

# ATR stop multiplier
ATR_STOP_MULTIPLIER = 2.0

# Minimum reward/risk ratio
MIN_REWARD_RISK = 2.0

# Indicator settings
EMA_FAST = 20
EMA_SLOW = 50
SMA_LONG = 200

RSI_PERIOD = 14
ATR_PERIOD = 14

# Volume confirmation
VOLUME_PERIOD = 20

# Don't buy if RSI is extremely extended
RSI_MIN = 50
RSI_MAX = 70


# ============================================================
# INDICATORS
# ============================================================

def calculate_ema(values, period):
    """
    Calculate exponential moving average.
    """
    if len(values) < period:
        return None

    multiplier = 2 / (period + 1)

    ema = sum(values[:period]) / period

    for price in values[period:]:
        ema = (price - ema) * multiplier + ema

    return ema


def calculate_sma(values, period):
    """
    Calculate simple moving average.
    """
    if len(values) < period:
        return None

    return sum(values[-period:]) / period


def calculate_rsi(closes, period=14):
    """
    Calculate RSI using Wilder-style smoothing.
    """

    if len(closes) < period + 1:
        return None

    gains = []
    losses = []

    for i in range(1, len(closes)):
        change = closes[i] - closes[i - 1]

        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for i in range(period, len(gains)):
        avg_gain = (
            (avg_gain * (period - 1)) + gains[i]
        ) / period

        avg_loss = (
            (avg_loss * (period - 1)) + losses[i]
        ) / period

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss

    return 100 - (100 / (1 + rs))


def calculate_atr(highs, lows, closes, period=14):
    """
    Calculate Average True Range.
    """

    if len(closes) < period + 1:
        return None

    true_ranges = []

    for i in range(1, len(closes)):

        high = highs[i]
        low = lows[i]
        previous_close = closes[i - 1]

        true_range = max(
            high - low,
            abs(high - previous_close),
            abs(low - previous_close)
        )

        true_ranges.append(true_range)

    atr = sum(true_ranges[:period]) / period

    for tr in true_ranges[period:]:
        atr = (
            (atr * (period - 1)) + tr
        ) / period

    return atr


# ============================================================
# ACCOUNT / POSITION CHECK
# ============================================================

def get_position():

    try:
        position = trading_client.get_open_position(SYMBOL)

        return (
            float(position.qty),
            float(position.avg_entry_price)
        )

    except Exception:
        return 0.0, 0.0


def get_open_orders():

    orders = trading_client.get_orders(
        filter=GetOrdersRequest(
            status=QueryOrderStatus.OPEN
        )
    )

    return [
        order
        for order in orders
        if order.symbol == SYMBOL
    ]


# ============================================================
# MAIN
# ============================================================

print()
print("========================================")
print("       AUTOMATED SPY BOT V2")
print("       PAPER TRADING ONLY")
print("========================================")
print()


# ------------------------------------------------------------
# CHECK POSITION
# ------------------------------------------------------------

shares_owned, entry_price = get_position()

print(f"SPY shares owned: {shares_owned}")

open_orders = get_open_orders()

print(f"Open SPY orders: {len(open_orders)}")


# ------------------------------------------------------------
# EXISTING POSITION
# ------------------------------------------------------------

if shares_owned > 0:

    print()
    print("Existing SPY position detected.")
    print(f"Entry price: ${entry_price:.2f}")

    protective_orders = [
        order
        for order in open_orders
        if order.side == OrderSide.SELL
    ]

    if protective_orders:

        print("Protective SELL order already exists.")
        print("No new protection will be created.")
        print()
        print("V2 CHECK COMPLETE.")

        exit()

    print()
    print("WARNING:")
    print("Existing position has NO protective SELL order.")
    print("V2 will NOT create a new trade.")
    print("Protection must be handled before continuing.")

    exit()


# ------------------------------------------------------------
# SAFETY CHECK
# ------------------------------------------------------------

if open_orders:

    print()
    print("Existing SPY order detected.")
    print("No new trade will be placed.")

    for order in open_orders:

        print(
            f"Order: {order.side} "
            f"{order.qty} "
            f"Status: {order.status}"
        )

    exit()


# ============================================================
# MARKET DATA
# ============================================================

end = datetime.now()

# Need enough history for 200-day SMA
start = end - timedelta(days=400)

request = StockBarsRequest(
    symbol_or_symbols=SYMBOL,
    timeframe=TimeFrame.Day,
    start=start,
    end=end,
    feed=DataFeed.IEX
)

bars = data_client.get_stock_bars(request).df


if bars.empty:

    print("ERROR: No market data received.")

    exit()


# ------------------------------------------------------------
# Extract OHLCV
# ------------------------------------------------------------

closes = bars["close"].tolist()
highs = bars["high"].tolist()
lows = bars["low"].tolist()
volumes = bars["volume"].tolist()


if len(closes) < SMA_LONG + 20:

    print(
        "ERROR: Not enough historical data."
    )

    exit()


current_price = closes[-1]


# ============================================================
# CALCULATE INDICATORS
# ============================================================

ema20 = calculate_ema(
    closes,
    EMA_FAST
)

ema50 = calculate_ema(
    closes,
    EMA_SLOW
)

sma200 = calculate_sma(
    closes,
    SMA_LONG
)

rsi = calculate_rsi(
    closes,
    RSI_PERIOD
)

atr = calculate_atr(
    highs,
    lows,
    closes,
    ATR_PERIOD
)

average_volume = sum(
    volumes[-VOLUME_PERIOD:]
) / VOLUME_PERIOD

current_volume = volumes[-1]


# ============================================================
# VALIDATION
# ============================================================

if any(
    value is None
    for value in [
        ema20,
        ema50,
        sma200,
        rsi,
        atr
    ]
):

    print("ERROR: Indicator calculation failed.")

    exit()


if atr <= 0:

    print("ERROR: Invalid ATR.")

    exit()


# ============================================================
# DISPLAY MARKET DATA
# ============================================================

print()
print("---------- MARKET DATA ----------")

print(f"SPY price:       ${current_price:.2f}")
print(f"EMA 20:          ${ema20:.2f}")
print(f"EMA 50:          ${ema50:.2f}")
print(f"SMA 200:         ${sma200:.2f}")
print(f"RSI 14:          {rsi:.2f}")
print(f"ATR 14:          ${atr:.2f}")

print(
    f"Volume:          {current_volume:,.0f}"
)

print(
    f"Avg volume:      {average_volume:,.0f}"
)


# ============================================================
# SIGNAL CONDITIONS
# ============================================================

trend_condition = (
    ema20 > ema50
    and ema50 > sma200
)

price_condition = (
    current_price > ema20
)

momentum_condition = (
    RSI_MIN <= rsi <= RSI_MAX
)

volume_condition = (
    current_volume >= average_volume
)


print()
print("---------- SIGNAL CHECK ----------")

print(
    f"Trend:       {'PASS' if trend_condition else 'FAIL'}"
)

print(
    f"Price:       {'PASS' if price_condition else 'FAIL'}"
)

print(
    f"Momentum:    {'PASS' if momentum_condition else 'FAIL'}"
)

print(
    f"Volume:      {'PASS' if volume_condition else 'FAIL'}"
)


# ============================================================
# BUY SIGNAL
# ============================================================

BUY_SIGNAL = (
    trend_condition
    and price_condition
    and momentum_condition
    and volume_condition
)


if not BUY_SIGNAL:

    print()
    print("SIGNAL: HOLD")
    print("Not all entry conditions are satisfied.")
    print()
    print("V2 CHECK COMPLETE.")

    exit()


# ============================================================
# RISK MANAGEMENT
# ============================================================

# ATR-based stop
stop_distance = atr * ATR_STOP_MULTIPLIER

stop_price = current_price - stop_distance


if stop_price <= 0:

    print(
        "ERROR: Calculated stop price is invalid."
    )

    exit()


# Reward target based on risk
target_distance = (
    stop_distance * MIN_REWARD_RISK
)

target_price = (
    current_price + target_distance
)


# ------------------------------------------------------------
# ACCOUNT EQUITY
# ------------------------------------------------------------

account = trading_client.get_account()

equity = float(account.equity)

max_risk_dollars = (
    equity * RISK_PER_TRADE
)


# ------------------------------------------------------------
# POSITION SIZE
# ------------------------------------------------------------

risk_per_share = (
    current_price - stop_price
)


if risk_per_share <= 0:

    print(
        "ERROR: Invalid risk per share."
    )

    exit()


risk_quantity = math.floor(
    max_risk_dollars / risk_per_share
)


value_quantity = math.floor(
    MAX_POSITION_VALUE / current_price
)


quantity = min(
    risk_quantity,
    value_quantity
)


# ============================================================
# POSITION SIZE SAFETY
# ============================================================

print()
print("---------- RISK MANAGEMENT ----------")

print(
    f"Account equity:       ${equity:,.2f}"
)

print(
    f"Max risk:             ${max_risk_dollars:,.2f}"
)

print(
    f"Risk/share:           ${risk_per_share:.2f}"
)

print(
    f"Stop price:           ${stop_price:.2f}"
)

print(
    f"Target price:         ${target_price:.2f}"
)

print(
    f"Calculated quantity:  {quantity}"
)


if quantity < 1:

    print()
    print(
        "POSITION TOO SMALL — HOLD"
    )

    exit()


estimated_position_value = (
    quantity * current_price
)

estimated_risk = (
    quantity * risk_per_share
)


print(
    f"Position value:       "
    f"${estimated_position_value:,.2f}"
)

print(
    f"Estimated max loss:   "
    f"${estimated_risk:,.2f}"
)


# ============================================================
# FINAL SAFETY CHECK
# ============================================================

if estimated_position_value > MAX_POSITION_VALUE:

    print()
    print(
        "SAFETY ERROR: Position exceeds maximum."
    )

    exit()


if estimated_risk > max_risk_dollars * 1.05:

    print()
    print(
        "SAFETY ERROR: Risk exceeds limit."
    )

    exit()


# ============================================================
# SUBMIT PAPER TRADE
# ============================================================

print()
print("========================================")
print("SIGNAL: BUY")
print("========================================")

print(f"Symbol:        {SYMBOL}")
print(f"Quantity:      {quantity}")
print(f"Entry:         ${current_price:.2f}")
print(f"Stop:          ${stop_price:.2f}")
print(f"Target:        ${target_price:.2f}")
print(
    f"Risk:          ${estimated_risk:.2f}"
)

print()
print("Submitting PAPER bracket order...")


order = MarketOrderRequest(
    symbol=SYMBOL,
    qty=quantity,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.GTC,
    order_class=OrderClass.BRACKET,
    take_profit=TakeProfitRequest(
        limit_price=round(target_price, 2)
    ),
    stop_loss=StopLossRequest(
        stop_price=round(stop_price, 2)
    )
)


trading_client.submit_order(
    order_data=order
)


print()
print("========================================")
print("PAPER TRADE SUBMITTED")
print("========================================")

print(
    "GTC bracket protection attached."
)

print(
    f"Stop-loss:   ${stop_price:.2f}"
)

print(
    f"Take-profit: ${target_price:.2f}"
)

print()
print("V2 COMPLETE.")
