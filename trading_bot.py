# Railway deployment test
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

API_KEY = os.environ["ALPACA_API_KEY"]
SECRET_KEY = os.environ["ALPACA_SECRET_KEY"]

trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

SYMBOL = "SPY"

MAX_POSITION_VALUE = 2500
STOP_LOSS_PERCENT = 0.05
TAKE_PROFIT_PERCENT = 0.10

print()
print("===== AUTOMATED SPY BOT =====")
account = trading_client.get_account()
equity = float(account.equity)
last_equity = float(account.last_equity)
daily_pnl = equity - last_equity

print(f"Account equity: ${equity:,.2f}")
print(f"Today's P&L: ${daily_pnl:+,.2f}")

# ---------------------------------------------------------
# CHECK CURRENT POSITION
# ---------------------------------------------------------

try:
    position = trading_client.get_open_position(SYMBOL)
    shares_owned = float(position.qty)
    entry_price = float(position.avg_entry_price)
except Exception:
    shares_owned = 0
    entry_price = 0

print(f"SPY shares owned: {shares_owned}")

# ---------------------------------------------------------
# CHECK OPEN ORDERS
# ---------------------------------------------------------

open_orders = trading_client.get_orders(
    filter=GetOrdersRequest(
        status=QueryOrderStatus.OPEN
    )
)

sp_orders = [
    order for order in open_orders
    if order.symbol == SYMBOL
]

print(f"Open SPY orders: {len(sp_orders)}")

# ---------------------------------------------------------
# EXISTING POSITION
# ---------------------------------------------------------

if shares_owned > 0:

    stop_price = round(
        entry_price * (1 - STOP_LOSS_PERCENT), 2
    )

    target_price = round(
        entry_price * (1 + TAKE_PROFIT_PERCENT), 2
    )

    print(f"Entry price: ${entry_price:.2f}")
    print(f"Stop-loss: ${stop_price:.2f}")
    print(f"Take-profit: ${target_price:.2f}")

    # Check whether an existing SELL protection order exists
    protective_orders = [
        order for order in sp_orders
        if order.side == OrderSide.SELL
    ]

    if protective_orders:
        print("Protective SELL order already exists.")
        print("No new protection will be created.")
        exit()

    # -----------------------------------------------------
    # CREATE OCO PROTECTION
    # -----------------------------------------------------

    print()
    print("NO PROTECTIVE ORDERS FOUND.")
    print("Creating OCO protection for existing position.")

    oco_order = LimitOrderRequest(
        symbol=SYMBOL,
        qty=shares_owned,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.GTC,
        limit_price=target_price,
        order_class=OrderClass.OCO,
        take_profit=TakeProfitRequest(
            limit_price=target_price
        ),
        stop_loss=StopLossRequest(
            stop_price=stop_price
        )
    )

    trading_client.submit_order(
        order_data=oco_order
    )

    print()
    print("OCO PROTECTION SUBMITTED.")
    print(f"Take-profit: ${target_price:.2f}")
    print(f"Stop-loss: ${stop_price:.2f}")

    exit()

# ---------------------------------------------------------
# NO POSITION / CHECK FOR EXISTING ORDERS
# ---------------------------------------------------------

if sp_orders:

    print()
    print("Existing SPY order detected.")
    print("NO NEW TRADE WILL BE PLACED.")

    for order in sp_orders:
        print(
            "Existing order:",
            order.side,
            order.qty,
            order.status
        )

    exit()

# ---------------------------------------------------------
# GET MARKET DATA
# ---------------------------------------------------------

end = datetime.now()
start = end - timedelta(days=80)

request = StockBarsRequest(
    symbol_or_symbols=SYMBOL,
    timeframe=TimeFrame.Day,
    start=start,
    end=end,
    feed=DataFeed.IEX
)

bars = data_client.get_stock_bars(request).df

prices = bars["close"].values

if len(prices) < 50:
    print("Not enough market data.")
    exit()

ma20 = prices[-20:].mean()
ma50 = prices[-50:].mean()
current_price = prices[-1]

print()
print(f"Current price: ${current_price:.2f}")
print(f"20-day average: ${ma20:.2f}")
print(f"50-day average: ${ma50:.2f}")

# ---------------------------------------------------------
# BUY SIGNAL
# ---------------------------------------------------------

if ma20 > ma50:

    quantity = int(MAX_POSITION_VALUE / current_price)

    if quantity < 1:
        print("SPY price is too high for the position limit.")
        print("SIGNAL: HOLD")
        exit()

    stop_price = round(
        current_price * (1 - STOP_LOSS_PERCENT), 2
    )

    target_price = round(
        current_price * (1 + TAKE_PROFIT_PERCENT), 2
    )

    print()
    print("SIGNAL: BUY")
    print(f"Quantity: {quantity}")
    print(f"Stop-loss: ${stop_price:.2f}")
    print(f"Take-profit: ${target_price:.2f}")

    order = MarketOrderRequest(
        symbol=SYMBOL,
        qty=quantity,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
        order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(
            limit_price=target_price
        ),
        stop_loss=StopLossRequest(
            stop_price=stop_price
        )
    )

    trading_client.submit_order(
        order_data=order
    )

    print()
    print("PAPER TRADE SUBMITTED.")
    print("GTC stop-loss and take-profit attached.")

# ---------------------------------------------------------
# HOLD
# ---------------------------------------------------------

else:

    print()
    print("SIGNAL: HOLD")
    print("No trade placed.")
