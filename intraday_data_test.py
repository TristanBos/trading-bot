import os
from datetime import datetime, timedelta, timezone

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.data.enums import DataFeed

client = StockHistoricalDataClient(
    os.environ["ALPACA_API_KEY"],
    os.environ["ALPACA_SECRET_KEY"]
)

end = datetime.now(timezone.utc)
start = end - timedelta(days=7)

for feed in [DataFeed.IEX, DataFeed.SIP]:
    print(f"\n===== {feed} =====")

    try:
        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start,
            end=end,
            feed=feed,
            limit=10000,
        )

        bars = client.get_stock_bars(request).df

        print(f"Rows: {len(bars):,}")

        if not bars.empty:
            print(f"First: {bars.index[0]}")
            print(f"Last:  {bars.index[-1]}")
            print(bars.tail(3).to_string())

    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")

print("\nTEST COMPLETE")
