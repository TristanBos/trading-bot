# SPY V2 Research Engine

Research-only intraday backtester for SPY. It does not submit, cancel, or modify orders.

## First smoke test
```bash
python3 research_engine.py --smoke --strategy all
```

## Full research run
```bash
python3 research_engine.py --strategy all --start 2020-01-01 --end 2026-09-23 --timeframe 5Min --output-dir outputs
```

The engine uses Alpaca historical bars with IEX by default in this project because the current account cannot query recent SIP data. IEX represents a single exchange, while SIP is consolidated US exchange data, so IEX-based research is a known data limitation.

Backtest assumptions:
- Signals use completed bars only.
- Entries occur at the next bar open.
- Stop/target collisions are resolved conservatively with the stop first.
- Slippage is configurable (5 bps default).
- Risk sizing is 0.5% of current equity by default, capped at $2,500 notional.
- One position at a time; both long and short are supported.
- Intraday positions are flattened before the regular session close.
- No leverage by default.

Results are research evidence, not proof of live profitability.
