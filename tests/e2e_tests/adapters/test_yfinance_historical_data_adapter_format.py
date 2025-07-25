from datetime import datetime
from adapters.historical_data_adapters.yfinance_historical_data_adapter import YFinanceHistoricalDataAdapter

EXPECTED_COLUMNS = ["DateTime", "open", "close", "high", "low", "volume"]
EXPECTED_TYPES = {
    "DateTime": None,  # Accepts datetime or string, just check presence
    "open": float,
    "close": float,
    "high": float,
    "low": float,
    "volume": int,
}

if __name__ == "__main__":
    adapter = YFinanceHistoricalDataAdapter()
    ticker = "AAPL"
    increments = ["daily", "weekly", "monthly", "annually"]
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 10)

    for inc in increments:
        print(f"\n--- {ticker} | {inc} ---")
        data = adapter.get_historical_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            tick_increment=inc
        )
        if not data:
            print("No data returned.")
            continue
        # Check columns
        for i, record in enumerate(data):
            missing = [col for col in EXPECTED_COLUMNS if col not in record]
            if missing:
                print(f"Record {i}: Missing columns: {missing}")
            # Check types and rounding
            for col, typ in EXPECTED_TYPES.items():
                val = record.get(col)
                if col == "DateTime":
                    if val is None:
                        print(f"Record {i}: DateTime is missing or None")
                elif val is not None:
                    if typ and not isinstance(val, typ):
                        print(f"Record {i}: {col} is not {typ.__name__} (got {type(val).__name__})")
                    if typ is float and val is not None:
                        # Check rounding to 2 decimals
                        if round(val, 2) != val:
                            print(f"Record {i}: {col} is not rounded to 2 decimals: {val}")
            # Only check the first 5 records for brevity
            if i >= 4:
                break
        print("Format check complete for this increment.") 