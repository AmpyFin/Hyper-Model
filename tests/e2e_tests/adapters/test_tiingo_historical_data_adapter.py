from datetime import datetime
from adapters.historical_data_adapters.tiingo_historical_data_adapter import TiingoHistoricalDataAdapter

if __name__ == "__main__":
    adapter = TiingoHistoricalDataAdapter()
    tickers = ["AAPL", "MSFT", "TSLA", "META"]
    increments = ["daily", "weekly", "monthly", "annually"]
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 1, 10)

    for ticker in tickers:
        for inc in increments:
            print(f"\n--- {ticker} | {inc} ---")
            try:
                data = adapter.get_historical_data(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    tick_increment=inc
                )
                print(data)
            except Exception as e:
                print(f"Error: {e}")

    # Demonstrate invalid tick_increment
    print("\n--- Invalid tick_increment test ---")
    try:
        adapter.get_historical_data(
            ticker="AAPL",
            start_date=start_date,
            end_date=end_date,
            tick_increment="1d"
        )
    except Exception as e:
        print(f"Expected error for invalid tick_increment: {e}") 