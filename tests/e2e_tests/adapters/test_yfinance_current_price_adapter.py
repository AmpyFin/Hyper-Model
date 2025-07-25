from adapters.current_price_adapters.yfinance_current_price_adapter import YFinanceCurrentPriceAdapter

if __name__ == "__main__":
    adapter = YFinanceCurrentPriceAdapter()
    tickers = ["AAPL", "MSFT", "TSLA"]
    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        price = adapter.get_current_price(ticker)
        print(price) 