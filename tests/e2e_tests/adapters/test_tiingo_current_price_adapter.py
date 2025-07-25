from adapters.current_price_adapters.tiingo_current_price_adapter import TiingoCurrentPriceAdapter

if __name__ == "__main__":
    adapter = TiingoCurrentPriceAdapter()
    tickers = ["AAPL", "MSFT", "TSLA"]
    for ticker in tickers:
        print(f"\n--- {ticker} ---")
        price = adapter.get_current_price(ticker)
        print(price) 