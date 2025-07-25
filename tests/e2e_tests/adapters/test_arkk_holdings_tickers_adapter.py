from adapters.tickers_adapters.arkk_holdings_tickers_adapter import ARKKHoldingsTickersAdapter

if __name__ == "__main__":
    adapter = ARKKHoldingsTickersAdapter()
    tickers = adapter.fetch_tickers()
    print(tickers)