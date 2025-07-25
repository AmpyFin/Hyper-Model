from adapters.current_price_adapters.yfinance_current_price_adapter import YFinanceCurrentPriceAdapter
from adapters.historical_data_adapters.tiingo_historical_data_adapter import TiingoHistoricalDataAdapter
from adapters.tickers_adapters.wiki_SPY_500_ticker_adapter import WikiSPY500TickerAdapter

# We can import different adapters but the left side of the variable must remain the same.

current_price_adapter = YFinanceCurrentPriceAdapter()
historical_data_adapter = TiingoHistoricalDataAdapter()
tickers_adapter = WikiSPY500TickerAdapter()






