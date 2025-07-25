# Data Standardization Documentation

## Adapter Types Overview
The system includes three main types of adapters:
1. Ticker Adapters (`tickers_adapters/`)
2. Current Price Adapters (`current_price_adapters/`)
3. Historical Data Adapters (`historical_data_adapters/`)

## Ticker Adapters Standard
- **Type:** List[str]
- **Format:** List of uppercase stock ticker symbols
- **Validation:**
  - All tickers must be strings
  - No whitespace in tickers
  - No empty or None values
- **Examples:**
  - NASDAQ-100: `["AAPL", "MSFT", "GOOGL", ...]` (exactly 100 tickers)
  - S&P 500: `["MMM", "ABT", "ABBV", ...]` (exactly 500 tickers)
  - ARKK Holdings: `["TSLA", "COIN", "ROKU", ...]` (variable number of tickers)
- **Description:** All ticker adapters must return a list of valid stock ticker symbols as strings. The number of tickers returned depends on the index or fund being tracked.

## Current Price Standard
- **Type:** float
- **Precision:** Rounded to 2 decimal places
- **Description:** All current price adapters must return the latest available price as a float rounded to 2 decimal places.
- **Examples:**
  - `156.39`
  - `1234.50`
  - `0.75`
- **Error Handling:**
  - Return None if price is unavailable
  - Raise RuntimeError with descriptive message for API failures

## Historical Data Standard
- **Type:** List of dictionaries (or pandas DataFrame)
- **Columns:**
  - `DateTime` (datetime): Timestamp of the data point
  - `open` (float, rounded to 2 decimal places)
  - `close` (float, rounded to 2 decimal places)
  - `high` (float, rounded to 2 decimal places)
  - `low` (float, rounded to 2 decimal places)
  - `volume` (int, rounded)
- **Description:** All historical data adapters must return a list of records (or DataFrame) with the above columns. All price columns must be floats rounded to 2 decimal places, and volume must be an int.
- **Example Record:**
  ```python
  {
      "DateTime": datetime(2023, 7, 21, 16, 0),
      "open": 156.39,
      "high": 157.85,
      "low": 155.98,
      "close": 156.75,
      "volume": 1234567
  }
  ```

## Implementation Requirements
1. **Type Consistency:**
   - Adapters must handle type conversion internally
   - Return data in the specified format
   - Validate data types before returning

2. **Error Handling:**
   - Use proper exception handling
   - Raise RuntimeError with descriptive messages
   - Log errors using the logging module, not print statements

3. **Data Validation:**
   - Check for missing or invalid values
   - Apply rounding rules consistently
   - Filter out invalid or malformed data

4. **API Keys and Authentication:**
   - Read API keys from config.py
   - Handle authentication errors gracefully
   - Log authentication failures appropriately

5. **Rate Limiting:**
   - Implement retry logic with exponential backoff
   - Handle rate limit errors gracefully
   - Log rate limit issues for monitoring

## Testing Requirements
1. **Unit Tests:**
   - Test data type validation
   - Test rounding rules
   - Test error handling

2. **Integration Tests:**
   - Test API connectivity
   - Test rate limiting handling
   - Test end-to-end data flow

3. **Logging:**
   - Log all API calls
   - Log errors and warnings
   - Log rate limit issues
   - No print statements allowed
