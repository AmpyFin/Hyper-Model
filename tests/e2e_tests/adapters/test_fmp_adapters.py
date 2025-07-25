from adapters.tickers_adapters.fmp_NDAQ_100_ticker_adapter import FMPNDAQ100TickerAdapter
from adapters.tickers_adapters.wiki_SPY_500_ticker_adapter import WikiSPY500TickerAdapter
import os
import json
from datetime import datetime

def ensure_logs_directory():
    """Create logs directory if it doesn't exist"""
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    return logs_dir

def save_tickers_to_log(tickers, name, logs_dir):
    """Save tickers to a log file with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name.lower().replace(' ', '_')}_{timestamp}.json"
    filepath = os.path.join(logs_dir, filename)
    
    data = {
        "timestamp": timestamp,
        "adapter": name,
        "count": len(tickers),
        "tickers": tickers
    }
    
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    
    return filepath

def test_adapter(adapter_class, expected_count, name, logs_dir):
    print(f"\nTesting {name}...")
    try:
        adapter = adapter_class()
        tickers = adapter.fetch_tickers()
        
        print(f"Number of tickers fetched: {len(tickers)}")
        print(f"First 5 tickers: {tickers[:5]}")
        
        assert len(tickers) > 0, f"No tickers returned from {name}"
        assert len(tickers) <= expected_count, f"Too many tickers returned from {name}"
        assert all(isinstance(t, str) for t in tickers), f"Non-string tickers found in {name}"
        assert all(t.strip() == t for t in tickers), f"Tickers with whitespace found in {name}"
        
        # Save tickers to log file
        log_file = save_tickers_to_log(tickers, name, logs_dir)
        print(f"Tickers saved to: {log_file}")
        
        print(f"{name} test passed successfully!")
        return True
        
    except Exception as e:
        print(f"{name} test failed: {str(e)}")
        return False

def main():
    logs_dir = ensure_logs_directory()
    
    adapters = [
        (FMPNDAQ100TickerAdapter, 100, "NASDAQ-100"),
        (WikiSPY500TickerAdapter, 500, "S&P 500")
    ]
    
    success = True
    for adapter_class, expected_count, name in adapters:
        if not test_adapter(adapter_class, expected_count, name, logs_dir):
            success = False
    
    if success:
        print("\nAll adapter tests passed!")
        print(f"Log files can be found in the {logs_dir}/ directory")
    else:
        print("\nSome adapter tests failed!")

if __name__ == "__main__":
    main() 