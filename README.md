# AmpyFin Trading System - Hyper-Model

| Category | Badges |
|----------|--------|
| **Project** | [![Team Size](https://img.shields.io/badge/Team%20Size-6-blue)](https://github.com/yeonholee50/AmpyFin/graphs/contributors) [![Proprietary](https://img.shields.io/badge/Proprietary-Yes-red)](https://github.com/yeonholee50/AmpyFin) |
| **CI/CD** | [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) [![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/yeonholee50/AmpyFin/pre-commit.yml?branch=main&label=linting)](https://github.com/yeonholee50/AmpyFin/actions/workflows/pre-commit.yml) |
| **Quality** | [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![Quality Gate Status](https://img.shields.io/badge/Quality%20Gate-Passing-success)](https://github.com/yeonholee50/AmpyFin) |
| **Package** | [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/) |
| **Meta** | [![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](https://opensource.org/licenses/MIT) |

## Introduction

AmpyFin Hyper-Model is a specialized proprietary subset of the AmpyFin trading system specifically designed to be more opportunistic and less pragmatic in its trading approach. While the standard AmpyFin system balances risk and reward, the Hyper-Model is configured through `control.py` to take more calculated risks, potentially yielding higher returns in favorable market conditions.

This proprietary model leverages an advanced AI-powered trading framework with carefully adjusted parameters to identify and capitalize on higher-potential trading opportunities within the NASDAQ-100. The Hyper-Model employs more aggressive strategy weighting, specialized profit/loss ratios, and optimized asset allocation limits that favor opportunistic positions.

### Mission

The primary goal of AmpyFin Hyper-Model is to:

- **Deliver Superior Trading Performance**: Implement proprietary trading algorithms for higher-risk, higher-reward scenarios
- **Leverage Advanced Analytics**: Apply cutting-edge machine learning techniques to financial markets
- **Optimize for Opportunistic Trading**: Fine-tune parameters to capture high-potential market movements
- **Provide Competitive Advantage**: Offer a sophisticated trading system that outperforms standard approaches
- **Balance Aggression with Intelligence**: Create a more opportunistic model without sacrificing data-driven decision making

AmpyFin Hyper-Model represents a significant advancement in algorithmic trading, offering a proprietary implementation specifically designed for traders seeking more aggressive market opportunities.

## Tutorial Playlist

For a comprehensive guide on how to use AmpyFin, check out our [YouTube tutorial playlist](https://www.youtube.com/playlist?list=PL7hzGb_OBFXaB7VGaXXiJk5uq2CXQqi-O).

## Features

### Data Collection

- **yfinance**: Primary source for historical price data and technical indicators
- **Financial Modeling Prep API**: Retrieves NASDAQ-100 tickers for market insights
- **Polygon API**: Alternative source for real-time market data if needed

### Data Storage

- **MongoDB**: Securely stores all data and trading logs for historical analysis

### Machine Learning

AmpyFin uses a ranked ensemble learning system that dynamically evaluates each strategy's performance and adjusts their influence in the final decision accordingly.

### Trading Strategies

- **Mean Reversion**: Predicts asset prices will return to their historical average
- **Momentum**: Capitalizes on prevailing market trends
- **Arbitrage**: Identifies and exploits price discrepancies between related assets
- **AI-Driven Custom Strategies**: Continuously refined through machine learning

### Dynamic Ranking System

Each strategy is evaluated based on performance metrics and assigned a weight using a sophisticated ranking algorithm. The system adapts to changing market conditions by prioritizing high-performing strategies.

## Architecture

### Core Components

| Component | Description |
|-----------|-------------|
| `control.py` | Configuration interface for trading parameters |
| `trading_client.py` | Executes trades based on algorithmic decisions |
| `ranking_client.py` | Evaluates and ranks trading strategies |
| `TradeSim/main.py` | Training and testing environment for strategies |
| `strategies/*` | Implementation of various trading algorithms |
| `helper_files/*` | Utility functions for client operations |
| `utils/*` | General utility functions for data processing |

## Installation

### Prerequisites

- Python 3.8+
- MongoDB
- TA-Lib

### Setup

1. **Clone the repository**

```bash
git clone https://github.com/AmpyFin/Hyper-Model.git
cd Hyper-Model
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Install TA-Lib**

TA-Lib is required for technical indicators. Installation options:
- [TA-Lib Python Original](https://github.com/TA-Lib/ta-lib-python)
- [TA-Lib Python Easy Installation](https://github.com/cgohlke/talib-build/releases)

4. **Configure API keys**

You need to sign up for the following services to obtain API keys:

- [Polygon.io](https://polygon.io/) - For market data
- [Financial Modeling Prep](https://financialmodelingprep.com/) - For financial data
- [Alpaca](https://alpaca.markets/) - For trading execution
- [Weights & Biases](https://wandb.ai/) - For experiment tracking

Create a `config.py` file based on the template:

```python
POLYGON_API_KEY = "your_polygon_api_key"
FINANCIAL_PREP_API_KEY = "your_fmp_api_key"
MONGO_DB_USER = "your_mongo_user"
MONGO_DB_PASS = "your_mongo_password"
API_KEY = "your_alpaca_api_key"
API_SECRET = "your_alpaca_secret_key"
BASE_URL = "https://paper-api.alpaca.markets"  # Paper trading (safe for testing)
# BASE_URL = "https://api.alpaca.markets"      # Live trading (uses real money)
mongo_url = "your_mongo_connection_string"
```

> ⚠️ **IMPORTANT**: The default configuration uses Alpaca's paper trading environment. To switch to live trading (using real money), change the BASE_URL to "https://api.alpaca.markets". Only do this once you've thoroughly tested your strategies and understand the risks involved.

5. **Set up MongoDB**

- Create a MongoDB cluster (e.g., via MongoDB Atlas)
- Configure network access for your IP address
- Update the connection string in `config.py`

6. **Run the setup script**

```bash
python setup.py
```

This initializes the database structure required for AmpyFin.

7. **Set up Weights & Biases**

```bash
wandb login 'wandb_api_key'
```

## Usage

### Running the System

Start the ranking and trading systems in separate terminals:

```bash
python ranking_client.py
python trading_client.py
```

### Training and Testing

These are separate operations that can be performed with the TradeSim module:

#### Training

1. Set the mode in `control.py`:
```python
mode = 'train'
```

2. Run the training module:
```bash
python TradeSim/main.py
```

#### Testing

1. Set the mode in `control.py`:
```python
mode = 'test'
```

2. Run the testing module:
```bash
python TradeSim/main.py
```

### Deploying a Model

1. Set the mode in `control.py`:
```python
mode = 'push'
```

2. Push your model to MongoDB:
```bash
python TradeSim/main.py
```

## Important Notes

For live trading, it's recommended to:
1. Train the system by running `ranking_client.py` for at least two weeks, or
2. Train using the TradeSim module and push changes to MongoDB before executing trades

This ensures the system has properly ranked strategies before making investment decisions.

## Logging

- `system.log`: Tracks major system events and errors
- `rank_system.log`: Records ranking-related events and updates

## Contributing

This is a proprietary trading system and is not open for public contributions. The codebase is maintained by an internal development team. If you have suggestions or have identified issues, please contact the system administrators directly.

## License

This project is proprietary software. All rights reserved. Unauthorized copying, distribution, modification, public display, or public performance of this proprietary software is an infringement of the copyright holder's rights.
