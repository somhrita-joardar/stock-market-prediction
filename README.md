
# Stock Market Prediction Tool

![Python](https://img.shields.io/badge/python-3.8+-blue)
![Libraries](https://img.shields.io/badge/libraries-yfinance%20|%20pandas%20|%20vaderSentiment%20|%20scikit--learn-green)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A customizable Python-based **stock market prediction and analysis tool** that combines technical indicators, fundamental data, sentiment analysis, and machine learning to generate insights and trading recommendations.

This repository includes an example script focused on analyzing a group of stocks (currently major Tata Group companies on NSE) to compute:
- A composite **group health score**
- Short-term **buy/sell/hold recommendation** for a flagship stock using Random Forest
- Competitor comparison, industry rating, and news sentiment

The framework is designed to be easily adapted for any stock, sector, or group of stocks.

## Features

- **Live stock data fetching** via `yfinance` (historical prices, technical indicators like SMA and RSI)
- **Fundamental metrics integration** (sales growth, profit growth, PE ratio, debt-to-equity, beta, etc.)
- **News sentiment analysis** using VADER
- **Custom weighted health score** for overall group/company strength
- **Machine learning prediction** (Random Forest Classifier) for 30-day price movement (>5% gain)
- **Competitor analysis**, industry growth rating, and basic filters
- Highly modular – change tickers and fundamentals to analyze any stock or portfolio

**Current Example**: Analyzes key Tata Group listed companies (TCS, Tata Motors, Tata Steel, Tata Consumer, Tata Communications) as a proxy for group health, with a specific trading signal for TCS.NS.

## Requirements

- Python 3.8+
- Libraries:
  - `yfinance`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `vaderSentiment`

Install via pip:
```bash
pip install yfinance pandas numpy scikit-learn vaderSentiment
```

## Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/somhrita-joardar/stock-market-prediction.git
   cd stock-market-prediction
   ```

2. Run the script:
   ```bash
   python main.py
   ```

Sample output includes:
- Group Health Score
- Stock Recommendation (Buy/Hold/Sell)
- Competitor comparison table
- Industry rating
- News sentiment feed
- Technical filter result

## Customization

- Change the `tickers` list to analyze different stocks.
- Update the `fundamentals` dictionary with latest quarterly data (script includes placeholders as of Q4 FY25).
- Modify weights in the health score calculation.
- Adjust the ML target (e.g., prediction horizon or return threshold).

## Disclaimer

This tool is for **educational and research purposes only**. It is not financial advice. Stock market predictions are inherently uncertain, and past performance does not guarantee future results. Always conduct your own research and consult qualified professionals before making investment decisions.
