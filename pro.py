import yfinance as yf
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Define RSI function
def compute_rsi(data, periods=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# Define tickers and map to fundamentals keys
tickers = ["TCS.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TATACONSUM.NS", "TATACOMM.NS"]
ticker_to_fundamental = {
    "TCS.NS": "TCS", "TATAMOTORS.NS": "TATAMOTORS", "TATASTEEL.NS": "TATASTEEL",
    "TATACONSUM.NS": "TATACONSUM", "TATACOMM.NS": "TATACOMM"
}

# Fetch subsidiary data (time-series for TCS)
data = {ticker: yf.Ticker(ticker).history(period="5y") for ticker in tickers}
tcs_data = data["TCS.NS"].copy()
tcs_data['SMA_50'] = tcs_data['Close'].rolling(window=50).mean()
tcs_data['RSI'] = compute_rsi(tcs_data['Close'], 14)

# Financial data (updated with TCS Q4 FY25)
fundamentals = {
    "TCS": {
        "Sales_Growth": 5.29,  # Q4 FY25 YoY
        "Profit_Growth": -1.69,  # Q4 FY25 YoY
        "PAT_Growth": -1.69,  # Same as profit
        "Debt_to_Equity": 0.0,
        "PE_Ratio": 25.46,
        "Beta": 0.5
    },
    "TATAMOTORS": {
        "Sales_Growth": 26.6,
        "Profit_Growth": 100,  # Capped
        "PAT_Growth": 100,  # Capped
        "Debt_to_Equity": 0.5,
        "PE_Ratio": 7.7,
        "Beta": 1.2
    },
    "TATASTEEL": {
        "Sales_Growth": 4.6,
        "Profit_Growth": 22.1,
        "PAT_Growth": 22.1,
        "Debt_to_Equity": 0.3,
        "PE_Ratio": 6.3,
        "Beta": 1.1
    },
    "TATACONSUM": {
        "Sales_Growth": 17.35,
        "Profit_Growth": 59.19,
        "PAT_Growth": 59.19,
        "Debt_to_Equity": 0.1,
        "PE_Ratio": 77.08,
        "Beta": 0.6
    },
    "TATACOMM": {
        "Sales_Growth": 8,
        "Profit_Growth": 10,
        "PAT_Growth": 10,
        "Debt_to_Equity": 0.4,
        "PE_Ratio": 35.2,
        "Beta": 0.9
    }
}

# Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
news = [
    "Tata Sons exits RBI registration",  # Neutral
    "Tata Neu struggles with synergy",  # Negative
    "Tata Capital IPO planned"  # Positive
]
sentiments = [analyzer.polarity_scores(text)['compound'] for text in news]
management_sentiment = sum(sentiments) / len(sentiments)  # e.g., -0.1

# Create features DataFrame for Tata Sons
features = pd.DataFrame({
    "Sales_Growth": [sum(fundamentals[ticker_to_fundamental[t]]["Sales_Growth"] for t in tickers) / len(tickers)],
    "Profit_Growth": [sum(fundamentals[ticker_to_fundamental[t]]["Profit_Growth"] for t in tickers) / len(tickers)],
    "PAT_Growth": [sum(fundamentals[ticker_to_fundamental[t]]["PAT_Growth"] for t in tickers) / len(tickers)],
    "Debt_to_Equity": [sum(fundamentals[ticker_to_fundamental[t]]["Debt_to_Equity"] for t in tickers) / len(tickers)],
    "PE_Ratio": [sum(fundamentals[ticker_to_fundamental[t]]["PE_Ratio"] for t in tickers) / len(tickers)],
    "CapEx_to_Revenue": [0.1],  # ₹30,000 Cr infusion
    "Business_Outcome_Score": [3],  # Mixed
    "Portfolio_Diversity": [5],  # High
    "Product_Growth_Score": [4],  # Strong EV/AI
    "Management_Sentiment": [management_sentiment],
    "Team_Stability": [3],  # Board changes
    "Beta": [sum(fundamentals[ticker_to_fundamental[t]]["Beta"] for t in tickers) / len(tickers)],
    "Industry_Growth": [10]
})

# Health score
weights = {
    "Sales_Growth": 0.15, "Profit_Growth": 0.2, "PAT_Growth": 0.25, "Debt_to_Equity": -0.15,
    "PE_Ratio": -0.1, "CapEx_to_Revenue": 0.15, "Business_Outcome_Score": 0.1,
    "Portfolio_Diversity": 0.15, "Product_Growth_Score": 0.1, "Management_Sentiment": 0.1,
    "Team_Stability": 0.1, "Beta": -0.05, "Industry_Growth": 0.15
}
health_score = sum(min(float(features[k].iloc[0]), 100) * v for k, v in weights.items()) * 100
print(f"Tata Sons Health Score: {health_score:.2f}")

# Buy/Sell for TCS
tcs_features = tcs_data[["Close", "SMA_50", "RSI"]].dropna()
tcs_features["Sales_Growth"] = fundamentals["TCS"]["Sales_Growth"]
tcs_features["Profit_Growth"] = fundamentals["TCS"]["Profit_Growth"]
tcs_features["PAT_Growth"] = fundamentals["TCS"]["PAT_Growth"]
tcs_features["Debt_to_Equity"] = fundamentals["TCS"]["Debt_to_Equity"]
tcs_features["PE_Ratio"] = fundamentals["TCS"]["PE_Ratio"]
tcs_features["Beta"] = fundamentals["TCS"]["Beta"]
tcs_features["CapEx_to_Revenue"] = 0.1
tcs_features["Business_Outcome_Score"] = 3
tcs_features["Portfolio_Diversity"] = 5
tcs_features["Product_Growth_Score"] = 4
tcs_features["Industry_Growth"] = 10
tcs_features["Management_Sentiment"] = management_sentiment

X = tcs_features.drop(columns=["Management_Sentiment"])
y = (tcs_data["Close"].shift(-30) > tcs_data["Close"] * 1.05).astype(int).loc[X.index]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
train_size = int(0.8 * len(X_scaled))
if train_size > 0:
    model = RandomForestClassifier(random_state=42)
    model.fit(X_scaled[:train_size], y[:train_size])
    recommendation = model.predict(X_scaled[-1:])[0]
    print(f"TCS Recommendation: {'Buy' if recommendation == 1 else 'Sell' if recommendation == -1 else 'Hold'}")
else:
    print("TCS Recommendation: Insufficient data for prediction")

# Competitor analysis
competitors = {
    "Infosys": {"Sales_Growth": 7, "PE_Ratio": 22.6, "Beta": 0.6},
    "Mahindra": {"Sales_Growth": 15, "PE_Ratio": 15, "Beta": 1.0}
}
print("Competitors:", pd.DataFrame(competitors).T)

# Industry rating
industry_rating = features["Industry_Growth"].iloc[0] * 10
print(f"Industry Rating: {industry_rating:.2f} (High Potential)")

# News feed
news_feed = [{"title": t, "sentiment": s} for t, s in zip(news, sentiments)]
print("Industry News Feed:", news_feed)

# Technical filter
filters = (features["Debt_to_Equity"] < 0.5) & (features["Sales_Growth"] > 10) & (features["PE_Ratio"] < 30)
print("Technical Filter: Tata Sons", "Passes" if filters.iloc[0] else "Fails")

