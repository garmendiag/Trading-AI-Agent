# requirements.txt
streamlit
pandas
numpy
matplotlib
seaborn
yfinance
nltk
scikit-learn
requests

# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_collection.market_index import get_market_data
from signal_generation.signal_manager import generate_signals
from sentiment_analysis.news_sentiment import analyze_news
from trading.trade_execution import execute_trade
from utils.logger import setup_logger

st.set_page_config(page_title="Trading AI Agent - Market Analysis", layout="wide")
st.title("Trading AI Agent")

logger = setup_logger()

# Fetch Market Data
market_data = get_market_data()
st.subheader("Market Index Data")
st.dataframe(market_data)

# Generate Trading Signals
signals = generate_signals(market_data)
st.subheader("Trading Signals")
st.dataframe(signals)

# Sentiment Analysis
sentiment = analyze_news()
st.subheader("News Sentiment Analysis")
st.write(sentiment)

# Trade Execution Simulation
st.subheader("Trade Execution")
if st.button("Execute Trade"):
    execute_trade(signals)
    st.success("Trade Executed Successfully!")

# data_collection/market_index.py
import yfinance as yf

def get_market_data():
    ticker = "^GSPC"  # S&P 500 Index
    data = yf.download(ticker, period="1mo", interval="1d")
    return data

# signal_generation/signal_manager.py
import matplotlib.pyplot as plt
import streamlit as st

def generate_signals(market_data):
    market_data["Signal"] = market_data["Close"].pct_change().apply(lambda x: "Buy" if x > 0 else "Sell")
    
    # Simple visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(market_data.index, market_data["Close"], label="Close Price", color="blue")
    ax.scatter(market_data.index, market_data["Close"], c=market_data["Signal"].apply(lambda x: "green" if x == "Buy" else "red"), label="Signals")
    ax.legend()
    ax.set_title("Market Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.grid()
    st.pyplot(fig)
    
    return market_data[["Close", "Signal"]]

# sentiment_analysis/news_sentiment.py
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def analyze_news():
    nltk.download('vader_lexicon')
    analyzer = SentimentIntensityAnalyzer()
    sample_text = "Market is looking strong today with bullish momentum."
    return analyzer.polarity_scores(sample_text)

# trading/trade_execution.py
def execute_trade(signals):
    if signals.iloc[-1]["Signal"] == "Buy":
        print("Executing Buy Order")
    else:
        print("Executing Sell Order")

# utils/logger.py
import logging

def setup_logger():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger("TradingAI")
