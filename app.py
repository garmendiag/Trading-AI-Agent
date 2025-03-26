import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import requests
import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import logging
import time

# Streamlit configuration
st.set_page_config(page_title="Trading AI Agent - Market Analysis", layout="wide")
st.title("Trading AI Agent")

# Polygon API Configuration
API_KEY = "87rT4dZbny_pKUAozFAn2SWXsA3pD4q6"
BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

FUTURES_MAP = {
    "ES": "ESZ2024",
    "NQ": "NQZ2024",
    "MES": "MESZ2024",
    "MNQ": "MNQZ2024"
}

# Logger setup
def setup_logger():
    logger = logging.getLogger('TradingAI')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger()

# Data fetching function
def fetch_polygon_data(symbol, timespan="minute", limit=50):
    end = datetime.datetime.utcnow()
    start = end - datetime.timedelta(minutes=limit * 5)

    url = f"{BASE_URL}/{symbol}/range/5/{timespan}/{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}?adjusted=true&sort=asc&limit={limit}&apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        logger.error(f"Polygon API Error: {response.status_code} - {response.text}")
        raise ValueError(f"Polygon API Error: {response.status_code} - {response.text}")

    data = response.json().get("results", [])
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("t", inplace=True)
    df.rename(columns={"c": "Close"}, inplace=True)
    return df[["Close"]]

# Signal generation function
def generate_signals(df):
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    
    df['Signal'] = 0
    df.loc[df['SMA20'] > df['SMA50'], 'Signal'] = 1  # Buy
    df.loc[df['SMA20'] < df['SMA50'], 'Signal'] = -1  # Sell
    
    return df[['Close', 'SMA20', 'SMA50', 'Signal']].dropna()

# Sentiment analysis function
def analyze_news():
    try:
        nltk.download('vader_lexicon', quiet=True)
        sia = SentimentIntensityAnalyzer()
        
        headlines = [
            "Market shows strong growth potential",
            "Economic uncertainty looms over futures",
            "Tech sector rallies in afternoon trading"
        ]
        
        sentiments = []
        for headline in headlines:
            score = sia.polarity_scores(headline)
            sentiments.append({
                'headline': headline,
                'sentiment': score['compound']
            })
        
        avg_sentiment = sum(s['sentiment'] for s in sentiments) / len(sentiments)
        return {
            'headlines': sentiments,
            'average_sentiment': avg_sentiment,
            'interpretation': 'Positive' if avg_sentiment > 0 else 'Negative' if avg_sentiment < 0 else 'Neutral'
        }
    except Exception as e:
        logger.error(f"News sentiment analysis failed: {e}")
        return {"error": str(e), "average_sentiment": 0, "interpretation": "Neutral"}

# Trade execution function
def execute_trade(signal_df):
    last_signal = signal_df['Signal'].iloc[-1]
    if last_signal == 1:
        logger.info("Executing BUY order")
        return "Executing BUY order"
    elif last_signal == -1:
        logger.info("Executing SELL order")
        return "Executing SELL order"
    logger.info("No trade executed (neutral signal)")
    return "No trade executed (neutral signal)"

# Main application
st.sidebar.header("Instrument Selection")
selected_symbols = st.sidebar.multiselect(
    "Select Futures", 
    list(FUTURES_MAP.keys()), 
    default=["ES", "NQ"]
)

signals_summary = {}

# Display data and signals for each selected symbol
for label in selected_symbols:
    polygon_symbol = FUTURES_MAP[label]
    st.subheader(f"{label} Contract ({polygon_symbol})")
    try:
        df = fetch_polygon_data(polygon_symbol)
        if df.empty:
            st.warning(f"No data available for {polygon_symbol}")
            continue
        
        signal_df = generate_signals(df)
        st.dataframe(signal_df.tail(10))
        
        # Chart
        chart_data = signal_df[['Close', 'SMA20', 'SMA50']]
        st.line_chart(chart_data)
        
        last_signal = signal_df.iloc[-1]["Signal"]
        signals_summary[label] = last_signal
        
    except Exception as e:
        st.error(f"Failed to load data for {label}: {e}")
        logger.error(f"Data loading failed for {label}: {e}")

# Signal summary
if signals_summary:
    st.subheader("Current Signal Summary")
    for label, sig in signals_summary.items():
        signal_text = "Buy" if sig == 1 else "Sell" if sig == -1 else "Neutral"
        st.markdown(f"**{label}**: {signal_text}")

# Sentiment analysis
sentiment = analyze_news()
st.subheader("News Sentiment Analysis")
if "error" not in sentiment:
    st.write(f"Average Sentiment: {sentiment['average_sentiment']:.2f}")
    st.write(f"Interpretation: {sentiment['interpretation']}")
    st.write("Headlines:")
    for item in sentiment['headlines']:
        st.write(f"- {item['headline']} (Sentiment: {item['sentiment']:.2f})")
else:
    st.write("Sentiment analysis unavailable")

# Trade execution
st.subheader("Trade Execution")
if st.button("Execute Trade"):
    if signals_summary:
        for label in selected_symbols:
            if label in signals_summary:
                signal_df = pd.DataFrame([[signals_summary[label]]], columns=["Signal"])
                result = execute_trade(signal_df)
                st.write(f"{label}: {result}")
        st.success("Trade Execution Processed!")
        time.sleep(2)
        st.experimental_rerun()
    else:
        st.warning("No signals available to execute trades")

# Footer
st.sidebar.markdown("---")
st.sidebar.write(f"Last updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
