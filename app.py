# requirements.txt
pandas
numpy
matplotlib
seaborn
yfinance
nltk
scikit-learn
requests

# polygon_data.py
import requests
import pandas as pd
import datetime

API_KEY = "87rT4dZbny_pKUAozFAn2SWXsA3pD4q6"
BASE_URL = "https://api.polygon.io/v2/aggs/ticker"

FUTURES_MAP = {
    "ES": "ESZ2024",
    "NQ": "NQZ2024",
    "MES": "MESZ2024",
    "MNQ": "MNQZ2024"
}

def fetch_polygon_data(symbol, timespan="minute", limit=50):
    end = datetime.datetime.utcnow()
    start = end - datetime.timedelta(minutes=limit * 5)

    url = f"{BASE_URL}/{symbol}/range/5/{timespan}/{start.date()}?adjusted=true&sort=asc&limit={limit}&apiKey={API_KEY}"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Polygon API Error: {response.status_code} - {response.text}")

    data = response.json().get("results", [])
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["t"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("t", inplace=True)
    df.rename(columns={"c": "Close"}, inplace=True)
    return df[["Close"]]

# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys

sys.path.append(".")

from polygon_data import FUTURES_MAP, fetch_polygon_data
from signal_generation.signal_manager import generate_signals
from sentiment_analysis.news_sentiment import analyze_news
from trading.trade_execution import execute_trade
from utils.logger import setup_logger

st.set_page_config(page_title="Trading AI Agent - Market Analysis", layout="wide")
st.title("Trading AI Agent")

logger = setup_logger()

st.sidebar.header("Instrument Selection")
selected_symbols = st.sidebar.multiselect("Select Futures", list(FUTURES_MAP.keys()), default=["ES", "NQ"])

signals_summary = {}

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
        last_signal = signal_df.iloc[-1]["Signal"]
        signals_summary[label] = last_signal
    except Exception as e:
        st.error(f"Failed to load data for {label}: {e}")

if signals_summary:
    st.subheader("Current Signal Summary")
    for label, sig in signals_summary.items():
        st.markdown(f"**{label}**: {sig}")

sentiment = analyze_news()
st.subheader("News Sentiment Analysis")
st.write(sentiment)

st.subheader("Trade Execution")
if st.button("Execute Trade"):
    for label in selected_symbols:
        if signals_summary.get(label):
            execute_trade(pd.DataFrame([[signals_summary[label]]], columns=["Signal"]))
    st.success("Trade Executed Successfully!")
    time.sleep(2)
    st.experimental_rerun()
