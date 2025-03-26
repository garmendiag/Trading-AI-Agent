import streamlit as st

# Try importing required modules with error handling
try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import yfinance as yf
    import nltk
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import logging
except ImportError as e:
    st.error(f"Failed to import a required module: {e}")
    st.write("Please ensure all dependencies are listed in requirements.txt and installed.")
    st.stop()

# Streamlit configuration
st.set_page_config(page_title="Trading AI Agent - Market Analysis", layout="wide")
st.title("Trading AI Agent")

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TradingAI")

# Market data function
def get_market_data():
    ticker = "^GSPC"  # S&P 500 Index
    data = yf.download(ticker, period="1mo", interval="1d")
    return data

# Signal generation function
def generate_signals(market_data):
    market_data["Signal"] = market_data["Close"].pct_change().apply(lambda x: "Buy" if x > 0 else "Sell")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(market_data.index, market_data["Close"], label="Close Price", color="blue")
    ax.scatter(market_data.index, market_data["Close"], 
               c=market_data["Signal"].apply(lambda x: "green" if x == "Buy" else "red"), 
               label="Signals")
    ax.legend()
    ax.set_title("Market Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.grid()
    st.pyplot(fig)
    return market_data[["Close", "Signal"]]

# Sentiment analysis function
def analyze_news():
    nltk.download('vader_lexicon', quiet=True)
    analyzer = SentimentIntensityAnalyzer()
    sample_text = "Market is looking strong today with bullish momentum."
    return analyzer.polarity_scores(sample_text)

# Trade execution function
def execute_trade(signals):
    if signals.iloc[-1]["Signal"] == "Buy":
        logger.info("Executing Buy Order")
    else:
        logger.info("Executing Sell Order")

# Main application
try:
    market_data = get_market_data()
    st.subheader("Market Index Data")
    st.dataframe(market_data)

    signals = generate_signals(market_data)
    st.subheader("Trading Signals")
    st.dataframe(signals)

    sentiment = analyze_news()
    st.subheader("News Sentiment Analysis")
    st.write(sentiment)

    st.subheader("Trade Execution")
    if st.button("Execute Trade"):
        execute_trade(signals)
        st.success("Trade Executed Successfully!")
except Exception as e:
    st.error(f"Error in application: {e}")
    logger.error(f"Application error: {e}")
