import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import sys
import json
from datetime import datetime, timedelta
import time

# Must be the first Streamlit command
st.set_page_config(
    page_title="MANUS Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
    }
    .card {
        border-radius: 5px;
        padding: 20px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .market-summary {
        background-color: #e3f2fd;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 20px;
    }
    .buy-signal {
        color: #4CAF50;
        font-weight: bold;
    }
    .sell-signal {
        color: #F44336;
        font-weight: bold;
    }
    .neutral-signal {
        color: #FF9800;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Dashboard title
st.markdown("<h1 class='main-header'>MANUS Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Market Analysis and Trading Signals</p>", unsafe_allow_html=True)

# Sidebar for controls
with st.sidebar:
    st.title("Dashboard Controls")
    
    # Stock selection
    stock_options = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", "V", "WMT"]
    selected_stock = st.selectbox("Select Stock", stock_options)
    
    # Time range selection
    time_range_options = ["1mo", "3mo", "6mo", "1y", "2y", "5y"]
    selected_range = st.selectbox("Select Time Range", time_range_options)
    
    # Interval selection
    interval_options = ["1d", "1wk", "1mo"]
    selected_interval = st.selectbox("Select Interval", interval_options)
    
    # Chart type selection
    chart_type = st.radio("Chart Type", ["Candlestick", "Line", "OHLC"])
    
    # Technical indicators
    st.markdown("### Technical Indicators")
    show_ma = st.checkbox("Moving Averages", value=True)
    ma_periods = st.multiselect("MA Periods", [5, 10, 20, 50, 100, 200], default=[20, 50])
    
    show_volume = st.checkbox("Volume", value=True)
    
    # Demo mode toggle
    demo_mode = st.checkbox("Demo Mode", value=True)
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This dashboard provides market analysis and trading signals using Yahoo Finance data.")
