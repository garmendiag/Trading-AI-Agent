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

# Function to fetch stock data from Yahoo Finance API
def fetch_stock_data(symbol, interval, range_val, use_demo=True):
    """
    Fetch stock data from Yahoo Finance API or generate demo data
    """
    if not use_demo:
        try:
            # Use the Yahoo Finance API
            sys.path.append('/opt/.manus/.sandbox-runtime')
            from data_api import ApiClient
            client = ApiClient()
            response = client.call_api('YahooFinance/get_stock_chart', 
                                query={'symbol': symbol, 
                                      'interval': interval, 
                                      'range': range_val,
                                      'includeAdjustedClose': True})
            
            # Extract data from response
            chart_data = response['chart']['result'][0]
            timestamps = chart_data['timestamp']
            quote_data = chart_data['indicators']['quote'][0]
            
            # Convert timestamps to datetime
            dates = [datetime.fromtimestamp(ts) for ts in timestamps]
            
            # Create DataFrame
            df = pd.DataFrame({
                'Date': dates,
                'Open': quote_data['open'],
                'High': quote_data['high'],
                'Low': quote_data['low'],
                'Close': quote_data['close'],
                'Volume': quote_data['volume']
            })
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            return df
            
        except Exception as e:
            st.error(f"Error fetching data from Yahoo Finance API: {e}")
            st.warning("Falling back to demo data...")
            return generate_demo_data(symbol, interval, range_val)
    else:
        return generate_demo_data(symbol, interval, range_val)

# Function to generate demo data
def generate_demo_data(symbol, interval, range_val):
    """Generate sample stock data for demo purposes"""
    end_date = datetime.now()
    
    if range_val == "1mo":
        days = 30
    elif range_val == "3mo":
        days = 90
    elif range_val == "6mo":
        days = 180
    elif range_val == "1y":
        days = 365
    elif range_val == "2y":
        days = 730
    else:  # 5y
        days = 1825
    
    start_date = end_date - timedelta(days=days)
    
    # Generate dates
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate price data with some randomness but trending
    np.random.seed(42)  # For reproducibility
    
    # Base price depends on the selected stock
    base_prices = {
        "AAPL": 180, "MSFT": 420, "GOOGL": 170, "AMZN": 180, 
        "META": 500, "TSLA": 175, "NVDA": 950, "JPM": 200, 
        "V": 280, "WMT": 60
    }
    
    base_price = base_prices.get(symbol, 100)
    
    # Generate a trend with some randomness
    trend = np.linspace(0, 0.3, len(date_range))  # Upward trend
    noise = np.random.normal(0, 0.02, len(date_range))  # Random noise
    
    # Create price series
    close_prices = base_price * (1 + trend + noise)
    
    # Add some volatility
    volatility = 0.015  # 1.5% daily volatility
    high_prices = close_prices * (1 + np.random.uniform(0, volatility, len(date_range)))
    low_prices = close_prices * (1 - np.random.uniform(0, volatility, len(date_range)))
    open_prices = close_prices * (1 + np.random.uniform(-volatility, volatility, len(date_range)))
    
    # Volume data
    volume = np.random.randint(1000000, 10000000, len(date_range))
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': date_range,
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    })
    
    # Resample based on interval
    if interval == "1wk":
        df = df.resample('W', on='Date').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).reset_index()
    elif interval == "1mo":
        df = df.resample('M', on='Date').agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        }).reset_index()
    
    return df

# Function to fetch stock insights from Yahoo Finance API
def fetch_stock_insights(symbol, use_demo=True):
    """
    Fetch stock insights from Yahoo Finance API or generate demo insights
    """
    if not use_demo:
        try:
            # Use the Yahoo Finance API
            sys.path.append('/opt/.manus/.sandbox-runtime')
            from data_api import ApiClient
            client = ApiClient()
            response = client.call_api('YahooFinance/get_stock_insights', 
                                query={'symbol': symbol})
            
            # Extract insights data
            finance_data = response['finance']['result']
            
            # Technical events data
            tech_events = finance_data['instrumentInfo']['technicalEvents']
            
            insights = {
                "shortTerm": {
                    "direction": tech_events['shortTermOutlook']['direction'],
                    "score": tech_events['shortTermOutlook']['score']
                },
                "intermediateTerm": {
                    "direction": tech_events['intermediateTermOutlook']['direction'],
                    "score": tech_events['intermediateTermOutlook']['score']
                },
                "longTerm": {
                    "direction": tech_events['longTermOutlook']['direction'],
                    "score": tech_events['longTermOutlook']['score']
                }
            }
            
            # Key technicals
            key_tech = finance_data['instrumentInfo']['keyTechnicals']
            insights["support"] = key_tech['support']
            insights["resistance"] = key_tech['resistance']
            
            # Add some additional metrics (these would need to be calculated or fetched from elsewhere)
            insights["rsi"] = 50  # Placeholder
            insights["macd"] = 0  # Placeholder
            
            return insights
            
        except Exception as e:
            st.error(f"Error fetching insights from Yahoo Finance API: {e}")
            st.warning("Falling back to demo insights...")
            return generate_demo_insights(symbol)
    else:
        return generate_demo_insights(symbol)

# Function to generate demo insights
def generate_demo_insights(symbol):
    """Generate sample insights for demo purposes"""
    directions = ["Bullish", "Bearish", "Neutral"]
    scores = [7, 4, 5]
    
    insights = {
        "shortTerm": {
            "direction": np.random.choice(directions),
            "score": np.random.choice(scores)
        },
        "intermediateTerm": {
            "direction": np.random.choice(directions),
            "score": np.random.choice(scores)
        },
        "longTerm": {
            "direction": np.random.choice(directions),
            "score": np.random.choice(scores)
        },
        "support": round(np.random.uniform(0.85, 0.95), 2),
        "resistance": round(np.random.uniform(1.05, 1.15), 2),
        "rsi": round(np.random.uniform(30, 70), 1),
        "macd": round(np.random.uniform(-2, 2), 2)
    }
    
    return insights

# Function to generate trading signals
def generate_signals(df):
    """Generate simple trading signals based on moving averages"""
    # Calculate moving averages for all selected periods
    for period in [5, 10, 20, 50, 100, 200]:
        if len(df) >= period:
            df[f'MA{period}'] = df['Close'].rolling(window=period).mean()
    
    # Generate signals based on MA20 and MA50 crossover
    if 'MA20' in df.columns and 'MA50' in df.columns:
        df['Signal'] = 0
        df.loc[df['MA20'] > df['MA50'], 'Signal'] = 1  # Buy signal
        df.loc[df['MA20'] < df['MA50'], 'Signal'] = -1  # Sell signal
        
        # Calculate signal changes
        df['SignalChange'] = df['Signal'].diff()
        
        # Generate signal points
        buy_signals = df[df['SignalChange'] == 1]
        sell_signals = df[df['SignalChange'] == -1]
    else:
        # Not enough data for signals
        buy_signals = pd.DataFrame()
        sell_signals = pd.DataFrame()
    
    return df, buy_signals, sell_signals

# Function to calculate performance metrics
def calculate_performance(df):
    """Calculate performance metrics based on signals"""
    # Initial investment
    initial_investment = 10000
    
    # Calculate returns
    df['Return'] = df['Close'].pct_change()
    
    # Calculate strategy returns (only invested when signal is 1)
    if 'Signal' in df.columns:
        df['StrategyReturn'] = df['Return'] * df['Signal'].shift(1)
    else:
        df['StrategyReturn'] = 0
    
    # Calculate cumulative returns
    df['CumulativeReturn'] = (1 + df['Return']).cumprod()
    df['CumulativeStrategyReturn'] = (1 + df['StrategyReturn']).cumprod()
    
    # Calculate metrics
    total_return = df['CumulativeReturn'].iloc[-1] - 1 if len(df) > 0 else 0
    strategy_return = df['CumulativeStrategyReturn'].iloc[-1] - 1 if len(df) > 0 else 0
    
    # Calculate annualized returns
    days = (df['Date'].iloc[-1] - df['Date'].iloc[0]).days if len(df) > 1 else 1
    years = days / 365
    
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    annualized_strategy_return = (1 + strategy_return) ** (1 / years) - 1 if years > 0 else 0
    
    # Calculate max drawdown
    df['Peak'] = df['CumulativeReturn'].cummax()
    df['Drawdown'] = (df['CumulativeReturn'] - df['Peak']) / df['Peak']
    max_drawdown = df['Drawdown'].min()
    
    df['StrategyPeak'] = df['CumulativeStrategyReturn'].cummax()
    df['StrategyDrawdown'] = (df['CumulativeStrategyReturn'] - df['StrategyPeak']) / df['StrategyPeak']
    strategy_max_drawdown = df['StrategyDrawdown'].min()
    
    return {
        'total_return': total_return,
        'strategy_return': strategy_return,
        'annualized_return': annualized_return,
        'annualized_strategy_return': annualized_strategy_return,
        'max_drawdown': max_drawdown,
        'strategy_max_drawdown': strategy_max_drawdown
    }

# Function to calculate additional technical indicators
def calculate_technical_indicators(df):
    """Calculate additional technical indicators"""
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    
    # Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['20dSTD'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['20dSTD'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['20dSTD'] * 2)
    
    return df

# Main dashboard content
st.markdown("## Market Overview")

# Fetch data
with st.spinner(f"Fetching data for {selected_stock}..."):
    df = fetch_stock_data(selected_stock, selected_interval, selected_range, use_demo=demo_mode)
    df = calculate_technical_indicators(df)

# Market summary
st.markdown("<div class='market-summary'>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

# Current price and change
current_price = df['Close'].iloc[-1]
prev_price = df['Close'].iloc[-2] if len(df) > 1 else df['Close'].iloc[-1]
price_change = current_price - prev_price
price_change_pct = (price_change / prev_price) * 100

with col1:
    st.metric(
        "Current Price", 
        f"${current_price:.2f}", 
        f"{price_change_pct:.2f}%"
    )

# Trading volume
current_volume = df['Volume'].iloc[-1]
avg_volume = df['Volume'].mean()
volume_change_pct = ((current_volume - avg_volume) / avg_volume) * 100

with col2:
    st.metric(
        "Volume", 
        f"{current_volume:,.0f}", 
        f"{volume_change_pct:.2f}% vs Avg"
    )

# 52-week range
if len(df) >= 252:  # Approximately 252 trading days in a year
    year_data = df.tail(252)
    year_low = year_data['Low'].min()
    year_high = year_data['High'].max()
else:
    year_low = df['Low'].min()
    year_high = df['High'].max()

with col3:
    st.metric(
        "52-Week Range", 
        f"${year_low:.2f} - ${year_high:.2f}"
    )

# Latest signal
if 'Signal' in df.columns:
    latest_signal = df['Signal'].iloc[-1]
    signal_text = "BUY" if latest_signal == 1 else "SELL" if latest_signal == -1 else "NEUTRAL"
    signal_class = "buy-signal" if latest_signal == 1 else "sell-signal" if latest_signal == -1 else "neutral-signal"
else:
    signal_text = "NEUTRAL"
    signal_class = "neutral-signal"

with col4:
    st.markdown(f"**Latest Signal**")
    st.markdown(f"<span class='{signal_class}'>{signal_text}</span>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Display stock chart
st.markdown(f"### {selected_stock} Stock Price")

# Create the chart based on selected type
fig = go.Figure()

if chart_type == "Candlestick":
    fig.add_trace(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))
elif chart_type == "Line":
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Close'],
        mode='lines',
        name='Close Price',
        line=dict(color='blue', width=2)
    ))
else:  # OHLC
    fig.add_trace(go.Ohlc(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))

# Add moving averages if selected
if show_ma:
    for period in ma_periods:
        if f'MA{period}' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Date'],
                y=df[f'MA{period}'],
                mode='lines',
                name=f'{period}-day MA',
                line=dict(width=1.5)
            ))

# Add Bollinger Bands
if 'Upper_Band' in df.columns and 'Lower_Band' in df.columns:
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Upper_Band'],
        mode='lines',
        name='Upper Band',
        line=dict(color='rgba(173, 216, 230, 0.7)', width=1, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Lower_Band'],
        mode='lines',
        name='Lower Band',
        line=dict(color='rgba(173, 216, 230, 0.7)', width=1, dash='dash'),
        fill='tonexty',
        fillcolor='rgba(173, 216, 230, 0.1)'
    ))

fig.update_layout(
    title=f"{selected_stock} Stock Price ({selected_interval} interval, {selected_range} range)",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    height=500,
    xaxis_rangeslider_visible=False,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
st.plotly_chart(fig, use_container_width=True)

# Volume chart
if show_volume:
    st.markdown("### Trading Volume")
    volume_fig = px.bar(
        df, 
        x='Date', 
        y='Volume',
        title=f"{selected_stock} Trading Volume"
    )
    volume_fig.update_layout(
        height=300,
        xaxis_title="Date",
        yaxis_title="Volume"
    )
    st.plotly_chart(volume_fig, use_container_width=True)

# Technical indicators
st.markdown("### Technical Indicators")
col1, col2 = st.columns(2)

with col1:
    # RSI Chart
    if 'RSI' in df.columns:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=1.5)
        ))
        
        # Add overbought/oversold lines
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
        
        rsi_fig.update_layout(
            title="Relative Strength Index (RSI)",
            xaxis_title="Date",
            yaxis_title="RSI",
            height=300,
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(rsi_fig, use_container_width=True)

with col2:
    # MACD Chart
    if 'MACD' in df.columns and 'Signal_Line' in df.columns:
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['MACD'],
            mode='lines',
            name='MACD',
            line=dict(color='blue', width=1.5)
        ))
        macd_fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Signal_Line'],
            mode='lines',
            name='Signal Line',
            line=dict(color='red', width=1.5)
        ))
        
        # Add MACD Histogram
        colors = ['green' if val >= 0 else 'red' for val in df['MACD_Histogram']]
        macd_fig.add_trace(go.Bar(
            x=df['Date'],
            y=df['MACD_Histogram'],
            name='Histogram',
            marker_color=colors
        ))
        
        macd_fig.update_layout(
            title="MACD (Moving Average Convergence Divergence)",
            xaxis_title="Date",
            yaxis_title="Value",
            height=300
        )
        st.plotly_chart(macd_fig, use_container_width=True)

# Trading Signals
st.markdown("## Trading Signals")

# Generate signals
df, buy_signals, sell_signals = generate_signals(df)

# Display signals chart
signal_fig = go.Figure()
signal_fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Close'],
    mode='lines',
    name='Close Price'
))

# Add moving averages used for signals
if 'MA20' in df.columns:
    signal_fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['MA20'],
        mode='lines',
        name='20-day MA',
        line=dict(color='orange')
    ))

if 'MA50' in df.columns:
    signal_fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['MA50'],
        mode='lines',
        name='50-day MA',
        line=dict(color='green')
    ))

# Add buy/sell signals
if not buy_signals.empty:
    signal_fig.add_trace(go.Scatter(
        x=buy_signals['Date'],
        y=buy_signals['Close'],
        mode='markers',
        name='Buy Signal',
        marker=dict(color='green', size=10, symbol='triangle-up')
    ))

if not sell_signals.empty:
    signal_fig.add_trace(go.Scatter(
        x=sell_signals['Date'],
        y=sell_signals['Close'],
        mode='markers',
        name='Sell Signal',
        marker=dict(color='red', size=10, symbol='triangle-down')
    ))

signal_fig.update_layout(
    title=f"{selected_stock} Trading Signals",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    height=500,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
st.plotly_chart(signal_fig, use_container_width=True)

# Signal summary
st.markdown("### Signal Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Buy Signals")
    if not buy_signals.empty:
        for _, row in buy_signals.tail(3).iterrows():
            st.markdown(f"**{row['Date'].strftime('%Y-%m-%d')}**: ${row['Close']:.2f}")
    else:
        st.markdown("No recent buy signals")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Sell Signals")
    if not sell_signals.empty:
        for _, row in sell_signals.tail(3).iterrows():
            st.markdown(f"**{row['Date'].strftime('%Y-%m-%d')}**: ${row['Close']:.2f}")
    else:
        st.markdown("No recent sell signals")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("#### Signal Statistics")
    buy_count = len(buy_signals)
    sell_count = len(sell_signals)
    total_signals = buy_count + sell_count
    
    st.markdown(f"**Total Signals**: {total_signals}")
    st.markdown(f"**Buy Signals**: {buy_count}")
    st.markdown(f"**Sell Signals**: {sell_count}")
    st.markdown("</div>", unsafe_allow_html=True)

# Performance metrics
st.markdown("## Performance Analysis")

# Calculate performance
performance = calculate_performance(df)

# Display metrics in columns
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Total Return", f"{performance['total_return']:.2%}")
    st.metric("Strategy Return", f"{performance['strategy_return']:.2%}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Annualized Return", f"{performance['annualized_return']:.2%}")
    st.metric("Strategy Annualized Return", f"{performance['annualized_strategy_return']:.2%}")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Max Drawdown", f"{performance['max_drawdown']:.2%}")
    st.metric("Strategy Max Drawdown", f"{performance['strategy_max_drawdown']:.2%}")
    st.markdown("</div>", unsafe_allow_html=True)

# Performance chart
st.markdown("### Equity Curve")
equity_fig = go.Figure()
equity_fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['CumulativeReturn'],
    mode='lines',
    name='Buy & Hold'
))
equity_fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['CumulativeStrategyReturn'],
    mode='lines',
    name='Strategy',
    line=dict(color='green')
))
equity_fig.update_layout(
    title=f"{selected_stock} Equity Curve",
    xaxis_title="Date",
    yaxis_title="Cumulative Return",
    height=400,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
st.plotly_chart(equity_fig, use_container_width=True)

# Drawdown chart
st.markdown("### Drawdown Analysis")
drawdown_fig = go.Figure()
drawdown_fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['Drawdown'],
    mode='lines',
    name='Buy & Hold Drawdown',
    line=dict(color='red')
))
drawdown_fig.add_trace(go.Scatter(
    x=df['Date'],
    y=df['StrategyDrawdown'],
    mode='lines',
    name='Strategy Drawdown',
    line=dict(color='orange')
))
drawdown_fig.update_layout(
    title=f"{selected_stock} Drawdown Analysis",
    xaxis_title="Date",
    yaxis_title="Drawdown",
    height=400,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    )
)
st.plotly_chart(drawdown_fig, use_container_width=True)

# Technical Insights
st.markdown("## Technical Insights")

# Fetch insights
with st.spinner(f"Fetching insights for {selected_stock}..."):
    insights = fetch_stock_insights(selected_stock, use_demo=demo_mode)

# Display insights in columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Short Term Outlook")
    direction = insights['shortTerm']['direction']
    direction_class = "buy-signal" if direction == "Bullish" else "sell-signal" if direction == "Bearish" else "neutral-signal"
    st.markdown(f"**Direction:** <span class='{direction_class}'>{direction}</span>", unsafe_allow_html=True)
    st.markdown(f"**Score:** {insights['shortTerm']['score']}/10")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Intermediate Term Outlook")
    direction = insights['intermediateTerm']['direction']
    direction_class = "buy-signal" if direction == "Bullish" else "sell-signal" if direction == "Bearish" else "neutral-signal"
    st.markdown(f"**Direction:** <span class='{direction_class}'>{direction}</span>", unsafe_allow_html=True)
    st.markdown(f"**Score:** {insights['intermediateTerm']['score']}/10")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("### Long Term Outlook")
    direction = insights['longTerm']['direction']
    direction_class = "buy-signal" if direction == "Bullish" else "sell-signal" if direction == "Bearish" else "neutral-signal"
    st.markdown(f"**Direction:** <span class='{direction_class}'>{direction}</span>", unsafe_allow_html=True)
    st.markdown(f"**Score:** {insights['longTerm']['score']}/10")
    st.markdown("</div>", unsafe_allow_html=True)

# Technical indicators
st.markdown("### Technical Indicators")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Support Level", f"${round(df['Close'].iloc[-1] * insights['support'], 2)}")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.metric("Resistance Level", f"${round(df['Close'].iloc[-1] * insights['resistance'], 2)}")
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    rsi_value = df['RSI'].iloc[-1] if 'RSI' in df.columns else insights['rsi']
    # Fixed delta_color to use only accepted values: 'normal', 'inverse', or 'off'
    rsi_color = "inverse" if rsi_value < 30 or rsi_value > 70 else "normal"
    st.metric("RSI", f"{rsi_value:.1f}", delta_color=rsi_color)
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    macd_value = df['MACD'].iloc[-1] if 'MACD' in df.columns else insights['macd']
    macd_signal = df['Signal_Line'].iloc[-1] if 'Signal_Line' in df.columns else 0
    macd_diff = macd_value - macd_signal
    st.metric("MACD", f"{macd_value:.2f}", f"{macd_diff:.2f}")
    st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("### Dashboard Information")
st.info("""
This dashboard provides a comprehensive view of stock market data and trading signals.
- **Market Overview**: Displays stock price, volume data, and key technical indicators
- **Trading Signals**: Shows buy/sell signals based on moving average crossovers
- **Performance Analysis**: Compares strategy performance against buy & hold
- **Technical Insights**: Provides technical analysis and outlook

**Note**: When Demo Mode is enabled, simulated data is used. Disable Demo Mode to fetch real data from Yahoo Finance API.
""")

# Last updated timestamp
st.markdown(f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
