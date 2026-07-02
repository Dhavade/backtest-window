import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import datetime
import pytz

# =============================================================================
# 1. PAGE SETUP & STYLING
# =============================================================================
st.set_page_config(
    page_title="NIFTY 5m ORB + EMA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
    <style>
    .main-title {
        font-size: 32px;
        font-weight: 800;
        color: #2196F3;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 20px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    .metric-box {
        background-color: #0e1117;
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.15);
    }
    .metric-label {
        font-size: 14px;
        color: #8b949e;
        margin-bottom: 5px;
    }
    .metric-value {
        font-size: 24px;
        font-weight: 700;
    }
    .positive-value {
        color: #26a69a;
    }
    .negative-value {
        color: #ef5350;
    }
    .neutral-value {
        color: #58a6ff;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📈 NIFTY 5m ORB + EMA Trading Strategy Dashboard</div>', unsafe_allow_html=True)

# =============================================================================
# 2. SIDEBAR CONFIGURATIONS
# =============================================================================
st.sidebar.header("🔧 Strategy Parameters")

# Date range selection (Yahoo Finance allows 5m data up to 60 days)
today = datetime.date.today()
min_date = today - datetime.timedelta(days=59)
default_start = today - datetime.timedelta(days=20)

st.sidebar.subheader("📅 Backtest Window (Max 60 Days)")
start_date = st.sidebar.date_input("Start Date", default_start, min_value=min_date, max_value=today)
end_date = st.sidebar.date_input("End Date", today, min_value=min_date, max_value=today)

# General trade configuration
st.sidebar.subheader("⚙️ General Settings")
allow_multiple = st.sidebar.checkbox("Allow concurrent EMA & ORB trades", value=False)

# ORB strategy parameters
st.sidebar.subheader("🛡️ ORB Setup")
enable_orb = st.sidebar.checkbox("Enable ORB Strategy", value=True)
use_inside_bar = st.sidebar.checkbox("Require 2nd candle Inside Bar", value=True)
orb_sl_type = st.sidebar.selectbox("ORB SL Type", ["Opposite Range", "Fixed Points", "Percentage"])
orb_sl_value = st.sidebar.number_input("ORB SL Value (Points/%)", min_value=1.0, value=30.0, step=5.0)
orb_tp_type = st.sidebar.selectbox("ORB TP Type", ["Risk Reward", "Fixed Points", "Percentage"])
orb_tp_value = st.sidebar.number_input("ORB TP Value (Points/%/RR)", min_value=0.5, value=2.0, step=0.5)

# EMA strategy parameters
st.sidebar.subheader("⚡ EMA Crossover Setup")
enable_ema = st.sidebar.checkbox("Enable EMA Strategy", value=True)
fast_ema_len = st.sidebar.number_input("Fast EMA Length", min_value=2, value=9)
slow_ema_len = st.sidebar.number_input("Slow EMA Length", min_value=5, value=20)
adx_len = st.sidebar.number_input("ADX Length", min_value=5, value=14)
adx_threshold = st.sidebar.number_input("ADX Threshold Filter", min_value=0, value=20)
ema20_lookback = st.sidebar.number_input("EMA 20 Slope Lookback", min_value=1, value=5)
ema_sl_type = st.sidebar.selectbox("EMA SL Type", ["Fixed Points", "Percentage"])
ema_sl_value = st.sidebar.number_input("EMA SL Value (Points/%)", min_value=1.0, value=40.0, step=5.0)
ema_tp_type = st.sidebar.selectbox("EMA TP Type", ["Risk Reward", "Fixed Points", "Percentage"])
ema_tp_value = st.sidebar.number_input("EMA TP Value (Points/%/RR)", min_value=0.5, value=2.0, step=0.5)

# Intraday timings
st.sidebar.subheader("⏰ Trading Session (IST)")
sq_off_hour = st.sidebar.number_input("Square-off Hour", min_value=9, max_value=16, value=15)
sq_off_min = st.sidebar.number_input("Square-off Minute", min_value=0, max_value=59, value=15)

# =============================================================================
# 3. INDICATOR CALCULATIONS
# =============================================================================
def calculate_adx(df, period=14):
    df_calc = df.copy()
    df_calc['tr1'] = df_calc['High'] - df_calc['Low']
    df_calc['tr2'] = (df_calc['High'] - df_calc['Close'].shift(1)).abs()
    df_calc['tr3'] = (df_calc['Low'] - df_calc['Close'].shift(1)).abs()
    df_calc['tr'] = df_calc[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    df_calc['up_move'] = df_calc['High'] - df_calc['High'].shift(1)
    df_calc['down_move'] = df_calc['Low'].shift(1) - df_calc['Low']
    
    df_calc['plus_dm'] = np.where((df_calc['up_move'] > df_calc['down_move']) & (df_calc['up_move'] > 0), df_calc['up_move'], 0.0)
    df_calc['minus_dm'] = np.where((df_calc['down_move'] > df_calc['up_move']) & (df_calc['down_move'] > 0), df_calc['down_move'], 0.0)
    
    df_calc['atr'] = df_calc['tr'].ewm(alpha=1/period, adjust=False).mean()
    df_calc['plus_di'] = 100 * df_calc['plus_dm'].ewm(alpha=1/period, adjust=False).mean() / df_calc['atr']
    df_calc['minus_di'] = 100 * df_calc['minus_dm'].ewm(alpha=1/period, adjust=False).mean() / df_calc['atr']
    
    df_calc['dx'] = 100 * (df_calc['plus_di'] - df_calc['minus_di']).abs() / (df_calc['plus_di'] + df_calc['minus_di'])
    df_calc['adx'] = df_calc['dx'].ewm(alpha=1/period, adjust=False).mean()
    return df_calc['plus_di'], df_calc['minus_di'], df_calc['adx']

# =============================================================================
# 4. DATA LOADING
# =============================================================================
@st.cache_data
def fetch_nifty_data(start, end):
    ticker = "^NSEI"
    # Download data with 5m interval
    df = yf.download(ticker, start=start, end=end, interval="5m")
    if df.empty:
        return pd.DataFrame()
    
    # Clean index and column headers (if MultiIndex columns returned by newer yfinance versions)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    # Timezone adjustments
    if df.index.tz is None:
        df = df.tz_localize('UTC').tz_convert('Asia/Kolkata')
    else:
        df = df.tz_convert('Asia/Kolkata')
        
    df['Date'] = df.index.date
    df['Time'] = df.index.time
    return df

with st.spinner("Fetching NIFTY 5-minute data from Yahoo Finance..."):
    # Formatting start/end dates for yfinance
    y_start = start_date.strftime("%Y-%m-%d")
    y_end = (end_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    df = fetch_nifty_data(y_start, y_end)

if df.empty:
    st.error("⚠️ No data was found for the selected date range. Ensure the dates selected are weekdays and market days.")
    st.stop()

# Calculate indicator values
df['ema9'] = df['Close'].ewm(span=fast_ema_len, adjust=False).mean()
df['ema20'] = df['Close'].ewm(span=slow_ema_len, adjust=False).mean()
df['plus_di'], df['minus_di'], df['adx'] = calculate_adx(df, adx_len)

# Precalculate offsets for logic
df['ema20_prev_slope'] = df['ema20'].shift(ema20_lookback)
df['ema9_prev'] = df['ema9'].shift(1)
df['ema20_prev'] = df['ema20'].shift(1)

# Initialize columns to log signal indicators
df['orb_high_line'] = np.nan
df['orb_low_line'] = np.nan
df['entry_long_marker'] = np.nan
df['entry_short_marker'] = np.nan
df['exit_marker'] = np.nan
df['exit_reason_marker'] = ""

# =============================================================================
# 5. BACKTEST ENGINE
# =============================================================================
trades = []
daily_stats = []

for date, group in df.groupby('Date'):
    group = group.sort_index()
    if len(group) < 3:
        continue
    
    # State flags
    first_high = None
    first_low = None
    inside_bar = False
    bo_taken = False
    active_trade = None
    
    for i, (idx, row) in enumerate(group.iterrows()):
        bar_time = idx.time()
        
        # 1. Identify first candle (starts at 09:15)
        if i == 0:
            first_high = row['High']
            first_low = row['Low']
            df.loc[idx, 'orb_high_line'] = first_high
            df.loc[idx, 'orb_low_line'] = first_low
            daily_stats.append({
                'Date': date,
                'ORB High': round(first_high, 2),
                'ORB Low': round(first_low, 2),
                'Inside Bar': False,
                'ORB Taken': False
            })
            continue
        
        # Keep track of first levels for plotting throughout the day
        df.loc[idx, 'orb_high_line'] = first_high
        df.loc[idx, 'orb_low_line'] = first_low
        
        # 2. Check second candle (starts at 09:20)
        if i == 1:
            inside_bar = (row['High'] <= first_high) and (row['Low'] >= first_low)
            daily_stats[-1]['Inside Bar'] = inside_bar
        
        # 3. Check Exits
        if active_trade is not None:
            # End of day square off
            if bar_time >= datetime.time(sq_off_hour, sq_off_min):
                exit_price = row['Close']
                pnl = (exit_price - active_trade['entry_price']) if active_trade['direction'] == 'Long' else (active_trade['entry_price'] - exit_price)
                pnl_pct = (pnl / active_trade['entry_price']) * 100
                trades.append({
                    'Entry Time': active_trade['entry_time'],
                    'Exit Time': idx,
                    'Type': active_trade['direction'],
                    'Origin': active_trade['origin'],
                    'Entry Price': round(active_trade['entry_price'], 2),
                    'Exit Price': round(exit_price, 2),
                    'Exit Reason': 'Intraday Square-off',
                    'PnL (Points)': round(pnl, 2),
                    'PnL (%)': round(pnl_pct, 2)
                })
                df.loc[idx, 'exit_marker'] = exit_price
                df.loc[idx, 'exit_reason_marker'] = "SqOff"
                active_trade = None
                continue
            
            # Stop Loss & Take Profit limits
            if active_trade['direction'] == 'Long':
                if row['Low'] <= active_trade['sl_price']:
                    exit_price = active_trade['sl_price']
                    pnl = exit_price - active_trade['entry_price']
                    pnl_pct = (pnl / active_trade['entry_price']) * 100
                    trades.append({
                        'Entry Time': active_trade['entry_time'],
                        'Exit Time': idx,
                        'Type': 'Long',
                        'Origin': active_trade['origin'],
                        'Entry Price': round(active_trade['entry_price'], 2),
                        'Exit Price': round(exit_price, 2),
                        'Exit Reason': 'Stop Loss',
                        'PnL (Points)': round(pnl, 2),
                        'PnL (%)': round(pnl_pct, 2)
                    })
                    df.loc[idx, 'exit_marker'] = exit_price
                    df.loc[idx, 'exit_reason_marker'] = "SL"
                    active_trade = None
                elif row['High'] >= active_trade['tp_price']:
                    exit_price = active_trade['tp_price']
                    pnl = exit_price - active_trade['entry_price']
                    pnl_pct = (pnl / active_trade['entry_price']) * 100
                    trades.append({
                        'Entry Time': active_trade['entry_time'],
                        'Exit Time': idx,
                        'Type': 'Long',
                        'Origin': active_trade['origin'],
                        'Entry Price': round(active_trade['entry_price'], 2),
                        'Exit Price': round(exit_price, 2),
                        'Exit Reason': 'Take Profit',
                        'PnL (Points)': round(pnl, 2),
                        'PnL (%)': round(pnl_pct, 2)
                    })
                    df.loc[idx, 'exit_marker'] = exit_price
                    df.loc[idx, 'exit_reason_marker'] = "TP"
                    active_trade = None
                    
            elif active_trade['direction'] == 'Short':
                if row['High'] >= active_trade['sl_price']:
                    exit_price = active_trade['sl_price']
                    pnl = active_trade['entry_price'] - exit_price
                    pnl_pct = (pnl / active_trade['entry_price']) * 100
                    trades.append({
                        'Entry Time': active_trade['entry_time'],
                        'Exit Time': idx,
                        'Type': 'Short',
                        'Origin': active_trade['origin'],
                        'Entry Price': round(active_trade['entry_price'], 2),
                        'Exit Price': round(exit_price, 2),
                        'Exit Reason': 'Stop Loss',
                        'PnL (Points)': round(pnl, 2),
                        'PnL (%)': round(pnl_pct, 2)
                    })
                    df.loc[idx, 'exit_marker'] = exit_price
                    df.loc[idx, 'exit_reason_marker'] = "SL"
                    active_trade = None
                elif row['Low'] <= active_trade['tp_price']:
                    exit_price = active_trade['tp_price']
                    pnl = active_trade['entry_price'] - exit_price
                    pnl_pct = (pnl / active_trade['entry_price']) * 100
                    trades.append({
                        'Entry Time': active_trade['entry_time'],
                        'Exit Time': idx,
                        'Type': 'Short',
                        'Origin': active_trade['origin'],
                        'Entry Price': round(active_trade['entry_price'], 2),
                        'Exit Price': round(exit_price, 2),
                        'Exit Reason': 'Take Profit',
                        'PnL (Points)': round(pnl, 2),
                        'PnL (%)': round(pnl_pct, 2)
                    })
                    df.loc[idx, 'exit_marker'] = exit_price
                    df.loc[idx, 'exit_reason_marker'] = "TP"
                    active_trade = None
        
        # 4. Check Entries
        can_enter = (active_trade is None) or allow_multiple
        if not can_enter:
            continue
            
        # No entries after square-off time
        if bar_time >= datetime.time(sq_off_hour, sq_off_min):
            continue
            
        # A. ORB BREAKOUT
        allow_orb = (inside_bar and i >= 2) if use_inside_bar else (i >= 1)
        if enable_orb and allow_orb and not bo_taken:
            close_price = row['Close']
            
            if close_price > first_high:
                # Calculate SL
                if orb_sl_type == "Opposite Range":
                    sl_price = first_low
                elif orb_sl_type == "Fixed Points":
                    sl_price = close_price - orb_sl_value
                else:
                    sl_price = close_price * (1 - orb_sl_value / 100)
                
                # Calculate TP
                risk = close_price - sl_price
                if orb_tp_type == "Risk Reward":
                    tp_price = close_price + (risk * orb_tp_value)
                elif orb_tp_type == "Fixed Points":
                    tp_price = close_price + orb_tp_value
                else:
                    tp_price = close_price * (1 + orb_tp_value / 100)
                
                active_trade = {
                    'entry_time': idx,
                    'direction': 'Long',
                    'origin': 'ORB',
                    'entry_price': close_price,
                    'sl_price': sl_price,
                    'tp_price': tp_price
                }
                df.loc[idx, 'entry_long_marker'] = close_price
                bo_taken = True
                daily_stats[-1]['ORB Taken'] = True
                continue
                
            elif close_price < first_low:
                # Calculate SL
                if orb_sl_type == "Opposite Range":
                    sl_price = first_high
                elif orb_sl_type == "Fixed Points":
                    sl_price = close_price + orb_sl_value
                else:
                    sl_price = close_price * (1 + orb_sl_value / 100)
                
                # Calculate TP
                risk = sl_price - close_price
                if orb_tp_type == "Risk Reward":
                    tp_price = close_price - (risk * orb_tp_value)
                elif orb_tp_type == "Fixed Points":
                    tp_price = close_price - orb_tp_value
                else:
                    tp_price = close_price * (1 - orb_tp_value / 100)
                    
                active_trade = {
                    'entry_time': idx,
                    'direction': 'Short',
                    'origin': 'ORB',
                    'entry_price': close_price,
                    'sl_price': sl_price,
                    'tp_price': tp_price
                }
                df.loc[idx, 'entry_short_marker'] = close_price
                bo_taken = True
                daily_stats[-1]['ORB Taken'] = True
                continue
                
        # B. EMA CROSSOVER (After 12:00 PM)
        ema_time_ok = (bar_time >= datetime.time(12, 0))
        if enable_ema and ema_time_ok:
            ema_crossover  = (row['ema9_prev'] <= row['ema20_prev']) and (row['ema9'] > row['ema20'])
            ema_crossunder = (row['ema9_prev'] >= row['ema20_prev']) and (row['ema9'] < row['ema20'])
            
            ema20_slope_up   = row['ema20'] > row['ema20_prev_slope']
            ema20_slope_down = row['ema20'] < row['ema20_prev_slope']
            
            price_above = (row['Close'] > row['ema9']) and (row['Close'] > row['ema20'])
            price_below = (row['Close'] < row['ema9']) and (row['Close'] < row['ema20'])
            
            adx_trend = row['adx'] > adx_threshold
            close_price = row['Close']
            
            if ema_crossover and ema20_slope_up and price_above and adx_trend:
                if ema_sl_type == "Fixed Points":
                    sl_price = close_price - ema_sl_value
                else:
                    sl_price = close_price * (1 - ema_sl_value / 100)
                
                risk = close_price - sl_price
                if ema_tp_type == "Risk Reward":
                    tp_price = close_price + (risk * ema_tp_value)
                elif ema_tp_type == "Fixed Points":
                    tp_price = close_price + ema_tp_value
                else:
                    tp_price = close_price * (1 + ema_tp_value / 100)
                    
                active_trade = {
                    'entry_time': idx,
                    'direction': 'Long',
                    'origin': 'EMA',
                    'entry_price': close_price,
                    'sl_price': sl_price,
                    'tp_price': tp_price
                }
                df.loc[idx, 'entry_long_marker'] = close_price
                continue
                
            elif ema_crossunder and ema20_slope_down and price_below and adx_trend:
                if ema_sl_type == "Fixed Points":
                    sl_price = close_price + ema_sl_value
                else:
                    sl_price = close_price * (1 + ema_sl_value / 100)
                
                risk = sl_price - close_price
                if ema_tp_type == "Risk Reward":
                    tp_price = close_price - (risk * ema_tp_value)
                elif ema_tp_type == "Fixed Points":
                    tp_price = close_price - ema_tp_value
                else:
                    tp_price = close_price * (1 - ema_tp_value / 100)
                    
                active_trade = {
                    'entry_time': idx,
                    'direction': 'Short',
                    'origin': 'EMA',
                    'entry_price': close_price,
                    'sl_price': sl_price,
                    'tp_price': tp_price
                }
                df.loc[idx, 'entry_short_marker'] = close_price
                continue

# Convert daily stats and trades to dataframes for presentation
df_trades = pd.DataFrame(trades)
df_daily_stats = pd.DataFrame(daily_stats)

# =============================================================================
# 6. METRICS COMPUTATIONS
# =============================================================================
total_trades = len(trades)
winning_trades = len([t for t in trades if t['PnL (Points)'] > 0])
losing_trades = len([t for t in trades if t['PnL (Points)'] <= 0])
win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0.0

gross_profit = sum([t['PnL (Points)'] for t in trades if t['PnL (Points)'] > 0])
gross_loss = sum([t['PnL (Points)'] for t in trades if t['PnL (Points)'] < 0])
net_points = sum([t['PnL (Points)'] for t in trades])
profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else (gross_profit if gross_profit > 0 else 1.0)

if total_trades > 0:
    pnl_series = pd.Series([t['PnL (Points)'] for t in trades])
    cum_pnl = pnl_series.cumsum()
    running_max = cum_pnl.cummax()
    drawdown = running_max - cum_pnl
    max_dd = drawdown.max()
else:
    max_dd = 0.0

# Display Metrics Cards
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Net Profit (Points)</div>
            <div class="metric-value {'positive-value' if net_points >= 0 else 'negative-value'}">{net_points:+.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Win Rate</div>
            <div class="metric-value neutral-value">{win_rate:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Total Trades</div>
            <div class="metric-value neutral-value">{total_trades}</div>
        </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Profit Factor</div>
            <div class="metric-value neutral-value">{profit_factor:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

with col5:
    st.markdown(f"""
        <div class="metric-box">
            <div class="metric-label">Max Drawdown (Pts)</div>
            <div class="metric-value negative-value">{max_dd:.2f}</div>
        </div>
    """, unsafe_allow_html=True)

st.write("")

# =============================================================================
# 7. CHART & VISUALIZATION BREAKDOWNS
# =============================================================================
st.markdown('<div class="subheader">📊 Strategy Chart Visualization</div>', unsafe_allow_html=True)

# Select mode for charts
chart_mode = st.radio("Chart View Mode:", ["Single Day Details (Recommended)", "Full Period Continuous"], horizontal=True)

if chart_mode == "Single Day Details (Recommended)":
    # Populate dates from Nifty index
    unique_dates = df_daily_stats['Date'].tolist()
    if unique_dates:
        selected_date = st.selectbox("Select Date to View:", unique_dates, index=len(unique_dates)-1)
        # Filter dataframe for selected day
        df_day = df[df['Date'] == selected_date].sort_index()
        
        # Create Candlestick Trace
        fig = go.Figure()
        
        # Candlesticks
        fig.add_trace(go.Candlestick(
            x=df_day.index,
            open=df_day['Open'],
            high=df_day['High'],
            low=df_day['Low'],
            close=df_day['Close'],
            name="Candlesticks"
        ))
        
        # EMAs
        fig.add_trace(go.Scatter(x=df_day.index, y=df_day['ema9'], name="EMA 9", line=dict(color='orange', width=2)))
        fig.add_trace(go.Scatter(x=df_day.index, y=df_day['ema20'], name="EMA 20", line=dict(color='black', width=2)))
        
        # ORB Lines
        fig.add_trace(go.Scatter(x=df_day.index, y=df_day['orb_high_line'], name="ORB High", line=dict(color='rgba(38, 166, 154, 0.8)', width=1.5, dash='dash')))
        fig.add_trace(go.Scatter(x=df_day.index, y=df_day['orb_low_line'], name="ORB Low", line=dict(color='rgba(239, 83, 80, 0.8)', width=1.5, dash='dash')))
        
        # Markers
        long_entries = df_day[df_day['entry_long_marker'].notna()]
        short_entries = df_day[df_day['entry_short_marker'].notna()]
        exits = df_day[df_day['exit_marker'].notna()]
        
        if not long_entries.empty:
            fig.add_trace(go.Scatter(
                x=long_entries.index, y=long_entries['entry_long_marker'] - 5,
                mode='markers+text', name="BUY Entry",
                marker=dict(symbol='triangle-up', size=14, color='green'),
                text="BUY", textposition="bottom center", textfont=dict(color='green', weight='bold')
            ))
        if not short_entries.empty:
            fig.add_trace(go.Scatter(
                x=short_entries.index, y=short_entries['entry_short_marker'] + 5,
                mode='markers+text', name="SELL Entry",
                marker=dict(symbol='triangle-down', size=14, color='red'),
                text="SELL", textposition="top center", textfont=dict(color='red', weight='bold')
            ))
        if not exits.empty:
            for idx, r in exits.iterrows():
                fig.add_trace(go.Scatter(
                    x=[idx], y=[r['exit_marker']],
                    mode='markers+text', name=f"EXIT ({r['exit_reason_marker']})",
                    marker=dict(symbol='x', size=10, color='blue'),
                    text=f"EXIT ({r['exit_reason_marker']})", textposition="bottom center"
                ))
        
        fig.update_layout(
            title=f"NIFTY 5m Chart: {selected_date}",
            yaxis_title="Points (INR)",
            xaxis_rangeslider_visible=False,
            height=600,
            template="plotly_dark",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display stats of the day
        day_stats_row = df_daily_stats[df_daily_stats['Date'] == selected_date].iloc[0]
        st.info(f"📅 **Date Details**: ORB Range: **{day_stats_row['ORB Low']} - {day_stats_row['ORB High']}** | Inside Bar? **{'Yes' if day_stats_row['Inside Bar'] else 'No'}** | ORB Trade Executed? **{'Yes' if day_stats_row['ORB Taken'] else 'No'}**")
    else:
        st.write("No date selection available.")

else: # Full Period Continuous
    fig = go.Figure()
    
    # Simple Close Price Plot for performance overview
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name="NIFTY Close", line=dict(color='#2196F3', width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema9'], name="EMA 9", line=dict(color='orange', width=1)))
    fig.add_trace(go.Scatter(x=df.index, y=df['ema20'], name="EMA 20", line=dict(color='black', width=1)))
    
    # Plot entries
    long_entries = df[df['entry_long_marker'].notna()]
    short_entries = df[df['entry_short_marker'].notna()]
    if not long_entries.empty:
        fig.add_trace(go.Scatter(x=long_entries.index, y=long_entries['entry_long_marker'], mode='markers', name="Buy Entry", marker=dict(symbol='triangle-up', size=10, color='green')))
    if not short_entries.empty:
        fig.add_trace(go.Scatter(x=short_entries.index, y=short_entries['entry_short_marker'], mode='markers', name="Sell Entry", marker=dict(symbol='triangle-down', size=10, color='red')))
        
    fig.update_layout(
        title="NIFTY Continuous Backtest Chart",
        yaxis_title="Points (INR)",
        height=500,
        template="plotly_dark",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# 8. DATA TABLES DISPLAY
# =============================================================================
tab1, tab2 = st.tabs(["📝 Detailed Trade Log", "🗓️ Daily Stats Summary"])

with tab1:
    if not df_trades.empty:
        # Format datetimes
        df_trades_styled = df_trades.copy()
        df_trades_styled['Entry Time'] = df_trades_styled['Entry Time'].dt.strftime('%Y-%m-%d %H:%M')
        df_trades_styled['Exit Time'] = df_trades_styled['Exit Time'].dt.strftime('%Y-%m-%d %H:%M')
        
        # Color PnL columns
        def color_pnl(val):
            color = '#26a69a' if val > 0 else '#ef5350'
            return f'color: {color}; font-weight: bold;'
            
        st.dataframe(
            df_trades_styled.style.map(color_pnl, subset=['PnL (Points)', 'PnL (%)']),
            use_container_width=True
        )
    else:
        st.info("No trades were executed with the current parameter configurations.")

with tab2:
    if not df_daily_stats.empty:
        st.dataframe(df_daily_stats, use_container_width=True)
    else:
        st.write("No daily summary available.")
