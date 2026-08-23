"""
Stock Technical Analysis Web App
=================================
A friendly, dark-mode Streamlit app for non-technical users.
Shows candlesticks + SMA50, OBV+MA, and a TTM-Squeeze-style momentum
oscillator with squeeze detection and signal annotations.

Run with:  streamlit run app.py
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ----------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Chart Helper",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------
# TRANSLATIONS
# ----------------------------------------------------------------------
TXT = {
    "en": {
        "title": "📈 Stock Chart Helper",
        "subtitle": "Type a stock symbol below to see how it's doing.",
        "lang_toggle": "中文 (Mandarin)",
        "ticker_label": "Stock Symbol",
        "ticker_help": "Example: AAPL for Apple, TSLA for Tesla",
        "period_label": "Time Range",
        "interval_label": "Chart Type",
        "daily": "Daily",
        "weekly": "Weekly",
        "loading": "Fetching data...",
        "error_no_data": "No data found for this symbol. Please check the spelling and try again.",
        "price": "Price",
        "change": "Today's Change",
        "state": "Current Signal",
        "state_bull_accel": "🔴 Bullish Acceleration",
        "state_bull_decel": "🟤 Bullish Deceleration (Warning)",
        "state_bear_accel": "🔵 Bearish Acceleration",
        "state_bear_decel": "🔷 Bearish Deceleration (Bottoming)",
        "state_squeeze": "🟢 Squeeze Active (Low Volatility)",
        "state_neutral": "⚪ Neutral",
        "chart1_title": "Price Chart",
        "chart2_title": "Volume Flow (OBV)",
        "chart3_title": "Momentum & Squeeze",
        "footer": "This tool is for educational purposes only and is not financial advice.",
        "gamma": "⚡Gamma Squeeze",
        "decel": "🛑Deceleration",
        "breakout": "💥Breakout",
        "golden_cross": "OBV Golden Cross",
    },
    "zh": {
        "title": "📈 股票图表助手",
        "subtitle": "在下方输入股票代码，查看它的走势。",
        "lang_toggle": "English",
        "ticker_label": "股票代码",
        "ticker_help": "例如：AAPL 代表苹果公司，TSLA 代表特斯拉",
        "period_label": "时间范围",
        "interval_label": "图表类型",
        "daily": "每日",
        "weekly": "每周",
        "loading": "正在获取数据...",
        "error_no_data": "找不到该股票代码的数据，请检查拼写后重试。",
        "price": "价格",
        "change": "今日涨跌",
        "state": "当前信号",
        "state_bull_accel": "🔴 看涨加速",
        "state_bull_decel": "🟤 看涨减速（警告）",
        "state_bear_accel": "🔵 看跌加速",
        "state_bear_decel": "🔷 看跌减速（触底）",
        "state_squeeze": "🟢 挤压中（波动率低）",
        "state_neutral": "⚪ 中性",
        "chart1_title": "价格图",
        "chart2_title": "资金流向 (OBV)",
        "chart3_title": "动能与挤压",
        "footer": "本工具仅供学习参考，不构成投资建议。",
        "gamma": "⚡伽玛挤压",
        "decel": "🛑减速",
        "breakout": "💥突破",
        "golden_cross": "OBV 黄金交叉",
    },
}

PERIOD_OPTIONS = {"6M": "6mo", "1Y": "1y", "2Y": "2y"}

# ----------------------------------------------------------------------
# STATE / LANGUAGE TOGGLE
# ----------------------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"

top_l, top_r = st.columns([5, 1])
with top_r:
    if st.button(TXT[st.session_state.lang]["lang_toggle"], use_container_width=True):
        st.session_state.lang = "zh" if st.session_state.lang == "en" else "en"
        st.rerun()

L = TXT[st.session_state.lang]

# ----------------------------------------------------------------------
# DARK MODE STYLING
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0E1117; color: #FAFAFA; }
    div[data-testid="stMetric"] {
        background-color: #1A1D24;
        border: 1px solid #2A2E38;
        border-radius: 12px;
        padding: 16px;
    }
    .big-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .subtitle { color: #9AA0A6; font-size: 1.05rem; margin-top: 4px; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<div class='big-title'>{L['title']}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>{L['subtitle']}</div>", unsafe_allow_html=True)
st.write("")

# ----------------------------------------------------------------------
# TOP CONTROLS
# ----------------------------------------------------------------------
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    ticker_input = st.text_input(
        L["ticker_label"], value="AAPL", help=L["ticker_help"]
    ).strip().upper()
with c2:
    period_label = st.selectbox(L["period_label"], list(PERIOD_OPTIONS.keys()), index=1)
with c3:
    interval_choice = st.radio(
        L["interval_label"], [L["daily"], L["weekly"]], horizontal=True
    )

interval = "1wk" if interval_choice == L["weekly"] else "1d"
period = PERIOD_OPTIONS[period_label]

# ----------------------------------------------------------------------
# DATA FETCH (cached)
# ----------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_data(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(how="all")
    return df


# ----------------------------------------------------------------------
# INDICATOR CALCULATIONS
# ----------------------------------------------------------------------
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- SMA 50 ---
    df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()

    # --- OBV ---
    price_diff = df["Close"].diff()
    direction = np.sign(price_diff).fillna(0)
    signed_volume = direction * df["Volume"]
    df["OBV_RAW"] = signed_volume.cumsum()
    df["MA_OBV"] = df["OBV_RAW"].rolling(window=20, min_periods=1).mean()

    # OBV Golden Cross: OBV_RAW crosses above MA_OBV
    obv_above = df["OBV_RAW"] > df["MA_OBV"]
    df["OBV_GOLDEN_CROSS"] = obv_above & (~obv_above.shift(1).fillna(False))

    # --- Bollinger Bands (20, 2.0) ---
    bb_mid = df["Close"].rolling(window=20, min_periods=1).mean()
    bb_std = df["Close"].rolling(window=20, min_periods=1).std(ddof=0)
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std

    # --- Keltner Channels (20, 1.5 ATR) ---
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=20, min_periods=1).mean()
    kc_mid = df["Close"].rolling(window=20, min_periods=1).mean()
    kc_upper = kc_mid + 1.5 * atr
    kc_lower = kc_mid - 1.5 * atr

    # --- Squeeze condition: Bollinger Bands inside Keltner Channels ---
    df["SQ_ON"] = (bb_upper < kc_upper) & (bb_lower > kc_lower)

    # --- Momentum ---
    hhv20 = df["High"].rolling(window=20, min_periods=1).max()
    llv20 = df["Low"].rolling(window=20, min_periods=1).min()
    ma20 = df["Close"].rolling(window=20, min_periods=1).mean()
    mom_source = df["Close"] - ((hhv20 + llv20) / 2 + ma20) / 2
    df["MOM"] = mom_source.ewm(span=12, adjust=False).mean()

    # --- 4-color classification ---
    prev_mom = df["MOM"].shift(1)
    conditions = [
        (df["MOM"] > 0) & (df["MOM"] > prev_mom),   # bright red - bullish accel
        (df["MOM"] > 0) & (df["MOM"] <= prev_mom),  # dark red - bullish decel
        (df["MOM"] < 0) & (df["MOM"] < prev_mom),   # bright cyan - bearish accel
        (df["MOM"] < 0) & (df["MOM"] >= prev_mom),  # dark blue - bearish decel
    ]
    colors = ["#FF0000", "#8B0000", "#00FFFF", "#00008B"]
    df["MOM_COLOR"] = np.select(conditions, colors, default="#777777")
    df["MOM_STATE"] = np.select(
        conditions,
        ["bull_accel", "bull_decel", "bear_accel", "bear_decel"],
        default="neutral",
    )

    return df


def detect_signals(df: pd.DataFrame, L: dict):
    """Return lists of (date, y, text) annotation tuples."""
    gamma_pts, decel_pts, breakout_pts = [], [], []

    prev_state = df["MOM_STATE"].shift(1)
    prev_sq = df["SQ_ON"].shift(1).fillna(False)

    for i in range(1, len(df)):
        idx = df.index[i]
        state = df["MOM_STATE"].iloc[i]
        pstate = prev_state.iloc[i]
        was_squeeze = bool(prev_sq.iloc[i])
        mom = df["MOM"].iloc[i]
        prev_mom = df["MOM"].iloc[i - 1]

        # Gamma squeeze: MOM turns bright red immediately after a squeeze
        if state == "bull_accel" and pstate != "bull_accel" and was_squeeze:
            gamma_pts.append((idx, mom, L["gamma"]))

        # Deceleration: first day MOM turns bright red -> dark red
        if state == "bull_decel" and pstate == "bull_accel":
            decel_pts.append((idx, mom, L["decel"]))

        # Breakout: bearish zero-cross (MOM crosses below 0)
        if prev_mom is not None and prev_mom >= 0 and mom < 0:
            breakout_pts.append((idx, mom, L["breakout"]))

    return gamma_pts, decel_pts, breakout_pts


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
if not ticker_input:
    st.stop()

with st.spinner(L["loading"]):
    raw_df = fetch_data(ticker_input, period, interval)

if raw_df.empty or len(raw_df) < 5:
    st.error(L["error_no_data"])
    st.stop()

df = compute_indicators(raw_df)
gamma_pts, decel_pts, breakout_pts = detect_signals(df, L)

# ----------------------------------------------------------------------
# SUMMARY CARD
# ----------------------------------------------------------------------
last_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else last_close
pct_change = ((last_close - prev_close) / prev_close * 100) if prev_close else 0.0

last_state = df["MOM_STATE"].iloc[-1]
last_squeeze = bool(df["SQ_ON"].iloc[-1])

if last_squeeze:
    state_display = L["state_squeeze"]
elif last_state == "bull_accel":
    state_display = L["state_bull_accel"]
elif last_state == "bull_decel":
    state_display = L["state_bull_decel"]
elif last_state == "bear_accel":
    state_display = L["state_bear_accel"]
elif last_state == "bear_decel":
    state_display = L["state_bear_decel"]
else:
    state_display = L["state_neutral"]

m1, m2, m3 = st.columns(3)
m1.metric(f"{L['price']} ({ticker_input})", f"${last_close:,.2f}")
m2.metric(L["change"], f"{pct_change:+.2f}%")
m3.metric(L["state"], state_display)

st.write("")

# ----------------------------------------------------------------------
# CHART
# ----------------------------------------------------------------------
fig = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=[0.55, 0.22, 0.23],
    subplot_titles=(L["chart1_title"], L["chart2_title"], L["chart3_title"]),
)

# --- Row 1: Candlestick + SMA50 ---
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name=ticker_input,
        increasing_line_color="#26A69A",
        decreasing_line_color="#EF5350",
        showlegend=False,
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["SMA_50"],
        name="SMA 50",
        line=dict(color="yellow", width=1.5),
    ),
    row=1,
    col=1,
)

# --- Row 2: OBV ---
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["OBV_RAW"],
        name="OBV",
        line=dict(color="#FF3333", width=1.5),
    ),
    row=2,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=df.index,
        y=df["MA_OBV"],
        name="MA OBV (20)",
        line=dict(color="#FFCC00", width=1.5, dash="dash"),
    ),
    row=2,
    col=1,
)
gc = df[df["OBV_GOLDEN_CROSS"]]
if not gc.empty:
    fig.add_trace(
        go.Scatter(
            x=gc.index,
            y=gc["OBV_RAW"],
            mode="markers",
            name=L["golden_cross"],
            marker=dict(symbol="star", size=10, color="gold", line=dict(width=1, color="black")),
        ),
        row=2,
        col=1,
    )

# --- Row 3: Momentum histogram ---
fig.add_trace(
    go.Bar(
        x=df.index,
        y=df["MOM"],
        marker_color=df["MOM_COLOR"],
        name="Momentum",
        showlegend=False,
    ),
    row=3,
    col=1,
)

# Squeeze band on zero line: green bar segments where SQ_ON is True
sq = df[df["SQ_ON"]]
if not sq.empty:
    fig.add_trace(
        go.Scatter(
            x=sq.index,
            y=[0] * len(sq),
            mode="markers",
            marker=dict(symbol="line-ew", size=10, color="#00FF00", line=dict(width=4, color="#00FF00")),
            name="Squeeze",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

# Annotations for signals
for idx, y, text in gamma_pts:
    fig.add_annotation(
        x=idx, y=y, row=3, col=1, text=text, showarrow=True,
        arrowhead=2, ay=-30, font=dict(size=10, color="#FF0000"),
    )
for idx, y, text in decel_pts:
    fig.add_annotation(
        x=idx, y=y, row=3, col=1, text=text, showarrow=True,
        arrowhead=2, ay=30, font=dict(size=10, color="#8B0000"),
    )
for idx, y, text in breakout_pts:
    fig.add_annotation(
        x=idx, y=y, row=3, col=1, text=text, showarrow=True,
        arrowhead=2, ay=30, font=dict(size=10, color="#00FFFF"),
    )

fig.add_hline(y=0, line=dict(color="#555555", width=1), row=3, col=1)

# --- Layout ---
fig.update_layout(
    template="plotly_dark",
    height=900,
    plot_bgcolor="#0E1117",
    paper_bgcolor="#0E1117",
    font=dict(color="#FAFAFA"),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=40, r=40, t=60, b=20),
    xaxis3=dict(rangeslider=dict(visible=False)),
)
fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="grey", spikethickness=1)
fig.update_yaxes(showspikes=True, spikethickness=1)

st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

st.caption(L["footer"])
