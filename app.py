"""
Stock Technical Analysis Web App
=================================
A friendly, dark-mode Streamlit app for non-technical users.
Chart mode shows candlesticks + SMA50, OBV+MA, and a TTM-Squeeze-style
momentum oscillator with squeeze detection and signal annotations.
Signals mode scans a watchlist for buy signals (price drawdown vs.
options put-wall support) and monitors an open position for sell
signals (stop-loss / take-profit vs. options call-wall resistance).

Run with:  streamlit run app.py
"""

import concurrent.futures
import json
import os

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from fpdf import FPDF

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
        "title": "Stock Chart Helper",
        "subtitle": "Type a stock symbol below to see how it's doing.",
        "lang_toggle": "中文 (Mandarin)",
        "ticker_label": "Stock Symbol",
        "ticker_help": "Example: AAPL for Apple, TSLA for Tesla",
        "ticker_picker_label": ":material/search: Search & pick a stock (or type below)",
        "ticker_picker_placeholder": "— Type your own below —",
        "period_label": "Time Range",
        "interval_label": "Chart Type",
        "daily": "Daily",
        "weekly": "Weekly",
        "monthly": "Monthly",
        "loading": "Fetching data...",
        "error_no_data": "No data found for this symbol. Please check the spelling and try again.",
        "price": "Price",
        "change": "Today's Change",
        "state": "Current Signal",
        "state_bull_accel": ":material/trending_up: Bullish Acceleration",
        "state_bull_decel": ":material/trending_flat: Bullish Deceleration (Warning)",
        "state_bear_accel": ":material/trending_down: Bearish Acceleration",
        "state_bear_decel": ":material/north_east: Bearish Deceleration (Bottoming)",
        "state_squeeze": ":material/compress: Squeeze Active (Low Volatility)",
        "state_neutral": ":material/remove: Neutral",
        "chart1_title": "Price Chart",
        "chart2_title": "Volume Flow (OBV)",
        "chart3_title": "Momentum & Squeeze",
        "footer": "This tool is for educational purposes only and is not financial advice.",
        "gamma": "Gamma Squeeze",
        "decel": "Deceleration",
        "breakout": "Breakout",
        "golden_cross": "OBV Golden Cross",
        "put_wall_line_label": "Put Wall (Support)",
        "call_wall_line_label": "Call Wall (Resistance)",
        "buy_arrow_label": "Buy Signal",
        "sell_arrow_label": "Sell/Caution Signal",
        # --- mode toggle ---
        "to_signals_btn": ":material/candlestick_chart: Buy/Sell Signals",
        "to_chart_btn": ":material/show_chart: Back to Chart",
        # --- signals page ---
        "signal_title": "Buy/Sell Signal Scanner",
        "signal_subtitle": "Rule-based signals from price drawdown and options positioning.",
        "signal_disclaimer": ":material/warning: This is an automated heuristic tool, not financial advice. Always do your own research before buying or selling.",
        "buy_section_title": "Buy Signal Scanner",
        "buy_section_desc": "Enter one or more stock symbols to check if now looks like a good time to consider buying.",
        "watchlist_label": "Stock Symbols (comma-separated)",
        "scan_button": ":material/search: Scan",
        "sell_section_title": "Sell / Take-Profit Monitor",
        "sell_section_desc": "Track a position you already own and see if it's time to sell.",
        "sell_ticker_label": "Stock Symbol",
        "entry_price_label": "Your Buy Price ($)",
        "take_profit_label": "Take-Profit Target (%)",
        "stop_loss_label": "Stop-Loss Limit (%)",
        "check_button": ":material/check_circle: Check",
        "current_price_label": "Current Price",
        "drawdown_label": "Drop From 52-Week High",
        "put_wall_label": "Support Level (Put Wall)",
        "put_wall_oi_label": "Put Wall Contracts",
        "call_wall_oi_label": "Call Wall Contracts",
        "buy_zone_label": "Buy Zone (90%–105% of Put Wall)",
        "pcr_label": "Put/Call Ratio",
        "expiration_label": "Options Expiration Used",
        "call_wall_label": "Resistance Level (Call Wall)",
        "pnl_label": "Current Profit/Loss",
        "entry_price_display_label": "Your Buy Price",
        "buy_verdict_signal_title": ":material/track_changes: High-Confidence Buy Signal",
        "buy_verdict_signal_detail": "Down {drawdown:.1f}% from its high and trading near the support level (${wall:.2f}).",
        "buy_verdict_broken_title": ":material/warning: Support Broken — Wait",
        "buy_verdict_broken_detail": "Price has fallen more than 10% below the support level (${wall:.2f}).",
        "buy_verdict_wait_title": ":material/hourglass_top: Not Yet — Keep Watching",
        "buy_verdict_wait_detail": "Price is down a lot, but hasn't reached the support level (~${wall:.2f}) yet.",
        "buy_verdict_none_title": ":material/block: Not Oversold",
        "buy_verdict_none_detail": "Only down {drawdown:.1f}% from its high (this tool looks for -25% or more).",
        "sell_verdict_stoploss_title": ":material/error: Stop-Loss Triggered",
        "sell_verdict_stoploss_detail": "Loss of {pnl:.1f}% has hit your stop-loss limit ({limit:.1f}%). Consider selling.",
        "sell_verdict_break_title": ":material/error: Support Broken — Stop-Loss",
        "sell_verdict_break_detail": "Price has fallen more than 5% below the support level (${wall_put:.2f}).",
        "sell_verdict_resistance_title": ":material/celebration: Near Resistance — Take Profit",
        "sell_verdict_resistance_detail": "Price is approaching the resistance level (${wall_call:.2f}). Consider selling part of your position.",
        "sell_verdict_target_title": ":material/celebration: Target Reached — Take Profit",
        "sell_verdict_target_detail": "Gain of {pnl:.1f}% has hit your target ({target:.1f}%). Consider taking some profit.",
        "sell_verdict_hold_title": ":material/pause_circle: Hold",
        "sell_verdict_hold_detail": "No exit conditions met yet.",
        "err_no_price": "Couldn't fetch price data for {ticker}.",
        "err_no_options": "No options data available for {ticker}.",
        "err_incomplete_chain": "Options chain data is incomplete for {ticker}.",
        "err_generic": "Something went wrong fetching {ticker}: {error}",
        "weekly_section_title": "Weekly Signal Scanner",
        "weekly_section_desc": "Scans a list of stocks for weekly breakout/divergence patterns using RSI, volume, and price structure.",
        "weekly_pool_label": "Stock Pool (comma-separated)",
        "weekly_scan_button": ":material/search: Scan Weekly Signals",
        "weekly_col_ticker": "Ticker",
        "weekly_col_close": "Close",
        "weekly_col_market_cap": "Market Cap ($B)",
        "weekly_col_float": "Float Shares (M)",
        "weekly_col_rsi": "RSI (14)",
        "weekly_col_signals": "Signals",
        "weekly_no_signals": "No weekly signals triggered this week for the entered stock pool.",
        "weekly_errors_caption": "Couldn't check: {tickers}",
        "weekly_screened_caption": "Didn't meet the size screen (Market Cap ≥ $1.6B, Float ≥ 400M): {tickers}",
        "weekly_download_button": ":material/download: Download Results (CSV)",
        "full_scan_title": "Full Market Scan (Pre-Screened List)",
        "full_scan_desc": "Runs the scanner across all {count} stocks that already passed the size screen (Market Cap ≥ $1.6B, Float ≥ 400M) — no need to type anything. This takes a few minutes.",
        "full_scan_interval_label": "Candle Interval & Lookback",
        "full_scan_option_daily": "Daily — past 12 months ({start} to {end})",
        "full_scan_option_weekly": "Weekly — past 52 weeks ({start} to {end})",
        "full_scan_option_monthly": "Monthly — past 36 months ({start} to {end})",
        "full_scan_button": ":material/rocket_launch: Scan All {count} Pre-Screened Stocks",
        "full_scan_pdf_button": ":material/download: Download Full Report (PDF)",
        "weekly_full_scan_progress": "Scanning {done}/{total} — {ticker}",
        "err_insufficient_data": "Not enough price history for {ticker}.",
        "pattern_bull_div": "Bullish Divergence",
        "pattern_sos_breakout": "SOS Breakout",
        "pattern_bear_div": "Bearish Divergence",
    },
    "zh": {
        "title": "股票图表助手",
        "subtitle": "在下方输入股票代码，查看它的走势。",
        "lang_toggle": "English",
        "ticker_label": "股票代码",
        "ticker_help": "例如：AAPL 代表苹果公司，TSLA 代表特斯拉",
        "ticker_picker_label": ":material/search: 搜索并选择股票（或在下方手动输入）",
        "ticker_picker_placeholder": "— 在下方手动输入 —",
        "period_label": "时间范围",
        "interval_label": "图表类型",
        "daily": "每日",
        "weekly": "每周",
        "monthly": "每月",
        "loading": "正在获取数据...",
        "error_no_data": "找不到该股票代码的数据，请检查拼写后重试。",
        "price": "价格",
        "change": "今日涨跌",
        "state": "当前信号",
        "state_bull_accel": ":material/trending_up: 看涨加速",
        "state_bull_decel": ":material/trending_flat: 看涨减速（警告）",
        "state_bear_accel": ":material/trending_down: 看跌加速",
        "state_bear_decel": ":material/north_east: 看跌减速（触底）",
        "state_squeeze": ":material/compress: 挤压中（波动率低）",
        "state_neutral": ":material/remove: 中性",
        "chart1_title": "价格图",
        "chart2_title": "资金流向 (OBV)",
        "chart3_title": "动能与挤压",
        "footer": "本工具仅供学习参考，不构成投资建议。",
        "gamma": "伽玛挤压",
        "decel": "减速",
        "breakout": "突破",
        "golden_cross": "OBV 黄金交叉",
        "put_wall_line_label": "Put 防守墙（支撑）",
        "call_wall_line_label": "Call 阻力墙（阻力）",
        "buy_arrow_label": "买入信号",
        "sell_arrow_label": "卖出/谨慎信号",
        # --- mode toggle ---
        "to_signals_btn": ":material/candlestick_chart: 买卖信号",
        "to_chart_btn": ":material/show_chart: 返回图表",
        # --- signals page ---
        "signal_title": "买卖信号扫描器",
        "signal_subtitle": "基于价格回撤和期权持仓的规则化信号。",
        "signal_disclaimer": ":material/warning: 这是一个自动化的启发式工具，不构成投资建议。做出买卖决定前请自行研究。",
        "buy_section_title": "买入信号扫描",
        "buy_section_desc": "输入一个或多个股票代码，查看现在是否适合考虑买入。",
        "watchlist_label": "股票代码（用逗号分隔）",
        "scan_button": ":material/search: 扫描",
        "sell_section_title": "卖出 / 止盈监控",
        "sell_section_desc": "追踪你已持有的仓位，查看是否该卖出了。",
        "sell_ticker_label": "股票代码",
        "entry_price_label": "买入价格 ($)",
        "take_profit_label": "止盈目标 (%)",
        "stop_loss_label": "止损限制 (%)",
        "check_button": ":material/check_circle: 检查",
        "current_price_label": "当前价格",
        "drawdown_label": "距52周高点跌幅",
        "put_wall_label": "支撑位（Put 防守墙）",
        "put_wall_oi_label": "Put 防守墙持仓量",
        "call_wall_oi_label": "Call 阻力墙持仓量",
        "buy_zone_label": "买入区间（Put 防守墙的 90%–105%）",
        "pcr_label": "认沽/认购比率 (PCR)",
        "expiration_label": "所用期权到期日",
        "call_wall_label": "阻力位（Call 阻力墙）",
        "pnl_label": "当前盈亏",
        "entry_price_display_label": "买入价格",
        "buy_verdict_signal_title": ":material/track_changes: 高置信度买入信号",
        "buy_verdict_signal_detail": "较高点已下跌 {drawdown:.1f}%，且接近支撑位 (${wall:.2f})。",
        "buy_verdict_broken_title": ":material/warning: 支撑位已破位 — 观望",
        "buy_verdict_broken_detail": "价格已跌破支撑位 (${wall:.2f}) 10% 以上。",
        "buy_verdict_wait_title": ":material/hourglass_top: 尚未到位 — 继续观察",
        "buy_verdict_wait_detail": "价格已大幅下跌，但尚未到达支撑位（约 ${wall:.2f}）。",
        "buy_verdict_none_title": ":material/block: 未达超跌条件",
        "buy_verdict_none_detail": "较高点仅下跌 {drawdown:.1f}%（本工具寻找 -25% 或以上）。",
        "sell_verdict_stoploss_title": ":material/error: 触发固定止损",
        "sell_verdict_stoploss_detail": "亏损 {pnl:.1f}% 已触及止损限制 ({limit:.1f}%)，建议卖出。",
        "sell_verdict_break_title": ":material/error: 支撑位破位 — 止损",
        "sell_verdict_break_detail": "价格已跌破支撑位 (${wall_put:.2f}) 5% 以上。",
        "sell_verdict_resistance_title": ":material/celebration: 接近阻力位 — 止盈",
        "sell_verdict_resistance_detail": "价格正接近阻力位 (${wall_call:.2f})，建议卖出部分仓位。",
        "sell_verdict_target_title": ":material/celebration: 达到目标 — 止盈",
        "sell_verdict_target_detail": "收益 {pnl:.1f}% 已达到目标 ({target:.1f}%)，建议获利了结部分仓位。",
        "sell_verdict_hold_title": ":material/pause_circle: 继续持有",
        "sell_verdict_hold_detail": "尚未触发任何卖出条件。",
        "err_no_price": "无法获取 {ticker} 的价格数据。",
        "err_no_options": "没有 {ticker} 的期权数据。",
        "err_incomplete_chain": "{ticker} 的期权链数据不完整。",
        "err_generic": "获取 {ticker} 时出错：{error}",
        "weekly_section_title": "周线信号扫描器",
        "weekly_section_desc": "使用 RSI、成交量与价格结构，扫描一组股票的周线突破/背离形态。",
        "weekly_pool_label": "股票池（用逗号分隔）",
        "weekly_scan_button": ":material/search: 扫描周线信号",
        "weekly_col_ticker": "股票代码",
        "weekly_col_close": "收盘价",
        "weekly_col_market_cap": "市值（十亿美元）",
        "weekly_col_float": "流通股（百万股）",
        "weekly_col_rsi": "RSI（14）",
        "weekly_col_signals": "信号",
        "weekly_no_signals": "所输入股票池本周未触发任何周线信号。",
        "weekly_errors_caption": "无法检查：{tickers}",
        "weekly_screened_caption": "未通过规模筛选（市值 ≥ 16亿美元，流通股 ≥ 4亿股）：{tickers}",
        "weekly_download_button": ":material/download: 下载结果（CSV）",
        "full_scan_title": "全市场扫描（预筛选列表）",
        "full_scan_desc": "对已通过规模筛选（市值 ≥ 16亿美元，流通股 ≥ 4亿股）的全部 {count} 支股票运行扫描 — 无需手动输入。此过程需要几分钟。",
        "full_scan_interval_label": "K线周期与回溯范围",
        "full_scan_option_daily": "每日 — 过去12个月（{start} 至 {end}）",
        "full_scan_option_weekly": "每周 — 过去52周（{start} 至 {end}）",
        "full_scan_option_monthly": "每月 — 过去36个月（{start} 至 {end}）",
        "full_scan_button": ":material/rocket_launch: 扫描全部 {count} 支预筛选股票",
        "full_scan_pdf_button": ":material/download: 下载完整报告（PDF）",
        "weekly_full_scan_progress": "正在扫描 {done}/{total} — {ticker}",
        "err_insufficient_data": "{ticker} 的历史数据不足。",
        "pattern_bull_div": "看涨背离",
        "pattern_sos_breakout": "放量突破",
        "pattern_bear_div": "看跌背离",
    },
}

PERIOD_OPTIONS = {"6M": "6mo", "1Y": "1y", "2Y": "2y"}

# ----------------------------------------------------------------------
# TICKER PICKER UNIVERSE
# Major index funds + the S&P 500's largest, most recognizable
# constituents across sectors. Not an exhaustive/always-current S&P 500
# list -- anything not listed here can still be typed directly into the
# ticker box, this is just a convenience shortlist for the dropdown.
# ----------------------------------------------------------------------
INDEX_FUNDS = [
    ("SPY", "SPDR S&P 500 ETF Trust"),
    ("VOO", "Vanguard S&P 500 ETF"),
    ("IVV", "iShares Core S&P 500 ETF"),
    ("QQQ", "Invesco QQQ Trust (Nasdaq-100)"),
    ("DIA", "SPDR Dow Jones Industrial Average ETF"),
    ("IWM", "iShares Russell 2000 ETF"),
    ("VTI", "Vanguard Total Stock Market ETF"),
    ("VEA", "Vanguard FTSE Developed Markets ETF"),
    ("VWO", "Vanguard FTSE Emerging Markets ETF"),
    ("VXUS", "Vanguard Total International Stock ETF"),
    ("AGG", "iShares Core U.S. Aggregate Bond ETF"),
    ("BND", "Vanguard Total Bond Market ETF"),
    ("GLD", "SPDR Gold Shares"),
    ("XLK", "Technology Select Sector SPDR Fund"),
    ("XLF", "Financial Select Sector SPDR Fund"),
    ("XLE", "Energy Select Sector SPDR Fund"),
    ("XLV", "Health Care Select Sector SPDR Fund"),
]

SP500_LARGE_CAP = [
    ("AAPL", "Apple"), ("MSFT", "Microsoft"), ("NVDA", "NVIDIA"), ("AMZN", "Amazon"),
    ("GOOGL", "Alphabet (Class A)"), ("GOOG", "Alphabet (Class C)"), ("META", "Meta Platforms"),
    ("TSLA", "Tesla"), ("AVGO", "Broadcom"), ("BRK-B", "Berkshire Hathaway"),
    ("JPM", "JPMorgan Chase"), ("LLY", "Eli Lilly"), ("V", "Visa"), ("UNH", "UnitedHealth Group"),
    ("XOM", "Exxon Mobil"), ("MA", "Mastercard"), ("COST", "Costco"), ("HD", "Home Depot"),
    ("PG", "Procter & Gamble"), ("JNJ", "Johnson & Johnson"), ("NFLX", "Netflix"),
    ("BAC", "Bank of America"), ("ABBV", "AbbVie"), ("CRM", "Salesforce"), ("ORCL", "Oracle"),
    ("KO", "Coca-Cola"), ("AMD", "Advanced Micro Devices"), ("CVX", "Chevron"), ("WMT", "Walmart"),
    ("PEP", "PepsiCo"), ("MRK", "Merck"), ("ADBE", "Adobe"), ("TMO", "Thermo Fisher Scientific"),
    ("ACN", "Accenture"), ("LIN", "Linde"), ("MCD", "McDonald's"), ("CSCO", "Cisco Systems"),
    ("ABT", "Abbott Laboratories"), ("WFC", "Wells Fargo"), ("DHR", "Danaher"), ("IBM", "IBM"),
    ("GE", "GE Aerospace"), ("QCOM", "Qualcomm"), ("TXN", "Texas Instruments"), ("PM", "Philip Morris International"),
    ("VZ", "Verizon"), ("CAT", "Caterpillar"), ("INTU", "Intuit"), ("NOW", "ServiceNow"),
    ("AMGN", "Amgen"), ("ISRG", "Intuitive Surgical"), ("SPGI", "S&P Global"), ("BKNG", "Booking Holdings"),
    ("AXP", "American Express"), ("PFE", "Pfizer"), ("DIS", "Walt Disney"), ("GS", "Goldman Sachs"),
    ("RTX", "RTX Corporation"), ("HON", "Honeywell"), ("UNP", "Union Pacific"), ("LOW", "Lowe's"),
    ("T", "AT&T"), ("COP", "ConocoPhillips"), ("MS", "Morgan Stanley"), ("BLK", "BlackRock"),
    ("ELV", "Elevance Health"), ("SCHW", "Charles Schwab"), ("BSX", "Boston Scientific"),
    ("SYK", "Stryker"), ("NEE", "NextEra Energy"), ("PGR", "Progressive"), ("UPS", "UPS"),
    ("TJX", "TJX Companies"), ("VRTX", "Vertex Pharmaceuticals"), ("C", "Citigroup"),
    ("ADP", "Automatic Data Processing"), ("MU", "Micron Technology"), ("PLD", "Prologis"),
    ("MDT", "Medtronic"), ("LMT", "Lockheed Martin"), ("CB", "Chubb"), ("SBUX", "Starbucks"),
    ("REGN", "Regeneron"), ("AMT", "American Tower"), ("MMC", "Marsh & McLennan"),
    ("PANW", "Palo Alto Networks"), ("ETN", "Eaton"), ("FI", "Fiserv"), ("BMY", "Bristol-Myers Squibb"),
    ("GILD", "Gilead Sciences"), ("SO", "Southern Company"), ("DE", "Deere & Company"),
    ("CI", "Cigna"), ("ANET", "Arista Networks"), ("ZTS", "Zoetis"), ("DUK", "Duke Energy"),
    ("SHW", "Sherwin-Williams"), ("MO", "Altria Group"), ("CME", "CME Group"), ("KLAC", "KLA Corporation"),
    ("EOG", "EOG Resources"), ("ICE", "Intercontinental Exchange"), ("SNPS", "Synopsys"),
    ("TT", "Trane Technologies"), ("APH", "Amphenol"), ("CDNS", "Cadence Design Systems"),
    ("WM", "Waste Management"), ("CL", "Colgate-Palmolive"), ("ITW", "Illinois Tool Works"),
    ("PYPL", "PayPal"), ("MCK", "McKesson"), ("CSX", "CSX Corporation"), ("EMR", "Emerson Electric"),
    ("NOC", "Northrop Grumman"), ("MSI", "Motorola Solutions"), ("APD", "Air Products"),
    ("FDX", "FedEx"), ("HCA", "HCA Healthcare"), ("GD", "General Dynamics"), ("ORLY", "O'Reilly Automotive"),
    ("MAR", "Marriott International"), ("ROP", "Roper Technologies"), ("PSA", "Public Storage"),
    ("AJG", "Arthur J. Gallagher"), ("NSC", "Norfolk Southern"), ("ADI", "Analog Devices"),
    ("TDG", "TransDigm"), ("CVS", "CVS Health"), ("PH", "Parker Hannifin"), ("AON", "Aon"),
    ("MMM", "3M"), ("F", "Ford Motor"), ("GM", "General Motors"), ("DELL", "Dell Technologies"),
    ("PCAR", "PACCAR"), ("WELL", "Welltower"), ("USB", "U.S. Bancorp"), ("TGT", "Target"),
    ("SRE", "Sempra"), ("MET", "MetLife"), ("AFL", "Aflac"), ("NXPI", "NXP Semiconductors"),
    ("CTAS", "Cintas"), ("COF", "Capital One"), ("AZO", "AutoZone"), ("PNC", "PNC Financial Services"),
    ("SPG", "Simon Property Group"), ("JCI", "Johnson Controls"), ("ECL", "Ecolab"),
    ("KMB", "Kimberly-Clark"), ("OXY", "Occidental Petroleum"), ("CARR", "Carrier Global"),
    ("MPC", "Marathon Petroleum"), ("SLB", "Schlumberger"), ("PSX", "Phillips 66"),
    ("TFC", "Truist Financial"), ("ALL", "Allstate"), ("DLR", "Digital Realty"),
    ("KMI", "Kinder Morgan"), ("AMP", "Ameriprise Financial"), ("O", "Realty Income"),
    ("URI", "United Rentals"), ("MSCI", "MSCI Inc."), ("CMG", "Chipotle Mexican Grill"),
    ("FTNT", "Fortinet"), ("TRV", "Travelers Companies"), ("D", "Dominion Energy"),
    ("HLT", "Hilton Worldwide"), ("PAYX", "Paychex"), ("AEP", "American Electric Power"),
    ("ROST", "Ross Stores"), ("YUM", "Yum! Brands"), ("MCHP", "Microchip Technology"),
    ("EW", "Edwards Lifesciences"), ("PCG", "PG&E"), ("PRU", "Prudential Financial"),
    ("STZ", "Constellation Brands"), ("VRSK", "Verisk Analytics"), ("KDP", "Keurig Dr Pepper"),
    ("EXC", "Exelon"), ("ADSK", "Autodesk"), ("A", "Agilent Technologies"), ("CTVA", "Corteva"),
    ("XEL", "Xcel Energy"), ("CPRT", "Copart"), ("FICO", "Fair Isaac"), ("HUM", "Humana"),
    ("EA", "Electronic Arts"), ("IQV", "IQVIA Holdings"), ("GWW", "W.W. Grainger"),
    ("NUE", "Nucor"), ("SYY", "Sysco"), ("WMB", "Williams Companies"), ("HPQ", "HP Inc."),
    ("DD", "DuPont"), ("ODFL", "Old Dominion Freight Line"), ("GEHC", "GE HealthCare"),
    ("VMC", "Vulcan Materials"), ("KVUE", "Kenvue"), ("FAST", "Fastenal"),
    ("IDXX", "IDEXX Laboratories"), ("MLM", "Martin Marietta Materials"), ("DOW", "Dow Inc."),
    ("VLO", "Valero Energy"), ("KR", "Kroger"), ("HES", "Hess Corporation"), ("BK", "BNY Mellon"),
    ("BIIB", "Biogen"), ("EBAY", "eBay"), ("RMBS", "Rambus"),
]

@st.cache_data(ttl=86400, show_spinner=False)
def load_screened_universe_full() -> list:
    """Pre-screened US stocks (Market Cap >= $1.6B, Float >= 400M) built
    offline by build_ticker_universe.py -- see that script to refresh.
    Returns full records (ticker, name, market_cap, float_shares).
    Empty list if the data file is ever missing (callers fall back
    appropriately -- the picker to SP500_LARGE_CAP, the full-scan
    button just won't have anything to scan)."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "us_stocks_screened.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def load_screened_universe() -> list:
    """(ticker, name) pairs for the ticker-picker dropdown."""
    full = load_screened_universe_full()
    if not full:
        return SP500_LARGE_CAP
    return [(r["ticker"], r["name"]) for r in full]


TICKER_UNIVERSE = sorted(
    {(t.upper(), name) for t, name in INDEX_FUNDS + load_screened_universe()}, key=lambda x: x[0]
)


def ticker_picker_options(L: dict) -> list:
    return [L["ticker_picker_placeholder"]] + [f"{t} — {name}" for t, name in TICKER_UNIVERSE]


def make_ticker_picker_callback(picker_key: str, target_key: str, L: dict):
    """Streamlit selectbox widgets can't accept free text, so this picker is a
    convenience shortlist that writes into the real (free-text) ticker field
    rather than replacing it -- anything not in TICKER_UNIVERSE can still be
    typed directly."""
    def _cb():
        picked = st.session_state.get(picker_key, "")
        if picked and picked != L["ticker_picker_placeholder"]:
            st.session_state[target_key] = picked.split(" — ")[0]
    return _cb


# ----------------------------------------------------------------------
# STATE
# ----------------------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "mode" not in st.session_state:
    st.session_state.mode = "chart"
if "buy_results" not in st.session_state:
    st.session_state.buy_results = []
if "sell_result" not in st.session_state:
    st.session_state.sell_result = None
if "weekly_results" not in st.session_state:
    st.session_state.weekly_results = None
if "full_scan_results" not in st.session_state:
    st.session_state.full_scan_results = None

L = TXT[st.session_state.lang]

# ----------------------------------------------------------------------
# TOP CONTROLS: mode + language toggle
# ----------------------------------------------------------------------
top_l, top_mode, top_lang = st.columns([4, 1.6, 1.6])
with top_mode:
    mode_btn_label = L["to_signals_btn"] if st.session_state.mode == "chart" else L["to_chart_btn"]
    if st.button(mode_btn_label, use_container_width=True):
        st.session_state.mode = "signals" if st.session_state.mode == "chart" else "chart"
        st.rerun()
with top_lang:
    if st.button(L["lang_toggle"], use_container_width=True):
        st.session_state.lang = "zh" if st.session_state.lang == "en" else "en"
        st.rerun()

# ----------------------------------------------------------------------
# DARK MODE STYLING
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #171C16; color: #F2F0E6; }
    div[data-testid="stMetric"] {
        background-color: #212820;
        border: 1px solid #333D2E;
        border-radius: 12px;
        padding: 16px;
    }
    .big-title { font-size: 2.2rem; font-weight: 700; margin-bottom: 0; }
    .subtitle { color: #A9B29C; font-size: 1.05rem; margin-top: 4px; }
    footer { visibility: hidden; }
    #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

if st.session_state.mode == "chart":
    st.title(f":material/show_chart: {L['title']}")
    st.markdown(f"<div class='subtitle'>{L['subtitle']}</div>", unsafe_allow_html=True)
else:
    st.title(f":material/candlestick_chart: {L['signal_title']}")
    st.markdown(f"<div class='subtitle'>{L['signal_subtitle']}</div>", unsafe_allow_html=True)
st.write("")

# ----------------------------------------------------------------------
# DATA FETCH (cached) -- chart mode
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
# INDICATOR CALCULATIONS -- chart mode
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
    colors = ["#FF0000", "#D2691E", "#00FFFF", "#4682B4"]
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
# SIGNAL SCANNER CALCULATIONS -- signals mode
# ----------------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def compute_buy_signal(ticker_symbol: str) -> dict:
    try:
        tk = yf.Ticker(ticker_symbol)
        hist = tk.history(period="1y")
        if hist.empty:
            return {"ok": False, "ticker": ticker_symbol, "error_key": "err_no_price"}

        current_price = float(hist["Close"].iloc[-1])
        high_52w = float(hist["High"].max())
        drawdown = (current_price - high_52w) / high_52w * 100

        expirations = tk.options
        if not expirations:
            return {"ok": False, "ticker": ticker_symbol, "error_key": "err_no_options"}

        today = datetime.now()
        far_exps = [e for e in expirations if 120 <= (datetime.strptime(e, "%Y-%m-%d") - today).days <= 360]
        target_exp = far_exps[-1] if far_exps else expirations[-1]

        opt = tk.option_chain(target_exp)
        puts = opt.puts.dropna(subset=["openInterest"]).copy()
        calls = opt.calls.dropna(subset=["openInterest"]).copy()
        if puts.empty or calls.empty:
            return {"ok": False, "ticker": ticker_symbol, "error_key": "err_incomplete_chain"}

        total_call_oi = calls["openInterest"].sum()
        total_put_oi = puts["openInterest"].sum()
        far_pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 0

        max_put_idx = puts["openInterest"].idxmax()
        put_wall_price = float(puts.loc[max_put_idx, "strike"])
        put_wall_oi = int(puts.loc[max_put_idx, "openInterest"])

        max_call_idx = calls["openInterest"].idxmax()
        call_wall_price = float(calls.loc[max_call_idx, "strike"])
        call_wall_oi = int(calls.loc[max_call_idx, "openInterest"])

        buy_zone_low = put_wall_price * 0.90
        buy_zone_high = put_wall_price * 1.05

        if drawdown <= -25:
            if buy_zone_low <= current_price <= buy_zone_high:
                verdict = "signal"
            elif current_price < buy_zone_low:
                verdict = "broken"
            else:
                verdict = "wait"
        else:
            verdict = "none"

        return {
            "ok": True,
            "ticker": ticker_symbol,
            "current_price": current_price,
            "drawdown": drawdown,
            "target_exp": target_exp,
            "far_pcr": far_pcr,
            "put_wall_price": put_wall_price,
            "put_wall_oi": put_wall_oi,
            "call_wall_price": call_wall_price,
            "call_wall_oi": call_wall_oi,
            "buy_zone_low": buy_zone_low,
            "buy_zone_high": buy_zone_high,
            "verdict": verdict,
        }
    except Exception as e:
        return {"ok": False, "ticker": ticker_symbol, "error_key": "err_generic", "error": str(e)}


@st.cache_data(ttl=300, show_spinner=False)
def compute_sell_signal(ticker_symbol: str, entry_price: float, take_profit_pct: float, stop_loss_pct: float) -> dict:
    try:
        tk = yf.Ticker(ticker_symbol)
        hist = tk.history(period="5d")
        if hist.empty:
            return {"ok": False, "ticker": ticker_symbol, "error_key": "err_no_price"}

        current_price = float(hist["Close"].iloc[-1])
        pnl_pct = (current_price - entry_price) / entry_price * 100

        expirations = tk.options
        if not expirations:
            return {"ok": False, "ticker": ticker_symbol, "error_key": "err_no_options"}

        today = datetime.now()
        far_exps = [e for e in expirations if 90 <= (datetime.strptime(e, "%Y-%m-%d") - today).days <= 360]
        target_exp = far_exps[0] if far_exps else expirations[-1]

        opt = tk.option_chain(target_exp)
        calls = opt.calls.dropna(subset=["openInterest"]).copy()
        puts = opt.puts.dropna(subset=["openInterest"]).copy()

        call_wall_price = float(calls.loc[calls["openInterest"].idxmax(), "strike"]) if not calls.empty else None
        put_wall_price = float(puts.loc[puts["openInterest"].idxmax(), "strike"]) if not puts.empty else None

        if pnl_pct <= stop_loss_pct:
            verdict = "stoploss"
        elif put_wall_price and current_price < put_wall_price * 0.95:
            verdict = "break"
        elif call_wall_price and current_price >= call_wall_price * 0.98:
            verdict = "resistance"
        elif pnl_pct >= take_profit_pct:
            verdict = "target"
        else:
            verdict = "hold"

        return {
            "ok": True,
            "ticker": ticker_symbol,
            "entry_price": entry_price,
            "current_price": current_price,
            "pnl_pct": pnl_pct,
            "call_wall_price": call_wall_price,
            "put_wall_price": put_wall_price,
            "verdict": verdict,
            "take_profit_pct": take_profit_pct,
            "stop_loss_pct": stop_loss_pct,
        }
    except Exception as e:
        return {"ok": False, "ticker": ticker_symbol, "error_key": "err_generic", "error": str(e)}


MIN_MARKET_CAP = 1.6e9   # $1.6 billion
MIN_FLOAT_SHARES = 4.0e8  # 400 million shares


# Lookback window per interval, in calendar days back from today. Monthly
# needs much more calendar time than daily/weekly to accumulate the same
# number of bars for the 12/14/20-period rolling calculations.
INTERVAL_LOOKBACK_DAYS = {"1d": 365, "1wk": 364, "1mo": 1095}


def interval_lookback_range(interval: str):
    """(start, end) datetimes for the given interval's lookback window --
    the single source of truth used both to fetch data and to label the
    interval dropdown, so the displayed range always matches what's
    actually fetched."""
    end = datetime.now()
    start = end - pd.Timedelta(days=INTERVAL_LOOKBACK_DAYS[interval])
    return start, end


@st.cache_data(ttl=1800, show_spinner=False)
def compute_weekly_pattern(
    ticker_symbol: str, market_cap: float = 0.0, float_shares: float = 0.0, interval: str = "1wk"
) -> dict:
    """Technical pattern check only -- no fundamentals fetch. Works on
    daily ("1d"), weekly ("1wk"), or monthly ("1mo") candles; the same
    12/14/20-period rule set just means 12/14/20 days/weeks/months
    depending on interval. Pass market_cap/float_shares through if
    already known (e.g. from the pre-screened universe) so they ride
    along in the result without an extra API call."""
    try:
        start, end = interval_lookback_range(interval)
        df = yf.Ticker(ticker_symbol).history(start=start, end=end, interval=interval)
        if df.empty:
            return {"ok": False, "ticker": ticker_symbol, "error_key": "err_insufficient_data"}

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # yfinance's most recent bar is the CURRENT, still-forming period
        # and keeps updating live until that period actually closes -- on
        # a Monday, a "weekly" bar is really just that one day's OHLCV
        # masquerading as a full week (same idea for "today" on daily
        # candles during market hours, or "this month" on monthly ones).
        # Drop it so every calculation below only sees fully completed
        # periods.
        now_local = pd.Timestamp.now(tz=df.index.tz)
        if interval == "1wk":
            is_current_period = df.index[-1].isocalendar()[:2] == now_local.isocalendar()[:2]
        elif interval == "1mo":
            is_current_period = (df.index[-1].year, df.index[-1].month) == (now_local.year, now_local.month)
        else:
            is_current_period = df.index[-1].date() == now_local.date()
        if is_current_period:
            df = df.iloc[:-1]

        if len(df) < 30:
            return {"ok": False, "ticker": ticker_symbol, "error_key": "err_insufficient_data"}

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]
        open_p = df["Open"]

        # Weekly 14-period RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi14 = 100 - (100 / (1 + rs))

        # 12-week volume MA & 12-week resistance level
        vol_ma12 = volume.rolling(12).mean()
        tr_high_12w = high.shift(1).rolling(12).max()

        curr_close = float(close.iloc[-1])
        curr_open = float(open_p.iloc[-1])
        curr_high = float(high.iloc[-1])
        curr_low = float(low.iloc[-1])
        curr_vol = float(volume.iloc[-1])
        curr_rsi = float(rsi14.iloc[-1])

        prev_tr_high = float(tr_high_12w.iloc[-1])
        avg_vol = float(vol_ma12.iloc[-1])

        signal_keys = []

        # Signal 1: Weekly Bullish Divergence
        # (Price hits a 12-week low + RSI is above its prior 12-week minimum by >3
        # + RSI < 45 + a rebound candle confirms it)
        low_12w_prev = float(low.shift(1).rolling(12).min().iloc[-1])
        rsi_12w_prev_min = float(rsi14.shift(1).rolling(12).min().iloc[-1])
        is_bottom_rebound = (curr_close > curr_open) or (curr_close > curr_low * 1.01)

        if (curr_low < low_12w_prev and curr_rsi > rsi_12w_prev_min + 3
                and curr_rsi < 45 and is_bottom_rebound):
            signal_keys.append("bull_div")

        # Signal 2: Weekly SOS Breakout
        # (Close breaks the 12-week high + weekly body gain > 2% + volume > 1.2x
        # the 12-week average)
        if (curr_close > prev_tr_high and curr_close > curr_open * 1.02
                and curr_vol > avg_vol * 1.20):
            signal_keys.append("sos_breakout")

        # Signal 3: Weekly Bearish Divergence
        # (Price hits a 20-week high + RSI weakens vs. its prior 20-week max +
        # weekly red candle + RSI still > 58)
        high_20w_prev = float(high.shift(1).rolling(20).max().iloc[-1])
        rsi_20w_prev_max = float(rsi14.shift(1).rolling(20).max().iloc[-1])
        prev_close = float(close.iloc[-2])

        if (curr_high > high_20w_prev and curr_rsi < rsi_20w_prev_max - 4
                and curr_close < prev_close and curr_rsi > 58):
            signal_keys.append("bear_div")

        return {
            "ok": True,
            "ticker": ticker_symbol,
            "close": curr_close,
            "rsi": curr_rsi,
            "signal_keys": signal_keys,
            "market_cap": market_cap,
            "float_shares": float_shares,
            "interval": interval,
        }
    except Exception as e:
        return {"ok": False, "ticker": ticker_symbol, "error_key": "err_generic", "error": str(e)}


@st.cache_data(ttl=1800, show_spinner=False)
def compute_weekly_signals(ticker_symbol: str) -> dict:
    """Manual-pool version: screens fundamentals first (ticker isn't
    known to be pre-screened), then delegates to compute_weekly_pattern."""
    try:
        info = yf.Ticker(ticker_symbol).info
        market_cap = info.get("marketCap", 0) or 0
        float_shares = info.get("floatShares") or info.get("sharesOutstanding", 0) or 0

        if market_cap < MIN_MARKET_CAP or float_shares < MIN_FLOAT_SHARES:
            return {
                "ok": False,
                "ticker": ticker_symbol,
                "error_key": "err_fundamentals_screen",
                "market_cap": market_cap,
                "float_shares": float_shares,
            }
    except Exception as e:
        return {"ok": False, "ticker": ticker_symbol, "error_key": "err_generic", "error": str(e)}

    return compute_weekly_pattern(ticker_symbol, market_cap, float_shares)


def render_buy_card(r: dict, L: dict):
    with st.container(border=True):
        st.markdown(f"#### {r['ticker']}")
        if not r["ok"]:
            st.error(L[r["error_key"]].format(ticker=r["ticker"], error=r.get("error", "")))
            return

        c1, c2 = st.columns(2)
        c1.metric(L["current_price_label"], f"${r['current_price']:.2f}")
        c2.metric(L["drawdown_label"], f"{r['drawdown']:.1f}%")
        c3, c4 = st.columns(2)
        c3.metric(L["put_wall_label"], f"${r['put_wall_price']:.2f}")
        c4.metric(L["call_wall_label"], f"${r['call_wall_price']:.2f}")
        st.caption(
            f"{L['buy_zone_label']}: "
            f"\\${r['buy_zone_low']:.2f} (90%) – "
            f"\\${r['put_wall_price']:.2f} (100%) – "
            f"\\${r['buy_zone_high']:.2f} (105%)"
        )
        st.caption(
            f"{L['put_wall_oi_label']}: {r['put_wall_oi']:,} · "
            f"{L['call_wall_oi_label']}: {r['call_wall_oi']:,} · "
            f"{L['pcr_label']}: {r['far_pcr']:.2f} · "
            f"{L['expiration_label']}: {r['target_exp']}"
        )

        verdict = r["verdict"]
        title = L[f"buy_verdict_{verdict}_title"]
        detail = L[f"buy_verdict_{verdict}_detail"].format(
            drawdown=abs(r["drawdown"]), wall=r["put_wall_price"]
        )
        box = {"signal": st.success, "broken": st.warning, "wait": st.info, "none": st.info}[verdict]
        box(f"**{title}**\n\n{detail}")


def render_sell_card(r: dict, L: dict):
    with st.container(border=True):
        st.markdown(f"#### {r['ticker']}")
        if not r["ok"]:
            st.error(L[r["error_key"]].format(ticker=r["ticker"], error=r.get("error", "")))
            return

        c1, c2, c3 = st.columns(3)
        c1.metric(L["entry_price_display_label"], f"${r['entry_price']:.2f}")
        c2.metric(L["current_price_label"], f"${r['current_price']:.2f}")
        c3.metric(L["pnl_label"], f"{r['pnl_pct']:+.2f}%")

        wall_bits = []
        if r["call_wall_price"] is not None:
            wall_bits.append(f"{L['call_wall_label']}: \\${r['call_wall_price']:.2f}")
        if r["put_wall_price"] is not None:
            wall_bits.append(f"{L['put_wall_label']}: \\${r['put_wall_price']:.2f}")
        if wall_bits:
            st.caption(" · ".join(wall_bits))

        verdict = r["verdict"]
        title = L[f"sell_verdict_{verdict}_title"]
        detail = L[f"sell_verdict_{verdict}_detail"].format(
            pnl=abs(r["pnl_pct"]),
            limit=abs(r["stop_loss_pct"]),
            target=r["take_profit_pct"],
            wall_put=r["put_wall_price"] if r["put_wall_price"] is not None else 0.0,
            wall_call=r["call_wall_price"] if r["call_wall_price"] is not None else 0.0,
        )
        box = {
            "stoploss": st.error,
            "break": st.error,
            "resistance": st.success,
            "target": st.success,
            "hold": st.info,
        }[verdict]
        box(f"**{title}**\n\n{detail}")


INTERVAL_PERIOD_KEY = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}


def format_signal_names(signal_keys: list, interval: str, L: dict) -> str:
    period_word = L[INTERVAL_PERIOD_KEY.get(interval, "weekly")]
    return " | ".join(f"{period_word} {L[f'pattern_{k}']}" for k in signal_keys)


def render_weekly_results(results: list, L: dict):
    matched = [r for r in results if r.get("ok") and r["signal_keys"]]
    ok_no_signal = [r for r in results if r.get("ok") and not r["signal_keys"]]
    screened_out = [r for r in results if not r.get("ok") and r.get("error_key") == "err_fundamentals_screen"]
    errored = [r for r in results if not r.get("ok") and r.get("error_key") != "err_fundamentals_screen"]

    if matched:
        matched.sort(key=lambda r: r.get("market_cap", 0), reverse=True)
        table_rows = [
            {
                L["weekly_col_ticker"]: r["ticker"],
                L["weekly_col_close"]: f"${r['close']:.2f}",
                L["weekly_col_market_cap"]: f"{r['market_cap'] / 1e9:.2f}",
                L["weekly_col_float"]: f"{r['float_shares'] / 1e6:.1f}",
                L["weekly_col_rsi"]: f"{r['rsi']:.1f}",
                L["weekly_col_signals"]: format_signal_names(r["signal_keys"], r.get("interval", "1wk"), L),
            }
            for r in matched
        ]
        results_df = pd.DataFrame(table_rows)
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        st.download_button(
            L["weekly_download_button"],
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="weekly_signals.csv",
            mime="text/csv",
        )
    elif ok_no_signal:
        st.info(L["weekly_no_signals"])

    if screened_out:
        st.caption(L["weekly_screened_caption"].format(tickers=", ".join(r["ticker"] for r in screened_out)))
    if errored:
        st.caption(L["weekly_errors_caption"].format(tickers=", ".join(r["ticker"] for r in errored)))


# PDF text is kept plain-ASCII English regardless of UI language: fpdf2's
# built-in core fonts (Helvetica etc.) only support latin-1, so neither
# emoji nor Chinese characters render correctly without embedding a custom
# Unicode font. The on-screen results stay fully translated either way.
PDF_PATTERN_NAMES = {
    "bull_div": "Bullish Divergence",
    "sos_breakout": "SOS Breakout",
    "bear_div": "Bearish Divergence",
}


PDF_PERIOD_WORD = {"1d": "Daily", "1wk": "Weekly", "1mo": "Monthly"}


def format_signal_names_pdf(signal_keys: list, interval: str) -> str:
    period_word = PDF_PERIOD_WORD.get(interval, "Weekly")
    return ", ".join(f"{period_word} {PDF_PATTERN_NAMES[k]}" for k in signal_keys)


def build_weekly_pdf(matched: list, universe_size: int, interval: str = "1wk") -> bytes:
    period_word = PDF_PERIOD_WORD.get(interval, "Weekly")
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, f"{period_word} Signal Scan Report", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Candle interval: {period_word}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Universe scanned: {universe_size} US stocks (Market Cap >= $1.6B, Float Shares >= 400M)",
             new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"Signals triggered: {len(matched)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    if matched:
        col_widths = [18, 20, 24, 24, 15, 90]
        headers = ["Ticker", "Close", "MCap ($B)", "Float (M)", "RSI", "Signals"]
        pdf.set_font("Helvetica", "B", 9)
        for w, h in zip(col_widths, headers):
            pdf.cell(w, 7, h, border=1)
        pdf.ln()

        pdf.set_font("Helvetica", "", 8)
        for r in matched:
            row = [
                r["ticker"],
                f"${r['close']:.2f}",
                f"{r['market_cap'] / 1e9:.2f}",
                f"{r['float_shares'] / 1e6:.1f}",
                f"{r['rsi']:.1f}",
                format_signal_names_pdf(r["signal_keys"], r.get("interval", "1wk")),
            ]
            for w, val in zip(col_widths, row):
                pdf.cell(w, 6, str(val), border=1)
            pdf.ln()
    else:
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(0, 6, "No signals triggered this week.", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 5, "This tool is for educational purposes only and is not financial advice.")

    return bytes(pdf.output())


def run_full_universe_scan(universe: list, L: dict, interval: str = "1wk") -> list:
    """Runs the technical pattern check across the full pre-screened
    universe with a live progress bar. Modest concurrency (5 workers)
    to stay gentle on Yahoo Finance's rate limits -- this hits the
    price-history endpoint, not the heavier .info endpoint, and the
    universe is pre-screened so no fundamentals calls are needed here."""
    total = len(universe)
    progress_bar = st.progress(0.0)
    status = st.empty()
    results = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futures = {
            ex.submit(compute_weekly_pattern, r["ticker"], r["market_cap"], r["float_shares"], interval): r["ticker"]
            for r in universe
        }
        for fut in concurrent.futures.as_completed(futures):
            done += 1
            ticker = futures[fut]
            progress_bar.progress(done / total)
            status.text(L["weekly_full_scan_progress"].format(done=done, total=total, ticker=ticker))
            results.append(fut.result())
    status.empty()
    progress_bar.empty()
    return results


# ========================================================================
# MODE: CHART
# ========================================================================
if st.session_state.mode == "chart":
    st.selectbox(
        L["ticker_picker_label"],
        options=ticker_picker_options(L),
        index=0,
        key="chart_ticker_picker",
        on_change=make_ticker_picker_callback("chart_ticker_picker", "ticker_text_input", L),
    )

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        ticker_input = st.text_input(
            L["ticker_label"], value="AAPL", help=L["ticker_help"], key="ticker_text_input"
        ).strip().upper()
    with c2:
        period_label = st.selectbox(L["period_label"], list(PERIOD_OPTIONS.keys()), index=1)
    with c3:
        interval_choice = st.radio(
            L["interval_label"], [L["daily"], L["weekly"]], horizontal=True
        )

    interval = "1wk" if interval_choice == L["weekly"] else "1d"
    period = PERIOD_OPTIONS[period_label]

    if not ticker_input:
        st.stop()

    with st.spinner(L["loading"]):
        raw_df = fetch_data(ticker_input, period, interval)

    if raw_df.empty or len(raw_df) < 5:
        st.error(L["error_no_data"])
        st.stop()

    df = compute_indicators(raw_df)
    gamma_pts, decel_pts, breakout_pts = detect_signals(df, L)
    wall_data = compute_buy_signal(ticker_input)

    # --- Summary card ---
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

    # --- Chart ---
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=(L["chart1_title"], L["chart2_title"], L["chart3_title"]),
    )

    # Row 1: Candlestick + SMA50
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

    # Put wall / call wall lines + buy/sell arrow, reusing the Buy Scanner's
    # cached options-derived walls and verdict logic (same formula, no
    # separate/duplicate signal definition).
    if wall_data.get("ok"):
        put_wall_price = wall_data["put_wall_price"]
        call_wall_price = wall_data["call_wall_price"]

        fig.add_hline(
            y=put_wall_price, row=1, col=1,
            line=dict(color="#4CAF50", dash="dash", width=1),
            annotation_text=f"{L['put_wall_line_label']} ${put_wall_price:.2f}",
            annotation_position="bottom left",
            annotation_font=dict(size=10, color="#4CAF50"),
        )
        fig.add_hline(
            y=call_wall_price, row=1, col=1,
            line=dict(color="#FF6B6B", dash="dash", width=1),
            annotation_text=f"{L['call_wall_line_label']} ${call_wall_price:.2f}",
            annotation_position="top left",
            annotation_font=dict(size=10, color="#FF6B6B"),
        )

        last_idx = df.index[-1]
        if wall_data["verdict"] == "signal":
            fig.add_trace(
                go.Scatter(
                    x=[last_idx], y=[float(df["Low"].iloc[-1]) * 0.98],
                    mode="markers", name=L["buy_arrow_label"],
                    marker=dict(symbol="triangle-up", size=16, color="#4CAF50", line=dict(width=1, color="black")),
                ),
                row=1, col=1,
            )
        elif last_close >= call_wall_price * 0.98 or last_close < put_wall_price * 0.95:
            fig.add_trace(
                go.Scatter(
                    x=[last_idx], y=[float(df["High"].iloc[-1]) * 1.02],
                    mode="markers", name=L["sell_arrow_label"],
                    marker=dict(symbol="triangle-down", size=16, color="#FF6B6B", line=dict(width=1, color="black")),
                ),
                row=1, col=1,
            )

    # Row 2: OBV
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

    # Row 3: Momentum histogram
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
            arrowhead=2, ay=30, font=dict(size=10, color="#D2691E"),
        )
    for idx, y, text in breakout_pts:
        fig.add_annotation(
            x=idx, y=y, row=3, col=1, text=text, showarrow=True,
            arrowhead=2, ay=30, font=dict(size=10, color="#00FFFF"),
        )

    fig.add_hline(y=0, line=dict(color="#555555", width=1), row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        height=900,
        plot_bgcolor="#171C16",
        paper_bgcolor="#171C16",
        font=dict(color="#F2F0E6"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=60, b=20),
        xaxis3=dict(rangeslider=dict(visible=False)),
    )
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikecolor="grey", spikethickness=1)
    fig.update_yaxes(showspikes=True, spikethickness=1)

    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False}, theme=None)

# ========================================================================
# MODE: SIGNALS
# ========================================================================
else:
    st.info(L["signal_disclaimer"])
    st.write("")

    # --- Buy signal scanner ---
    st.subheader(L["buy_section_title"])
    st.caption(L["buy_section_desc"])
    watchlist_input = st.text_input(L["watchlist_label"], value="BMY, GM, RMBS", key="watchlist_input")
    if st.button(L["scan_button"], key="scan_btn"):
        tickers = [t.strip().upper() for t in watchlist_input.split(",") if t.strip()]
        with st.spinner(L["loading"]):
            st.session_state.buy_results = [compute_buy_signal(t) for t in tickers]

    for r in st.session_state.buy_results:
        render_buy_card(r, L)

    st.write("")
    st.divider()

    # --- Sell / take-profit monitor ---
    st.subheader(L["sell_section_title"])
    st.caption(L["sell_section_desc"])
    st.selectbox(
        L["ticker_picker_label"],
        options=ticker_picker_options(L),
        index=0,
        key="sell_ticker_picker",
        on_change=make_ticker_picker_callback("sell_ticker_picker", "sell_ticker", L),
    )
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        sell_ticker = st.text_input(L["sell_ticker_label"], value="BMY", key="sell_ticker")
    with s2:
        entry_price = st.number_input(L["entry_price_label"], value=42.50, min_value=0.0, step=0.5, key="entry_price")
    with s3:
        tp_pct = st.number_input(L["take_profit_label"], value=30.0, step=1.0, key="tp_pct")
    with s4:
        sl_pct = st.number_input(L["stop_loss_label"], value=-15.0, step=1.0, key="sl_pct")

    if st.button(L["check_button"], key="check_btn"):
        with st.spinner(L["loading"]):
            st.session_state.sell_result = compute_sell_signal(
                sell_ticker.strip().upper(), entry_price, tp_pct, sl_pct
            )

    if st.session_state.sell_result:
        render_sell_card(st.session_state.sell_result, L)

    st.write("")
    st.divider()

    # --- Weekly signal scanner ---
    st.subheader(L["weekly_section_title"])
    st.caption(L["weekly_section_desc"])
    weekly_pool_input = st.text_input(
        L["weekly_pool_label"],
        value="AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AMD, NFLX, QCOM, RMBS, GM, BMY",
        key="weekly_pool_input",
    )
    if st.button(L["weekly_scan_button"], key="weekly_scan_btn"):
        tickers = [t.strip().upper() for t in weekly_pool_input.split(",") if t.strip()]
        with st.spinner(L["loading"]):
            st.session_state.weekly_results = [compute_weekly_signals(t) for t in tickers]

    if st.session_state.weekly_results is not None:
        render_weekly_results(st.session_state.weekly_results, L)

    st.write("")
    st.divider()

    # --- Full pre-screened universe scan ---
    st.subheader(L["full_scan_title"])
    full_universe = load_screened_universe_full()
    st.caption(L["full_scan_desc"].format(count=len(full_universe)))

    full_scan_interval_options = [
        ("1d", "full_scan_option_daily"),
        ("1wk", "full_scan_option_weekly"),
        ("1mo", "full_scan_option_monthly"),
    ]
    full_scan_interval_labels = [
        L[label_key].format(
            start=interval_lookback_range(code)[0].strftime("%Y-%m-%d"),
            end=interval_lookback_range(code)[1].strftime("%Y-%m-%d"),
        )
        for code, label_key in full_scan_interval_options
    ]
    full_scan_interval_idx = st.selectbox(
        L["full_scan_interval_label"],
        options=range(len(full_scan_interval_options)),
        format_func=lambda i: full_scan_interval_labels[i],
        index=1,  # default to Weekly
        key="full_scan_interval",
    )
    full_scan_interval = full_scan_interval_options[full_scan_interval_idx][0]

    if st.button(L["full_scan_button"].format(count=len(full_universe)), key="full_scan_btn", disabled=not full_universe):
        st.session_state.full_scan_results = run_full_universe_scan(full_universe, L, full_scan_interval)

    if st.session_state.full_scan_results is not None:
        render_weekly_results(st.session_state.full_scan_results, L)
        matched = [r for r in st.session_state.full_scan_results if r.get("ok") and r["signal_keys"]]
        matched.sort(key=lambda r: r.get("market_cap", 0), reverse=True)
        result_interval = matched[0]["interval"] if matched else full_scan_interval
        pdf_bytes = build_weekly_pdf(matched, len(full_universe), result_interval)
        st.download_button(
            L["full_scan_pdf_button"],
            data=pdf_bytes,
            file_name="weekly_signal_scan_report.pdf",
            mime="application/pdf",
        )

st.caption(L["footer"])
