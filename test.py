import streamlit as st
import pandas as pd
import sqlite3
import websocket
import json
import threading
import matplotlib.pyplot as plt
import time

from statsmodels.regression.linear_model import OLS
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.norms import HuberT
import statsmodels.api as sm

# =========================
# CONFIG
# =========================
AVAILABLE_SYMBOLS = ["btcusdt", "ethusdt", "bnbusdt", "solusdt", "xrpusdt"]

# =========================
# DB SETUP
# =========================
conn = sqlite3.connect("ticks.db", check_same_thread=False)

conn.execute("""
CREATE TABLE IF NOT EXISTS ticks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT,
    ts TIMESTAMP,
    price REAL,
    size REAL
)
""")
conn.commit()

if "db_cleared" not in st.session_state:
    conn.execute("DELETE FROM ticks")
    conn.execute("DELETE FROM sqlite_sequence WHERE name='ticks'")
    conn.commit()
    st.session_state["db_cleared"] = True

# =========================
# WEBSOCKET
# =========================
def normalize(j):
    ts = pd.to_datetime(j["T"], unit="ms")
    return {
        "symbol": j["s"],
        "ts": ts.isoformat(),
        "price": float(j["p"]),
        "size": float(j["q"]),
    }

def on_message(ws, message):
    j = json.loads(message)
    if j.get("e") == "trade":
        t = normalize(j)
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO ticks (symbol, ts, price, size) VALUES (?, ?, ?, ?)",
                (t["symbol"], t["ts"], t["price"], t["size"]),
            )
            conn.commit()
        except Exception as e:
            print("DB insert error:", e)

def on_error(ws, error):
    print("WS error:", error)

def on_close(ws, code, msg):
    print("WS closed:", code, msg)

def start_ws(symbols):
    def run_ws(url):
        ws = websocket.WebSocketApp(
            url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws.run_forever()

    for sym in symbols:
        url = f"wss://fstream.binance.com/ws/{sym}@trade"
        t = threading.Thread(target=run_ws, args=(url,), daemon=True)
        t.start()

# =========================
# ANALYTICS FUNCS
# =========================
def resample(df, freq="1T"):
    return df.resample(freq).agg({"price": "mean", "size": "sum"})

def zscore(series):
    return (series - series.mean()) / series.std()

def hedge_ratio_ols(y, x):
    x_const = sm.add_constant(x)
    model = OLS(y, x_const).fit()
    return model.params[1]

def hedge_ratio_huber(y, x):
    x_const = sm.add_constant(x)
    model = RLM(y, x_const, M=HuberT())
    results = model.fit()
    return results.params[1]

def mini_backtest(z):
    position = 0
    positions = []
    for val in z:
        if val > 2:
            position = -1
        elif val < -2:
            position = 1
        elif abs(val) < 0.1:
            position = 0
        positions.append(position)
    return pd.Series(positions, index=z.index)

# =========================
# UI
# =========================
st.title("📊 Real-Time Quant Analytics Dashboard")

stream_symbols = st.multiselect(
    "Select symbols to stream",
    AVAILABLE_SYMBOLS,
    default=["btcusdt", "ethusdt"],
)

if st.button("🚀 Start Analyzer"):
    if "ws_started" not in st.session_state:
        start_ws(stream_symbols)
        st.session_state["ws_started"] = True
        st.success("WebSocket(s) started")
    else:
        st.info("WebSockets already running")

# auto-refresh
REFRESH_INTERVAL = 5
if "last_refresh" not in st.session_state:
    st.session_state["last_refresh"] = time.time()
if time.time() - st.session_state["last_refresh"] > REFRESH_INTERVAL:
    st.session_state["last_refresh"] = time.time()
    st.rerun()

# =========================
# LOAD DATA
# =========================
df = pd.read_sql("SELECT * FROM ticks", conn)

if not df.empty and "ts" in df.columns:
    df["ts"] = pd.to_datetime(df["ts"])
    df.set_index("ts", inplace=True)

# =========================
# RAW TICKS (all symbols)
# =========================
st.subheader("Raw Tick Price (all symbols)")
if not df.empty and "price" in df.columns:
    st.line_chart(df["price"])
else:
    st.info("Waiting for live data... click Start Analyzer and wait a few seconds")

# =========================
# TWO SINGLE-SYMBOL GRAPHS
# =========================
if not df.empty and "symbol" in df.columns:
    all_syms = sorted(df["symbol"].unique().tolist())

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Symbol 1")
        sym1 = st.selectbox("Select symbol for left graph", all_syms, index=0)
        df1 = df[df["symbol"] == sym1]
        if not df1.empty:
            st.line_chart(df1["price"])

    with col2:
        st.subheader("Symbol 2")
        # if at least 2 symbols, default to second one, else use first
        default_index = 1 if len(all_syms) > 1 else 0
        sym2 = st.selectbox("Select symbol for right graph", all_syms, index=default_index)
        df2 = df[df["symbol"] == sym2]
        if not df2.empty:
            st.line_chart(df2["price"])

# =========================
# PAIR ANALYTICS (spread/backtest)
# =========================
if not df.empty and "symbol" in df.columns and len(stream_symbols) >= 2:
    s1, s2 = stream_symbols[:2]

    if s1 in df["symbol"].unique() and s2 in df["symbol"].unique():
        d1 = resample(df[df["symbol"] == s1])
        d2 = resample(df[df["symbol"] == s2])

        merged = pd.merge(
            d1,
            d2,
            left_index=True,
            right_index=True,
            suffixes=(f"{s1}", f"{s2}"),
        ).dropna()

        if not merged.empty:
            merged["spread"] = merged[f"price_{s1}"] - merged[f"price_{s2}"]
            merged["z"] = zscore(merged["spread"])

            beta_ols = hedge_ratio_ols(merged[f"price_{s1}"], merged[f"price_{s2}"])
            beta_huber = hedge_ratio_huber(merged[f"price_{s1}"], merged[f"price_{s2}"])

            st.subheader("📐 Hedge Ratios")
            st.write(f"{s1}/{s2} OLS Hedge Ratio: {beta_ols:.4f}")
            st.write(f"{s1}/{s2} Robust (Huber) Hedge Ratio: {beta_huber:.4f}")

            st.subheader("📉 Spread & Z-Score")
            st.line_chart(merged[["spread", "z"]])

            z_alert = st.slider("Z-Score Alert", 0.5, 5.0, 2.0)
            if abs(merged["z"].iloc[-1]) > z_alert:
                st.warning("⚠️ Z-Score threshold breached")

            merged["position"] = mini_backtest(merged["z"])
            st.subheader("📈 Mini Mean-Reversion Backtest")
            st.line_chart(merged["position"])

# footer
st.caption("Session-scoped • No trading • Research-grade analytics")