# streamlit_app.py
import streamlit as st
import pandas as pd
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from fredapi import Fred

# ---------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------
st.set_page_config(page_title="USD/CHF Macro Correlation Dashboard", layout="wide")
st.title("💹 USD/CHF Correlation Dashboard")
st.markdown("Analyze how USD/CHF correlates with key U.S. macroeconomic indicators.")

# ---------------------------------------------
# Configuration
# ---------------------------------------------
# Replace with your own FRED API key if you have one
# You can get one free from https://fred.stlouisfed.org/
FRED_API_KEY = st.secrets.get("FRED_API_KEY", None)
fred = Fred(api_key=FRED_API_KEY) if FRED_API_KEY else None

# ---------------------------------------------
# Helper Functions
# ---------------------------------------------
@st.cache_data(ttl=3600)
def get_fx_data():
   """Fetch USD/CHF monthly data from Yahoo Finance."""
   try:
       fx = yf.download("CHF=X", start="1995-01-01", interval="1mo", progress=False)
       fx = fx[['Close']].rename(columns={'Close': 'USDCHF'})
       fx.index = fx.index.to_period('M').to_timestamp()
       return fx
   except Exception as e:
       st.error(f"Error fetching FX data: {e}")
       return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fred_data(symbol, start, end):
   """Fetch FRED data using fredapi."""
   if not fred:
       st.warning("No FRED API key found. Some data may not load.")
       return pd.DataFrame()
   try:
       series = fred.get_series(symbol)
       df = pd.DataFrame(series, columns=[symbol])
       df = df.loc[start:end]
       return df
   except Exception as e:
       st.warning(f"Could not fetch {symbol}: {e}")
       return pd.DataFrame()

def merge_data(fx_df, fred_dict):
   """Combine FX and FRED macro data into one DataFrame."""
   if fx_df.empty:
       return pd.DataFrame()
   df = fx_df.copy() 
 # Flatten MultiIndex columns if present
   if isinstance(df.columns, pd.MultiIndex):
      df.columns = ['_'.join(map(str, col)).strip() for col in df.columns.values]
   for symbol, data in fred_dict.items():      
      if not data.empty:
         # Flatten FRED data columns if needed
         if isinstance(data.columns, pd.MultiIndex):
             data.columns = ['_'.join(map(str, col)).strip() for col in data.columns.values]
         df = df.join(data, how="left")
   df = df.ffill().dropna()
   return df

def compute_correlation(df):
   """Compute correlation matrix between USDCHF and macro variables."""
   if df.empty:
       return pd.DataFrame()
   df_ret = df.pct_change().dropna()
   return df_ret.corr()

# ---------------------------------------------
# Sidebar
# ---------------------------------------------
with st.sidebar:
   st.header("Settings")

   today = datetime.date.today()
   default_start = today - datetime.timedelta(days=5 * 365)
   start_date = st.date_input("Start Date", default_start)
   end_date = st.date_input("End Date", today)

   st.subheader("Select Macro Indicators (FRED Symbols)")
   macros = {
       "US CPI": "CPIAUCSL",
       "Unemployment Rate": "UNRATE",
       "10Y Treasury Yield": "DGS10",
       "Fed Funds Rate": "FEDFUNDS",
   }
   selected_macros = [symbol for label, symbol in macros.items()
                      if st.checkbox(f"{label} ({symbol})", value=True)]

   custom_symbol = st.text_input("Add Custom FRED Symbol (optional)").strip().upper()
   if custom_symbol:
       selected_macros.append(custom_symbol)

# ---------------------------------------------
# Fetch and Display Data
# ---------------------------------------------
st.info("Fetching data...")

fx_data = get_fx_data()
if fx_data.empty:
   st.stop()

fred_data = {symbol: get_fred_data(symbol, start_date, end_date) for symbol in selected_macros}
merged = merge_data(fx_data, fred_data)

if merged.empty:
   st.error("No merged data available to display.")
   st.stop()

st.subheader("📊 Data Preview")
st.dataframe(merged.tail(10))

corr_matrix = compute_correlation(merged)
if not corr_matrix.empty:
   st.subheader("📈 Correlation Matrix (Returns)")
   st.dataframe(corr_matrix.style.background_gradient(cmap="RdBu_r", axis=None))
else:
   st.warning("Not enough data to compute correlation matrix.")

# ---------------------------------------------
# Plot Section
# ---------------------------------------------
st.subheader("📉 USD/CHF vs Macroeconomic Indicators")

fig, ax1 = plt.subplots(figsize=(10, 4))
st.write("Merged columns:", merged.columns.tolist())
ax1.plot(merged.index, merged["<actual_column_name>"], label="USD/CHF", color="blue")
ax1.set_ylabel("USD/CHF", color="blue")
ax1.tick_params(axis="y", labelcolor="blue")
ax1.grid(True)

ax2 = ax1.twinx()
for symbol in selected_macros:
   if symbol in merged.columns:
       ax2.plot(merged.index, merged[symbol], label=symbol)
ax2.set_ylabel("Macro Indicators", color="gray")
ax2.tick_params(axis="y", labelcolor="gray")

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="upper left")

st.pyplot(fig)
st.caption("Data sources: Yahoo Finance (USD/CHF) and FRED (macroeconomic indicators).")
