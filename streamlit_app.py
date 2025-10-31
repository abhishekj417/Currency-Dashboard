# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
# ---- Config ----
st.set_page_config(page_title="USD/CHF Correlation Dashboard", layout="wide")
# ---- Helpers ----
@st.cache_data(ttl=3600)
def fetch_fx(ticker="CHF=X", start=None, end=None):
   """
   Fetch FX price series via yfinance.
   Ticker 'CHF=X' is commonly used for USD/CHF (check your data provider if you disagree).
   Returns daily Close prices as dataframe with column 'Close'.
   """
   start = start or (datetime.date.today() - datetime.timedelta(days=365*3))
   end = end or datetime.date.today()
   try:
       df = yf.download(ticker, start=start, end=end, progress=False)
       if df is None or df.empty:
           return pd.DataFrame()
       df = df[['Close']].rename(columns={'Close': 'USDCHF'})
       df.index = pd.to_datetime(df.index)
       return df
   except Exception as e:
       st.error(f"Failed to fetch FX data: {e}")
       return pd.DataFrame()
@st.cache_data(ttl=3600)
def fetch_fred_series(symbol, start=None, end=None):
   """
   Fetch macro series from FRED via pandas_datareader.
   Example symbols: 'CPIAUCSL' (US CPI), 'UNRATE' (Unemployment Rate), 'DGS10' (10yr Treasury).
   """
   start = start or (datetime.date.today() - datetime.timedelta(days=365*5))
   end = end or datetime.date.today()
   try:
       df = pdr.DataReader(symbol, 'fred', start, end)
       if df is None or df.empty:
           return pd.DataFrame()
       df.index = pd.to_datetime(df.index)
       return df.rename(columns={df.columns[0]: symbol})
   except Exception as e:
       st.warning(f"Could not fetch {symbol} from FRED: {e}")
       return pd.DataFrame()
def prepare_data(fx_df, macro_dfs):
   """
   Align FX and macro on business days and compute pct_change or level as appropriate.
   Returns a merged DataFrame with daily aligned data and % returns where applicable.
   """
   # Resample quarterly/monthly macro to forward-fill to daily (common practice)
   # Convert macro frequency to daily by forward filling
   merged = fx_df.copy()
   for name, df in macro_dfs.items():
       if df.empty:
           continue
       # upsample macro to daily and forward fill
       df_daily = df.reindex(pd.date_range(df.index.min(), fx_df.index.max(), freq='D'))
       df_daily = df_daily.ffill()
       merged = merged.join(df_daily, how='left')
   merged = merged.dropna(how='all')
   return merged
def compute_correlations(df, window=None):
   """
   Compute correlation matrix of daily returns (for FX) and changes for macros.
   If window is provided (int days), compute rolling correlation between USDCHF returns and each macro series.
   """
   result = {}
   # Convert FX to returns
   df_returns = pd.DataFrame()
   if 'USDCHF' in df.columns:
       df_returns['USDCHF_ret'] = df['USDCHF'].pct_change()
   # For macro series, compute pct_change if numeric (use pct_change to capture movement)
   for col in df.columns:
       if col == 'USDCHF':
           continue
       series = df[col]
       # if series has many repeating values (like daily forward-filled monthly), pct_change is OK
       df_returns[f"{col}_chg"] = series.pct_change()
   corr_matrix = df_returns.corr()
   result['corr_matrix'] = corr_matrix
   if window:
       rolling = {}
       for col in df_returns.columns:
           if col == 'USDCHF_ret':
               continue
           rolling[col] = df_returns['USDCHF_ret'].rolling(window).corr(df_returns[col])
       result['rolling'] = pd.DataFrame(rolling)
   else:
       result['rolling'] = pd.DataFrame()
   return result
# ---- UI ----
st.title("USD/CHF Correlation Dashboard — Quick & Dirty (robust)")
with st.sidebar:
   st.header("Settings")
   today = datetime.date.today()
   default_start = today - datetime.timedelta(days=365*3)
   start_date = st.date_input("Start date", default_start)
   end_date = st.date_input("End date", today)
   window = st.number_input("Rolling window (days, 0 = none)", min_value=0, max_value=365, value=90, step=1)
   st.markdown("**Macro series to include (FRED symbols)**")
   # Provide default list but allow editing
   default_macros = {
       "US CPI (CPIAUCSL)": "CPIAUCSL",
       "US Unemployment Rate (UNRATE)": "UNRATE",
       "US 10Y Treasury Yield (DGS10)": "DGS10"
   }
   # Allow user to select which ones to include
   macros_selected = []
   for label, symbol in default_macros.items():
       if st.checkbox(f"{label} — {symbol}", value=True):
           macros_selected.append(symbol)
   # Allow adding a custom FRED symbol
   custom_symbol = st.text_input("Add custom FRED symbol (optional)", value="")
   if custom_symbol.strip():
       macros_selected.append(custom_symbol.strip().upper())
st.info("Fetching data... (this may take a few seconds)")
# Fetch FX
fx_df = fetch_fx("CHF=X", start=start_date, end=end_date)
if fx_df.empty:
   st.error("No FX data. Make sure you have internet access and that the ticker is correct ('CHF=X' used).")
   st.stop()
# Fetch macros
macro_dfs = {}
for symbol in set(macros_selected):
   df = fetch_fred_series(symbol, start=start_date, end=end_date)
   macro_dfs[symbol] = df
merged = prepare_data(fx_df, macro_dfs)
st.subheader("Data preview")
st.dataframe(merged.tail(10))
corrs = compute_correlations(merged, window=(window if window > 0 else None))
st.subheader("Correlation matrix (returns/changes)")
if 'corr_matrix' in corrs and not corrs['corr_matrix'].empty:
   st.dataframe(corrs['corr_matrix'])
else:
   st.write("No usable correlation data (not enough data points).")
if window > 0 and not corrs['rolling'].empty:
   st.subheader(f"Rolling correlation with USD/CHF returns — window {window} days")
   rolling_df = corrs['rolling'].dropna(how='all')
   if rolling_df.empty:
       st.write("Not enough data for rolling correlations.")
   else:
       # Show interactive plot per series
       st.line_chart(rolling_df)
# Plot price series
st.subheader("Price / Macro series (time series)")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(merged.index, merged['USDCHF'], label='USD/CHF')
ax.set_ylabel("USD/CHF (price)")
ax.set_title("USD/CHF price")
ax.grid(True)
ax.legend()
st.pyplot(fig)
st.markdown("---")
st.caption("If any series failed to load, you'll see warnings above. Expand settings to change dates or add symbols.")
