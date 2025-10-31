# streamlit_app.py
import streamlit as st
import pandas as pd
import datetime
import yfinance as yf
from pandas_datareader import data as pdr
import matplotlib.pyplot as plt

# ---------------------------------------------
# Streamlit Page Setup
# ---------------------------------------------
st.set_page_config(
   page_title="USD/CHF Macro Correlation Dashboard",
   layout="wide"
)

st.title("💹 USD/CHF Correlation Dashboard")
st.markdown("Analyze the relationship between USD/CHF and key macroeconomic indicators.")

# ---------------------------------------------
# Helper functions
# ---------------------------------------------

@st.cache_data(ttl=3600)
def get_fx_data():
   """
   Fetch USD/CHF monthly data from Yahoo Finance.
   Uses CHF=X ticker (USD/CHF pair).
   """
   try:
       fx = yf.download("CHF=X", start="1995-01-01", interval="1mo", progress=False)
       if fx.empty:
           st.error("No FX data returned from Yahoo Finance.")
           return pd.DataFrame()
       fx = fx['Close']
       fx.name = "USDCHF"
       fx.index = fx.index.to_period('M').to_timestamp()
       return fx
   except Exception as e:
       st.error(f"Error fetching FX data: {e}")
       return pd.DataFrame()

@st.cache_data(ttl=3600)
def get_fred_data(symbol, start, end):
   """
   Fetch macroeconomic data from FRED using pandas_datareader.
   """
   try:
       df = pdr.DataReader(symbol, "fred", start, end)
       if df.empty:
           st.warning(f"No data found for {symbol}")
           return pd.DataFrame()
       df.index = pd.to_datetime(df.index)
       df.columns = [symbol]
       return df
   except Exception as e:
       st.warning(f"Failed to fetch {symbol} from FRED: {e}")
       return pd.DataFrame()

def merge_data(fx_series, fred_dict):
   """
   Combine FX and FRED macro data into one DataFrame.
   """
   if fx_series.empty:
       return pd.DataFrame()

   df = fx_series.to_frame()
   for symbol, macro_df in fred_dict.items():
       if not macro_df.empty:
           df = df.join(macro_df, how="left")
   df = df.ffill().dropna()
   return df

def compute_correlation(df):
   """
   Compute static correlation matrix between USDCHF and macro variables.
   """
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
   default_start = today - datetime.timedelta(days=5*365)
   start_date = st.date_input("Start Date", default_start)
   end_date = st.date_input("End Date", today)

   st.subheader("Select Macro Indicators (FRED Symbols)")
   macros = {
       "US CPI": "CPIAUCSL",
       "Unemployment Rate": "UNRATE",
       "10Y Treasury Yield": "DGS10",
       "Fed Funds Rate": "FEDFUNDS"
   }

   selected_macros = [symbol for label, symbol in macros.items()
                      if st.checkbox(f"{label} ({symbol})", value=True)]

   custom_symbol = st.text_input("Add Custom FRED Symbol (optional)").strip().upper()
   if custom_symbol:
       selected_macros.append(custom_symbol)

# ---------------------------------------------
# Data Fetching
# ---------------------------------------------
st.info("Fetching data...")

fx_data = get_fx_data()
if fx_data.empty:
   st.stop()

fred_data = {}
for symbol in selected_macros:
   fred_data[symbol] = get_fred_data(symbol, start_date, end_date)

merged = merge_data(fx_data, fred_data)

if merged.empty:
   st.error("No merged data available to display.")
   st.stop()

# ---------------------------------------------
# Display Data
# ---------------------------------------------
st.subheader("📊 Data Preview")
st.dataframe(merged.tail(10))

corr_matrix = compute_correlation(merged)
if not corr_matrix.empty:
   st.subheader("📈 Correlation Matrix (Daily Returns)")
   st.dataframe(corr_matrix.style.background_gradient(cmap="RdBu_r", axis=None))
else:
   st.warning("Not enough data to compute correlation matrix.")

# ---------------------------------------------
# Plot USD/CHF and Macro Series
# ---------------------------------------------
st.subheader("📉 USD/CHF vs Macroeconomic Indicators")

fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(merged.index, merged["USDCHF"], label="USD/CHF", color="blue")
ax1.set_ylabel("USD/CHF", color="blue")
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)

ax2 = ax1.twinx()
for symbol in selected_macros:
   if symbol in merged.columns:
       ax2.plot(merged.index, merged[symbol], label=symbol)
ax2.set_ylabel("Macro Indicators", color="gray")
ax2.tick_params(axis='y', labelcolor='gray')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc="upper left")

st.pyplot(fig)

st.caption("Data sources: Yahoo Finance (USD/CHF) and FRED (Macroeconomic Indicators).")
