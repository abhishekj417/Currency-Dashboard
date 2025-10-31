import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
import plotly.express as px
import os

# ---------------------------
# PAGE SETUP
# ---------------------------
st.set_page_config(page_title="USD/CHF Macro Dashboard", layout="wide")
st.title("📊 USD/CHF Exchange Rate - 30 Year Macro Dashboard")

# ---------------------------
# FRED API Setup
# ---------------------------
fred_key = os.getenv("FRED_API_KEY")
fred = Fred(api_key=fred_key) if fred_key else None

# ---------------------------
# Data Fetch Functions
# ---------------------------

@st.cache_data
def get_fx_data():
    fx = yf.download("CHF=X", start="1995-01-01", interval="1mo")
    fx = fx['Close'].rename("USDCHF")
    fx.index = fx.index.to_period('M').to_timestamp()
    return fx

@st.cache_data
def get_yfinance_data(tickers):
    data = yf.download(tickers, start="1995-01-01", interval="1mo")['Close']
    data.index = data.index.to_period('M').to_timestamp()
    return data

@st.cache_data
def get_fred_data(series_id, label):
    if fred is None:
        return pd.Series(dtype=float, name=label)
    data = fred.get_series(series_id)
    df = pd.DataFrame(data, columns=[label])
    df.index = pd.to_datetime(df.index)
    df = df.resample('M').last()
    return df[label]

# ---------------------------
# Fetch all datasets
# ---------------------------

fx = get_fx_data()
ydata = get_yfinance_data(['GC=F', 'BZ=F', '^GSPC', 'URTH'])  # Gold, Brent, S&P500, MSCI World

# FRED-based data (if key available)
rates_us = get_fred_data('FEDFUNDS', 'US_Interest_Rate')
rates_ch = get_fred_data('IR3TIB01CHM156N', 'CH_Interest_Rate')
cpi_us = get_fred_data('CPIAUCSL', 'US_CPI')
cpi_ch = get_fred_data('CPHPTT01CHM659N', 'CH_CPI')
gdp_ch = get_fred_data('NAEXKP01CHQ657S', 'CH_GDP')
exports_ch = get_fred_data('XTEXVA01CHM667S', 'CH_Exports')

# Merge all into one dataframe
df = pd.concat([fx, ydata, rates_us, rates_ch, cpi_us, cpi_ch, gdp_ch, exports_ch], axis=1)
df = df.sort_index().fillna(method='ffill')

# ---------------------------
# Dashboard Visuals
# ---------------------------

tab1, tab2, tab3 = st.tabs(["📈 Overview", "📊 Correlations", "📉 Relationships"])

with tab1:
    st.subheader("USD/CHF Exchange Rate Over Time")
    fig = px.line(df, x=df.index, y='USDCHF', title="USD/CHF Exchange Rate (Monthly)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Macro Variables Overview")
    st.line_chart(df[['US_Interest_Rate', 'CH_Interest_Rate', 'US_CPI', 'CH_CPI', 'GC=F', 'BZ=F']])

with tab2:
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix", aspect="auto")
    st.plotly_chart(fig_corr, use_container_width=True)

with tab3:
    st.subheader("USD/CHF vs Key Variables")
    chosen = st.selectbox("Select Variable:", df.columns.drop('USDCHF'))
    fig_scatter = px.scatter(df, x=chosen, y='USDCHF', trendline='ols',
                             title=f"USD/CHF vs {chosen}")
    st.plotly_chart(fig_scatter, use_container_width=True)

st.download_button("Download Data as CSV", df.to_csv().encode('utf-8'), "usdchf_data.csv", "text/csv")
st.success("Dashboard loaded successfully! Data updates automatically each time you visit.")
