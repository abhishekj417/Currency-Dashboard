# USD/CHF 30-Year Macro Dashboard (Stable Version)

This dashboard visualizes USD/CHF exchange rate trends and their relationship with global macro variables for the past 30 years.

## 🧩 Data Sources
- **FRED API** — Interest rates, CPI, GDP, Exports
- **Yahoo Finance** — USD/CHF FX rate, Gold, Brent Oil, S&P 500, MSCI World

## ⚙️ How to Deploy
1. Create a [FRED API Key](https://fred.stlouisfed.org/docs/api/api_key.html)
2. Create a new GitHub repository (e.g., `usdchf-dashboard`)
3. Upload these three files:
   - `streamlit_app.py`
   - `requirements.txt`
   - `README.md`
4. Go to [Streamlit Cloud](https://streamlit.io/cloud)
   - Sign in with GitHub
   - Click “New App” → select your repo → choose `streamlit_app.py`
5. Add your FRED API Key in Streamlit:
   - Manage app → Edit Secrets → paste:
     ```
     FRED_API_KEY = your_api_key_here
     ```
6. Click “Deploy” — your app will go live automatically.

## 💰 Cost
| Tool | Purpose | Cost |
|------|----------|------|
| GitHub | Hosting files | Free |
| Streamlit Cloud | Dashboard hosting | Free |
| FRED API | Data | Free |
| yFinance | Market data | Free |

