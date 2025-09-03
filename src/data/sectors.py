import pandas as pd
from typing import Iterable

# Simple GICS-like buckets (close enough for neutrality / attribution)
_SECTOR_MAP = {
    # Info Tech
    "AAPL":"Information Technology","MSFT":"Information Technology","NVDA":"Information Technology",
    "AVGO":"Information Technology","AMD":"Information Technology","QCOM":"Information Technology",
    "CSCO":"Information Technology","ORCL":"Information Technology","ACN":"Information Technology",
    "ADBE":"Information Technology","CRM":"Information Technology","TXN":"Information Technology",
    "IBM":"Information Technology","INTC":"Information Technology","AMAT":"Information Technology",

    # Communication Services
    "GOOGL":"Communication Services","META":"Communication Services","NFLX":"Communication Services",
    "CMCSA":"Communication Services","VZ":"Communication Services",

    # Consumer Discretionary
    "AMZN":"Consumer Discretionary","TSLA":"Consumer Discretionary","HD":"Consumer Discretionary",
    "NKE":"Consumer Discretionary","MCD":"Consumer Discretionary","SBUX":"Consumer Discretionary",
    "BKNG":"Consumer Discretionary","LOW":"Consumer Discretionary",

    # Consumer Staples
    "KO":"Consumer Staples","PEP":"Consumer Staples","COST":"Consumer Staples",
    "PM":"Consumer Staples","WMT":"Consumer Staples",

    # Health Care
    "UNH":"Health Care","MRK":"Health Care","ABBV":"Health Care","PFE":"Health Care",
    "TMO":"Health Care","ABT":"Health Care","DHR":"Health Care","GILD":"Health Care",

    # Energy
    "CVX":"Energy","XOM":"Energy",

    # Financials (treat V/MA as Financials)
    "JPM":"Financials","V":"Financials","MA":"Financials",

    # Industrials
    "UPS":"Industrials","CAT":"Industrials","HON":"Industrials",

    # Materials
    "LIN":"Materials",
}

def get_sector_map(tickers: Iterable[str]) -> pd.Series:
    """Return a Series index=tickers, values=sector (unknown -> 'Other')."""
    s = pd.Series({t: _SECTOR_MAP.get(t, "Other") for t in tickers})
    s.index.name = "Ticker"
    s.name = "Sector"
    return s
