# projects/alpha_run/data.py
from __future__ import annotations

from typing import Dict, Any, List, Tuple
import pandas as pd

from src.data.prices import get_prices
from src.data.sectors import get_sector_map


def load_universe(
    tickers: List[str],
    start: str,
    end: str,
    market_ticker: str,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str], List[str]]:
    """
    Fetch prices for the universe and market series, and return a sector map.
    Returns: prices_df, market_series, sector_map, available_tickers
    """
    prices = get_prices(tickers, start=start, end=end)
    # filter to available names
    available = [c for c in tickers if c in prices.columns]
    prices = prices[available]

    market = get_prices(market_ticker, start=start, end=end)[market_ticker]
    sector_map = get_sector_map(available)

    return prices, market, sector_map, available
