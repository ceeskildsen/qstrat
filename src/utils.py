# src/utils.py
import pandas as pd

def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Percentage returns with no forward/backward fill (avoids FutureWarning in pandas).
    Rows with any NaN are dropped to keep downstream math stable.
    """
    return prices.pct_change(fill_method=None).dropna()


def trade_pnl(entry_price: float, exit_price: float, notional: float = 1.0, side: str = "long") -> float:
    """
    Simple PnL for one trade:
      - long: profit if price rises
      - short: profit if price falls
    """
    r = (exit_price - entry_price) / entry_price
    if side == "long":
        return notional * r
    elif side == "short":
        return notional * (-r)
    else:
        raise ValueError("side must be 'long' or 'short'")

# src/utils.py (append)

import pandas as pd

def daily_to_period_rf(daily_rf: pd.Series, period_end_index: pd.DatetimeIndex) -> pd.Series:
    """
    Convert daily risk-free returns (e.g., BIL daily returns) into per-period
    returns that align to your backtest's rebalance dates (period_end_index).

    It compounds the daily RF between consecutive period endpoints using a
    cumulative-product trick.
    """
    daily_rf = daily_rf.sort_index().dropna()
    if daily_rf.empty or len(period_end_index) == 0:
        return pd.Series(index=period_end_index, dtype=float)

    cum = (1.0 + daily_rf).cumprod()
    aligned = cum.reindex(period_end_index.union(cum.index)).ffill().reindex(period_end_index)
    per = aligned.pct_change().fillna(0.0)
    per.name = "risk_free_periodic"
    return per
