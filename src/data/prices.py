# src/data/prices.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import pandas as pd

try:
    import yfinance as yf
except Exception as e:
    yf = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None

# ----------------- tuning knobs -----------------
BATCH_SIZE = 20                    # how many tickers per yf.download call
RETRY_MAX = 3                      # retries for batch & single
SLEEP_BASE = 0.7                   # backoff base (seconds)
CACHE_DIR = Path(".cache/prices")  # on-disk cache for speed
# ------------------------------------------------


# --------------- helpers / cleaning ---------------

def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df is ... or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    idx = pd.to_datetime(df.index, utc=True, errors="coerce").tz_localize(None)
    df = df.copy()
    df.index = idx
    # de-duplicate dates (first wins)
    df = df.loc[~df.index.duplicated(keep="first")]
    return df.sort_index()

def _clean_prices(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize a yfinance/stooq DataFrame to single Close column named <ticker>."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame(columns=[ticker], dtype=float)

    # yfinance multi/single variants
    if isinstance(df.columns, pd.MultiIndex):
        # expect (field) or (ticker, field). Try both
        if ticker in df.columns.get_level_values(0):
            sub = df[ticker]
            col = "Close" if "Close" in sub.columns else sub.columns[-1]
            px = sub[col].rename(ticker).to_frame()
        else:
            # sometimes first level is field
            col = ("Close", ticker) if ("Close", ticker) in df.columns else df.columns[-1]
            px = df[col].rename(ticker).to_frame()
    else:
        if "Close" in df.columns:
            px = df["Close"].rename(ticker).to_frame()
        else:
            s = df.squeeze()
            px = (s if isinstance(s, pd.Series) else df.iloc[:, -1]).rename(ticker).to_frame()

    px = _ensure_dt_index(px).dropna()
    return px


# ------------------ fallbacks (single) ------------------

def _dl_one_yf_download(ticker: str, start: str, end: str,
                        max_tries: int = RETRY_MAX, sleep_base: float = SLEEP_BASE) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError(f"yfinance import failed: {_IMPORT_ERR!r}")

    last_exc = None
    for k in range(max_tries):
        try:
            df = yf.download(
                ticker, start=start, end=end,
                auto_adjust=True, progress=False, threads=False,
            )
            px = _clean_prices(df, ticker)
            if not px.empty:
                return px
        except Exception as e:
            last_exc = e
        if k < max_tries - 1:
            time.sleep(sleep_base * (1.5 ** k))
    if last_exc is not None:
        raise last_exc
    return pd.DataFrame(columns=[ticker], dtype=float)

def _dl_one_yf_history(ticker: str, start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError(f"yfinance import failed: {_IMPORT_ERR!r}")
    tkr = yf.Ticker(ticker)
    df = tkr.history(start=start, end=end, auto_adjust=True, actions=False, interval="1d")
    return _clean_prices(df, ticker)

def _dl_one_yf_history_max(ticker: str, start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError(f"yfinance import failed: {_IMPORT_ERR!r}")
    tkr = yf.Ticker(ticker)
    df = tkr.history(period="max", auto_adjust=True, actions=False, interval="1d")
    px = _clean_prices(df, ticker)
    if not px.empty:
        s, e = pd.to_datetime(start), pd.to_datetime(end)
        px = px.loc[(px.index >= s) & (px.index <= e)]
    return px

def _dl_one_stooq(ticker: str, start: str, end: str) -> pd.DataFrame:
    sym = ticker.lower()
    if "." not in sym:
        sym = f"{sym}.us"
    url = f"https://stooq.com/q/d/l/?s={sym}&i=d"
    df = pd.read_csv(url, parse_dates=["Date"])
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame(columns=[ticker], dtype=float)
    df = df.rename(columns={"Date": "date", "Close": ticker})
    px = df[["date", ticker]].set_index("date").sort_index().dropna()
    px.index = pd.to_datetime(px.index, utc=True).tz_localize(None)
    s, e = pd.to_datetime(start), pd.to_datetime(end)
    return px.loc[(px.index >= s) & (px.index <= e)]

def _dl_one(ticker: str, start: str, end: str) -> pd.DataFrame:
    last_error = None
    try:
        px = _dl_one_yf_download(ticker, start, end);  return px if not px.empty else px
    except KeyboardInterrupt:  raise
    except Exception as e:      last_error = e

    try:
        px = _dl_one_yf_history(ticker, start, end);   return px if not px.empty else px
    except KeyboardInterrupt:  raise
    except Exception as e:      last_error = e

    try:
        px = _dl_one_yf_history_max(ticker, start, end);  return px if not px.empty else px
    except KeyboardInterrupt:  raise
    except Exception as e:      last_error = e

    try:
        px = _dl_one_stooq(ticker, start, end);        return px if not px.empty else px
    except KeyboardInterrupt:  raise
    except Exception as e:      last_error = e

    print(f"[WARN] failed to download {ticker} after fallbacks"
          + (f" (last error: {last_error!r})" if last_error else ""))
    return pd.DataFrame(columns=[ticker], dtype=float)


# ------------------ batch download ------------------

def _batch_download(tickers: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """
    Try to fetch a batch via yf.download (multi-ticker).
    Returns dict[ticker] -> Close price DataFrame (may be empty if missing).
    """
    out: Dict[str, pd.DataFrame] = {t: pd.DataFrame(columns=[t], dtype=float) for t in tickers}
    if yf is None or not tickers:
        return out

    last_exc = None
    for k in range(RETRY_MAX):
        try:
            df = yf.download(
                tickers, start=start, end=end, auto_adjust=True,
                progress=False, threads=True, group_by="ticker",
            )
            if df is None or df.empty:
                break
            # If a single ticker, yfinance may return single-level columns
            if not isinstance(df.columns, pd.MultiIndex):
                # treat as single; figure out the only symbol
                t = tickers[0]
                out[t] = _clean_prices(df, t)
                break

            # MultiIndex expected: level 0 ticker, level 1 field
            for t in tickers:
                if t in df.columns.get_level_values(0):
                    sub = df[t]
                    col = "Close" if "Close" in sub.columns else sub.columns[-1]
                    out[t] = _clean_prices(sub[[col]], t)
            break
        except Exception as e:
            last_exc = e
            if k < RETRY_MAX - 1:
                time.sleep(SLEEP_BASE * (1.5 ** k))
    if last_exc and all(out[t].empty for t in tickers):
        # all failed; let caller fall back per-ticker
        pass
    return out


# ------------------ simple parquet cache ------------------

def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.upper()}.parquet"

def _cache_load(ticker: str) -> pd.DataFrame:
    p = _cache_path(ticker)
    if not p.exists():
        return pd.DataFrame(columns=[ticker], dtype=float)
    try:
        df = pd.read_parquet(p)
        return _ensure_dt_index(df[[ticker]])
    except Exception:
        return pd.DataFrame(columns=[ticker], dtype=float)

def _cache_save(ticker: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    p = _cache_path(ticker)
    try:
        df = _ensure_dt_index(df[[ticker]])
        df.to_parquet(p)
    except Exception:
        pass

def _merge_cache(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if old is None or old.empty:
        return _ensure_dt_index(new)
    if new is None or new.empty:
        return _ensure_dt_index(old)
    both = pd.concat([old, new], axis=0)
    both = _ensure_dt_index(both)
    return both.loc[~both.index.duplicated(keep="last")].sort_index()


# ------------------ public API ------------------

def get_prices(tickers: Union[str, Iterable[str]], start: str, end: str, *, use_cache: bool = True) -> pd.DataFrame:
    """
    Robust, fast price downloader with caching.
    - Accepts a single ticker or iterable of tickers
    - Tries batch yfinance first (chunks of BATCH_SIZE), then per-ticker fallback
    - Uses a per-ticker Parquet cache on disk; later calls only fetch missing dates
    Returns: DataFrame (date x tickers) of Close prices (adjusted where available)
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers: List[str] = list(dict.fromkeys(tickers))  # unique, preserve order

    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)

    results: Dict[str, pd.DataFrame] = {}
    need_fetch: List[str] = []

    # 1) load cache slices
    if use_cache:
        for t in tickers:
            cached = _cache_load(t)
            if not cached.empty:
                # If cache fully covers window, slice and done; else mark for top-up
                have_start, have_end = cached.index.min(), cached.index.max()
                if have_start <= start_ts and have_end >= end_ts:
                    results[t] = cached.loc[(cached.index >= start_ts) & (cached.index <= end_ts)]
                else:
                    # keep cached piece; we will merge later
                    results[t] = cached
                    need_fetch.append(t)
            else:
                need_fetch.append(t)
    else:
        need_fetch = tickers[:]

    # 2) figure missing ranges per ticker
    def _missing_window(t: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        if t not in results or results[t].empty:
            return start_ts, end_ts
        have_start, have_end = results[t].index.min(), results[t].index.max()
        miss_start = min(start_ts, have_start) if start_ts < have_start else have_end + pd.Timedelta(days=1)
        miss_end = end_ts if end_ts > have_end else have_end
        return miss_start, miss_end

    # 3) batch download for those needing fetch
    to_fetch = need_fetch[:]
    for i in range(0, len(to_fetch), BATCH_SIZE):
        batch = to_fetch[i:i + BATCH_SIZE]
        # For each ticker in batch, compute the missing window we want to hit with yf.download
        # Use the broadest window among them to keep a single request.
        miss_windows = [_missing_window(t) for t in batch]
        miss_start = min(w[0] for w in miss_windows)
        miss_end = max(w[1] for w in miss_windows)

        fetched = _batch_download(batch, start=str(miss_start.date()), end=str(miss_end.date()))
        # Merge into results; track batch misses to single-fallback
        single_fallback: List[str] = []
        for t in batch:
            px = fetched.get(t, pd.DataFrame())
            if px is None or px.empty:
                single_fallback.append(t)
                continue
            if t in results and not results[t].empty:
                results[t] = _merge_cache(results[t], px)
            else:
                results[t] = px

        # per-ticker fallbacks for failures
        for t in single_fallback:
            miss_start, miss_end = _missing_window(t)
            if miss_start > miss_end:
                continue
            px = _dl_one(t, start=str(miss_start.date()), end=str(miss_end.date()))
            if t in results and not results[t].empty:
                results[t] = _merge_cache(results[t], px)
            else:
                results[t] = px

        # small breather to be gentle with API
        time.sleep(0.2)

    # 4) final clip to requested window + save cache
    frames: List[pd.DataFrame] = []
    failed: List[str] = []
    for t in tickers:
        df = results.get(t, pd.DataFrame(columns=[t], dtype=float))
        df = _ensure_dt_index(df)
        df = df.loc[(df.index >= start_ts) & (df.index <= end_ts)]
        if df.empty:
            failed.append(t)
        else:
            frames.append(df)
            if use_cache:
                # Save full (merged) history for future runs
                _cache_save(t, results[t])

    prices = pd.concat(frames, axis=1).sort_index() if frames else pd.DataFrame()
    prices = prices.loc[~prices.index.duplicated(keep="first")]
    prices = prices.dropna(how="all")

    if failed:
        print(f"\n{len(failed)} Failed downloads (after all fallbacks):")
        for i in range(0, len(failed), 3):
            print(f"{failed[i:i+3]}")

    return prices
