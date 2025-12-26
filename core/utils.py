import re
import numpy as np
import pandas as pd

def find_col(df, candidates):
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        for c in df.columns:
            if cand.lower() == c.lower():
                return c
        for low, orig in cols_lower.items():
            if cand.lower() in low:
                return orig
    return None

def make_unique(names):
    seen = {}
    out = []
    for n in names:
        if n in seen:
            seen[n] += 1
            out.append(f"{n}__{seen[n]}")
        else:
            seen[n] = 0
            out.append(n)
    return out

def _clean_header_token(x):
    if x is None:
        return ""
    s = str(x).strip()
    if re.match(r"(?i)^unnamed.*", s) or s.lower() in {"nan", "none"}:
        return ""
    return s

def fuse_top_two_rows_as_header(df_raw):
    if df_raw.shape[0] < 2:
        return df_raw
    top = df_raw.iloc[0].tolist()
    sec = df_raw.iloc[1].tolist()
    headers = []
    for a, b in zip(top, sec):
        a_clean = _clean_header_token(a)
        b_clean = _clean_header_token(b)
        if a_clean and b_clean:
            h = f"{a_clean} {b_clean}"
        elif a_clean:
            h = a_clean
        elif b_clean:
            h = b_clean
        else:
            h = "col"
        headers.append(h.strip())
    headers = [re.sub(r"\s+", " ", h) for h in headers]
    headers = make_unique(headers)
    df = df_raw.iloc[2:].copy()
    df.columns = headers
    return df

def clean_numeric(df, cols):
    out = df.copy()
    if not cols:
        return out
    out[cols] = (
        out[cols]
        .replace('-', np.nan)
        .astype(str)
        .replace(r'\*', '', regex=True)
        .replace('', np.nan)
        .apply(pd.to_numeric, errors='coerce')
    )
    return out

def km_to_cm(series):
    return pd.to_numeric(series, errors='coerce') * 100_000.0

def ensure_sorted_unique_by_km(df, km_col):
    out = df.sort_values(km_col, kind="mergesort")
    out = out.drop_duplicates(subset=[km_col], keep="first")
    return out
