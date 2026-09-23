"""Point-in-time FRED / OECD government yield + curve-slope loaders for FX research.

Long-term government bond yields (OECD ``IRLTLT01*``) and short rates
(``IRSTCI01*`` via ``fred_rates``) form a simple term-structure panel:

    slope_c = LT_c − ST_c
    slope_diff_vs_usd = slope_c − slope_USD

Literature framing (not a claim our Yahoo sample matches paper tables):
Chen & Tsang (2013) relative yield-curve factors; Ang–Chen style slope/carry;
Lustig–Stathopoulos–Verdelhan term structure of currency risk premia; Fama UIP.

Publication delay: monthly OECD series use ``pub_lag_months`` (default 1).
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import (
    FRED_IMMEDIATE_MONTHLY,
    apply_publication_lag,
    download_fred_series,
    load_currency_rates,
    load_fred_series,
    macro_dir,
)

# OECD long-term government bond yields (~10y), % p.a.
# EUR: prefer EZ aggregate; DE bund kept as documented fallback.
FRED_LT_GOVT_MONTHLY: dict[str, str] = {
    "USD": "IRLTLT01USM156N",
    "EUR": "IRLTLT01EZM156N",
    "GBP": "IRLTLT01GBM156N",
    "JPY": "IRLTLT01JPM156N",
    "AUD": "IRLTLT01AUM156N",
    "CAD": "IRLTLT01CAM156N",
    "CHF": "IRLTLT01CHM156N",
    "NZD": "IRLTLT01NZM156N",
}

FRED_LT_GOVT_FALLBACK: dict[str, str] = {
    "EUR": "IRLTLT01DEM156N",  # German bund if EZ sparse
}

# Optional OECD 3-month interest rates (money-market) for UIP secondary panel
FRED_IR3M_MONTHLY: dict[str, str] = {
    "USD": "IR3TIB01USM156N",
    "EUR": "IR3TIB01EZM156N",
    "GBP": "IR3TIB01GBM156N",
    "JPY": "IR3TIB01JPM156N",
    "AUD": "IR3TIB01AUM156N",
    "CAD": "IR3TIB01CAM156N",
    "CHF": "IR3TIB01CHM156N",
    "NZD": "IR3TIB01NZM156N",
}


def _to_month_start_utc(s: pd.Series) -> pd.Series:
    out = s.copy()
    idx = out.index.tz_convert("UTC").tz_localize(None) if out.index.tz is not None else out.index
    out.index = pd.DatetimeIndex(idx.to_period("M").to_timestamp(how="start"), tz="UTC")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out


def load_lt_yield_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = 1,
    download: bool = True,
    use_eur_fallback: bool = True,
) -> pd.DataFrame:
    """Panel of OECD long-term government bond yields (% p.a.), publication-lagged."""
    curs = [c.upper() for c in (currencies or FRED_LT_GOVT_MONTHLY.keys())]
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    missing: list[str] = []
    for c in curs:
        sid = FRED_LT_GOVT_MONTHLY.get(c)
        if not sid:
            missing.append(c)
            continue
        try:
            s = load_fred_series(sid, download=download)
            s = _to_month_start_utc(s)
            # EUR EZ series can lag; backfill gaps with DE bund if requested
            if c == "EUR" and use_eur_fallback:
                fb = FRED_LT_GOVT_FALLBACK.get("EUR")
                if fb:
                    try:
                        s_fb = _to_month_start_utc(load_fred_series(fb, download=download))
                        before = int(s.isna().sum())
                        s = s.combine_first(s_fb)
                        notes[c] = f"EZ primary; DE bund fill gaps (n_na_before≈{before})"
                    except Exception as exc:  # noqa: BLE001
                        notes[c] = f"EZ only; DE fallback failed: {exc}"[:120]
            s = apply_publication_lag(s, lag_months=pub_lag_months)
            cols[c] = s
        except Exception as exc:  # noqa: BLE001
            missing.append(c)
            notes[c] = f"load_fail: {exc}"[:120]
    if not cols:
        raise RuntimeError(f"No LT yield series loaded; missing={missing}")
    df = pd.DataFrame(cols).sort_index()
    df.attrs["source"] = "fred_oecd_lt_govt_monthly"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["units"] = "percent_per_annum"
    df.attrs["series_map"] = {c: FRED_LT_GOVT_MONTHLY.get(c) for c in cols}
    df.attrs["missing"] = missing
    df.attrs["notes"] = notes
    return df


def load_ir3m_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = 1,
    download: bool = True,
) -> pd.DataFrame:
    """OECD 3-month interest-rate panel (optional UIP secondary)."""
    curs = [c.upper() for c in (currencies or FRED_IR3M_MONTHLY.keys())]
    cols: dict[str, pd.Series] = {}
    missing: list[str] = []
    for c in curs:
        sid = FRED_IR3M_MONTHLY.get(c)
        if not sid:
            missing.append(c)
            continue
        try:
            s = apply_publication_lag(
                _to_month_start_utc(load_fred_series(sid, download=download)),
                lag_months=pub_lag_months,
            )
            cols[c] = s
        except Exception as exc:  # noqa: BLE001
            missing.append(c)
            continue
    if not cols:
        raise RuntimeError(f"No IR3M series loaded; missing={missing}")
    df = pd.DataFrame(cols).sort_index()
    df.attrs["source"] = "fred_oecd_ir3m_monthly"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["missing"] = missing
    return df


def load_curve_panels(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = 1,
    download: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load LT, short (immediate), slope, and differentials vs USD.

    Returns keys: ``lt``, ``st``, ``slope``, ``lt_diff``, ``st_diff``, ``slope_diff``.
    Short rates reuse ``load_currency_rates`` (IRSTCI01*).
    """
    curs = [c.upper() for c in (currencies or FRED_LT_GOVT_MONTHLY.keys())]
    # Ensure USD present for differentials
    if "USD" not in curs:
        curs = ["USD"] + curs

    lt = load_lt_yield_panel(curs, pub_lag_months=pub_lag_months, download=download)
    st = load_currency_rates(
        [c for c in curs if c in FRED_IMMEDIATE_MONTHLY or c == "USD"],
        freq="M",
        pub_lag_months=pub_lag_months,
        download=download,
    )
    # Align columns to intersection
    common = sorted(set(lt.columns) & set(st.columns))
    if "USD" not in common:
        raise RuntimeError("USD missing from LT or ST panel — cannot form differentials")
    lt = lt[common]
    st = st.reindex(lt.index).ffill()
    st = st[common]
    # Reindex LT to union of ST index for overlapping history
    idx = lt.index.intersection(st.dropna(how="all").index)
    lt = lt.reindex(idx)
    st = st.reindex(idx)

    slope = lt - st
    slope.attrs["definition"] = "lt_minus_st"
    slope.attrs["pub_lag_months"] = int(pub_lag_months)

    def _diff(panel: pd.DataFrame) -> pd.DataFrame:
        usd = panel["USD"]
        out = panel.drop(columns=["USD"]).sub(usd, axis=0)
        out.attrs["definition"] = "ccy_minus_usd"
        return out

    lt_diff = _diff(lt)
    st_diff = _diff(st)
    slope_diff = _diff(slope)

    return {
        "lt": lt,
        "st": st,
        "slope": slope,
        "lt_diff": lt_diff,
        "st_diff": st_diff,
        "slope_diff": slope_diff,
    }


def yield_coverage_table(
    *,
    download: bool = False,
    asof: str | None = None,
) -> pd.DataFrame:
    """Document LT / ST coverage and staleness for G10 curve panel."""
    asof_ts = (pd.Timestamp(asof, tz="UTC") if asof else pd.Timestamp.now("UTC")).tz_convert(None)
    rows = []
    for ccy, sid in FRED_LT_GOVT_MONTHLY.items():
        row = {
            "currency": ccy,
            "lt_series": sid,
            "st_series": FRED_IMMEDIATE_MONTHLY.get(ccy),
            "ir3m_series": FRED_IR3M_MONTHLY.get(ccy),
        }
        try:
            s = load_fred_series(sid, download=download)
            end = s.index.max()
            end_naive = end.tz_convert(None) if getattr(end, "tz", None) else end
            months_lag = (asof_ts.to_period("M") - pd.Timestamp(end_naive).to_period("M")).n
            row.update(
                {
                    "lt_n": int(s.notna().sum()),
                    "lt_start": str(s.index.min().date()),
                    "lt_end": str(s.index.max().date()),
                    "lt_months_since_end": int(months_lag),
                    "lt_stale": bool(months_lag > 4),
                    "lt_status": "ok",
                }
            )
        except Exception as exc:  # noqa: BLE001
            row.update(
                {
                    "lt_n": 0,
                    "lt_start": None,
                    "lt_end": None,
                    "lt_months_since_end": None,
                    "lt_stale": True,
                    "lt_status": f"fail:{str(exc)[:60]}",
                }
            )
        st_sid = FRED_IMMEDIATE_MONTHLY.get(ccy)
        if st_sid:
            try:
                s2 = load_fred_series(st_sid, download=download)
                end2 = s2.index.max()
                end2n = end2.tz_convert(None) if getattr(end2, "tz", None) else end2
                m2 = (asof_ts.to_period("M") - pd.Timestamp(end2n).to_period("M")).n
                row.update(
                    {
                        "st_n": int(s2.notna().sum()),
                        "st_end": str(s2.index.max().date()),
                        "st_months_since_end": int(m2),
                        "st_stale": bool(m2 > 4),
                    }
                )
            except Exception:  # noqa: BLE001
                row.update({"st_n": 0, "st_end": None, "st_months_since_end": None, "st_stale": True})
        else:
            row.update({"st_n": 0, "st_end": None, "st_months_since_end": None, "st_stale": True})
        note = ""
        if ccy == "EUR":
            note = "EZ LT; DE bund used as gap-fill fallback"
        row["note"] = note
        rows.append(row)
    return pd.DataFrame(rows).sort_values("currency").reset_index(drop=True)


def download_default_yield_panel(*, force: bool = False) -> dict[str, object]:
    """Ensure LT (+ optional IR3M) FRED CSVs exist under ``data/macro/``."""
    paths = {}
    for sid in set(FRED_LT_GOVT_MONTHLY.values()) | set(FRED_LT_GOVT_FALLBACK.values()) | set(
        FRED_IR3M_MONTHLY.values()
    ):
        paths[sid] = download_fred_series(sid, force=force)
    # touch macro_dir for side effect
    _ = macro_dir()
    return paths
