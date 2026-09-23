"""FRED / OECD MEI industrial production (PRINTO01 / PRMNTO01) panels for IP-growth FX.

Literature
----------
- Real-activity / growth differentials → FX (Molodtsova–Papell / growth-channel
  style; Dahlquist–Hasseltoft macro–FX framing applied to *coincident* IP).
- Primary prior: currencies with *high* relative IP YoY growth appreciate vs
  low-growth peers. Honesty alternate: long low IP / activity-stress debtor.
- Distinct from: macro_diff EW CPI/IP/UR blend, OECD CLI/BCI/CCI, employment
  §41, WUI, EPU/TPU, CA/TB, fiscal/debt, BIS, reserves, house-price, money,
  equity, IG OAS, commodity.

Free data (verified live on FRED ~2026-09-23)
--------------------------------------------
Prefer OECD MEI **industry excl. construction YoY growth** via FRED
``{ISO3}PRINTO01GYSAM`` (monthly, seasonally adjusted, growth rate same period
previous year). Where industry PRINTO01 is 404 (AUD/NZD/CHF), fall back to
**manufacturing** ``{ISO3}PRMNTO01GYSAQ`` (quarterly YoY). EUR: France
``FRAPRINTO01GYSAM`` proxy (``EA19PRINTO01GYSAM`` ends 2023-10; ``DEUPRINTO01GYSAM``
ends 2023-12 — both stale).

| Ccy | FRED id (primary)   | Freq | Notes                                              |
|-----|---------------------|------|----------------------------------------------------|
| USD | USAPRINTO01GYSAM    | M    | industry YoY; through ~2026-07                     |
| EUR | FRAPRINTO01GYSAM    | M    | France proxy (EA19/DEU GYSAM stale)                |
| GBP | GBRPRINTO01GYSAM    | M    | through ~2026-06                                   |
| JPY | JPNPRINTO01GYSAM    | M    | through ~2026-06                                   |
| CAD | CANPRINTO01GYSAM    | M    | through ~2026-06                                   |
| AUD | AUSPRMNTO01GYSAQ    | Q    | manufacturing YoY (AUSPRINTO01GYSAM **404**)       |
| NZD | NZLPRMNTO01GYSAQ    | Q    | manufacturing YoY (NZLPRINTO01GYSAM **404**)       |
| CHF | CHEPRMNTO01GYSAQ    | Q    | manufacturing YoY (CHEPRINTO01GYSAM **404**)       |

Tried / documented:
- ``PRINTO01*M661S`` / ``PRINTO01USM661S`` style — mostly **404**.
- ``*PROINDMISMEI`` index levels — end ~**2024-02..04** (stale — not primary).
- ``PRINTO01*A661N`` annual levels — end ~**2023** (stale).
- ``PRINTO01*A657S`` annual YoY — some live to 2025-01 but annual-only (not primary).
- ``EA19PRINTO01GYSAM`` / ``EA19PRINTO01IXOBSAM`` — end **2023-10** (stale).
- ``DEUPRINTO01GYSAM`` — end **2023-12** (stale) → France EUR proxy.
- ``AUS/NZL/CHEPRINTO01GYSAM`` / ``IXOBSAM`` — **404**.
- ``USAPRMNTO01GYSAM`` etc. manufacturing monthly — live alt (not mixed into
  industry panel for cross-country consistency except AUD/NZD/CHF).
- US ``INDPRO`` / ``IPMAN`` — live national alts (not mixed into OECD panel).

Unit choice (fixed a priori)
----------------------------
Panel stores **YoY growth rates as reported** (GYSAM / GYSAQ, percent). Strategy
scores use these YoY values directly for XS ranks — not raw index levels.
Acceleration = Δ12 of that YoY growth (after quarterly→monthly ffill).

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Industrial production typically lands ~1–2 months after reference. Default
``pub_lag_months=2`` (picked once a priori — IP faster than labour's 3).
Observation dated month-/quarter-start ``t`` is first known at ``t + 2 months``.
Quarterly series are pub-lagged then forward-filled to month-start (no
intra-quarter leak). Strategy modules add a **1 trading-day** weight lag
(``signal_lag_days=1``); no extra month signal lag (a priori for §42).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED OECD MEI IP YoY growth (industry monthly; manufacturing quarterly for AUD/NZD/CHF)
IP_SERIES: dict[str, str | None] = {
    "USD": "USAPRINTO01GYSAM",
    "EUR": "FRAPRINTO01GYSAM",  # France proxy; EA19/DEU GYSAM stale
    "GBP": "GBRPRINTO01GYSAM",
    "JPY": "JPNPRINTO01GYSAM",
    "CAD": "CANPRINTO01GYSAM",
    "AUD": "AUSPRMNTO01GYSAQ",  # manufacturing Q; industry PRINTO01 404
    "NZD": "NZLPRMNTO01GYSAQ",
    "CHF": "CHEPRMNTO01GYSAQ",
}

# Native frequency tag for each primary id (M=monthly, Q=quarterly)
IP_FREQ: dict[str, str] = {
    "USD": "M",
    "EUR": "M",
    "GBP": "M",
    "JPY": "M",
    "CAD": "M",
    "AUD": "Q",
    "NZD": "Q",
    "CHF": "Q",
}

IP_FAILED_OR_ALT: dict[str, list[str]] = {
    "printo01_m661s_mostly_404": [
        "PRINTO01USM661S",
        "PRINTO01GBM661S",
        "PRINTO01JPM661S",
        "PRINTO01CAM661S",
        "PRINTO01AUM661S",
        "PRINTO01NZM661S",
        "PRINTO01CHM661S",
        "PRINTO01DEM661S",
        "PRINTO01EZM661S",
    ],
    "proindmismei_stale_~2024_03": [
        "USAPROINDMISMEI",
        "GBRPROINDMISMEI",
        "JPNPROINDMISMEI",
        "CANPROINDMISMEI",
        "DEUPROINDMISMEI",
        "FRAPROINDMISMEI",
        "EA19PRMNTO01IXOBSAM",
    ],
    "printo01_annual_levels_stale_2023": [
        "PRINTO01USA661N",
        "PRINTO01GBA661N",
        "PRINTO01JPA661N",
        "PRINTO01CAA661N",
        "PRINTO01AUA661N",
        "PRINTO01NZA661N",
        "PRINTO01CHA661N",
        "PRINTO01DEA661N",
        "PRINTO01EZA661N",
    ],
    "eur_ea19_gysam_stale_2023_10": ["EA19PRINTO01GYSAM", "EA19PRINTO01IXOBSAM"],
    "eur_deu_gysam_stale_2023_12": ["DEUPRINTO01GYSAM"],
    "eur_france_proxy_primary": ["FRAPRINTO01GYSAM"],
    "eur_ita_esp_nld_alts": [
        "ITAPRINTO01GYSAM",
        "ESPPRINTO01GYSAM",
        "NLDPRINTO01GYSAM",
    ],
    "aud_nzd_chf_printo01_gysam_404": [
        "AUSPRINTO01GYSAM",
        "NZLPRINTO01GYSAM",
        "CHEPRINTO01GYSAM",
        "AUSPRINTO01IXOBSAM",
        "NZLPRINTO01IXOBSAM",
        "CHEPRINTO01IXOBSAM",
    ],
    "aud_nzd_chf_manufacturing_q_primary": [
        "AUSPRMNTO01GYSAQ",
        "NZLPRMNTO01GYSAQ",
        "CHEPRMNTO01GYSAQ",
    ],
    "us_national_alts_not_primary": ["INDPRO", "IPMAN"],
    "manufacturing_monthly_alts_not_mixed": [
        "USAPRMNTO01GYSAM",
        "GBRPRMNTO01GYSAM",
        "JPNPRMNTO01GYSAM",
        "CANPRMNTO01GYSAM",
        "FRAPRMNTO01GYSAM",
    ],
}

EUR_ITALY = "ITAPRINTO01GYSAM"
EUR_SPAIN = "ESPPRINTO01GYSAM"
EUR_NLD = "NLDPRINTO01GYSAM"
EUR_EA19_STALE = "EA19PRINTO01GYSAM"
EUR_DEU_STALE = "DEUPRINTO01GYSAM"
US_INDPRO_ALT = "INDPRO"

DEFAULT_PUB_LAG_MONTHS = 2  # a priori: IP ~1–2m; use 2 (faster than labour's 3)


def _to_month_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    ms = pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = ms
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _to_quarter_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    qs = pd.DatetimeIndex(naive.to_period("Q").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = qs
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _load_series(series_id: str, *, freq: str, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    raw = load_fred_series(series_id, download=False)
    if freq.upper().startswith("Q"):
        return _to_quarter_start(raw)
    return _to_month_start(raw)


def _apply_pub_lag_monthly(df: pd.DataFrame, *, pub_lag_months: int) -> pd.DataFrame:
    if df.empty or pub_lag_months <= 0:
        return df
    out = df.copy()
    out.index = out.index + pd.DateOffset(months=int(pub_lag_months))
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def _expand_quarterly_to_monthly(
    df: pd.DataFrame,
    *,
    pub_lag_months: int,
) -> pd.DataFrame:
    """Shift quarterly index by pub_lag, then ffill to month-start (no leak)."""
    if df.empty:
        return df
    out = df.copy()
    if pub_lag_months > 0:
        out.index = out.index + pd.DateOffset(months=int(pub_lag_months))
        out = out[~out.index.duplicated(keep="last")].sort_index()
    naive_min = out.index.min().tz_convert(None) if out.index.tz is not None else out.index.min()
    naive_max = out.index.max().tz_convert(None) if out.index.tz is not None else out.index.max()
    start = naive_min.to_period("M").to_timestamp(how="start")
    end = naive_max.to_period("M").to_timestamp(how="start")
    monthly_idx = pd.date_range(start, end, freq="MS", tz="UTC")
    out = out.reindex(monthly_idx.union(out.index)).sort_index()
    out = out.ffill()
    out = out.loc[monthly_idx]
    out.index = pd.DatetimeIndex(out.index)
    if out.index.tz is None:
        out.index = out.index.tz_localize("UTC")
    else:
        out.index = out.index.tz_convert("UTC")
    return out


def load_industrial_production_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """OECD MEI IP YoY growth → monthly PIT panel.

    Columns = ISO currency codes. Values = IP **YoY growth %** as reported
    (GYSAM / GYSAQ). Index = UTC month-start of the PIT *known* date after
    ``pub_lag_months``. Quarterly series are pub-lagged first, then
    forward-filled to month-start. Strategy modules XS-rank on these YoY values
    (foreign only).
    """
    curs = [c.upper() for c in (currencies or IP_SERIES.keys())]
    monthly_cols: dict[str, pd.Series] = {}
    quarterly_cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}
    freq_map: dict[str, str] = {}

    for c in curs:
        sid = IP_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_fred_printo01"
            continue
        freq = IP_FREQ.get(c, "M")
        try:
            s = _load_series(sid, freq=freq, download=download, force=force)
            series_map[c] = sid
            freq_map[c] = freq
            if c == "EUR":
                notes[c] = (
                    f"France proxy via {sid} (M) "
                    f"(EA19PRINTO01GYSAM ends 2023-10; DEUPRINTO01GYSAM ends 2023-12; "
                    f"ITA/ESP/NLD GYSAM alts)"
                )
            elif freq == "Q":
                notes[c] = (
                    f"OECD MEI manufacturing YoY {sid} (quarterly; "
                    f"industry PRINTO01 GYSAM 404)"
                )
            else:
                notes[c] = f"OECD MEI industry YoY {sid} (monthly)"
            if freq == "Q":
                quarterly_cols[c] = s
            else:
                monthly_cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    frames: list[pd.DataFrame] = []
    if monthly_cols:
        mdf = pd.DataFrame(monthly_cols).sort_index()
        mdf = _apply_pub_lag_monthly(mdf, pub_lag_months=pub_lag_months)
        frames.append(mdf)
    if quarterly_cols:
        qdf = pd.DataFrame(quarterly_cols).sort_index()
        qdf = _expand_quarterly_to_monthly(qdf, pub_lag_months=pub_lag_months)
        frames.append(qdf)

    if not frames:
        df = pd.DataFrame()
    else:
        df = pd.concat(frames, axis=1).sort_index()
        df = df.dropna(how="all")
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

    df.attrs["factor"] = "oecd_mei_industrial_production_printo01"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["freq_map"] = freq_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI {ISO3}PRINTO01GYSAM industry YoY monthly "
        "(USD/GBP/JPY/CAD; EUR=FRA France proxy — EA19/DEU GYSAM stale; "
        "AUD/NZD/CHF={ISO3}PRMNTO01GYSAQ manufacturing quarterly YoY — "
        "industry PRINTO01 404; PROINDMISMEI stale ~2024-03 — not primary; "
        "INDPRO/IPMAN US alts)"
    )
    df.attrs["unit"] = "ip_yoy_pct_as_reported"
    df.attrs["score_basis"] = "yoy_growth_as_reported"
    df.attrs["unmapped"] = [c for c in curs if c not in series_map]
    df.attrs["failed_or_alt"] = IP_FAILED_OR_ALT
    return df


def load_us_industrial_production(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US OECD MEI industry YoY (USAPRINTO01GYSAM), PIT-lagged."""
    panel = load_industrial_production_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US industrial production series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_IP"
    out.attrs["series_id"] = IP_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def industrial_production_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_industrial_production_panel(download=False)
    rows = []
    notes = panel.attrs.get("notes", {})
    smap = panel.attrs.get("series_map", {})
    fmap = panel.attrs.get("freq_map", {})
    for c in panel.columns:
        s = panel[c].dropna()
        rows.append(
            {
                "currency": c,
                "fred_id": smap.get(c, ""),
                "freq": fmap.get(c, ""),
                "n_obs": int(len(s)),
                "start": str(s.index.min().date()) if len(s) else None,
                "end": str(s.index.max().date()) if len(s) else None,
                "last": float(s.iloc[-1]) if len(s) else float("nan"),
                "note": notes.get(c, ""),
            }
        )
    for c in panel.attrs.get("unmapped", []):
        rows.append(
            {
                "currency": c,
                "fred_id": "",
                "freq": "",
                "n_obs": 0,
                "start": None,
                "end": None,
                "last": float("nan"),
                "note": notes.get(c, "unmapped_on_fred_printo01"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
