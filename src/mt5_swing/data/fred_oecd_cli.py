"""FRED / OECD MEI Composite Leading Indicator (CLI) panels for leading-activity FX.

Literature
----------
- OECD Composite Leading Indicators (amplitude-adjusted) summarise turning points
  in business-cycle activity ahead of coincident measures (IP, UR, CPI).
- Estrella & Mishkin (1998) and related leading-indicator / recession literature:
  leading activity forecasts risk and asset returns; FX inherits relative-activity
  differentials (Dahlquist–Hasseltoft-style macro–FX framing for *leading*, not
  coincident, activity).
- Primary prior: currencies with *high* relative lagged CLI growth appreciate vs
  low-CLI peers. Honesty alternate: long low CLI (stress / debtor premium).
- Distinct from: macro-diff CPI/IP/UR (coincident), house-price (§35), money-growth
  (§33), CA/TB, BIS REER/credit, reserves, IG OAS (§36), equity-diff, GPR/EPU.

Free data (no OECD portal key)
------------------------------
OECD MEI **amplitude-adjusted CLI** via FRED ``*LOLITOAASTSAM``:

| Ccy | FRED id (primary)   | Notes                                              |
|-----|---------------------|----------------------------------------------------|
| USD | USALOLITOAASTSAM    | through ~2026-08                                   |
| EUR | DEULOLITOAASTSAM    | Germany proxy (``EA19LOLITOAASTSAM`` ends 2022-11) |
| GBP | GBRLOLITOAASTSAM    | through ~2026-08                                   |
| JPY | JPNLOLITOAASTSAM    | through ~2026-08                                   |
| CAD | CANLOLITOAASTSAM    | through ~2026-08                                   |
| AUD | AUSLOLITOAASTSAM    | through ~2026-08                                   |
| NZD | None (primary)      | ``NZLLOLITOAASTSAM`` ends 2019-11 — stale          |
| CHF | None (primary)      | ``CHELOLITOAASTSAM`` ends 2022-11 — stale          |

Tried / documented:
- ``EA19LOLITOAASTSAM`` / ``EZ19LOLITOAASTSAM`` — EA19 live but ends **2022-11**;
  EZ19 **404**. Primary EUR = Germany ``DEULOLITOAASTSAM``.
- ``CHFLOLITOAASTSAM`` — **404**; Swiss is ``CHELOLITOAASTSAM`` (stale).
- ``FRALOLITOAASTSAM`` — live France alt (coverage / report only).
- ``OECDLOLITOAASTSAM`` — aggregate ends 2022-11 (coverage only).

Unit choice (fixed a priori)
----------------------------
Levels are amplitude-adjusted CLI (~100 = trend). Strategy scores use **YoY
diff** ``CLI_t − CLI_{t−12}`` for XS ranks (growth of leading activity), plus
trailing z of that YoY and Δ12 acceleration. Documented in panel attrs.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Monthly OECD CLI typically lands ~1–2 months after reference. Default
``pub_lag_months=2``: observation dated month-start ``t`` is first known at
``t + 2 months``. Strategy modules add a **1 trading-day** weight lag
(``signal_lag_days=1``); no extra month signal lag (a priori for this wave).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED OECD amplitude-adjusted CLI
CLI_SERIES: dict[str, str | None] = {
    "USD": "USALOLITOAASTSAM",
    "EUR": "DEULOLITOAASTSAM",  # Germany proxy (EA19 ends 2022-11)
    "GBP": "GBRLOLITOAASTSAM",
    "JPY": "JPNLOLITOAASTSAM",
    "CAD": "CANLOLITOAASTSAM",
    "AUD": "AUSLOLITOAASTSAM",
    "NZD": None,  # NZLLOLITOAASTSAM ends 2019-11
    "CHF": None,  # CHELOLITOAASTSAM ends 2022-11
}

CLI_FAILED_OR_ALT: dict[str, list[str]] = {
    "eur_ea19_stale_2022": ["EA19LOLITOAASTSAM"],
    "eur_ez19_404": ["EZ19LOLITOAASTSAM"],
    "eur_germany_primary": ["DEULOLITOAASTSAM"],
    "eur_france_secondary": ["FRALOLITOAASTSAM"],
    "nzd_stale_2019": ["NZLLOLITOAASTSAM"],
    "chf_stale_2022": ["CHELOLITOAASTSAM"],
    "chf_alt_404": ["CHFLOLITOAASTSAM"],
    "oecd_aggregate_stale": ["OECDLOLITOAASTSAM"],
}

EUR_SECONDARY = "FRALOLITOAASTSAM"
EA19_STALE = "EA19LOLITOAASTSAM"

DEFAULT_PUB_LAG_MONTHS = 2  # conservative monthly OECD CLI lag


def _to_month_start(s: pd.Series) -> pd.Series:
    s = s.dropna().sort_index()
    idx = pd.DatetimeIndex(s.index)
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    ms = pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start")).tz_localize("UTC")
    s = s.copy()
    s.index = ms
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s


def _load_monthly(series_id: str, *, download: bool, force: bool) -> pd.Series:
    if download:
        download_fred_series(series_id, force=force)
    return _to_month_start(load_fred_series(series_id, download=False))


def _apply_pub_lag(df: pd.DataFrame, *, pub_lag_months: int) -> pd.DataFrame:
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


def load_oecd_cli_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Monthly OECD amplitude-adjusted CLI → PIT panel (index levels ~100).

    Columns = ISO currency codes. Values = OECD MEI CLI amplitude-adjusted.
    Index = UTC month-start of the PIT *known* date after ``pub_lag_months``.
    Strategy modules convert to YoY diff for XS ranks.
    """
    curs = [c.upper() for c in (currencies or CLI_SERIES.keys())]
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}

    for c in curs:
        sid = CLI_SERIES.get(c)
        if sid is None:
            notes[c] = (
                "unmapped_on_primary_cli "
                "(NZL ends 2019-11; CHE ends 2022-11 — stale)"
            )
            continue
        try:
            s = _load_monthly(sid, download=download, force=force)
            series_map[c] = sid
            if c == "EUR":
                notes[c] = (
                    f"Germany proxy via {sid} "
                    f"(EA19LOLITOAASTSAM ends 2022-11; EZ19 404)"
                )
            elif c == "USD":
                notes[c] = f"OECD US CLI {sid}"
            cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    df = pd.DataFrame(cols).sort_index()
    df = _apply_pub_lag(df, pub_lag_months=pub_lag_months)

    df.attrs["factor"] = "oecd_cli_amplitude_adjusted"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED OECD MEI *LOLITOAASTSAM amplitude-adjusted CLI "
        "(EUR=DEU proxy; NZD/CHF stale — unmapped on primary)"
    )
    df.attrs["unit"] = "oecd_mei_cli_amplitude_adjusted"
    df.attrs["unmapped"] = [c for c in curs if c not in cols]
    df.attrs["failed_or_alt"] = CLI_FAILED_OR_ALT
    return df


def load_us_oecd_cli(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
) -> pd.Series:
    """US OECD CLI (USALOLITOAASTSAM), PIT-lagged."""
    panel = load_oecd_cli_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US OECD CLI series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_OECD_CLI"
    out.attrs["series_id"] = CLI_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    return out


def cli_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_oecd_cli_panel(download=False)
    rows = []
    notes = panel.attrs.get("notes", {})
    smap = panel.attrs.get("series_map", {})
    for c in panel.columns:
        s = panel[c].dropna()
        rows.append(
            {
                "currency": c,
                "fred_id": smap.get(c, ""),
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
                "n_obs": 0,
                "start": None,
                "end": None,
                "last": float("nan"),
                "note": notes.get(c, "unmapped_on_primary_cli"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
