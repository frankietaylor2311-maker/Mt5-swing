"""FRED / IMF IFS total-reserves panels for external-buffer FX (free, PIT).

Literature
----------
- External-buffer / reserve-adequacy channel (Aizenman–Jeanne–Rancière; IMF
  reserve-adequacy / Guidotti–Greenspan literature): currencies with *high*
  relative international reserves (or reserves/GDP) are more resilient →
  subsequent appreciation / lower crash risk.
- Honesty alternate: low-reserve / thin-buffer debtor premium (risk-premium sort).
- Distinct from CA/GDP (§21), CB-BS/QE (§22), debt/GDP (§29), fiscal (§28),
  TB (§30), BIS REER (§31), BIS credit (§32), money-growth (§33), funding-liq,
  macro-diff, equity-diff, AI-GPR, ToT, dollar-beta, crash-skew.

Free data (no Bloomberg / OECD portal key)
-----------------------------------------
IMF IFS **Total Reserves excluding Gold** (USD millions) via FRED ``TRESEG*M052N``:

| Ccy | FRED id (primary) | Notes                                              |
|-----|-------------------|----------------------------------------------------|
| USD | TRESEGUSM052N     | through 2026-08                                    |
| EUR | TRESEGDEM052N     | Germany proxy (``TRESEGEZM052N`` ends **2018-04**) |
| GBP | TRESEGGBM052N     | through 2026-08                                    |
| JPY | TRESEGJPM052N     | through 2026-08                                    |
| CAD | TRESEGCAM052N     | through 2026-08                                    |
| AUD | TRESEGAUM052N     | through 2026-08                                    |
| NZD | None              | ``TRESEGNZM052N`` / ``TRESEGNZL052N`` **404**      |
| CHF | None              | ``TRESEGCHM052N`` / ``TRESEGCHE052N`` **404**      |

Tried / documented:
- ``TRESEGEZM052N`` — euro-area aggregate exists historically but **ends 2018-04**.
- ``TRESEGFRM052N`` / ``TRESEGITM052N`` / ``TRESEGESM052N`` — live FR/IT/ES alts
  (coverage / report only; primary EUR = Germany).
- Reserves/%GDP World-Bank / IMF WEO mnemonics (``TRESEG*188N``, many
  ``MKTGDP*646NWDB``) — mostly **404** or US-only; free GDP units heterogeneous
  and often end ~2023 → **reserves/GDP not formed** as primary panel.
- US ``TOTRESNS`` / ``WRESBAL`` — *Fed reserve balances* (bank reserves at Fed),
  **not** IMF IFS external reserves; documented alt for US tilt research only.

Publication lag (conservative, fixed a priori — not holdout-tuned)
------------------------------------------------------------------
Monthly IMF IFS reserves typically land ~1–3 months after month-end. Default
``pub_lag_months=3``: observation dated month-start ``t`` is first known at
``t + 3 months``. Strategy modules add ``signal_lag`` months on top.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from mt5_swing.data.fred_rates import download_fred_series, load_fred_series

# Currency → FRED IMF IFS Total Reserves excl. Gold (USD mn, monthly)
RESERVES_SERIES: dict[str, str | None] = {
    "USD": "TRESEGUSM052N",
    "EUR": "TRESEGDEM052N",  # Germany proxy (EA aggregate stale 2018)
    "GBP": "TRESEGGBM052N",
    "JPY": "TRESEGJPM052N",
    "CAD": "TRESEGCAM052N",
    "AUD": "TRESEGAUM052N",
    "NZD": None,  # TRESEGNZM052N / TRESEGNZL052N 404
    "CHF": None,  # TRESEGCHM052N / TRESEGCHE052N 404
}

# Documented failed / alt / secondary mnemonics
RESERVES_FAILED_OR_ALT: dict[str, list[str]] = {
    "nzd_chf_404": [
        "TRESEGNZM052N",
        "TRESEGNZL052N",
        "TRESEGNZA052N",
        "TRESEGCHM052N",
        "TRESEGCHE052N",
        "TRESEGSWM052N",
    ],
    "ez_stale_2018": ["TRESEGEZM052N"],
    "eur_secondary_fr_it_es": ["TRESEGFRM052N", "TRESEGITM052N", "TRESEGESM052N"],
    "reserves_gdp_mostly_404": [
        "TRESEGUSA188N",
        "TRESEGDEA188N",
        "MKTGDPDEU646NWDB",
        "MKTGDPGBR646NWDB",
        "MKTGDPJPN646NWDB",
    ],
    "us_fed_bank_reserves_not_ifs": ["TOTRESNS", "WRESBAL"],
    "incl_gold_alt": ["TRESEGUSM194N"],
}

EUR_SECONDARY = "TRESEGFRM052N"
US_FED_BANK_RESERVES = "TOTRESNS"  # documented alt — NOT IFS external buffer

DEFAULT_PUB_LAG_MONTHS = 3  # conservative IMF IFS monthly lag


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


def load_reserves_panel(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
    use_log: bool = True,
) -> pd.DataFrame:
    """Monthly IMF IFS total reserves excl. gold → PIT panel.

    Columns = ISO currency codes. Values = log(USD mn) if ``use_log`` else raw
    USD millions. Index = UTC month-start of the PIT *known* date after
    ``pub_lag_months``.
    """
    curs = [c.upper() for c in (currencies or RESERVES_SERIES.keys())]
    cols: dict[str, pd.Series] = {}
    notes: dict[str, str] = {}
    series_map: dict[str, str] = {}

    for c in curs:
        sid = RESERVES_SERIES.get(c)
        if sid is None:
            notes[c] = "unmapped_on_primary_treseg (404 on free FRED TRESEG*M052N)"
            continue
        try:
            s = _load_monthly(sid, download=download, force=force)
            series_map[c] = sid
            if use_log:
                s = np.log(s.replace(0.0, np.nan))
            if c == "EUR":
                notes[c] = (
                    f"Germany proxy via {sid} "
                    "(TRESEGEZM052N ends 2018-04 — not primary)"
                )
            elif c == "USD":
                notes[c] = f"IMF IFS reserves excl gold {sid}"
            cols[c] = s
        except Exception as exc:  # noqa: BLE001
            notes[c] = f"load_fail:{exc}"[:120]

    df = pd.DataFrame(cols).sort_index()
    df = _apply_pub_lag(df, pub_lag_months=pub_lag_months)

    df.attrs["factor"] = "imf_ifs_treseg_excl_gold"
    df.attrs["pub_lag_months"] = int(pub_lag_months)
    df.attrs["series_map"] = series_map
    df.attrs["notes"] = notes
    df.attrs["source"] = (
        "FRED IMF IFS TRESEG*M052N total reserves excl. gold (USD mn); "
        "EUR=Germany TRESEGDEM052N; NZD/CHF 404; log levels for XS ranks"
    )
    df.attrs["unit"] = "log_usd_mn" if use_log else "usd_mn"
    df.attrs["use_log"] = bool(use_log)
    df.attrs["unmapped"] = [c for c in curs if c not in cols]
    df.attrs["failed_or_alt"] = RESERVES_FAILED_OR_ALT
    df.attrs["reserves_gdp"] = (
        "not_formed: free GDP units heterogeneous / many 404; "
        "primary is log TRESEG levels (rank ≡ level rank)"
    )
    return df


def load_us_reserves(
    *,
    pub_lag_months: int = DEFAULT_PUB_LAG_MONTHS,
    download: bool = True,
    force: bool = False,
    use_log: bool = True,
) -> pd.Series:
    """US IMF IFS total reserves excl. gold (TRESEGUSM052N), PIT-lagged."""
    panel = load_reserves_panel(
        ["USD"],
        pub_lag_months=pub_lag_months,
        download=download,
        force=force,
        use_log=use_log,
    )
    if "USD" not in panel.columns:
        raise RuntimeError("US reserves series failed to load")
    out = panel["USD"].dropna()
    out.name = "US_RESERVES"
    out.attrs["series_id"] = RESERVES_SERIES["USD"]
    out.attrs["pub_lag_months"] = int(pub_lag_months)
    out.attrs["unit"] = panel.attrs.get("unit")
    return out


def reserves_coverage(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-currency coverage summary for reports."""
    if panel is None:
        panel = load_reserves_panel(download=False)
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
                "note": notes.get(c, "unmapped_on_primary_treseg"),
            }
        )
    return pd.DataFrame(rows)


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"
