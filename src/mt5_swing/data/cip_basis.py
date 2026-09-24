"""Du–Keerati–Schreger government-bond CIP / U.S. Treasury premium loaders.

Literature
----------
- Du, Tepper & Verdelhan (2018), *JF* — CIP deviations as intermediary /
  dollar-funding stress (bank / Libor cross-currency basis).
- Du, Im & Schreger (2018) — G10 government-bond CIP as the
  **U.S. Treasury Premium** (convenience-yield differential).
- Du, Keerati & Schreger (2025), "Decoupling Dollar and Treasury Privilege"
  — public government-bond CIP panel (this dataset, v4).

Public data
-----------
CSV: ``https://jschreger.s3.us-east-2.amazonaws.com/cip_dataset_v4.csv``
Appendix: ``…/Data_Appendix_V4.pdf``

Variables (appendix §4):
- ``cip_govt`` — gov CIP deviation in **basis points**:
  ``x = y_i^Govt − ρ − y_USD^Govt``.
- ``diff_y`` — gov yield differential (pp); ``rho`` — forward premium (pp).
- Main series splices IBOR→alternative benchmark at ``cip_govt_break_date``.

Tenor prior (fixed a priori — **not** holdout-tuned)
----------------------------------------------------
``DEFAULT_TENOR = "5y"`` — medium tenor used in Du–Im–Schreger Treasury-premium
work; liquid CCS construction (≥1y); full multi-year G10 trade-currency coverage
(AUD/CAD/CHF/EUR/GBP/JPY/NZD from ~2000 through 2025-06). 1y is a documented
companion; we do **not** grid tenors on holdout.

PIT lag (frozen)
----------------
Daily market series → ``pub_lag_days=1`` (observation dated *t* first known at
*t+1*; matches TED/CPFF/OAS daily priors). Strategy modules add
``signal_lag`` months (default 1) on the month-end panel + 1 trading-day weight
lag.

Distinct from: funding_liq §20 (NFCI/TED/CPFF), fwd_carry §19 (rate-implied CIP
FD approximation — **not** observed basis), IG OAS §36, CB-BS §22.

Do **not** overlay onto the locked FTMO sleeve.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import pandas as pd

CIP_CSV_URL = "https://jschreger.s3.us-east-2.amazonaws.com/cip_dataset_v4.csv"
CIP_APPENDIX_URL = "https://jschreger.s3.us-east-2.amazonaws.com/Data_Appendix_V4.pdf"

# G10 trade currencies we map to USD majors (EUR = Germany in dataset)
TRADE_CURRENCIES: tuple[str, ...] = ("EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")

DEFAULT_TENOR = "5y"  # a priori: Du–Im–Schreger Treasury-premium medium tenor
DEFAULT_PUB_LAG_DAYS = 1
DEFAULT_VALUE_COL = "cip_govt"  # bps

G10_GROUP = "g10"


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"


def cip_raw_path() -> Path:
    return macro_dir() / "cip_dataset_v4.csv"


def cip_slim_panel_path(*, tenor: str = DEFAULT_TENOR) -> Path:
    return macro_dir() / f"cip_g10_{tenor}_govt_panel.csv"


def ensure_cip_dataset(*, force: bool = False) -> Path:
    """Download raw Du–Schreger CIP CSV into ``data/macro/`` if missing."""
    path = cip_raw_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force and path.stat().st_size > 1_000_000:
        return path
    print(f"Downloading Du–Schreger CIP v4 → {path} …")
    urlretrieve(CIP_CSV_URL, path)
    if path.stat().st_size < 1_000_000:
        raise RuntimeError(f"CIP download looks too small: {path} ({path.stat().st_size} bytes)")
    return path


def _parse_dates(s: pd.Series) -> pd.Series:
    """Parse appendix dates like ``08feb2007`` / ISO."""
    parsed = pd.to_datetime(s, format="%d%b%Y", errors="coerce")
    if parsed.isna().any():
        # Fallback only for residual non-%d%b%Y rows (avoids noisy dateutil warn on mixed)
        mask = parsed.isna()
        parsed = parsed.copy()
        parsed.loc[mask] = pd.to_datetime(s.loc[mask], format="mixed", errors="coerce")
    return parsed


def load_cip_long(
    *,
    download: bool = True,
    force: bool = False,
    currencies: Iterable[str] | None = None,
    tenors: Iterable[str] | None = None,
    group: str | None = G10_GROUP,
) -> pd.DataFrame:
    """Load (optionally filtered) long CIP panel from cache / download."""
    if download:
        ensure_cip_dataset(force=force)
    path = cip_raw_path()
    if not path.exists():
        raise FileNotFoundError(f"Missing CIP dataset: {path}")

    want_ccy = {c.upper() for c in (currencies or TRADE_CURRENCIES)}
    want_ten = {t.lower() for t in (tenors or (DEFAULT_TENOR,))}
    usecols = [
        "group",
        "currency",
        "tenor",
        "date",
        "diff_y",
        "rho",
        "cip_govt",
        "cip_govt_break_date",
    ]
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=400_000):
        if group is not None:
            chunk = chunk[chunk["group"].astype(str).str.lower() == group.lower()]
        chunk = chunk[chunk["currency"].astype(str).str.upper().isin(want_ccy)]
        chunk = chunk[chunk["tenor"].astype(str).str.lower().isin(want_ten)]
        if chunk.empty:
            continue
        frames.append(chunk)
    if not frames:
        return pd.DataFrame(columns=usecols)
    out = pd.concat(frames, ignore_index=True)
    out["currency"] = out["currency"].astype(str).str.upper()
    out["tenor"] = out["tenor"].astype(str).str.lower()
    out["date"] = _parse_dates(out["date"])
    out = out.dropna(subset=["date"]).sort_values(["currency", "tenor", "date"])
    out = out.drop_duplicates(subset=["currency", "tenor", "date"], keep="last")
    return out.reset_index(drop=True)


def _apply_pub_lag_daily(df: pd.DataFrame, pub_lag_days: int) -> pd.DataFrame:
    """Shift calendar index forward by pub_lag_days (PIT)."""
    out = df.copy()
    if pub_lag_days > 0:
        out.index = out.index + pd.Timedelta(days=int(pub_lag_days))
        out = out[~out.index.duplicated(keep="last")].sort_index()
    out.attrs["pub_lag_days"] = int(pub_lag_days)
    return out


def load_cip_wide_daily(
    *,
    tenor: str = DEFAULT_TENOR,
    value_col: str = DEFAULT_VALUE_COL,
    currencies: Iterable[str] | None = None,
    pub_lag_days: int = DEFAULT_PUB_LAG_DAYS,
    download: bool = True,
    force: bool = False,
) -> pd.DataFrame:
    """Wide daily panel: index=date (UTC), columns=currency, values=cip_govt (bps)."""
    ccys = [c.upper() for c in (currencies or TRADE_CURRENCIES)]
    long = load_cip_long(
        download=download,
        force=force,
        currencies=ccys,
        tenors=(tenor,),
        group=G10_GROUP,
    )
    if long.empty:
        wide = pd.DataFrame(columns=ccys)
        wide.attrs.update(
            {
                "tenor": tenor,
                "value_col": value_col,
                "pub_lag_days": int(pub_lag_days),
                "source": "du_keerati_schreger_cip_v4",
                "unit": "bps",
            }
        )
        return wide

    pivot = (
        long.dropna(subset=[value_col])
        .pivot_table(index="date", columns="currency", values=value_col, aggfunc="last")
        .sort_index()
    )
    pivot.index = pd.DatetimeIndex(pivot.index)
    if pivot.index.tz is None:
        pivot.index = pivot.index.tz_localize("UTC")
    else:
        pivot.index = pivot.index.tz_convert("UTC")
    for c in ccys:
        if c not in pivot.columns:
            pivot[c] = pd.NA
    pivot = pivot[[c for c in ccys if c in pivot.columns]]
    pivot = _apply_pub_lag_daily(pivot, pub_lag_days)
    pivot.attrs.update(
        {
            "tenor": tenor,
            "value_col": value_col,
            "pub_lag_days": int(pub_lag_days),
            "source": "du_keerati_schreger_cip_v4",
            "unit": "bps",
            "score_basis": "cip_govt_bps",
            "tenor_prior": (
                "5y a priori (Du–Im–Schreger Treasury-premium medium tenor; "
                "liquid CCS; multi-year G10 coverage) — not HO-tuned"
            ),
            "citations": [
                "Du, Tepper & Verdelhan (2018), JF — CIP / intermediary constraints",
                "Du, Im & Schreger (2018) — U.S. Treasury Premium",
                "Du, Keerati & Schreger (2025) — Decoupling Dollar and Treasury Privilege (v4 panel)",
            ],
            "distinct_from": [
                "funding_liquidity_20",
                "forward_carry_19",
                "ig_oas_36",
                "cb_balance_sheet_22",
            ],
            "end_date_note": "raw panel ends ~2025-06-30 (dataset vintage)",
        }
    )
    return pivot


def daily_to_month_end(panel: pd.DataFrame) -> pd.DataFrame:
    """Last valid observation per calendar month (UTC month-end index)."""
    if panel.empty:
        return panel.copy()
    p = panel.copy()
    p.index = pd.DatetimeIndex(p.index)
    if p.index.tz is None:
        p.index = p.index.tz_localize("UTC")
    # resample to month-end business; use last non-null per month
    monthly = p.resample("ME").last()
    monthly.index = (
        pd.DatetimeIndex(monthly.index)
        .tz_convert(None)
        .to_period("M")
        .to_timestamp(how="end")
        .tz_localize("UTC")
    )
    monthly = monthly[~monthly.index.duplicated(keep="last")].sort_index()
    monthly.attrs.update({k: v for k, v in p.attrs.items()})
    monthly.attrs["frequency"] = "month_end"
    return monthly


def load_cip_panel(
    *,
    tenor: str = DEFAULT_TENOR,
    value_col: str = DEFAULT_VALUE_COL,
    currencies: Iterable[str] | None = None,
    pub_lag_days: int = DEFAULT_PUB_LAG_DAYS,
    download: bool = True,
    force: bool = False,
    frequency: str = "month_end",
) -> pd.DataFrame:
    """Load CIP wide panel (daily or month-end) with PIT pub lag.

    ``frequency``: ``"daily"`` or ``"month_end"`` (default — matches sister XS waves).
    """
    daily = load_cip_wide_daily(
        tenor=tenor,
        value_col=value_col,
        currencies=currencies,
        pub_lag_days=pub_lag_days,
        download=download,
        force=force,
    )
    if frequency == "daily":
        return daily
    if frequency in ("month_end", "monthly", "ME"):
        return daily_to_month_end(daily)
    raise ValueError(f"Unknown frequency={frequency!r}")


def write_slim_g10_panel(
    *,
    tenor: str = DEFAULT_TENOR,
    pub_lag_days: int = 0,
    force_download: bool = False,
) -> Path:
    """Cache a slim G10 trade-currency daily ``cip_govt`` panel (pre-pub-lag).

    Written **without** pub lag so loaders can re-apply PIT lags reproducibly.
    """
    daily = load_cip_wide_daily(
        tenor=tenor,
        pub_lag_days=0,
        download=True,
        force=force_download,
    )
    path = cip_slim_panel_path(tenor=tenor)
    out = daily.copy()
    out.index.name = "date"
    out.to_csv(path)
    meta = {
        "tenor": tenor,
        "value_col": "cip_govt",
        "unit": "bps",
        "pub_lag_in_file": 0,
        "source": "du_keerati_schreger_cip_v4",
        "currencies": list(out.columns),
        "start": str(out.dropna(how="all").index.min().date()) if len(out) else None,
        "end": str(out.dropna(how="all").index.max().date()) if len(out) else None,
    }
    path.with_suffix(".meta.json").write_text(
        __import__("json").dumps(meta, indent=2)
    )
    return path


def load_ust_premium(
    cip_panel: pd.DataFrame | None = None,
    *,
    pub_lag_days: int = DEFAULT_PUB_LAG_DAYS,
    download: bool = True,
) -> pd.Series:
    """U.S. Treasury Premium ≈ −cross-sectional mean of foreign ``cip_govt`` (bps).

    Du–Im–Schreger (2018): average G10 gov CIP vs USD measures Treasury specialness.
    Positive premium → UST rich vs foreign gov after FX hedge.
    """
    panel = cip_panel
    if panel is None:
        panel = load_cip_panel(pub_lag_days=pub_lag_days, download=download, frequency="month_end")
    foreign = panel[[c for c in panel.columns if c.upper() != "USD"]].copy()
    prem = (-foreign.mean(axis=1, skipna=True)).rename("ust_premium")
    prem.attrs.update(
        {
            "definition": "-mean(cip_govt_foreign)",
            "unit": "bps",
            "source": "du_im_schreger_treasury_premium",
            "pub_lag_days": panel.attrs.get("pub_lag_days", pub_lag_days),
            "tenor": panel.attrs.get("tenor", DEFAULT_TENOR),
        }
    )
    return prem


def cip_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-currency observation counts and date span."""
    rows = []
    for c in panel.columns:
        s = panel[c].dropna()
        rows.append(
            {
                "currency": c,
                "n_obs": int(len(s)),
                "start": str(s.index.min().date()) if len(s) else None,
                "end": str(s.index.max().date()) if len(s) else None,
                "last": float(s.iloc[-1]) if len(s) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "CIP_CSV_URL",
    "DEFAULT_PUB_LAG_DAYS",
    "DEFAULT_TENOR",
    "DEFAULT_VALUE_COL",
    "TRADE_CURRENCIES",
    "cip_coverage",
    "cip_raw_path",
    "cip_slim_panel_path",
    "daily_to_month_end",
    "ensure_cip_dataset",
    "load_cip_long",
    "load_cip_panel",
    "load_cip_wide_daily",
    "load_ust_premium",
    "macro_dir",
    "write_slim_g10_panel",
]
