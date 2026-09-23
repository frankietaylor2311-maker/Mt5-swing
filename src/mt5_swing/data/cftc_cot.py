"""CFTC Commitments of Traders (COT) FX positioning panel — free public data.

Sources (no API key required)
-----------------------------
- Traders in Financial Futures (TFF) Futures-Only SODA:
  ``https://publicreporting.cftc.gov/resource/gpe5-46if.json``
- Legacy Futures-Only SODA:
  ``https://publicreporting.cftc.gov/resource/6dca-aqww.json``

FX CME / ICE contracts mapped to G10 currencies + U.S. Dollar Index.

Point-in-time / release lag (documented, frozen)
-----------------------------------------------
- ``report_date`` = Tuesday open-interest snapshot.
- CFTC typically releases the weekly COT on the following **Friday ~15:30 ET**.
- Conservative ``release_lag_days=3`` → treat Friday as the first calendar day
  the positions are known (same-day Friday trading after 15:30 is *not* assumed).
- Strategy layer then applies ``signal_lag`` trading days (≥1) on daily weight
  expansion so Monday is the earliest trade date under default settings.

Literature (positioning / speculative pressure)
-----------------------------------------------
Klitgaard & Weir (NY Fed EPR 2004); Sanders / Irwin / Merrin COT predictive-
content work; classic Briese / speculative-pressure constructions. Mixed
academic evidence on forecasting power — we pre-specify continuation *and*
extreme mean-reversion legs without holdout tuning.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

# CME / ICE FX futures contract market codes (CFTC)
FX_CONTRACT_CODES: dict[str, str] = {
    "EUR": "099741",
    "GBP": "096742",
    "JPY": "097741",
    "CAD": "090741",
    "CHF": "092741",
    "AUD": "232741",
    "NZD": "112741",
    "DX": "098662",  # U.S. Dollar Index — ICE
}

TFF_DATASET = "gpe5-46if"
LEGACY_DATASET = "6dca-aqww"
SOCRATA_BASE = "https://publicreporting.cftc.gov/resource"

# Default PIT: Tuesday report → Friday known
DEFAULT_RELEASE_LAG_DAYS = 3


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"


def _ensure_macro() -> Path:
    d = macro_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def tff_raw_path() -> Path:
    return macro_dir() / "cftc_tff_fx_raw.csv"


def legacy_raw_path() -> Path:
    return macro_dir() / "cftc_legacy_fx_raw.csv"


def cot_panel_path() -> Path:
    return macro_dir() / "cftc_cot_fx_panel.csv"


def cot_meta_path() -> Path:
    return macro_dir() / "cftc_cot_fx_panel.meta.json"


def _fetch_socrata(
    dataset: str,
    *,
    codes: list[str],
    select: str,
    page_size: int = 50_000,
) -> list[dict]:
    """Paginated SODA fetch for selected contract codes."""
    code_list = ",".join(f"'{c}'" for c in codes)
    where = f"cftc_contract_market_code in({code_list})"
    rows: list[dict] = []
    offset = 0
    while True:
        params = {
            "$select": select,
            "$where": where,
            "$order": "report_date_as_yyyy_mm_dd,cftc_contract_market_code",
            "$limit": str(page_size),
            "$offset": str(offset),
        }
        url = f"{SOCRATA_BASE}/{dataset}.json?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "mt5-swing-cot/1.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            chunk = json.loads(resp.read().decode())
        if not chunk:
            break
        rows.extend(chunk)
        if len(chunk) < page_size:
            break
        offset += page_size
    return rows


def download_tff_fx(*, force: bool = False) -> Path:
    """Download/cache TFF Futures-Only rows for FX + DX contracts."""
    path = tff_raw_path()
    if path.exists() and not force and path.stat().st_size > 500:
        return path
    _ensure_macro()
    select = (
        "report_date_as_yyyy_mm_dd,cftc_contract_market_code,contract_market_name,"
        "open_interest_all,"
        "lev_money_positions_long,lev_money_positions_short,"
        "asset_mgr_positions_long,asset_mgr_positions_short,"
        "dealer_positions_long_all,dealer_positions_short_all,"
        "other_rept_positions_long,other_rept_positions_short,"
        "nonrept_positions_long_all,nonrept_positions_short_all"
    )
    try:
        rows = _fetch_socrata(TFF_DATASET, codes=list(FX_CONTRACT_CODES.values()), select=select)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        if path.exists() and path.stat().st_size > 500:
            print(f"WARN TFF download failed ({exc}); keeping cache {path}")
            return path
        raise RuntimeError(f"CFTC TFF download failed and no cache at {path}: {exc}") from exc
    if not rows:
        raise RuntimeError("CFTC TFF returned zero rows for FX codes")
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def download_legacy_fx(*, force: bool = False) -> Path:
    """Download/cache Legacy Futures-Only rows for FX + DX (NonComm / Comm)."""
    path = legacy_raw_path()
    if path.exists() and not force and path.stat().st_size > 500:
        return path
    _ensure_macro()
    select = (
        "report_date_as_yyyy_mm_dd,cftc_contract_market_code,contract_market_name,"
        "open_interest_all,"
        "noncomm_positions_long_all,noncomm_positions_short_all,"
        "comm_positions_long_all,comm_positions_short_all,"
        "nonrept_positions_long_all,nonrept_positions_short_all"
    )
    try:
        rows = _fetch_socrata(
            LEGACY_DATASET, codes=list(FX_CONTRACT_CODES.values()), select=select
        )
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
        if path.exists() and path.stat().st_size > 500:
            print(f"WARN Legacy COT download failed ({exc}); keeping cache {path}")
            return path
        raise RuntimeError(f"CFTC Legacy download failed and no cache at {path}: {exc}") from exc
    if not rows:
        raise RuntimeError("CFTC Legacy returned zero rows for FX codes")
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def _code_to_ccy() -> dict[str, str]:
    return {v: k for k, v in FX_CONTRACT_CODES.items()}


def _normalize_contract_code(val) -> str:
    """SODA / CSV may drop leading zeros (099741 → 99741); pad to 6 digits."""
    s = str(val).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if s.isdigit():
        return s.zfill(6)
    return s


def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _parse_report_dates(raw: pd.Series) -> pd.DatetimeIndex:
    dt = pd.to_datetime(raw, utc=True, errors="coerce")
    return pd.DatetimeIndex(dt).normalize()


def build_cot_panel(
    *,
    release_lag_days: int = DEFAULT_RELEASE_LAG_DAYS,
    force_download: bool = False,
) -> pd.DataFrame:
    """Build wide panel indexed by *known_date* (report_date + release_lag).

    Columns (per currency EUR/GBP/... and DX):
    - ``{ccy}_lev_net``, ``{ccy}_lev_net_oi``  (TFF leveraged money)
    - ``{ccy}_am_net``, ``{ccy}_am_net_oi``    (TFF asset manager)
    - ``{ccy}_nc_net``, ``{ccy}_nc_net_oi``    (Legacy non-commercial)
    - ``{ccy}_oi``
    Plus ``report_date`` column for audit.
    """
    tff_path = download_tff_fx(force=force_download)
    leg_path = download_legacy_fx(force=force_download)
    tff = pd.read_csv(tff_path)
    leg = pd.read_csv(leg_path)
    code_map = _code_to_ccy()

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["report_date"] = _parse_report_dates(out["report_date_as_yyyy_mm_dd"])
        out["ccy"] = out["cftc_contract_market_code"].map(_normalize_contract_code).map(code_map)
        out = out.dropna(subset=["report_date", "ccy"])
        out["oi"] = _to_float(out["open_interest_all"])
        return out

    tff = _prep(tff)
    tff["lev_net"] = _to_float(tff["lev_money_positions_long"]) - _to_float(
        tff["lev_money_positions_short"]
    )
    tff["am_net"] = _to_float(tff["asset_mgr_positions_long"]) - _to_float(
        tff["asset_mgr_positions_short"]
    )
    tff["lev_net_oi"] = tff["lev_net"] / tff["oi"].replace(0.0, np.nan)
    tff["am_net_oi"] = tff["am_net"] / tff["oi"].replace(0.0, np.nan)

    leg = _prep(leg)
    leg["nc_net"] = _to_float(leg["noncomm_positions_long_all"]) - _to_float(
        leg["noncomm_positions_short_all"]
    )
    leg["nc_net_oi"] = leg["nc_net"] / leg["oi"].replace(0.0, np.nan)

    # Pivot TFF
    pieces: list[pd.DataFrame] = []
    for col in ("lev_net", "lev_net_oi", "am_net", "am_net_oi", "oi"):
        wide = tff.pivot_table(
            index="report_date", columns="ccy", values=col, aggfunc="last"
        )
        wide.columns = [f"{c}_{col}" for c in wide.columns]
        pieces.append(wide)

    for col in ("nc_net", "nc_net_oi"):
        wide = leg.pivot_table(
            index="report_date", columns="ccy", values=col, aggfunc="last"
        )
        wide.columns = [f"{c}_{col}" for c in wide.columns]
        pieces.append(wide)

    panel = pd.concat(pieces, axis=1, sort=True).sort_index()
    panel = panel[~panel.index.duplicated(keep="last")]
    # Release lag: known_date = report_date + lag
    known = panel.index + pd.Timedelta(days=int(release_lag_days))
    panel = panel.copy()
    panel.insert(0, "report_date", panel.index)
    panel.index = pd.DatetimeIndex(known, name="known_date")
    if panel.index.tz is None:
        panel.index = panel.index.tz_localize("UTC")
    else:
        panel.index = panel.index.tz_convert("UTC")
    panel = panel[~panel.index.duplicated(keep="last")].sort_index()

    path = cot_panel_path()
    _ensure_macro()
    panel.to_csv(path)
    meta = {
        "release_lag_days": int(release_lag_days),
        "tff_dataset": TFF_DATASET,
        "legacy_dataset": LEGACY_DATASET,
        "contracts": FX_CONTRACT_CODES,
        "n_rows": int(len(panel)),
        "start": str(panel.index.min()),
        "end": str(panel.index.max()),
        "note": (
            "Index = known_date (report_date + release_lag_days). "
            "CFTC Tuesday snapshot, Friday release; lag=3 → Friday known."
        ),
    }
    cot_meta_path().write_text(json.dumps(meta, indent=2))
    panel.attrs.update(meta)
    return panel


def load_cot_panel(
    *,
    release_lag_days: int = DEFAULT_RELEASE_LAG_DAYS,
    download: bool = True,
    force_download: bool = False,
) -> pd.DataFrame:
    """Load cached COT panel or rebuild from SODA."""
    path = cot_panel_path()
    if download or force_download or not path.exists():
        return build_cot_panel(
            release_lag_days=release_lag_days, force_download=force_download
        )
    panel = pd.read_csv(path, index_col=0, parse_dates=True)
    panel.index = pd.DatetimeIndex(panel.index)
    if panel.index.tz is None:
        panel.index = panel.index.tz_localize("UTC")
    else:
        panel.index = panel.index.tz_convert("UTC")
    if cot_meta_path().exists():
        meta = json.loads(cot_meta_path().read_text())
        panel.attrs.update(meta)
    panel.attrs["release_lag_days"] = int(release_lag_days)
    return panel


def cot_score_panel(
    panel: pd.DataFrame,
    *,
    field: str = "lev_net_oi",
    currencies: tuple[str, ...] = ("EUR", "GBP", "JPY", "CAD", "CHF", "AUD", "NZD"),
) -> pd.DataFrame:
    """Extract wide currency×time score matrix from the long-column panel."""
    cols = {}
    for ccy in currencies:
        key = f"{ccy}_{field}"
        if key in panel.columns:
            cols[ccy] = panel[key]
    if not cols:
        return pd.DataFrame(index=panel.index)
    out = pd.DataFrame(cols).sort_index()
    out.index = pd.DatetimeIndex(out.index)
    return out


def dx_series(panel: pd.DataFrame, *, field: str = "lev_net_oi") -> pd.Series:
    key = f"DX_{field}"
    if key not in panel.columns:
        return pd.Series(dtype=float, name="DX")
    s = panel[key].copy()
    s.name = "DX"
    return s


def cot_coverage_table(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    panel = panel if panel is not None else load_cot_panel(download=False)
    rows = []
    for ccy, code in FX_CONTRACT_CODES.items():
        for field in ("lev_net_oi", "nc_net_oi", "am_net_oi"):
            key = f"{ccy}_{field}"
            if key not in panel.columns:
                continue
            s = panel[key].dropna()
            rows.append(
                {
                    "ccy": ccy,
                    "code": code,
                    "field": field,
                    "n": int(len(s)),
                    "start": str(s.index.min()) if len(s) else None,
                    "end": str(s.index.max()) if len(s) else None,
                }
            )
    return pd.DataFrame(rows)
