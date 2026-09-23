"""Caldara–Iacoviello AI-GPR / role decompositions loader.

Source: ``data/macro/ai_gpr_daily.csv`` (AI newspaper GPR, threats/acts,
oil vs non-oil, oil-region roles). Distinct from classic daily GPRD and from
monthly ``GPRC_*`` country sorts.

PIT: ``pub_lag_days=1`` shifts the calendar date forward so day-t is known only
from t+1 onward (match other daily macro waves).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

AI_GPR_ROLE_COLS: tuple[str, ...] = (
    "GPR_AI",
    "GPR_AER",
    "GPR_OIL",
    "GPR_NONOIL",
    "THREATS_GPR_AI",
    "ACTS_GPR_AI",
    "GPR_OIL_THREATS",
    "GPR_OIL_ACTS",
    "GPR_OIL_MiddleEast",
    "GPR_OIL_Russia",
    "GPR_OIL_USA",
    "GPR_OIL_Venezuela",
    "GPR_OIL_Africa",
    "GPR_OIL_Americas",
    "GPR_OIL_Asia",
    "GPR_OIL_NorthSea",
)

OIL_STRESS_REGIONS: tuple[str, ...] = (
    "GPR_OIL_MiddleEast",
    "GPR_OIL_Russia",
    "GPR_OIL_Africa",
    "GPR_OIL_Americas",
    "GPR_OIL_NorthSea",
)

OIL_REGION_FX_NOTES: dict[str, str] = {
    "GPR_OIL_MiddleEast": "oil-supply risk; CAD proxy (no MENA FX on book)",
    "GPR_OIL_Russia": "oil/gas supply risk; CAD proxy (no RUB)",
    "GPR_OIL_USA": "US oil disruption; USD / energy channel",
    "GPR_OIL_Venezuela": "LatAm oil risk; CAD/Americas proxy",
    "GPR_OIL_Africa": "African oil-supply risk; CAD proxy",
    "GPR_OIL_Americas": "W Hemisphere oil risk → CAD",
    "GPR_OIL_Asia": "Asian oil geo; not mapped to JPY/AUD sort",
    "GPR_OIL_NorthSea": "NOK natural map — untraded (no USDNOK); enters stress EW only",
}


def macro_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "macro"


def ai_gpr_csv_path() -> Path:
    return macro_dir() / "ai_gpr_daily.csv"


def load_ai_gpr_panel(
    *,
    pub_lag_days: int = 1,
    columns: Iterable[str] | None = None,
    path: Path | None = None,
) -> pd.DataFrame:
    """Load AI-GPR daily panel with publication lag on the index."""
    p = Path(path) if path is not None else ai_gpr_csv_path()
    if not p.exists():
        raise FileNotFoundError(
            f"AI-GPR daily CSV missing at {p}. Place Caldara–Iacoviello "
            "ai_gpr_daily.csv under data/macro/ (Date + role columns)."
        )
    df = pd.read_csv(p)
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    idx = pd.to_datetime(df[date_col], utc=True)
    want = list(columns) if columns is not None else list(AI_GPR_ROLE_COLS)
    keep = [c for c in want if c in df.columns]
    if not keep:
        raise ValueError(f"None of requested AI-GPR columns present in {p}: {want}")
    out = df[keep].apply(pd.to_numeric, errors="coerce")
    out.index = idx
    out = out[~out.index.duplicated(keep="last")].sort_index()
    if pub_lag_days > 0:
        out.index = out.index + pd.Timedelta(days=int(pub_lag_days))
        out = out[~out.index.duplicated(keep="last")].sort_index()
    out.attrs["source"] = "ai_gpr_daily"
    out.attrs["pub_lag_days"] = int(pub_lag_days)
    out.attrs["path"] = str(p)
    out.attrs["distinct_from"] = (
        "classic_keyword_GPR_daily",
        "monthly_GPRC_country_indexes",
        "gpr_regime_aggregate_stress",
    )
    return out


def load_ai_gpr_series(
    col: str,
    *,
    pub_lag_days: int = 1,
    path: Path | None = None,
) -> pd.Series:
    """Single AI-GPR role series with publication lag."""
    panel = load_ai_gpr_panel(pub_lag_days=pub_lag_days, columns=[col], path=path)
    if col not in panel.columns:
        raise KeyError(f"AI-GPR column {col!r} not in panel")
    s = panel[col].dropna()
    s.name = col
    s.attrs["source"] = "ai_gpr_daily"
    s.attrs["pub_lag_days"] = int(pub_lag_days)
    return s


def oil_me_vs_non_series(
    panel: pd.DataFrame | None = None,
    *,
    pub_lag_days: int = 1,
) -> pd.Series:
    """MiddleEast oil-GPR minus non-oil GPR (bilateral oil-region role diff)."""
    if panel is None:
        panel = load_ai_gpr_panel(
            pub_lag_days=pub_lag_days,
            columns=("GPR_OIL_MiddleEast", "GPR_NONOIL"),
        )
    if "GPR_OIL_MiddleEast" not in panel.columns or "GPR_NONOIL" not in panel.columns:
        raise KeyError("Need GPR_OIL_MiddleEast and GPR_NONOIL for oil_me_vs_non")
    s = panel["GPR_OIL_MiddleEast"] - panel["GPR_NONOIL"]
    s.name = "OIL_ME_VS_NON"
    s.attrs["pub_lag_days"] = int(panel.attrs.get("pub_lag_days", pub_lag_days))
    return s


def build_role_series(panel: pd.DataFrame) -> dict[str, pd.Series]:
    """Frozen role constructions from a pub-lagged AI-GPR panel (pre signal_lag)."""
    out: dict[str, pd.Series] = {}
    if "THREATS_GPR_AI" in panel.columns and "ACTS_GPR_AI" in panel.columns:
        diff = panel["THREATS_GPR_AI"] - panel["ACTS_GPR_AI"]
        diff.name = "threat_minus_act"
        out["threat_minus_act"] = diff
        out["threats"] = panel["THREATS_GPR_AI"].rename("threats")
        out["acts"] = panel["ACTS_GPR_AI"].rename("acts")
    if "GPR_OIL" in panel.columns:
        out["oil"] = panel["GPR_OIL"].rename("oil")
    if "GPR_NONOIL" in panel.columns and "GPR_OIL" in panel.columns:
        ovs = panel["GPR_OIL"] - panel["GPR_NONOIL"]
        ovs.name = "oil_minus_nonoil"
        out["oil_minus_nonoil"] = ovs
    if "GPR_OIL_THREATS" in panel.columns and "GPR_OIL_ACTS" in panel.columns:
        od = panel["GPR_OIL_THREATS"] - panel["GPR_OIL_ACTS"]
        od.name = "oil_threat_minus_act"
        out["oil_threat_minus_act"] = od
    regs = [c for c in OIL_STRESS_REGIONS if c in panel.columns]
    if regs:
        stress = panel[regs].astype(float).mean(axis=1)
        stress.name = "oil_region_stress"
        out["oil_region_stress"] = stress
    if "GPR_AI" in panel.columns:
        out["gpr_ai"] = panel["GPR_AI"].rename("gpr_ai")
    if "GPR_OIL_MiddleEast" in panel.columns and "GPR_NONOIL" in panel.columns:
        me = panel["GPR_OIL_MiddleEast"] - panel["GPR_NONOIL"]
        me.name = "oil_me_vs_non"
        out["oil_me_vs_non"] = me
    return out


__all__ = [
    "AI_GPR_ROLE_COLS",
    "OIL_REGION_FX_NOTES",
    "OIL_STRESS_REGIONS",
    "ai_gpr_csv_path",
    "build_role_series",
    "load_ai_gpr_panel",
    "load_ai_gpr_series",
    "macro_dir",
    "oil_me_vs_non_series",
]
