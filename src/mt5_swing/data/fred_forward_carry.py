"""Point-in-time FRED / OECD money-market rates → CIP-implied forward discounts.

Classic FX carry sorts on *forward discounts* (Lustig–Roussanov–Verdelhan 2011;
Lustig–Verdelhan 2007). Under covered-interest parity (CIP),

    F/S ≈ (1 + i_quote) / (1 + i_base)

so the forward discount of the foreign currency vs USD is approximately the
interest differential. True FX swap / outright forward points are vendor data
(Bloomberg, Refinitiv, broker feeds). Free FRED OECD panels are **cash-rate /
money-market approximations**, not observed forward points.

Series hierarchy (best free proxy first for 1–3M carry horizon)
----------------------------------------------------------------
1. ``IR3TIB01*`` — OECD 3-month interest rates (money-market / IBOR-style).
   Closest free match to the tenor of academic 1M–3M forwards.
2. ``IRSTCI01*`` — OECD immediate / policy short rates (cash-rate carry used
   in earlier waves). Good for cross-section, coarser tenor match.
3. Daily overnight (``DFF``, ``ECBDFR``, ``IUDSOIA``) — sparse G10 coverage;
   documented only, not used as primary panel.

Honesty
-------
- CIP-implied FD from IR3M ≠ broker FX swap points (post-GFC CIP basis exists).
- Rank order of IR3M differentials vs exact CIP FD is nearly identical for G10;
  we still expose the exact discrete CIP formula for auditability.
- Publication lag default: 1 month (OECD MEI).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from mt5_swing.data.fred_rates import (
    FRED_IMMEDIATE_MONTHLY,
    FRED_OVERNIGHT_DAILY,
    apply_publication_lag,
    download_fred_series,
    load_currency_rates,
    load_fred_series,
    rate_differentials_vs_usd,
)
from mt5_swing.data.fred_yields import FRED_IR3M_MONTHLY, load_ir3m_panel, _to_month_start_utc


def cip_implied_forward_discount_vs_usd(
    rates_pct: pd.DataFrame,
    *,
    tenor_months: int = 3,
) -> pd.DataFrame:
    """CIP-implied foreign-vs-USD forward discount from money-market rates (% p.a.).

    For foreign currency ``c`` with annualised rate ``i_c`` (%), USD rate ``i_USD``,

        (1 + i_c/100)^(τ) / (1 + i_USD/100)^(τ) − 1

    with ``τ = tenor_months/12``, returned in **percent** (×100) so magnitudes are
    comparable to rate differentials. Positive ⇒ foreign rates above USD ⇒
    classic long-foreign carry score (same sign as ``i_c − i_USD``).

    This is still a *rate-implied* discount, **not** an observed FX forward point.
    """
    if "USD" not in rates_pct.columns:
        raise ValueError("rates panel must include USD")
    if tenor_months <= 0:
        raise ValueError("tenor_months must be positive")
    tau = float(tenor_months) / 12.0
    usd = rates_pct["USD"].astype(float) / 100.0
    out_cols = {}
    for c in rates_pct.columns:
        if c == "USD":
            continue
        i_c = rates_pct[c].astype(float) / 100.0
        # Exact discrete CIP ratio minus 1, scaled to percent
        fd = ((1.0 + i_c) ** tau) / ((1.0 + usd) ** tau) - 1.0
        out_cols[c] = fd * 100.0
    out = pd.DataFrame(out_cols, index=rates_pct.index)
    out.attrs["definition"] = "cip_implied_fd_pct_vs_usd"
    out.attrs["tenor_months"] = int(tenor_months)
    out.attrs["note"] = (
        "Rate-implied CIP forward discount from money-market / cash rates; "
        "NOT observed FX swap or outright forward points"
    )
    return out


def rate_diff_vs_usd(rates: pd.DataFrame) -> pd.DataFrame:
    """``i_ccy − i_USD`` (% p.a.) — linear carry score."""
    return rate_differentials_vs_usd(rates)


def load_forward_carry_panels(
    currencies: Iterable[str] | None = None,
    *,
    pub_lag_months: int = 1,
    download: bool = True,
    cip_tenor_months: int = 3,
) -> dict[str, pd.DataFrame]:
    """Load IRSTCI + IR3M panels and CIP-implied FD differentials.

    Returns keys:
      ``irstci``, ``ir3m``,
      ``irstci_diff``, ``ir3m_diff``,
      ``cip_fd_ir3m``, ``cip_fd_irstci``
    """
    curs = [c.upper() for c in (currencies or FRED_IR3M_MONTHLY.keys())]
    if "USD" not in curs:
        curs = ["USD"] + [c for c in curs if c != "USD"]

    irstci = load_currency_rates(
        [c for c in curs if c in FRED_IMMEDIATE_MONTHLY or c == "USD"],
        freq="M",
        pub_lag_months=pub_lag_months,
        download=download,
    )
    ir3m = load_ir3m_panel(curs, pub_lag_months=pub_lag_months, download=download)

    common = sorted(set(irstci.columns) & set(ir3m.columns))
    if "USD" not in common:
        raise RuntimeError("USD missing from IRSTCI or IR3M — cannot form carry diffs")
    # Align on intersection of dates with both panels
    idx = irstci.dropna(how="all").index.intersection(ir3m.dropna(how="all").index)
    irstci = irstci.reindex(idx)[common]
    ir3m = ir3m.reindex(idx)[common]

    irstci_diff = rate_diff_vs_usd(irstci)
    ir3m_diff = rate_diff_vs_usd(ir3m)
    cip_fd_ir3m = cip_implied_forward_discount_vs_usd(ir3m, tenor_months=cip_tenor_months)
    cip_fd_irstci = cip_implied_forward_discount_vs_usd(irstci, tenor_months=1)

    return {
        "irstci": irstci,
        "ir3m": ir3m,
        "irstci_diff": irstci_diff,
        "ir3m_diff": ir3m_diff,
        "cip_fd_ir3m": cip_fd_ir3m,
        "cip_fd_irstci": cip_fd_irstci,
    }


def forward_carry_coverage(
    *,
    download: bool = False,
    asof: str | None = None,
) -> pd.DataFrame:
    """Document IRSTCI / IR3M / overnight coverage for G10 forward-proxy carry."""
    asof_ts = (pd.Timestamp(asof, tz="UTC") if asof else pd.Timestamp.now("UTC")).tz_convert(None)
    rows = []
    for ccy in sorted(set(FRED_IMMEDIATE_MONTHLY) | set(FRED_IR3M_MONTHLY)):
        row: dict = {
            "currency": ccy,
            "irstci_series": FRED_IMMEDIATE_MONTHLY.get(ccy),
            "ir3m_series": FRED_IR3M_MONTHLY.get(ccy),
            "overnight_series": FRED_OVERNIGHT_DAILY.get(ccy),
        }
        for label, sid in (
            ("irstci", FRED_IMMEDIATE_MONTHLY.get(ccy)),
            ("ir3m", FRED_IR3M_MONTHLY.get(ccy)),
        ):
            if not sid:
                row[f"{label}_n"] = 0
                row[f"{label}_end"] = None
                row[f"{label}_stale"] = True
                continue
            try:
                s = load_fred_series(sid, download=download)
                end = s.index.max()
                end_n = end.tz_convert(None) if getattr(end, "tz", None) else end
                months_lag = (asof_ts.to_period("M") - pd.Timestamp(end_n).to_period("M")).n
                row[f"{label}_n"] = int(s.notna().sum())
                row[f"{label}_end"] = str(s.index.max().date())
                row[f"{label}_months_since_end"] = int(months_lag)
                row[f"{label}_stale"] = bool(months_lag > 4)
            except Exception as exc:  # noqa: BLE001
                row[f"{label}_n"] = 0
                row[f"{label}_end"] = None
                row[f"{label}_stale"] = True
                row[f"{label}_err"] = str(exc)[:80]
        note_parts = []
        if ccy in FRED_OVERNIGHT_DAILY:
            note_parts.append(f"daily ON={FRED_OVERNIGHT_DAILY[ccy]} (sparse; not primary)")
        if not FRED_IR3M_MONTHLY.get(ccy):
            note_parts.append("no IR3M — cash IRSTCI only")
        row["note"] = "; ".join(note_parts)
        row["approx_level"] = (
            "money_market_ir3m"
            if FRED_IR3M_MONTHLY.get(ccy)
            else "cash_immediate_only"
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("currency").reset_index(drop=True)


def download_forward_carry_panel(*, force: bool = False) -> dict[str, object]:
    """Ensure IRSTCI + IR3M FRED CSVs exist under ``data/macro/``."""
    paths = {}
    for sid in set(FRED_IMMEDIATE_MONTHLY.values()) | set(FRED_IR3M_MONTHLY.values()):
        paths[sid] = download_fred_series(sid, force=force)
    return paths


def rank_corr_irstci_vs_ir3m(panels: dict[str, pd.DataFrame] | None = None) -> float:
    """Mean cross-sectional Spearman rank correlation of IRSTCI vs IR3M diffs."""
    if panels is None:
        panels = load_forward_carry_panels(download=False)
    a = panels["irstci_diff"]
    b = panels["ir3m_diff"]
    common_cols = sorted(set(a.columns) & set(b.columns))
    common_idx = a.index.intersection(b.index)
    if not common_cols or len(common_idx) < 12:
        return float("nan")
    cors = []
    for dt in common_idx:
        x = a.loc[dt, common_cols]
        y = b.loc[dt, common_cols]
        mask = x.notna() & y.notna()
        if int(mask.sum()) < 4:
            continue
        # Rank+Pearson (no scipy)
        rx, ry = x[mask].rank(), y[mask].rank()
        if float(rx.std(ddof=0)) == 0.0 or float(ry.std(ddof=0)) == 0.0:
            continue
        cors.append(float(rx.corr(ry, method="pearson")))
    return float(np.nanmean(cors)) if cors else float("nan")
