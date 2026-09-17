"""Country-level Caldara–Iacoviello GPR → FX depreciation tests.

Hypothesis (literature-aligned, fixed prior — not holdout-tuned):
  Elevated *home-country* geopolitical risk is associated with depreciation of
  that currency versus the USD (capital flight / risk premium), after a
  publication lag.

Tests implemented
-----------------
1. **Lagged sort portfolio:** each month, rank currencies by lagged home GPR
   (z-score or level); long low-GPR / short high-GPR vs USD (equal-weight sides).
2. **Local-projection style:** for each currency, regress cumulative h-month
   FX returns on lagged home GPR z; report β and OLS t-stat (pooled and by ccy).

All signals use ``signal_lag`` on top of the loader's ``pub_lag_months``.
No technical overlays.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.data.macro_uncertainty import CURRENCY_USD_PAIR


@dataclass
class CountryGprFxConfig:
    """Fixed priors for country-GPR FX study."""

    signal_lag: int = 1  # extra months after publication-lagged GPR is known
    n_long: int = 2
    n_short: int = 2
    z_window: int = 60  # months for trailing z-score of country GPR
    min_periods: int = 24
    lp_horizons: tuple[int, ...] = (1, 3, 6)  # months
    use_zscore: bool = True


def foreign_vs_usd_returns(pair_ret: pd.DataFrame) -> pd.DataFrame:
    """Daily returns of foreign currency vs USD (positive = foreign appreciates)."""
    out = {}
    for ccy, (pair, sign) in CURRENCY_USD_PAIR.items():
        if pair not in pair_ret.columns:
            continue
        out[ccy] = sign * pair_ret[pair]
    return pd.DataFrame(out, index=pair_ret.index).sort_index()


def monthly_fx_returns(pair_ret: pd.DataFrame) -> pd.DataFrame:
    """Month-end foreign-vs-USD returns from daily pair returns."""
    fx = foreign_vs_usd_returns(pair_ret)
    if fx.empty:
        return fx
    eq = (1.0 + fx.fillna(0.0)).cumprod()
    return eq.resample("ME").last().pct_change()


def _align_month_start(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    naive = idx.tz_convert(None) if idx.tz is not None else idx
    return pd.DatetimeIndex(naive.to_period("M").to_timestamp(how="start"), tz="UTC")


def prepare_country_gpr_signal(
    country_gpr: pd.DataFrame,
    *,
    cfg: CountryGprFxConfig | None = None,
) -> pd.DataFrame:
    """Publication-lagged country GPR → optional z-score → signal_lag months."""
    cfg = cfg or CountryGprFxConfig()
    g = country_gpr.copy()
    g.index = _align_month_start(pd.DatetimeIndex(g.index))
    g = g[~g.index.duplicated(keep="last")].sort_index()
    if cfg.use_zscore:
        mu = g.rolling(cfg.z_window, min_periods=cfg.min_periods).mean()
        sd = g.rolling(cfg.z_window, min_periods=cfg.min_periods).std()
        sig = (g - mu) / sd.replace(0.0, np.nan)
    else:
        sig = g
    if cfg.signal_lag > 0:
        sig = sig.shift(int(cfg.signal_lag))
    sig.attrs["signal_lag"] = int(cfg.signal_lag)
    sig.attrs["use_zscore"] = bool(cfg.use_zscore)
    return sig


def country_gpr_sort_weights(
    gpr_signal: pd.DataFrame,
    *,
    cfg: CountryGprFxConfig | None = None,
    exclude: tuple[str, ...] = ("USD",),
) -> pd.DataFrame:
    """Monthly currency weights: long low home-GPR, short high home-GPR.

    Hypothesis: high home GPR → depreciate → we *short* high-GPR currencies.
    Weights sum to ≈0; each side |sum| = 0.5.
    """
    cfg = cfg or CountryGprFxConfig()
    cols = [c for c in gpr_signal.columns if c.upper() not in {x.upper() for x in exclude}]
    panel = gpr_signal[cols]
    rows = []
    for dt, row in panel.iterrows():
        s = row.dropna()
        if len(s) < cfg.n_long + cfg.n_short:
            continue
        ranked = s.sort_values()  # low GPR first
        longs = ranked.index[: cfg.n_long]
        shorts = ranked.index[-cfg.n_short :]
        w = pd.Series(0.0, index=cols)
        w.loc[list(longs)] = 0.5 / cfg.n_long
        w.loc[list(shorts)] = -0.5 / cfg.n_short
        w.name = dt
        rows.append(w)
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows).sort_index()


def currency_weights_to_pair_weights_fx(ccy_w: pd.DataFrame) -> pd.DataFrame:
    """Map currency weights (long foreign > 0) onto USD-major pair weights."""
    pairs = sorted({CURRENCY_USD_PAIR[c][0] for c in ccy_w.columns if c in CURRENCY_USD_PAIR})
    out = pd.DataFrame(0.0, index=ccy_w.index, columns=pairs)
    for ccy in ccy_w.columns:
        if ccy not in CURRENCY_USD_PAIR:
            continue
        pair, sign = CURRENCY_USD_PAIR[ccy]
        # long foreign (w>0): for XXXUSD (sign=+1) → +pair; for USDXXX (sign=-1) → -pair
        out[pair] = out[pair] + ccy_w[ccy] * sign
    return out


def expand_monthly_weights_to_daily(
    monthly_w: pd.DataFrame,
    daily_index: pd.DatetimeIndex,
    *,
    signal_lag_days: int = 1,
) -> pd.DataFrame:
    """Hold month-end weights through the next month; shift by signal_lag_days."""
    if monthly_w.empty:
        return pd.DataFrame(0.0, index=daily_index, columns=monthly_w.columns)
    w = monthly_w.copy()
    w.index = pd.DatetimeIndex(w.index)
    if w.index.tz is None:
        w.index = w.index.tz_localize("UTC")
    else:
        w.index = w.index.tz_convert("UTC")
    idx = daily_index.tz_convert("UTC") if daily_index.tz is not None else daily_index.tz_localize("UTC")
    daily = w.reindex(idx, method="ffill")
    if signal_lag_days > 0:
        daily = daily.shift(int(signal_lag_days))
    return daily.fillna(0.0)


def portfolio_returns_from_pair_weights(
    daily_w: pd.DataFrame,
    pair_ret: pd.DataFrame,
) -> pd.Series:
    cols = [c for c in daily_w.columns if c in pair_ret.columns]
    if not cols:
        return pd.Series(0.0, index=pair_ret.index, name="country_gpr_sort")
    r = (daily_w[cols] * pair_ret[cols]).sum(axis=1)
    r.name = "country_gpr_sort"
    return r


def country_gpr_sort_returns(
    country_gpr: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: CountryGprFxConfig | None = None,
) -> pd.Series:
    """End-to-end lagged sort: high home GPR → short that FX vs USD."""
    cfg = cfg or CountryGprFxConfig()
    sig = prepare_country_gpr_signal(country_gpr, cfg=cfg)
    # Align signal month index to month-end for weight application
    ccy_w = country_gpr_sort_weights(sig, cfg=cfg)
    if ccy_w.empty:
        return pd.Series(dtype=float, name="country_gpr_sort")
    # Use month-end timestamps for expand
    ccy_w.index = (
        pd.DatetimeIndex(ccy_w.index)
        .tz_convert(None)
        .to_period("M")
        .to_timestamp(how="end")
        .tz_localize("UTC")
    )
    pair_w = currency_weights_to_pair_weights_fx(ccy_w)
    daily_w = expand_monthly_weights_to_daily(pair_w, pair_ret.index, signal_lag_days=1)
    return portfolio_returns_from_pair_weights(daily_w, pair_ret)


def local_projection_beta(
    gpr_signal: pd.Series,
    fx_monthly: pd.Series,
    *,
    horizon: int = 1,
) -> dict:
    """Simple LP: cumret_{t→t+h} = a + b * GPR_signal_t + e.

    Returns dict with beta, tstat, n, r2. Uses OLS with iid SE (honest small-sample).
    """
    g = gpr_signal.copy()
    r = fx_monthly.copy()
    g.index = _align_month_start(pd.DatetimeIndex(g.index))
    r.index = _align_month_start(pd.DatetimeIndex(r.index))
    # cumulative h-month return from t+1 .. t+h (forward, after signal at t)
    fwd = pd.Series(index=r.index, dtype=float)
    for i in range(len(r) - horizon):
        fwd.iloc[i] = float(r.iloc[i + 1 : i + 1 + horizon].sum())
    df = pd.concat([g.rename("x"), fwd.rename("y")], axis=1).dropna()
    if len(df) < 24:
        return {
            "horizon": horizon,
            "beta": float("nan"),
            "tstat": float("nan"),
            "n": int(len(df)),
            "r2": float("nan"),
            "mean_y": float("nan"),
        }
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    x1 = np.column_stack([np.ones(len(x)), x])
    coef, _, _, _ = np.linalg.lstsq(x1, y, rcond=None)
    yhat = x1 @ coef
    resid = y - yhat
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y - y.mean(), y - y.mean()))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    # iid SE
    n, k = len(y), 2
    sigma2 = ss_res / max(n - k, 1)
    xtx_inv = np.linalg.inv(x1.T @ x1)
    se_b = float(np.sqrt(sigma2 * xtx_inv[1, 1]))
    tstat = float(coef[1] / se_b) if se_b > 0 else float("nan")
    return {
        "horizon": int(horizon),
        "beta": float(coef[1]),
        "alpha": float(coef[0]),
        "tstat": tstat,
        "n": int(n),
        "r2": float(r2),
        "mean_y": float(y.mean()),
    }


def local_projection_panel(
    country_gpr: pd.DataFrame,
    pair_ret: pd.DataFrame,
    *,
    cfg: CountryGprFxConfig | None = None,
) -> pd.DataFrame:
    """Per-currency and pooled LP of FX returns on lagged home GPR z.

    Sign: beta < 0 supports high home GPR → depreciation vs USD.
    """
    cfg = cfg or CountryGprFxConfig()
    sig = prepare_country_gpr_signal(country_gpr, cfg=cfg)
    fx_m = monthly_fx_returns(pair_ret)
    rows = []
    common = [c for c in sig.columns if c in fx_m.columns and c != "USD"]
    for h in cfg.lp_horizons:
        # per currency
        betas = []
        for ccy in common:
            res = local_projection_beta(sig[ccy], fx_m[ccy], horizon=h)
            res["currency"] = ccy
            res["scope"] = "currency"
            rows.append(res)
            if np.isfinite(res["beta"]):
                betas.append(res["beta"])
        # pooled (stack)
        xs, ys = [], []
        for ccy in common:
            g = sig[ccy].copy()
            r = fx_m[ccy].copy()
            g.index = _align_month_start(pd.DatetimeIndex(g.index))
            r.index = _align_month_start(pd.DatetimeIndex(r.index))
            fwd = pd.Series(index=r.index, dtype=float)
            for i in range(len(r) - h):
                fwd.iloc[i] = float(r.iloc[i + 1 : i + 1 + h].sum())
            df = pd.concat([g.rename("x"), fwd.rename("y")], axis=1).dropna()
            xs.append(df["x"].to_numpy())
            ys.append(df["y"].to_numpy())
        if xs:
            x = np.concatenate(xs)
            y = np.concatenate(ys)
            if len(x) >= 24:
                x1 = np.column_stack([np.ones(len(x)), x])
                coef, _, _, _ = np.linalg.lstsq(x1, y, rcond=None)
                resid = y - x1 @ coef
                n, k = len(y), 2
                sigma2 = float(np.dot(resid, resid) / max(n - k, 1))
                se_b = float(np.sqrt(sigma2 * np.linalg.inv(x1.T @ x1)[1, 1]))
                rows.append(
                    {
                        "horizon": h,
                        "currency": "POOLED",
                        "scope": "pooled",
                        "beta": float(coef[1]),
                        "alpha": float(coef[0]),
                        "tstat": float(coef[1] / se_b) if se_b > 0 else float("nan"),
                        "n": int(n),
                        "r2": float(
                            1.0
                            - np.dot(resid, resid) / max(np.dot(y - y.mean(), y - y.mean()), 1e-18)
                        ),
                        "mean_y": float(y.mean()),
                    }
                )
    return pd.DataFrame(rows)
