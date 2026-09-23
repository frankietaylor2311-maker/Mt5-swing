"""Literature-style multi-factor FX combo: carry + momentum + dollar TSMOM.

Regime conditioning (fixed priors — not holdout-tuned):
- Menkhoff et al. (2012 JF): reduce **carry** exposure in high FX-vol / high
  uncertainty states (VIX used as free proxy for global FX vol).
- Caldara–Iacoviello (2022): on elevated **GPR**, cool carry further and apply
  a modest **USD safe-haven tilt** (Lustig–Roussanov–Verdelhan dollar factor
  load in risk-off).

Blend weights are equal a-priori (1/3 each sleeve) after each sleeve is built
with ``signal_lag`` and FRED/GPR publication lags from the loaders. Gross
exposure is never hiked (cool ≤ 1; tilt is a reallocation toward USD).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from mt5_swing.strategies.carry_rank import (
    CarryRankConfig,
    carry_weights_from_rates,
    currency_weights_to_pair_weights,
    expand_weights_to_daily,
    portfolio_returns_from_weights,
)
from mt5_swing.strategies.fx_momentum import (
    FxMomentumConfig,
    dollar_factor_returns,
    momentum_weights_from_returns,
)
from mt5_swing.strategies.gpr_regime import (
    GprRegimeConfig,
    USD_LONG_PAIRS,
    align_macro_to_index,
    regime_z,
    risk_scale_from_z,
    usd_tilt_weights,
)


@dataclass
class ScholarlyComboConfig:
    """Fixed literature priors — do not grid-search on holdout."""

    carry_n_long: int = 2
    carry_n_short: int = 2
    mom_formation_days: int = 63
    mom_skip_days: int = 21
    mom_n_long: int = 2
    mom_n_short: int = 2
    dollar_lookback: int = 21
    signal_lag: int = 1
    # Sleeve mix (sums to 1)
    w_carry: float = 1.0 / 3.0
    w_mom: float = 1.0 / 3.0
    w_dollar: float = 1.0 / 3.0
    # Regime: cool carry only (momentum/dollar keep scale unless usd_tilt)
    carry_cool: float = 0.35
    z_high: float = 1.0
    z_low: float = 0.0
    z_window: int = 252
    min_periods: int = 60
    # GPR-driven USD tilt intensity (max additive weight toward USD longs)
    usd_tilt: float = 0.15
    # Use max(z_gpr, z_vix) for carry cool; USD tilt keyed on GPR z only when available
    use_gpr: bool = True
    use_vix: bool = True
    use_epu: bool = True  # Baker–Bloom–Davis EPU in carry cool (fixed prior)


def _dollar_tsmom_pair_weights(
    pair_ret: pd.DataFrame,
    *,
    lookback: int,
    signal_lag: int,
) -> pd.DataFrame:
    """Map lagged sign of dollar factor → equal-weight foreign vs USD positions.

    Positive dollar factor = foreign currencies rising vs USD → long foreign
    (short USD). Negative → long USD. Weights sum abs ≈ 1 across available pairs.
    """
    dol = dollar_factor_returns(pair_ret)
    sig = np.sign(dol.rolling(int(lookback), min_periods=max(5, lookback // 2)).mean())
    sig = sig.shift(int(signal_lag))
    # Equal weight across foreign legs; direction = +sig on currency-vs-USD
    # EURUSD etc: long foreign when sig>0 → +1 on XXXUSD / -1 on USDXXX
    cols = [c for c in pair_ret.columns if c.upper() in {p.upper() for p in USD_LONG_PAIRS}]
    if not cols:
        cols = list(pair_ret.columns)
    n = max(len(cols), 1)
    w = pd.DataFrame(0.0, index=pair_ret.index, columns=cols)
    for sym in cols:
        # USD_LONG_PAIRS[sym] = +1 means long symbol = long USD
        # When dollar_tsmom sig > 0 (long foreign): want short USD → opposite of USD_LONG
        usd_long_sign = USD_LONG_PAIRS.get(sym.upper(), 0)
        # long foreign weight contribution on symbol = -usd_long_sign * (1/n) * sig
        # sig=+1 → foreign long → weight = -usd_long_sign / n
        w[sym] = (-usd_long_sign / n) * sig
    w = w.fillna(0.0)
    w.attrs["strategy"] = "dollar_tsmom"
    return w


def carry_cool_scale(
    gpr: pd.Series | None,
    vix: pd.Series | None,
    index: pd.DatetimeIndex,
    *,
    cfg: ScholarlyComboConfig,
    epu: pd.Series | None = None,
) -> pd.Series:
    """Lagged risk scale in [carry_cool, 1] from max(z_gpr, z_vix, z_epu)."""
    gcfg = GprRegimeConfig(
        z_window=cfg.z_window,
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.carry_cool,
        signal_lag=cfg.signal_lag,
        use_gpr=cfg.use_gpr,
        use_vix=cfg.use_vix,
        use_epu=getattr(cfg, "use_epu", True),
        usd_tilt=0.0,
        min_periods=cfg.min_periods,
    )
    z = regime_z(gpr, vix, index, cfg=gcfg, epu=epu)
    return risk_scale_from_z(z, cfg=gcfg)


def gpr_only_z(
    gpr: pd.Series | None,
    index: pd.DatetimeIndex,
    *,
    cfg: ScholarlyComboConfig,
) -> pd.Series:
    """Lagged GPR z for USD-tilt intensity (VIX does not drive tilt)."""
    gcfg = GprRegimeConfig(
        z_window=cfg.z_window,
        z_high=cfg.z_high,
        z_low=cfg.z_low,
        cool=cfg.carry_cool,
        signal_lag=cfg.signal_lag,
        use_gpr=True,
        use_vix=False,
        usd_tilt=cfg.usd_tilt,
        min_periods=cfg.min_periods,
    )
    if gpr is None or not len(gpr):
        return pd.Series(0.0, index=index, name="gpr_z")
    return regime_z(gpr, None, index, cfg=gcfg)


def build_combo_weights(
    rates: pd.DataFrame,
    pair_ret: pd.DataFrame,
    gpr: pd.Series | None,
    vix: pd.Series | None,
    *,
    cfg: ScholarlyComboConfig | None = None,
    epu: pd.Series | None = None,
) -> pd.DataFrame:
    """Daily pair weights: EW blend with carry cool + GPR USD tilt."""
    cfg = cfg or ScholarlyComboConfig()
    idx = pair_ret.index

    # --- Carry sleeve (month-end → daily, already signal_lag in expand) ---
    ccfg = CarryRankConfig(
        n_long=cfg.carry_n_long,
        n_short=cfg.carry_n_short,
        signal_lag=cfg.signal_lag,
    )
    ccy_w = carry_weights_from_rates(rates, cfg=ccfg)
    carry_pw = currency_weights_to_pair_weights(ccy_w)
    carry_d = expand_weights_to_daily(carry_pw, idx, signal_lag=cfg.signal_lag)

    # --- Momentum sleeve ---
    mcfg = FxMomentumConfig(
        formation_days=cfg.mom_formation_days,
        skip_days=cfg.mom_skip_days,
        n_long=cfg.mom_n_long,
        n_short=cfg.mom_n_short,
        signal_lag=cfg.signal_lag,
    )
    mom_ccy = momentum_weights_from_returns(pair_ret, cfg=mcfg)
    mom_pw = currency_weights_to_pair_weights(mom_ccy)
    mom_d = expand_weights_to_daily(mom_pw, idx, signal_lag=cfg.signal_lag)

    # --- Dollar TSMOM sleeve ---
    dol_d = _dollar_tsmom_pair_weights(
        pair_ret, lookback=cfg.dollar_lookback, signal_lag=cfg.signal_lag
    )

    # Align columns
    all_cols = sorted(
        set(carry_d.columns) | set(mom_d.columns) | set(dol_d.columns) | set(pair_ret.columns)
    )
    carry_d = carry_d.reindex(columns=all_cols, fill_value=0.0).reindex(idx).fillna(0.0)
    mom_d = mom_d.reindex(columns=all_cols, fill_value=0.0).reindex(idx).fillna(0.0)
    dol_d = dol_d.reindex(columns=all_cols, fill_value=0.0).reindex(idx).fillna(0.0)

    # Cool carry in high VIX / high GPR states (literature)
    cool = carry_cool_scale(gpr, vix, idx, cfg=cfg, epu=epu)
    # Extra one-bar lag so scale known before return realization (matches gpr_regime)
    cool_lag = cool.shift(1).fillna(1.0)
    carry_scaled = carry_d.mul(cool_lag, axis=0)

    # Blend
    blended = (
        cfg.w_carry * carry_scaled + cfg.w_mom * mom_d + cfg.w_dollar * dol_d
    )

    # USD tilt on GPR spikes (additive, then mild renormalize of gross)
    if cfg.usd_tilt > 0 and gpr is not None and len(gpr):
        gz = gpr_only_z(gpr, idx, cfg=cfg)
        gcfg_tilt = GprRegimeConfig(
            z_high=cfg.z_high,
            usd_tilt=cfg.usd_tilt,
            signal_lag=cfg.signal_lag,
            z_window=cfg.z_window,
            min_periods=cfg.min_periods,
        )
        tilt = usd_tilt_weights(gz, all_cols, cfg=gcfg_tilt)
        tilt = tilt.reindex(columns=all_cols, fill_value=0.0).reindex(idx).fillna(0.0)
        # lag tilt one bar (gz already signal_lagged)
        tilt = tilt.shift(1).fillna(0.0)
        blended = blended + tilt

    # Cap gross |w| ≤ 1 (no RF hike); preserve signs
    gross = blended.abs().sum(axis=1).replace(0.0, np.nan)
    scale_g = (1.0 / gross).clip(upper=1.0).fillna(1.0)
    out = blended.mul(scale_g, axis=0)
    out.attrs["strategy"] = "scholarly_combo"
    out.attrs["signal_lag"] = cfg.signal_lag
    out.attrs["w_carry"] = cfg.w_carry
    out.attrs["carry_cool"] = cfg.carry_cool
    out.attrs["usd_tilt"] = cfg.usd_tilt
    return out


def combo_portfolio_returns(
    rates: pd.DataFrame,
    pair_ret: pd.DataFrame,
    gpr: pd.Series | None,
    vix: pd.Series | None,
    *,
    cfg: ScholarlyComboConfig | None = None,
    epu: pd.Series | None = None,
) -> pd.Series:
    """Daily portfolio returns from scholarly combo weights."""
    cfg = cfg or ScholarlyComboConfig()
    w = build_combo_weights(rates, pair_ret, gpr, vix, cfg=cfg, epu=epu)
    common = [c for c in w.columns if c in pair_ret.columns]
    r = portfolio_returns_from_weights(w[common], pair_ret[common])
    r.name = "scholarly_combo"
    return r


def sleeve_returns_bundle(
    rates: pd.DataFrame,
    pair_ret: pd.DataFrame,
    gpr: pd.Series | None,
    vix: pd.Series | None,
    *,
    cfg: ScholarlyComboConfig | None = None,
    epu: pd.Series | None = None,
) -> dict[str, pd.Series]:
    """Individual sleeves + combo + variants for the stats board."""
    cfg = cfg or ScholarlyComboConfig()
    idx = pair_ret.index

    ccfg = CarryRankConfig(
        n_long=cfg.carry_n_long, n_short=cfg.carry_n_short, signal_lag=cfg.signal_lag
    )
    carry_pw = currency_weights_to_pair_weights(carry_weights_from_rates(rates, cfg=ccfg))
    carry_d = expand_weights_to_daily(carry_pw, idx, signal_lag=cfg.signal_lag)
    common_c = [c for c in carry_d.columns if c in pair_ret.columns]
    carry_r = portfolio_returns_from_weights(carry_d[common_c], pair_ret[common_c])
    carry_r.name = "carry_rank"

    mcfg = FxMomentumConfig(
        formation_days=cfg.mom_formation_days,
        skip_days=cfg.mom_skip_days,
        n_long=cfg.mom_n_long,
        n_short=cfg.mom_n_short,
        signal_lag=cfg.signal_lag,
    )
    mom_pw = currency_weights_to_pair_weights(momentum_weights_from_returns(pair_ret, cfg=mcfg))
    mom_d = expand_weights_to_daily(mom_pw, idx, signal_lag=cfg.signal_lag)
    common_m = [c for c in mom_d.columns if c in pair_ret.columns]
    mom_r = portfolio_returns_from_weights(mom_d[common_m], pair_ret[common_m])
    mom_r.name = "fx_momentum"

    dol_d = _dollar_tsmom_pair_weights(
        pair_ret, lookback=cfg.dollar_lookback, signal_lag=cfg.signal_lag
    )
    common_d = [c for c in dol_d.columns if c in pair_ret.columns]
    dol_r = portfolio_returns_from_weights(dol_d[common_d], pair_ret[common_d])
    dol_r.name = "dollar_tsmom"

    # Equal-weight return blend without regime (baseline multi-factor)
    blend_raw = (cfg.w_carry * carry_r + cfg.w_mom * mom_r + cfg.w_dollar * dol_r).rename(
        "combo_ew_raw"
    )

    combo = combo_portfolio_returns(rates, pair_ret, gpr, vix, cfg=cfg, epu=epu)

    # Carry alone with cool (no mom/dollar)
    cool = carry_cool_scale(gpr, vix, idx, cfg=cfg, epu=epu).shift(1).fillna(1.0)
    carry_cooled = (carry_r * cool).rename("carry_cooled")

    return {
        "carry_rank": carry_r,
        "fx_momentum": mom_r,
        "dollar_tsmom": dol_r,
        "combo_ew_raw": blend_raw,
        "carry_cooled": carry_cooled,
        "scholarly_combo": combo,
    }
