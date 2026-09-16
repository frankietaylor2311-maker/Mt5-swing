# Quest wave: intraday→swing + cross-asset stack + missing-month diversifiers

**Data:** `approximate_non_ftmo`. **Locked tag unchanged:** `fx4plus_gbpcad_d1_voltarget_0025`.
**Pairs Δz proxy:** abandoned (illusory).
**RF:** 8% fixed. **signal_lag=1**. IS selection; HO/2026 confirm only.

## Board (IS)

- Stack ideas: **10**; soft pass: **0**; hard pass: **0**
- Missing-month diversifiers OK: **1**
- Promote passers: **0**

### Best IS

- `baseline_locked|frozen1.0` min_mo=**0.42%** soft=False hard=False
- Legs: USDCHF:H4:bbands_reversion,GBPUSD:H4:breakout_donchian,CADJPY:H4:mean_reversion_regime,AUDCAD:H4:mean_reversion_regime,GBPCAD:D1:bbands_reversion

## Promote table (top 12)

```
                            idea  mo_2024  mo_2025   mo_2026     mo_ho  pos_2024   pos_ho  promote                 reason
            lock+dual2|frozen1.0 0.010255 0.010688  0.016861  0.012647  0.727273 0.750000    False           holdout_fail
       lock+dual2|is_scale_1.000 0.010255 0.010688  0.016861  0.012647  0.727273 0.750000    False           holdout_fail
            lock+dual3|frozen1.0 0.008642 0.009278  0.013383  0.009195  0.545455 0.583333    False 2024 mean_mo=0.86% <1%
       lock+dual3|is_scale_1.000 0.008642 0.009278  0.013383  0.009195  0.545455 0.583333    False 2024 mean_mo=0.86% <1%
            lock+dual4|frozen1.0 0.007305 0.008627  0.011371  0.007931  0.545455 0.583333    False 2024 mean_mo=0.73% <1%
       lock+dual4|is_scale_1.000 0.007305 0.008627  0.011371  0.007931  0.545455 0.583333    False 2024 mean_mo=0.73% <1%
         scratch_dual4|frozen1.0 0.005993 0.006914 -0.006002 -0.010250  0.636364 0.333333    False 2024 mean_mo=0.60% <1%
    scratch_dual4|is_scale_1.000 0.005993 0.006914 -0.006002 -0.010250  0.636364 0.333333    False 2024 mean_mo=0.60% <1%
       baseline_locked|frozen1.0 0.004217 0.016278  0.023011  0.015232  0.545455 0.750000    False 2024 mean_mo=0.42% <1%
  baseline_locked|is_scale_1.000 0.004217 0.016278  0.023011  0.015232  0.545455 0.750000    False 2024 mean_mo=0.42% <1%
                 baseline_locked 0.004217 0.016278  0.023011  0.015232  0.545455 0.750000    False 2024 mean_mo=0.42% <1%
missdiv::USDJPY_D1_hybrid_regime 0.003005 0.018019  0.020617  0.012326  0.636364 0.750000    False 2024 mean_mo=0.30% <1%
```

## Evidence / Yahoo limits

- Locked baseline 2024 mean_mo=0.42% pos=55% (missing months=5); 2025_IS mean_mo=2.33% pos=75% (missing=2).
- Yahoo 15m FX history is ~60 calendar days — cannot cover 2024/2025/2026 calendars; M15 is probe-only on approximate_non_ftmo. FTMO MT5 M15 export is required for multi-year intraday-to-swing.
- M15 landed for ['EURUSD', 'GBPUSD', 'USDJPY']: 5666 bars [2026-06-24 23:00:00+00:00 → 2026-09-16 17:15:00+00:00] — no 2024/2025 coverage.
- H1 swing screen: 60 legs; dual-year gate+positive: 0. Best min_mo=nan%.
- No dual-confirm metals (XAUUSD has H4 passers but no D1 dual) — skipped.
- Missing-month diversifiers screened=10; IS dual-year %pos improvers=1.
- Stack board: 10 ideas; soft IS=0; hard IS=0; promote passers=0.
- Best IS idea=baseline_locked|frozen1.0 min_mo=0.42% soft=False hard=False.
- H4 Yahoo ≈730d from 1h — cannot extend multi-year H4 history via yfinance.
- H1 Yahoo ≈730d from ~2023-11 — 2024 calendar partial at start; useful as diversity probe only.
- Pairs Δz proxy abandoned (illusory ~180× vs real two-leg; confirm_min ~0.30%/mo).
- No FTMO MT5 exports in data/ftmo/ — all approximate_non_ftmo; spreads/swap/sessions ≠ FTMO CFD.
- Clean ≥1% mean monthly on EACH of 2024,2025,2026,holdout with ≥70% pos not achieved under no-look-ahead / no RF hike / consistency-first on Yahoo approx.
- Required unlock: FTMO MT5 History Center exports (H1/M15/H4/D1 + indices if offered) into data/ftmo/ tagged ftmo_mt5_export.

**Promote?** **No** — official tag unchanged.

