# Quest — Scholarly FX §44 ACM term premium (THREEFYTP10)

**Date:** 2026-09-23 Europe/London (BST)
**Path:** NY Fed ACM Treasury term premium → USD risk-appetite (Adrian–Crump–Moench / Lustig–Stathopoulos–Verdelhan / Hofmann–Shim–Shin)
**Primary:** `acm_tp_usd` — long USD when lagged z(`THREEFYTP10`) ≥ 1.0
**Data tag:** `approximate_non_ftmo` (Yahoo D1; `data/ftmo/` has README only — no go-live)
**PIT:** daily `pub_lag_days=1` + `signal_lag=1`; z_window=252, min_periods=60, z_high=1.0; costs 1.5 bps/side
**Locked sleeve:** `fx4plus_gbpcad_d1_voltarget_0025` **untouched PASS** (see `quest_locked_verify_acm_tp.md`)

## Board

| Metric | Value |
|--------|------:|
| n factors | 8 |
| soft_nw_pos (|t|≥1.5 & mean>0) | 0 |
| hard_nw_pos (|t|≥2.0 & mean>0) | 0 |
| promote | 0 |

## Primary `acm_tp_usd`

| Window | mean%/mo | %pos | gates |
|--------|--------:|-----:|:-----:|
| full-sample | +0.007% | 22% | PASS |
| 2024 | +0.14% | 36% | PASS |
| 2025 | −0.08% | 0% | PASS |
| 2026 | +0.03% | 62% | PASS |
| holdout_365d | −0.13% | 42% | PASS |

Full-sample NW t ≈ **+0.20**.

## Risk sweep (prop challenge — push near 10% static / 5% daily)

| Item | Value |
|------|------:|
| scale | ≈3.727 |
| binding | daily |
| scaled HO mean%/mo | ≈−0.48% |
| scaled HO %pos | 42% |
| scaled HO clears ~1%/mo | **NO** |

## Soft-best-ish

`acm_tp_lvl_usd` ≈ **+4.0 bp/mo**, NW t ≈ +1.05 (below soft |t|≥1.5 gate). `acm_tp5_usd` similar (~+4.0 bp/mo, NW t ≈ +1.12).

## Series

- `THREEFYTP10` live ~1990-01 → 2026-09 (~9165 rows after pub lag)
- `THREEFYTP5` companion live (same window)
- `ACMTP10` **404** — skipped

## Distinct from

Yield-curve slope §16 · real-rate/TIPS §23 · funding-liq §20 · IG OAS §36 · building-permits §43

## Verdict

ACM term premium is the right free-data *Treasury duration-risk / USD risk-appetite* structure after housing-activity §43, and is cleanly distinct from slope / TIPS / NFCI / corporate OAS. On Yahoo D1 G10 the primary USD-haven tilt is essentially **flat** (full-sample ≈ +0.7 bp/mo, NW t ≈ +0.20, sparse activity %pos 22%) — nowhere near 1%/mo + 70% hit-rate. Soft/hard NW boards empty. No go-live claim under `approximate_non_ftmo`.

## Next

FTMO MT5 CSV re-score when exports arrive; or country-GPR bilateral refinement; or live OECD retail if any free series found (SARTMISMEI stale). FX IV/RR + news NLP still **blocked**.

## Artifacts

- `reports/scholarly_fx_acm_tp_wave.md`
- `reports/scholarly_fx_acm_tp_factor_summary.csv`
- `reports/scholarly_fx_acm_tp_window_stats.csv`
- `reports/scholarly_fx_acm_tp_risk_sweep.csv`
- `reports/scholarly_fx_acm_tp_meta.json`
- `reports/quest_locked_verify_acm_tp.md`
