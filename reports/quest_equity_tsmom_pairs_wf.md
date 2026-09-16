# Equity TSMOM + pairs residual WF

**data_source:** `approximate_non_ftmo`  
**RF:** 0.08 (no hike)  
**Locked:** `fx4plus_gbpcad_d1_voltarget_0025` unchanged unless promote.  
**Candidates:** 204 | soft_pass=118 | hard_pass=0  
**Promote:** **NO**

## History expand

- AUDCAD_D1:keep:3909
- AUDUSD_D1:keep:3904
- EURAUD_D1:keep:3907
- EURCHF_D1:keep:3905
- EURJPY_D1:keep:3907
- EURUSD_D1:keep:3904
- GBPJPY_D1:keep:3906
- GBPUSD_D1:keep:3905
- NZDCAD_D1:keep:3906
- NZDUSD_D1:keep:3905
- USDCAD_D1:keep:3904
- USDCHF_D1:keep:3902
- skipped_locked_d1:GBPCAD

H4 Yahoo 1h cap (~730d) unchanged — cannot add more H4 years via yfinance.

## Best IS soft (selection; holdout excluded)

- `pairs_n3` score=0.0087 min_mo=1.01% max_top3=62% min_pos=75%

## Best raw IS min(mean_mo) (evidence)

- `pairs_n3` min_mo=1.05% max_top3=72% min_pos=75% soft=False

## Confirmation table

| Idea | 2024 mo/%pos | 2025 | 2026 | HO mo/%pos | Promote | Reason |
|------|-------------:|-----:|-----:|-----------:|:-------:|--------|
| `pairs_n3` | 1.35/100 | 1.12/91 | 0.88/88 | 0.95/92 | no | holdout_fail|years_fail |
| `pairs_n4` | 1.79/100 | 1.01/100 | 1.07/100 | 0.99/100 | no | holdout_fail |
| `pairs_n3` | 1.18/82 | 1.08/82 | 0.47/62 | 0.67/75 | no | holdout_fail|years_fail |
| `pairs_n2` | 1.51/91 | 0.97/100 | 1.08/100 | 0.98/100 | no | holdout_fail|years_fail |
| `pairs_n2` | 1.47/100 | 1.06/91 | 0.98/88 | 1.06/92 | no | years_fail |
| `pairs_n3` | 1.37/100 | 1.09/91 | 1.02/100 | 0.96/92 | no | is_soft_fail|holdout_fail |
| `pairs_n3` | 1.34/91 | 1.01/100 | 0.97/100 | 0.86/92 | no | is_soft_fail|holdout_fail|years_fail |


## Important: pairs Δz proxy vs real fills

IS/confirm tables above for `pairs_*` use a **residual Δz → return proxy** with hard
per-bar clips and RF/n risk — useful for monthly-distribution screening only.
Earlier `pairs_two_leg_basket.py` with real ATR sizing + spreads was **negative** on
holdout. Proxy near-misses (HO 0.95–0.99%, or 2026 0.88–0.98%) are **not** a promote
path without FTMO-realistic two-leg confirmation.

## Verdict

No promote. Equity-TSMOM / monthly-budget overlays and pairs residual sleeves did not jointly clear ≥1% mean_mo on **each** of 2024, 2025, 2026, holdout with ≥70% pos under FTMO gates without look-ahead / RF hike on `approximate_non_ftmo`.

Artifacts: `quest_equity_tsmom_pairs_board.csv`, `quest_equity_tsmom_pairs_wf.csv`, `quest_equity_tsmom_pairs_promote.csv`, `configs/quest_equity_tsmom_pairs_selected.json`.

