# NUMBERS

Canonical reference for every figure quoted elsewhere in this project. Every value below is read
directly from a stored notebook output or a source file — nothing here is recomputed or estimated.
If a figure can't be traced to one of those, it doesn't belong in this table.

| figure | value | source (file + cell id / line) | split |
|---|---|---|---|
| **— raw source scale —** | | | |
| members rows | 6,769,473 | `01_eda.ipynb#5bb1ef65` | — |
| transactions rows | 22,978,755 | `01_eda.ipynb#5bb1ef65` | — |
| user_logs rows | 410,502,905 | `01_eda.ipynb#5bb1ef65` | — |
| **— label validation (vs KKBox train.csv, Feb-2017 cohort) —** | | | |
| n scored | 788,237 | `01_eda.ipynb#8ec75cf4` | — |
| our churn rate | 0.039 | `01_eda.ipynb#8ec75cf4` | — |
| KKBox churn rate | 0.044 | `01_eda.ipynb#8ec75cf4` | — |
| precision vs KKBox | 0.9993 | `01_eda.ipynb#8ec75cf4` | — |
| recall vs KKBox | 0.8870 | `01_eda.ipynb#8ec75cf4` | — |
| kappa vs KKBox | 0.9379 | `01_eda.ipynb#8ec75cf4` | — |
| **— row counts & base rate, full population (`cohorts`) —** | | | |
| n / base rate | 9,970,382 / 0.1188 | `01_eda.ipynb#40fa735b` | train |
| n / base rate | 2,468,979 / 0.1363 | `01_eda.ipynb#40fa735b` | val |
| n / base rate | 2,358,012 / 0.0594 | `01_eda.ipynb#40fa735b` | test |
| **— row counts & base rate, sampled population (`cohorts_s`, all-feature scale) —** | | | |
| n / users / base rate | 300,000 / 256,907 / 0.1192 | `02_model.ipynb#334349a2` | train |
| n / users / base rate | 100,000 / 96,706 / 0.1368 | `02_model.ipynb#334349a2` | val |
| n / users / base rate | 100,000 / 96,314 / 0.0595 | `02_model.ipynb#334349a2` | test |
| **— paid-only population, sampled scale —** | | | |
| n / base rate | 89,089 / 0.0389 | `02_model.ipynb#02ef91bb` | val |
| n / base rate | 98,267 / 0.0465 | `02_model.ipynb#02ef91bb` | test |
| **— paid-only population, full scale —** | | | |
| n | 9,399,666 | `02_model.ipynb#59b0e904` | train |
| n | 2,202,951 | `02_model.ipynb#59b0e904` | val |
| n | 2,317,271 | `02_model.ipynb#59b0e904` | test |
| **— baseline models, ALL population (incl. free trials, is_free as feature), sampled scale —** | | | |
| logistic (commitment feats only), val PR-AUC | 0.8586 | `02_model.ipynb#559ff8de` | val |
| logistic (+engagement), val / test PR-AUC | 0.8969 / 0.5178 | `02_model.ipynb#20473ad5` | val / test |
| XGBoost (+engagement, all feats), val / test PR-AUC | 0.9176 / 0.5839 | `02_model.ipynb#91941562` | val / test |
| — same model, scored on PAID rows only, val / test PR-AUC | 0.4150 / 0.4010 | `02_model.ipynb#02ef91bb` | val / test |
| **— locked base12 model, paid-only, sampled scale —** | | | |
| logistic-paid, val / test PR-AUC | 0.3337 / 0.2840 | `02_model.ipynb#c656a3eb` | val / test |
| XGBoost-paid (base12), val / test PR-AUC | 0.4179 / 0.4021 | `02_model.ipynb#c656a3eb` | val / test |
| — safe slice (auto_renew=1) PR-AUC | val 0.0408 (n=77,645) | `02_model.ipynb#c656a3eb` | val |
| train / test PR-AUC (overfit check) | 0.3613 (base 0.075) / 0.4021 (base 0.046) | `02_model.ipynb#60f559c2` | train / test |
| **— lifecycle / paymethod ablation, paid-only, sampled scale —** | | | |
| base12, val PR-AUC / safe-slice PR-AUC | 0.4179 / 0.0408 | `02_model.ipynb#ca6da8d6` | val |
| base12+lifecycle, val PR-AUC / safe-slice PR-AUC | 0.4130 / 0.0883 | `02_model.ipynb#ca6da8d6` | val |
| base12+paymethod, val PR-AUC / safe-slice PR-AUC | 0.4582 / 0.2812 | `02_model.ipynb#ca6da8d6` | val |
| base12+lifecycle, test PR-AUC (report-only) | val 0.4130 / test 0.3941 | `02_model.ipynb#6d54f99d` | val / test |
| 14-feat train / test PR-AUC (overfit check) | 0.6656 (base 0.075) / 0.3941 (base 0.046) | `02_model.ipynb#60f559c2` | train / test |
| **— locked model, honest test open (sampled scale) —** | | | |
| base12 test PR-AUC (14-feat in parens) | 0.4021 (0.3941) | `02_model.ipynb#51bd3cd4` | test |
| base12 test PR-AUC + 95% CI (paired bootstrap, 2000 resamples) | 0.4021 [0.3867, 0.4176] | `02_model.ipynb#d37b9090` | test |
| lifecycle gain, topline (test bootstrap) | −0.0081 [−0.0168, +0.0005] — within noise | `02_model.ipynb#d37b9090` | test |
| lifecycle gain, safe-slice (test bootstrap) | +0.0419 [+0.0301, +0.0540] — clears noise | `02_model.ipynb#d37b9090` | test |
| **— full-population confirmation —** | | | |
| test PR-AUC / base rate / trees | 0.4124 / 0.0465 / 72 | `02_model.ipynb#59b0e904` | test (full-pop) |
| **— calibration (isotonic on val, applied to test) —** | | | |
| test PR-AUC raw → cal | 0.402115 → 0.392343 | `02_model.ipynb#f18a8371` | test |
| test Brier raw → cal | 0.03460 → 0.03435 | `02_model.ipynb#f18a8371` | test |
| top reliability bin: predicted vs observed | 0.2955 vs 0.2950 (n=9,406) | `02_model.ipynb#f18a8371` | test |
| cal mean vs actual (drift check) | val: 0.0389 vs 0.0389 · test: 0.0391 vs 0.0465 | `02_model.ipynb#f12d8095` | val / test |
| **— val cost selection, B1 (2-fold OOF isotonic, seed 42) —** | | | |
| val break-even / value(12mo) / median monthly paid | 0.303 / NT$1,650 / NT$138 | `02_model.ipynb#5c0b2843` | val |
| base12: contacted / precision / net / prec@1k / prec@3k | 2,601 / 0.481 / NT$228,773 / 0.631 / 0.456 | `02_model.ipynb#5c0b2843` | val |
| 14-feat: contacted / precision / net / prec@1k / prec@3k | 2,887 / 0.462 / NT$227,960 / 0.611 / 0.456 | `02_model.ipynb#5c0b2843` | val |
| **— val paired bootstrap, B2 (2000 resamples) —** | | | |
| base12 val PR-AUC + 95% CI | 0.4179 [0.4012, 0.4358] | `02_model.ipynb#cd107952` | val |
| lifecycle gain, topline | −0.0049 [−0.0142, +0.0045] — within noise | `02_model.ipynb#cd107952` | val |
| lifecycle gain, safe-slice | +0.0481 [+0.0322, +0.0650] — clears noise | `02_model.ipynb#cd107952` | val |
| val safe slice n / base rate | 77,645 / 0.0109 | `02_model.ipynb#cd107952` | val |
| **— safe-slice closure, B3 —** | | | |
| bin count (qcut, ties collapse) | 8 | `02_model.ipynb#ffd04094` | val |
| top bin: mean P / n / required save_rate | 0.0504 / 6,083 / 1.9217 | `02_model.ipynb#ffd04094` | val |
| n clearing the val backtest cut (0.303) | 0 / 77,645 | `02_model.ipynb#ffd04094` | val |
| max calibrated P in the safe slice | 0.1437 | `02_model.ipynb#ffd04094` | val |
| max EV-positive offer at top bin (save_rate 0.30) | NT$23 | `02_model.ipynb#ffd04094` | val |
| cheap-channel NT$20: n / mean P / net | 3,578 / 0.0624 / NT$32,111 | `02_model.ipynb#ffd04094` | val |
| cheap-channel NT$30: n / mean P / net | 993 / 0.0829 / NT$8,452 | `02_model.ipynb#ffd04094` | val |
| **— test-side evaluation, deployed cut 0.323 (MODEL SELECTION cell) —** | | | |
| deployed value(12mo) / median monthly paid / break-even | NT$1,548 / NT$129 / 0.323 | `02_model.ipynb#6d56c25f` | test |
| base12: contacted / precision / net / prec@1k / prec@3k | 2,984 / 0.526 / NT$281,972 / 0.706 / 0.525 | `02_model.ipynb#6d56c25f` | test |
| 14-feat: contacted / precision / net / prec@1k / prec@3k | 3,230 / 0.490 / NT$251,110 / 0.669 / 0.504 | `02_model.ipynb#6d56c25f` | test |
| **— test-side evaluation, val-derived cut 0.303, B4 —** | | | |
| contacted / precision / would_stay / net (deployed value) | 3,260 / 0.505 / 1,614 / NT$275,402 | `02_model.ipynb#2ceaa82d` | test |
| prec@1k / prec@3k | 0.706 / 0.525 | `02_model.ipynb#2ceaa82d` | test |
| net, revalued at val-period value NT$1,650 (sensitivity, same list) | NT$325,998 | `02_model.ipynb#2ceaa82d` | test |
| test PR-AUC raw → cal | 0.402115 → 0.392343 | `02_model.ipynb#2ceaa82d` | test |
| test Brier raw → cal | 0.03460 → 0.03435 | `02_model.ipynb#2ceaa82d` | test |
| **— deployed operating point constants —** | | | |
| OFFER | NT$150 | `src/config.py:39` | — |
| SAVE_RATE | 0.30 | `src/config.py:40` | — |
| HORIZON_MONTHS | 12 | `src/config.py:41` | — |
| MEDIAN_MONTHLY_PAID | NT$129 | `src/config.py:42` | — |
| value (= MEDIAN_MONTHLY_PAID × HORIZON_MONTHS) | NT$1,548 | derived, `src/config.py:39-42`; printed value matches `02_model.ipynb#6d56c25f` | — |
| break-even (= OFFER / (SAVE_RATE × value)) | 0.323 | derived, `src/config.py:39-42`; printed value matches `02_model.ipynb#6d56c25f` | — |
| **— Hillstrom uplift experiment (Phase D) —** | | | |
| naive ATE (any email vs none) | +0.0609 (control 0.1062, mailed 0.1670) | `04_uplift_experiment.ipynb#dac2436e` | test |
| T-learner test mean predicted uplift | +0.0633 | `04_uplift_experiment.ipynb#27b5b0d6` | test |
| Qini coefficient (area vs random) | 54.93 | `04_uplift_experiment.ipynb#afcd9d42` | test |
| total incremental visits, full test | 704.4 | `04_uplift_experiment.ipynb#afcd9d42` | test |
| top-30%-targeted incremental visits: model vs random | 285.5 vs 211.3 | `04_uplift_experiment.ipynb#afcd9d42` | test |
| tier Q5 (persuadable): n / realised uplift | 3,840 / 0.0731 | `04_uplift_experiment.ipynb#6a33bb9c` | test |
| tier Q4: n / realised uplift | 3,840 / 0.0699 | `04_uplift_experiment.ipynb#6a33bb9c` | test |
| tier Q3: n / realised uplift | 3,840 / 0.0518 | `04_uplift_experiment.ipynb#6a33bb9c` | test |
| tier Q2: n / realised uplift | 3,840 / 0.0395 | `04_uplift_experiment.ipynb#6a33bb9c` | test |
| tier Q1 (sleeping/low): n / realised uplift | 3,840 / 0.0401 | `04_uplift_experiment.ipynb#6a33bb9c` | test |

## Stability

Measured result (DECISIONS.md §6): **raw test PR-AUC is reproducible to 6 decimal places** (0.402121 in
one back-to-back pinned pair; 0.402115 in the run this table draws from — the same figure, different
pinned runs, agreeing to 4 decimal places). Everything downstream of the isotonic calibrator moves:

- **Reproducible run-to-run** (rank-based, unaffected by calibration): raw PR-AUC (val and test), the
  overfit/train-test gap check, the bootstrap CI *verdicts* (within noise / clears noise — the specific
  bounds shift but the call hasn't flipped across any run this session).
- **Moves run-to-run**: calibrated PR-AUC (differs in the 4th decimal), prec@budget (~0.005 swings),
  contact counts under any hard threshold — val selection, deployed-cut, val-derived-cut, and every
  closure/cheap-channel figure — by roughly 5% and up to ~17% in one observed case (B3's NT$20 case).
  The 14-feat comparison figures move more than base12's (172 trees vs 32, more accumulation order to
  diverge). The safe-slice bin count itself is unstable (7-10 bins observed across runs) because it's
  downstream of the same calibrator.
- **Mechanism** (DECISIONS.md §6): isotonic regression collapses tens of thousands of rows onto ~200-250
  distinct probability values; mass points of 100+ customers can sit within a thousandth of an operating
  cut, so a tiny upstream float shift flips a whole block across the line at once. The deployed cut
  (0.323) has no mass point within ±0.005 in either split and is structurally more stable than the
  val-derived backtest cut (0.303).
- **This table is point-in-time**: every figure above is from one pinned run (`n_jobs=1` sampled scale,
  `n_jobs=4` full-population). Treat contact-count-derived figures as illustrative of magnitude, not as
  exact reproducible constants — re-running the notebook will move them within the ranges above.
