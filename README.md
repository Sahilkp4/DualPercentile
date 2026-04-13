# DualPercentile — Quantitative Stock Filtering & ML Strategy Pipeline

A production-grade, multi-stage quantitative trading research pipeline that identifies consolidating stocks using a dual percentile filtering system, then discovers, validates, and hardens trading strategies through rigorous statistical testing and machine learning — all with strict temporal integrity to prevent data leakage.

---

## Technical Highlights

### Machine Learning & Statistical Methods
- **Logistic Regression & Random Forest classifiers** with AUC-ROC evaluation and feature importance ranking for predictive signal identification
- **K-Means clustering** with silhouette score optimization to discover natural trade regime archetypes across 32,000+ historical trades
- **Exhaustive grid search** across 320,271 strategy parameter combinations (stop-loss × take-profit × holding period)
- **Bootstrap confidence intervals** for robustness assessment of every strategy candidate
- **Welch's t-test & Mann-Whitney U test** for non-parametric hypothesis testing of feature significance
- **Cohen's d effect size** measurement to ensure practical (not just statistical) significance
- **Bonferroni correction** and **Benjamini-Hochberg FDR correction** for multiple hypothesis testing across thousands of simultaneous tests
- **Anchored walk-forward out-of-sample validation** — expanding training window with fixed test window to maintain strict temporal integrity
- **Block bootstrap hypothesis testing** within each walk-forward fold
- **Directed harm testing** for interaction effects between features, with average harm pruning for conflict resolution
- **ANOVA & Kruskal-Wallis** tests for multi-group earnings timing analysis
- **Levene's & Bartlett's tests** for variance homogeneity

### Software Engineering & Performance
- **Multi-core parallel processing** via `ProcessPoolExecutor` with automatic worker scaling — achieves 5–16x speedup across all pipeline stages
- **Vectorized percentile calculations** using pandas GroupBy operations — 10–50x faster than row-level iteration
- **Bulk SQL data fetching** (single query for entire universe vs. thousands of individual queries)
- **Per-process database connection pooling** to eliminate contention in parallel workloads
- **Thread-safe progress tracking** and result aggregation across worker processes
- **In-memory caching** of trading date lookups to avoid redundant database round-trips
- **YAML-driven configuration** for all pipeline stages — every threshold, path, and hyperparameter is externalized
- **Modular architecture** — Stage 2–4 each use dedicated submodule packages (data loading, fold generation, screening, aggregation, output)

### Anti-Overfitting & Data Integrity
- **Zero look-ahead bias** — outcome variables (max gain, max loss) are strictly excluded from ML feature sets and used only for trade simulation
- **Earnings window blackout** — configurable exclusion periods around quarterly reporting dates to isolate signal from noise
- **Per-strategy random seeding** for full reproducibility of parallel runs
- **FDR correction applied across all strategies** at the pipeline level, not just within individual folds
- **Out-of-sample degradation tracking** to flag strategies that don't generalize

---

## Architecture Overview

The system works in two phases:

**Phase 1 — Signal Generation:** The `DualPercentileFilter` scans a universe of stocks against a PostgreSQL database of OHLCV data with pre-computed peak/valley analysis. Stocks exhibiting tight consolidation (low short-term and long-term volatility percentiles) are flagged as breakout candidates. The `generate_ml_dataset` script runs this filter across every eligible trading week from 2020–2025, producing a 32,000+ trade dataset.

**Phase 2 — Strategy Optimization (Stages 1–4):** A four-stage pipeline progressively refines strategy parameters:

| Stage | Purpose | Method |
|-------|---------|--------|
| **Stage 1** | Find optimal SL/TP/HP | Exhaustive grid search (320K combos) + bootstrap CI |
| **Stage 2** | Select predictive features | Walk-forward validation + block bootstrap + BH-FDR |
| **Stage 3** | Tune entry thresholds | Multi-core threshold screening + FDR correction |
| **Stage 4** | Prune harmful categories | Categorical exclusion discovery + walk-forward + FDR |

---

## File Descriptions

### Core Filtering

**`DualPercentileFilter.py`** — Primary stock screening engine. Queries the `stocks_peaks_valleys_analysis` PostgreSQL table, computes dual percentile scores (short-term volatility vs. stock's own history, and vs. the full universe), and flags consolidation breakout candidates. Uses bulk data fetching and vectorized pandas operations for 10–50x speedup over naive implementations. Imports DB credentials from a separate `config.py` for security.

### Historical Dataset Generation

**`generate_ml_dataset.py`** — Runs the dual percentile filter across 5 years of trading data (Jan 2020 – Jan 2025) using `ProcessPoolExecutor` with per-worker database connections. Skips earnings blackout windows. Tracks forward-looking performance metrics for each signal. Outputs `2020_2025_DualPercentileResults.csv` (~32,000+ trade signals). DB credentials read from `DB_PASSWORD` environment variable.

**`2020_2025_DualPercentileResults.csv`** — Raw historical backtest output: one row per qualifying stock per week with entry conditions and outcome metrics.

**`2020_2025_DualPercentileResults_ML_Ready_WithWeeklyMetrics.csv`** — Enriched version with weekly max-gain/max-loss metrics, ready for direct ingestion by the Stage 1–4 pipeline.

**`2020_2025_DualPercentileResults_Encoding_Reference.txt`** — Lookup table mapping encoded integers back to human-readable labels (tickers, industries).

### Strategy Optimization Pipeline

**`stage_1_sl_tp_hp.py`** — Exhaustive grid search: 151 profit targets × 101 stop losses × 21 holding periods = 320,271 combinations. Simulates trades with realistic exit priority (uses `max_gain_before_max_loss` flag). Computes expectancy, win rate, profit factor, and Sharpe-like metrics. Bootstrap confidence intervals for robustness. Parallel execution. Configured via `stage1_config.yaml`.

**`stage_2_feature_selection.py`** — Statistical feature selection with anchored walk-forward OOS validation, block bootstrap hypothesis testing, BH-FDR correction within each fold, directed harm testing for interaction effects, and average harm pruning. Configured via `stage2_config.yaml`.

**`stage_3_threshold_discovery.py`** — Discovers optimal minimum price and buy-volume thresholds. Multi-core parallel processing at the strategy level (10–16x speedup). Per-strategy random seeding. Walk-forward fold generation → threshold screening → aggregation → FDR correction. Configured via `stage3_config.yaml`.

**`stage_3_threshold_discovery_ORIGINAL_BACKUP.py`** — Pre-optimization backup of Stage 3.

**`stage_4_exclusion_discovery.py`** — Identifies harmful categorical values (industries, weeks-after-earnings) to exclude from each strategy using anchored walk-forward validation with FDR correction. Modular subpackage architecture.

### Analysis & Visualization

**`comprehensive_backtest_analysis.py`** — Deep-dive analysis of the 32,000+ trade dataset. Welch's t-test, Mann-Whitney U, Cohen's d, Bonferroni correction, ANOVA, Kruskal-Wallis, Levene's test, Logistic Regression, Random Forest (with AUC-ROC), K-Means clustering with silhouette optimization, SL/TP grid search with multiprocessing, and matplotlib/seaborn visualizations.

**`earnings_timing_winrate.png`** — Win-rate by earnings timing window visualization.

### Configuration

**`stage1_config.yaml`** / **`stage2_config.yaml`** / **`stage3_config.yaml`** — YAML configs controlling data paths, grid search ranges, statistical thresholds, FDR alpha, parallelism, and output directories.

### Data Reference

**`SQL_DATA.txt`** / **`SQL_DATA_GUIDE`** — Complete 27-column reference for the `stocks_peaks_valleys_analysis` table: OHLCV data, peak/valley identification, percentage change metrics, ML-ready binary flags, and 5 rolling volatility averages.

### Tests

**`test_ml_dataset_generation.py`** — Validates bug fixes, multi-core correctness, and sequential-vs-parallel consistency (~30–45s runtime).

**`test_stage1_logic.py`** — Unit tests for trade simulation logic, exit priority, metric calculations, and edge cases.

**`test_stage4_small.py`** — Small-scale integration test for Stage 4.

**`check_results.py`** / **`verify_fixes.py`** — Spot-check utilities.

---

## Security Notes

Database credentials in `generate_ml_dataset.py` and `test_ml_dataset_generation.py` now read from the `DB_PASSWORD` environment variable:
```bash
export DB_PASSWORD="your_password_here"
```
`DualPercentileFilter.py` imports from `config.py` (not committed).

---

## Tech Stack

Python 3.9+ · PostgreSQL · psycopg2 · pandas · NumPy · SciPy · scikit-learn · statsmodels · matplotlib · seaborn · PyYAML · tqdm · concurrent.futures
