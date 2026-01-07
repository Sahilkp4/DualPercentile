#!/usr/bin/env python3
"""
STAGE 1: EXHAUSTIVE STOP LOSS / TARGET PROFIT / HOLDING PERIOD GRID SEARCH

This script implements a comprehensive grid search across all combinations of:
- Target Profit: 0.05% to 15.05% in 0.1% steps (151 values)
- Stop Loss: 0.05% to 10.05% in 0.1% steps (101 values)
- Holding Period: 1 to 21 weeks in 1 week steps (21 values)

Total combinations tested: 151 × 101 × 21 = 320,271 strategies

For each strategy, the script:
1. Simulates trades using weekly max_gain/max_loss data
2. Calculates performance metrics (expectancy, win rate, profit factor, etc.)
3. Computes bootstrap confidence intervals for robustness
4. Filters strategies by minimum trade count and quality thresholds
5. Saves ranked results to CSV files

Author: Professional Quant Trader
Date: 2025-12-26
"""

import pandas as pd
import numpy as np
import yaml
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class StrategyParams:
    """Parameters defining a single trading strategy"""
    target_profit_pct: float
    stop_loss_pct: float
    holding_period_weeks: int


@dataclass
class StrategyMetrics:
    """Performance metrics for a trading strategy"""
    target_profit_pct: float
    stop_loss_pct: float
    holding_period_weeks: int
    trade_count: int
    expectancy: float
    win_rate: float
    avg_win: float
    avg_loss: float
    avg_win_loss_ratio: float
    profit_factor: float
    ci_lower: float
    ci_upper: float
    total_return: float
    max_drawdown: float
    sharpe_ratio: float


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"ERROR: Failed to load config file: {e}")
        sys.exit(1)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(config: dict) -> logging.Logger:
    """Configure logging based on config settings"""
    log_config = config['logging']
    output_dir = config['data']['output_dir']

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Setup log file path
    log_file = log_config['log_file'].replace('{output_dir}', output_dir)

    # Create logger
    logger = logging.getLogger('Stage1')
    logger.setLevel(getattr(logging, log_config['log_level']))

    # Remove existing handlers
    logger.handlers = []

    # File handler
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setLevel(getattr(logging, log_config['log_level']))
    fh.setFormatter(logging.Formatter(
        log_config['log_format'],
        datefmt=log_config['date_format']
    ))
    logger.addHandler(fh)

    # Console handler
    if log_config['log_to_console']:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_config['log_level']))
        ch.setFormatter(logging.Formatter(
            log_config['log_format'],
            datefmt=log_config['date_format']
        ))
        logger.addHandler(ch)

    return logger


# ============================================================================
# DATA LOADING AND VALIDATION
# ============================================================================

def load_data(config: dict, logger: logging.Logger) -> pd.DataFrame:
    """Load and validate CSV data"""
    logger.info("Loading data from CSV...")

    try:
        df = pd.read_csv(config['data']['input_csv'])
        logger.info(f"Loaded {len(df)} rows from CSV")
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        sys.exit(1)

    # Validate data if enabled
    if config['data_handling']['validate_data']:
        logger.info("Validating data...")
        validate_data(df, logger)

    return df


def validate_data(df: pd.DataFrame, logger: logging.Logger):
    """Validate data structure and contents"""

    # Check for required columns
    required_base_cols = [
        'buy_date_encoded', 'sell_date_encoded', 'symbol_encoded',
        'quarter_period_encoded', 'weeks_after_earnings'
    ]

    missing_cols = [col for col in required_base_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        sys.exit(1)

    # Check for weekly metric columns (max_gain_Xweek, max_loss_Xweek, max_gain_before_max_loss_Xweek)
    weekly_cols_found = 0
    for week in range(1, 22):  # Check weeks 1-21
        gain_col = f'max_gain_{week}week'
        loss_col = f'max_loss_{week}week'
        precedence_col = f'max_gain_before_max_loss_{week}week'

        if gain_col in df.columns and loss_col in df.columns and precedence_col in df.columns:
            weekly_cols_found += 1

    logger.info(f"Found weekly metrics for {weekly_cols_found} weeks")

    if weekly_cols_found < 21:
        logger.warning(f"Expected 21 weeks of data, found {weekly_cols_found}")

    # Check for NaN patterns
    logger.info("Analyzing NaN patterns in weekly metrics...")
    for week in [1, 5, 10, 15, 21]:
        gain_col = f'max_gain_{week}week'
        loss_col = f'max_loss_{week}week'
        if gain_col in df.columns:
            nan_count = df[gain_col].isna().sum()
            pct = 100 * nan_count / len(df)
            logger.info(f"  Week {week}: {nan_count} NaNs ({pct:.1f}%)")

    logger.info("Data validation complete")


# ============================================================================
# TRADE SIMULATION
# ============================================================================

def simulate_strategy(
    df: pd.DataFrame,
    params: StrategyParams,
    logger: logging.Logger
) -> List[float]:
    """
    Simulate a single strategy across all rows in the dataset.

    Returns list of trade returns (one per valid trade).
    """

    returns = []

    for idx, row in df.iterrows():
        # Get the holding period
        hp = params.holding_period_weeks

        # Check if weekly data exists for this holding period
        gain_col = f'max_gain_{hp}week'
        loss_col = f'max_loss_{hp}week'
        precedence_col = f'max_gain_before_max_loss_{hp}week'

        if pd.isna(row[gain_col]) or pd.isna(row[loss_col]) or pd.isna(row[precedence_col]):
            # Skip this row - data not available for this holding period
            continue

        # Walk forward through weeks 1 to holding_period_weeks
        trade_return = None

        for week in range(1, hp + 1):
            gain_col_w = f'max_gain_{week}week'
            loss_col_w = f'max_loss_{week}week'
            precedence_col_w = f'max_gain_before_max_loss_{week}week'

            # Check if this week's data exists
            if pd.isna(row[gain_col_w]) or pd.isna(row[loss_col_w]):
                # Data doesn't exist for this week, skip to next row
                break

            max_gain = row[gain_col_w]
            max_loss = row[loss_col_w]
            gain_before_loss = row[precedence_col_w]  # 1 if gain came first, 0 if loss came first

            # Check if target profit is triggered
            target_triggered = max_gain >= params.target_profit_pct

            # Check if stop loss is triggered
            # Note: stop loss is stored as positive in params but compared to negative max_loss
            stop_triggered = max_loss <= -params.stop_loss_pct

            # Determine if we exit this week
            if target_triggered and stop_triggered:
                # Both triggered - use precedence to determine which came first
                if gain_before_loss == 1:
                    # Gain came first -> target profit hit
                    trade_return = params.target_profit_pct
                else:
                    # Loss came first -> stop loss hit
                    trade_return = -params.stop_loss_pct
                break
            elif target_triggered:
                # Only target profit triggered
                trade_return = params.target_profit_pct
                break
            elif stop_triggered:
                # Only stop loss triggered
                trade_return = -params.stop_loss_pct
                break

            # If we reach the final week without triggering, use median of final week
            if week == hp:
                median_return = np.median([max_gain, max_loss])
                trade_return = median_return
                break

        # If we got a valid return, add it to results
        if trade_return is not None:
            returns.append(trade_return)

    return returns


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(
    params: StrategyParams,
    returns: List[float],
    config: dict
) -> StrategyMetrics:
    """Calculate all performance metrics for a strategy"""

    if len(returns) == 0:
        # No trades - return metrics with NaN
        return StrategyMetrics(
            target_profit_pct=params.target_profit_pct,
            stop_loss_pct=params.stop_loss_pct,
            holding_period_weeks=params.holding_period_weeks,
            trade_count=0,
            expectancy=np.nan,
            win_rate=np.nan,
            avg_win=np.nan,
            avg_loss=np.nan,
            avg_win_loss_ratio=np.nan,
            profit_factor=np.nan,
            ci_lower=np.nan,
            ci_upper=np.nan,
            total_return=np.nan,
            max_drawdown=np.nan,
            sharpe_ratio=np.nan
        )

    returns_arr = np.array(returns)

    # Trade count
    trade_count = len(returns)

    # Win/loss separation
    winning_trades = returns_arr[returns_arr > 0]
    losing_trades = returns_arr[returns_arr < 0]

    # Win rate
    win_rate = len(winning_trades) / trade_count if trade_count > 0 else 0

    # Average win/loss
    avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
    avg_loss = abs(np.mean(losing_trades)) if len(losing_trades) > 0 else 0

    # Avg win/loss ratio
    avg_win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf

    # Expectancy
    p_win = win_rate
    p_loss = 1 - win_rate
    expectancy = (p_win * avg_win) - (p_loss * avg_loss)

    # Profit factor
    gross_profit = np.sum(winning_trades) if len(winning_trades) > 0 else 0
    gross_loss = abs(np.sum(losing_trades)) if len(losing_trades) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Total return
    total_return = np.sum(returns_arr)

    # Maximum drawdown
    cumulative = np.cumsum(returns_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    # Sharpe ratio
    mean_return = np.mean(returns_arr)
    std_return = np.std(returns_arr, ddof=1) if len(returns_arr) > 1 else 0
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0

    # Bootstrap confidence interval
    ci_lower, ci_upper = bootstrap_ci(
        returns_arr,
        n_iterations=config['bootstrap']['n_iterations'],
        confidence_level=config['bootstrap']['confidence_level'],
        random_seed=config['bootstrap']['random_seed']
    )

    return StrategyMetrics(
        target_profit_pct=params.target_profit_pct,
        stop_loss_pct=params.stop_loss_pct,
        holding_period_weeks=params.holding_period_weeks,
        trade_count=trade_count,
        expectancy=expectancy,
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        avg_win_loss_ratio=avg_win_loss_ratio,
        profit_factor=profit_factor,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        total_return=total_return,
        max_drawdown=max_drawdown,
        sharpe_ratio=sharpe_ratio
    )


def bootstrap_ci(
    returns: np.ndarray,
    n_iterations: int,
    confidence_level: float,
    random_seed: int
) -> Tuple[float, float]:
    """
    Calculate bootstrap confidence interval for expectancy.

    Returns (ci_lower, ci_upper) as 5th and 95th percentiles.
    """

    if len(returns) == 0:
        return np.nan, np.nan

    np.random.seed(random_seed)

    bootstrap_expectancies = []

    for _ in range(n_iterations):
        # Resample with replacement
        sample = np.random.choice(returns, size=len(returns), replace=True)

        # Calculate expectancy for this sample
        winning = sample[sample > 0]
        losing = sample[sample < 0]

        p_win = len(winning) / len(sample)
        p_loss = 1 - p_win
        avg_win = np.mean(winning) if len(winning) > 0 else 0
        avg_loss = abs(np.mean(losing)) if len(losing) > 0 else 0

        expectancy = (p_win * avg_win) - (p_loss * avg_loss)
        bootstrap_expectancies.append(expectancy)

    # Calculate percentiles
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_expectancies, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_expectancies, 100 * (1 - alpha / 2))

    return ci_lower, ci_upper


# ============================================================================
# PARALLEL PROCESSING
# ============================================================================

def process_strategy_batch(args):
    """Process a batch of strategies (for parallel execution)"""
    strategies_batch, df, config = args

    results = []

    for params in strategies_batch:
        # Simulate strategy
        returns = simulate_strategy(df, params, logging.getLogger('Stage1'))

        # Calculate metrics
        metrics = calculate_metrics(params, returns, config)

        results.append(metrics)

    return results


# ============================================================================
# GRID SEARCH
# ============================================================================

def generate_strategy_grid(config: dict) -> List[StrategyParams]:
    """Generate all strategy parameter combinations"""

    tp_config = config['grid_search']['target_profit']
    sl_config = config['grid_search']['stop_loss']
    hp_config = config['grid_search']['holding_period']

    # Generate value ranges
    target_profits = np.arange(
        tp_config['start'],
        tp_config['end'] + tp_config['step'] / 2,  # Add half step to include end
        tp_config['step']
    )

    stop_losses = np.arange(
        sl_config['start'],
        sl_config['end'] + sl_config['step'] / 2,
        sl_config['step']
    )

    holding_periods = np.arange(
        hp_config['start'],
        hp_config['end'] + 1,  # Include end
        hp_config['step'],
        dtype=int
    )

    # Generate all combinations
    strategies = []
    for tp in target_profits:
        for sl in stop_losses:
            for hp in holding_periods:
                strategies.append(StrategyParams(
                    target_profit_pct=round(tp, 2),
                    stop_loss_pct=round(sl, 2),
                    holding_period_weeks=int(hp)
                ))

    return strategies


def run_grid_search(
    df: pd.DataFrame,
    config: dict,
    logger: logging.Logger
) -> pd.DataFrame:
    """Execute grid search across all strategy combinations"""

    logger.info("Generating strategy grid...")
    strategies = generate_strategy_grid(config)
    logger.info(f"Total strategies to test: {len(strategies):,}")

    # Check if parallel processing is enabled
    use_parallel = config['performance']['use_multiprocessing']
    n_jobs = config['performance']['n_jobs']
    chunk_size = config['performance']['chunk_size']

    all_metrics = []

    if use_parallel:
        logger.info(f"Running parallel grid search with {n_jobs} workers...")

        # Split strategies into batches
        batches = [
            strategies[i:i + chunk_size]
            for i in range(0, len(strategies), chunk_size)
        ]

        logger.info(f"Split into {len(batches)} batches of ~{chunk_size} strategies")

        # Prepare arguments for each batch
        batch_args = [(batch, df, config) for batch in batches]

        # Process batches in parallel
        with ProcessPoolExecutor(max_workers=n_jobs if n_jobs > 0 else None) as executor:
            futures = {executor.submit(process_strategy_batch, args): i
                      for i, args in enumerate(batch_args)}

            with tqdm(total=len(batches), desc="Processing batches") as pbar:
                for future in as_completed(futures):
                    batch_results = future.result()
                    all_metrics.extend(batch_results)
                    pbar.update(1)
    else:
        logger.info("Running sequential grid search...")

        with tqdm(total=len(strategies), desc="Testing strategies") as pbar:
            for params in strategies:
                returns = simulate_strategy(df, params, logger)
                metrics = calculate_metrics(params, returns, config)
                all_metrics.append(metrics)
                pbar.update(1)

    logger.info(f"Grid search complete. Processed {len(all_metrics):,} strategies")

    # Convert to DataFrame
    results_df = pd.DataFrame([vars(m) for m in all_metrics])

    return results_df


# ============================================================================
# FILTERING AND RANKING
# ============================================================================

def filter_by_min_trades(
    results_df: pd.DataFrame,
    min_trades: int,
    logger: logging.Logger
) -> pd.DataFrame:
    """Filter strategies by minimum trade count"""

    initial_count = len(results_df)
    filtered_df = results_df[results_df['trade_count'] >= min_trades].copy()
    final_count = len(filtered_df)

    logger.info(f"Trade count filter ({min_trades}+): {initial_count:,} -> {final_count:,} strategies")

    return filtered_df


def apply_quality_filters(
    results_df: pd.DataFrame,
    config: dict,
    logger: logging.Logger
) -> pd.DataFrame:
    """Apply quality filters to select top candidates"""

    filters = config['quality_filters']

    logger.info("Applying quality filters...")
    initial_count = len(results_df)

    # Filter by expectancy
    mask = results_df['expectancy'] > filters['min_expectancy']
    logger.info(f"  Expectancy > {filters['min_expectancy']}: {mask.sum():,} pass")

    # Filter by CI lower bound
    mask = mask & (results_df['ci_lower'] > filters['min_ci_lower'])
    logger.info(f"  CI Lower > {filters['min_ci_lower']}: {mask.sum():,} pass")

    # Filter by profit factor
    mask = mask & (results_df['profit_factor'] > filters['min_profit_factor'])
    logger.info(f"  Profit Factor > {filters['min_profit_factor']}: {mask.sum():,} pass")

    # Filter by win rate
    mask = mask & (results_df['win_rate'] > filters['min_win_rate'])
    logger.info(f"  Win Rate > {filters['min_win_rate']}: {mask.sum():,} pass")

    # Filter by avg_win_loss_ratio (if not NaN)
    if not np.isnan(filters['min_avg_win_loss_ratio']):
        mask = mask & (results_df['avg_win_loss_ratio'] > filters['min_avg_win_loss_ratio'])
        logger.info(f"  Avg Win/Loss > {filters['min_avg_win_loss_ratio']}: {mask.sum():,} pass")

    filtered_df = results_df[mask].copy()
    final_count = len(filtered_df)

    logger.info(f"Quality filters: {initial_count:,} -> {final_count:,} strategies")

    return filtered_df


def rank_strategies(results_df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """Rank strategies by expectancy (descending)"""

    ranked_df = results_df.sort_values('expectancy', ascending=False).reset_index(drop=True)
    logger.info("Strategies ranked by expectancy")

    return ranked_df


# ============================================================================
# OUTPUT
# ============================================================================

def save_results(
    total_results_df: pd.DataFrame,
    top_candidates_df: pd.DataFrame,
    config: dict,
    logger: logging.Logger
):
    """Save results to CSV files"""

    output_dir = config['data']['output_dir']
    min_trades = config['filtering']['min_trades']

    # Total results file
    total_file = os.path.join(
        output_dir,
        config['output']['total_results_file'].replace('{min_trades}', str(min_trades))
    )

    total_results_df.to_csv(total_file, index=False)
    logger.info(f"Saved total results ({len(total_results_df):,} strategies) to: {total_file}")

    # Top candidates file
    top_file = os.path.join(output_dir, config['output']['top_candidates_file'])
    top_candidates_df.to_csv(top_file, index=False)
    logger.info(f"Saved top candidates ({len(top_candidates_df):,} strategies) to: {top_file}")

    # Print summary statistics
    logger.info("\n" + "="*80)
    logger.info("STAGE 1 SUMMARY")
    logger.info("="*80)
    logger.info(f"Total strategies tested: {len(total_results_df):,}")
    logger.info(f"Top candidates: {len(top_candidates_df):,}")

    if len(top_candidates_df) > 0:
        logger.info("\nTop 5 Strategies by Expectancy:")
        for i, row in top_candidates_df.head(5).iterrows():
            logger.info(f"\n  Rank {i+1}:")
            logger.info(f"    TP: {row['target_profit_pct']:.2f}%, SL: {row['stop_loss_pct']:.2f}%, HP: {row['holding_period_weeks']} weeks")
            logger.info(f"    Trades: {row['trade_count']}, Expectancy: {row['expectancy']:.4f}%")
            logger.info(f"    Win Rate: {row['win_rate']:.2%}, Profit Factor: {row['profit_factor']:.2f}")
            logger.info(f"    Sharpe: {row['sharpe_ratio']:.2f}, Max DD: {row['max_drawdown']:.2f}%")

    logger.info("="*80 + "\n")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    # Load configuration
    script_dir = Path(__file__).parent
    config_path = script_dir / 'stage1_config.yaml'

    print(f"Loading configuration from: {config_path}")
    config = load_config(str(config_path))

    # Setup logging
    logger = setup_logging(config)
    logger.info("="*80)
    logger.info("STAGE 1: EXHAUSTIVE SL/TP/HP GRID SEARCH")
    logger.info("="*80)
    logger.info(f"Configuration loaded from: {config_path}")

    # Check if results already exist
    output_dir = config['data']['output_dir']
    min_trades = config['filtering']['min_trades']
    total_results_file = os.path.join(
        output_dir,
        config['output']['total_results_file'].replace('{min_trades}', str(min_trades))
    )

    if os.path.exists(total_results_file):
        logger.info(f"\nFound existing results file: {total_results_file}")
        logger.info("Loading existing results instead of re-running grid search...")
        logger.info("(To re-run grid search, delete this file or change min_trades in config)")

        total_results_df = pd.read_csv(total_results_file)
        logger.info(f"Loaded {len(total_results_df):,} strategies from existing results")
    else:
        # Load data
        df = load_data(config, logger)

        # Run grid search
        start_time = time.time()
        total_results_df = run_grid_search(df, config, logger)
        elapsed = time.time() - start_time
        logger.info(f"Grid search completed in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")

        # Filter by minimum trades
        total_results_df = filter_by_min_trades(total_results_df, min_trades, logger)

        # Rank by expectancy
        total_results_df = rank_strategies(total_results_df, logger)

    # Apply quality filters
    top_candidates_df = apply_quality_filters(total_results_df, config, logger)
    top_candidates_df = rank_strategies(top_candidates_df, logger)

    # Save results
    save_results(total_results_df, top_candidates_df, config, logger)

    logger.info("Stage 1 execution complete!")


if __name__ == '__main__':
    main()
