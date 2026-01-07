"""
Stage 3 - Threshold Discovery Pipeline (Main Orchestrator)

This script coordinates all Stage 3 modules to discover minimum price and buy_volume
thresholds for strategies that passed Stage 2.

Pipeline Flow:
1. Load configuration and data
2. For each Stage 2 strategy:
   a. Build tradable trades (apply Stage2 features)
   b. Generate walk-forward folds
   c. For each variable (price, buy_volume):
      - Screen thresholds in each fold
      - Aggregate fold results to strategy level
   d. Return results
3. Apply FDR correction across all strategies
4. Generate output CSVs

Usage:
    python stage_3_threshold_discovery.py [--config stage3_config.yaml]
"""

import argparse
import sys
import yaml
from pathlib import Path
import logging
import pandas as pd
import numpy as np
from typing import Dict, List

# Import Stage 3 modules
from stage3_modules.data_loader import DataLoader
from stage3_modules.strategy_filter_applier import StrategyFilterApplier
from stage3_modules.threshold_screener import ThresholdScreener
from stage3_modules.strategy_aggregator import StrategyAggregator
from stage3_modules.fdr_controller import FDRController
from stage3_modules.output_generator import OutputGenerator

# Import Stage 2 modules (reused)
from stage2_modules.fold_generator import FoldGenerator, InsufficientFoldsError


def load_config(config_path: str = "stage3_config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : str
        Path to configuration file

    Returns
    -------
    dict
        Configuration dictionary
    """
    config_file = Path(__file__).parent / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Replace path placeholders
    output_dir = config['data']['output_dir']
    config['logging']['log_file'] = config['logging']['log_file'].replace('{output_dir}', output_dir)

    return config


def setup_logging(config: dict) -> logging.Logger:
    """
    Setup logging for Stage 3.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    logger = logging.getLogger('Stage3')
    logger.setLevel(config['logging']['log_level'])
    logger.handlers = []  # Clear existing handlers

    # File handler
    log_file = Path(config['logging']['log_file'])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter(
        config['logging']['log_format'],
        datefmt=config['logging']['date_format']
    ))
    logger.addHandler(fh)

    # Console handler (if enabled)
    if config['logging']['log_to_console']:
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter(
            config['logging']['log_format'],
            datefmt=config['logging']['date_format']
        ))
        logger.addHandler(ch)

    return logger


def process_single_strategy(
    strategy_idx: int,
    strategy: pd.Series,
    raw_data: pd.DataFrame,
    stage2_row: pd.Series,
    config: dict,
    logger: logging.Logger
) -> Dict:
    """
    Process a single strategy through Stage 3 pipeline.

    Steps:
    1. Build tradable trades set (apply Stage2 feature filters)
    2. Generate anchored walk-forward folds
    3. For each variable (price, buy_volume):
        a. Screen thresholds in each fold (train + test)
        b. Aggregate fold results to strategy-level decision
        c. Compute full-sample metrics
    4. Return results dict

    Parameters
    ----------
    strategy_idx : int
        Strategy index (for logging)
    strategy : pd.Series
        Strategy parameters from Stage2 output
    raw_data : pd.DataFrame
        Raw dataset (filtered for missing values)
    stage2_row : pd.Series
        Corresponding Stage2 row (same as strategy)
    config : dict
        Configuration dictionary
    logger : logging.Logger
        Logger instance

    Returns
    -------
    dict
        {
            'strategy': strategy,
            'price_summary': StrategyThresholdSummary,
            'buy_volume_summary': StrategyThresholdSummary,
            'tradable_trades': pd.DataFrame,
            'fold_results_price': List[ThresholdTestResult],
            'fold_results_volume': List[ThresholdTestResult]
        }
    """
    # Log progress (reduced logging - every 50 strategies)
    if strategy_idx % 50 == 0 or strategy_idx == 0:
        logger.info(f"Processing strategy {strategy_idx + 1}...")
        logger.info(f"  TP={strategy['target_profit_pct']:.2f}, SL={strategy['stop_loss_pct']:.2f}, HP={int(strategy['holding_period_weeks'])}")

    # Step 1: Build tradable trades (apply Stage2 filters)
    filter_applier = StrategyFilterApplier(config, logger)

    try:
        tradable_trades = filter_applier.build_tradable_trades(
            raw_data, strategy, stage2_row
        )
    except Exception as e:
        logger.error(f"Strategy {strategy_idx}: Failed to build tradable trades: {e}")
        return create_empty_result(strategy)

    if len(tradable_trades) == 0:
        logger.warning(f"Strategy {strategy_idx}: No tradable trades")
        return create_empty_result(strategy)

    # Step 2: Generate walk-forward folds
    fold_generator = FoldGenerator(config, logger)

    try:
        folds = fold_generator.generate_folds(tradable_trades)
    except InsufficientFoldsError as e:
        logger.warning(f"Strategy {strategy_idx}: {e}")
        return create_empty_result(strategy)

    # Step 3: Process each variable (price, buy_volume)
    screener = ThresholdScreener(config, logger)
    aggregator = StrategyAggregator(config, logger)

    results = {
        'strategy': strategy,
        'tradable_trades': tradable_trades,
    }

    for variable in config['threshold_variables']['variables']:
        # Screen thresholds in all folds
        fold_results = screener.screen_all_folds(tradable_trades, folds, variable)

        # Aggregate to strategy level
        summary = aggregator.aggregate_variable(fold_results, tradable_trades, variable)

        # Store results
        results[f'{variable}_summary'] = summary
        results[f'fold_results_{variable}'] = fold_results

    return results


def create_empty_result(strategy: pd.Series) -> Dict:
    """
    Create empty result for strategies that failed processing.

    Parameters
    ----------
    strategy : pd.Series
        Strategy parameters

    Returns
    -------
    dict
        Empty result dict with default values
    """
    from stage3_modules.strategy_aggregator import StrategyThresholdSummary

    return {
        'strategy': strategy,
        'price_summary': StrategyThresholdSummary(
            variable='price',
            n_folds_eligible=0,
            n_folds_with_accepted_threshold=0,
            fold_proposed_thresholds=[],
            n_folds_oos_eligible=0,
            n_folds_oos_passed=0,
            oos_pass_rate=0.0,
            strategy_threshold=0.0,
            threshold_approved=False,
            rejection_reason="insufficient_tradable_trades_or_folds"
        ),
        'buy_volume_summary': StrategyThresholdSummary(
            variable='buy_volume',
            n_folds_eligible=0,
            n_folds_with_accepted_threshold=0,
            fold_proposed_thresholds=[],
            n_folds_oos_eligible=0,
            n_folds_oos_passed=0,
            oos_pass_rate=0.0,
            strategy_threshold=0.0,
            threshold_approved=False,
            rejection_reason="insufficient_tradable_trades_or_folds"
        ),
        'tradable_trades': pd.DataFrame(),
        'fold_results_price': [],
        'fold_results_volume': []
    }


def main():
    """Main entry point for Stage 3 pipeline."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Stage 3: Threshold Discovery Pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='stage3_config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()

    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)

    # Setup logging
    logger = setup_logging(config)

    logger.info("=" * 70)
    logger.info("STAGE 3 - THRESHOLD DISCOVERY PIPELINE")
    logger.info("=" * 70)
    logger.info(f"Configuration: {args.config}")

    # Set random seed for reproducibility
    np.random.seed(config['bootstrap']['random_seed'])
    logger.info(f"Random seed: {config['bootstrap']['random_seed']}")

    # Step 1: Load data
    data_loader = DataLoader(config, logger)
    raw_data, stage2_strategies = data_loader.load_and_validate()

    # Step 2: Process each strategy
    logger.info("=" * 70)
    logger.info(f"PROCESSING {len(stage2_strategies)} STRATEGIES")
    logger.info("=" * 70)

    all_results = []

    # Sequential processing (can be parallelized later)
    for idx, (_, strategy) in enumerate(stage2_strategies.iterrows()):
        result = process_single_strategy(
            idx, strategy, raw_data, strategy, config, logger
        )
        all_results.append(result)

    # Step 3: Apply FDR correction across all strategies
    fdr_controller = FDRController(config, logger)
    all_results = fdr_controller.apply_fdr_correction(all_results)

    # Step 4: Generate outputs
    output_generator = OutputGenerator(config, logger)
    output_generator.generate_outputs(all_results)

    logger.info("=" * 70)
    logger.info("STAGE 3 PIPELINE COMPLETE")
    logger.info("=" * 70)

    print("\nStage 3 pipeline completed successfully!")
    print(f"Check output files in: {config['data']['output_dir']}")
    print(f"Log file: {config['logging']['log_file']}")


if __name__ == "__main__":
    main()
