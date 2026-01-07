"""
Stage 4: Categorical Exclusion Discovery Pipeline

Main orchestrator that discovers harmful categorical values (industry_encoded, weeks_after_earnings)
to exclude from trading strategies using anchored walk-forward validation and FDR correction.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Any
import yaml
import pandas as pd
import numpy as np

# Import Stage 4 modules
from stage4_modules.data_loader import DataLoader
from stage4_modules.strategy_filter_applier import StrategyFilterApplier
from stage4_modules.categorical_candidate_generator import CategoricalCandidateGenerator
from stage4_modules.categorical_screener import CategoricalScreener
from stage4_modules.strategy_aggregator import StrategyAggregator, StrategyExclusionSummary
from stage4_modules.fdr_controller import FDRController
from stage4_modules.output_generator import OutputGenerator

# Import reusable Stage 2 modules
from stage2_modules.fold_generator import FoldGenerator, InsufficientFoldsError


def load_config(config_path: str = "stage4_config.yaml") -> dict:
    """
    Load YAML configuration file.

    Parameters:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(config: dict) -> logging.Logger:
    """
    Setup logging (thread-safe for multiprocessing).

    Parameters:
        config: Configuration dictionary

    Returns:
        Logger instance
    """
    logger = logging.getLogger('Stage4')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # Console handler (always)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler (only if not multiprocessing to avoid conflicts)
    if not config['performance']['use_multiprocessing']:
        log_path = Path(config['data']['output_dir']) / config['logging']['log_file']
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(console_formatter)
        logger.addHandler(file_handler)

    return logger


def create_empty_result(strategy: pd.Series) -> Dict[str, Any]:
    """
    Return empty result for failed strategies.

    Parameters:
        strategy: Strategy parameters (Series with TP/SL/HP)

    Returns:
        Dict with empty summary and fold results
    """
    return {
        'summary': StrategyExclusionSummary(
            target_profit_pct=strategy['target_profit_pct'],
            stop_loss_pct=strategy['stop_loss_pct'],
            holding_period_weeks=strategy['holding_period_weeks'],
            excluded_industry_values=[],
            excluded_weeks_values=[],
            n_excluded_industries=0,
            n_excluded_weeks=0,
            expectancy_base_strategy=0.0,
            expectancy_after_industry_exclusions=0.0,
            expectancy_after_weeks_exclusions=0.0,
            expectancy_after_both_exclusions=0.0,
            lift_industry=0.0,
            lift_weeks=0.0,
            lift_both=0.0,
            industry_exclusion_filter_rate=0.0,
            weeks_exclusion_filter_rate=0.0,
            both_exclusion_filter_rate=0.0,
            eligible_folds_industry_median=0.0,
            eligible_folds_weeks_median=0.0,
            industry_p_value=1.0,
            weeks_p_value=1.0,
            industry_exclusions_applied=False,
            weeks_exclusions_applied=False
        ),
        'fold_results_industry': [],
        'fold_results_weeks': []
    }


def process_single_strategy(
    strategy_idx: int,
    strategy: pd.Series,
    raw_data: pd.DataFrame,
    stage2_row: pd.Series,
    config: dict,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Process one strategy through full Stage 4 pipeline.

    Parameters:
        strategy_idx: Strategy index (for random seed)
        strategy: Strategy parameters (Series with TP/SL/HP)
        raw_data: Full raw data DataFrame
        stage2_row: Stage 2 output row for this strategy
        config: Configuration dictionary
        logger: Logger instance

    Returns:
        Dict with 'summary' (StrategyExclusionSummary) and fold results
    """
    # Set per-strategy random seed for reproducibility
    np.random.seed(config['bootstrap']['random_seed'] + strategy_idx)

    try:
        # Step 1: Build BASE strategy trades (apply TP/SL/HP only, no feature/threshold filters)
        filter_applier = StrategyFilterApplier(config, logger)
        tradable_trades = filter_applier.build_tradable_trades(raw_data, strategy, stage2_row)

        if len(tradable_trades) == 0:
            return create_empty_result(strategy)

        # Step 2: Generate folds
        fold_generator = FoldGenerator(config, logger)
        try:
            folds = fold_generator.generate_folds(tradable_trades)
        except InsufficientFoldsError:
            return create_empty_result(strategy)

        # Step 3: Generate candidate values
        candidate_gen = CategoricalCandidateGenerator(config, logger)
        candidates = candidate_gen.generate_all_candidates(tradable_trades)

        if not candidates['industry_encoded'] and not candidates['weeks_after_earnings']:
            return create_empty_result(strategy)

        # Step 4: Screen categorical values for each variable
        screener = CategoricalScreener(config, logger)

        fold_results_industry = screener.screen_all_folds(
            tradable_trades, folds, 'industry_encoded', candidates['industry_encoded']
        )

        fold_results_weeks = screener.screen_all_folds(
            tradable_trades, folds, 'weeks_after_earnings', candidates['weeks_after_earnings']
        )

        # Step 5: Aggregate to strategy level
        aggregator = StrategyAggregator(config, logger)
        summary = aggregator.aggregate_strategy(
            fold_results_industry,
            fold_results_weeks,
            tradable_trades,
            strategy
        )

        return {
            'summary': summary,
            'fold_results_industry': fold_results_industry,
            'fold_results_weeks': fold_results_weeks
        }

    except Exception as e:
        logger.error(
            f"Error processing strategy {strategy_idx} "
            f"(TP={strategy['target_profit_pct']}, SL={strategy['stop_loss_pct']}, "
            f"HP={strategy['holding_period_weeks']}): {e}"
        )
        return create_empty_result(strategy)


def main():
    """Main pipeline orchestrator."""
    print("=" * 80)
    print("Stage 4: Categorical Exclusion Discovery Pipeline")
    print("=" * 80)

    # 1. Load configuration
    print("\n[1/5] Loading configuration...")
    config = load_config()
    logger = setup_logging(config)
    logger.info("Configuration loaded successfully.")

    # 2. Load data
    print("[2/5] Loading data...")
    data_loader = DataLoader(config, logger)
    raw_data, stage2_strategies, stage3_strategies = data_loader.load_and_validate()

    logger.info(f"Loaded {len(raw_data)} raw trades")
    logger.info(f"Loaded {len(stage3_strategies)} strategies from Stage 3")

    # 3. Process strategies
    print("[3/5] Processing strategies...")
    use_multiprocessing = config['performance']['use_multiprocessing']

    if use_multiprocessing:
        from joblib import Parallel, delayed
        logger.info(f"Using multiprocessing (n_jobs=-1)...")

        results = Parallel(n_jobs=-1, verbose=10)(
            delayed(process_single_strategy)(
                idx,
                stage3_strategies.iloc[idx],
                raw_data,
                stage2_strategies.iloc[idx],
                config,
                logger
            )
            for idx in range(len(stage3_strategies))
        )
    else:
        logger.info("Using serial processing...")
        results = []
        for idx in range(len(stage3_strategies)):
            if idx % 50 == 0:
                logger.info(f"Processing strategy {idx + 1}/{len(stage3_strategies)}...")

            result = process_single_strategy(
                idx,
                stage3_strategies.iloc[idx],
                raw_data,
                stage2_strategies.iloc[idx],
                config,
                logger
            )
            results.append(result)

    logger.info(f"Processed {len(results)} strategies.")

    # 4. Apply FDR correction
    print("[4/5] Applying FDR correction...")
    strategy_summaries = [r['summary'] for r in results]

    fdr_controller = FDRController(config, logger)
    fdr_controller.apply_fdr_correction(strategy_summaries)

    # Count final exclusions
    n_industry_exclusions = sum(1 for s in strategy_summaries if s.industry_exclusions_applied)
    n_weeks_exclusions = sum(1 for s in strategy_summaries if s.weeks_exclusions_applied)

    logger.info(f"Final exclusions: {n_industry_exclusions} industry, {n_weeks_exclusions} weeks")

    # 5. Generate outputs
    print("[5/5] Generating output files...")

    # Build all_fold_results dict for diagnostics
    all_fold_results = {}
    for idx, result in enumerate(results):
        strategy = stage3_strategies.iloc[idx]
        strategy_id = (
            f"TP{strategy['target_profit_pct']}_"
            f"SL{strategy['stop_loss_pct']}_"
            f"HP{strategy['holding_period_weeks']}"
        )
        all_fold_results[strategy_id] = {
            'industry_encoded': result['fold_results_industry'],
            'weeks_after_earnings': result['fold_results_weeks']
        }

    output_generator = OutputGenerator(config, logger)
    output_generator.generate_outputs(strategy_summaries, all_fold_results)

    # Final summary
    print("\n" + "=" * 80)
    print("Stage 4 Complete!")
    print("=" * 80)
    print(f"Total strategies processed: {len(strategy_summaries)}")
    print(f"Strategies with industry exclusions: {n_industry_exclusions}")
    print(f"Strategies with weeks exclusions: {n_weeks_exclusions}")
    print("\nOutput files:")
    print(f"  - {config['output']['optimal_exclusions_file']}")
    print(f"  - {config['output']['diagnostics_file']}")
    print(f"  - {config['logging']['log_file']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
