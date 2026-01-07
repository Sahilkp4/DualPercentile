"""
Test Stage 4 on a small subset of strategies (10 strategies).

This script is for validation before running the full pipeline.
"""

import sys
import logging
from pathlib import Path
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
from stage2_modules.fold_generator import FoldGenerator, InsufficientFoldsError


def load_config(config_path: str = "stage4_config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict) -> logging.Logger:
    """Setup logging."""
    logger = logging.getLogger('Stage4Test')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def create_empty_result(strategy: pd.Series) -> dict:
    """Return empty result for failed strategies."""
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
) -> dict:
    """Process one strategy through full Stage 4 pipeline."""
    np.random.seed(config['bootstrap']['random_seed'] + strategy_idx)

    try:
        # Step 1: Build BASE strategy trades
        filter_applier = StrategyFilterApplier(config, logger)
        tradable_trades = filter_applier.build_tradable_trades(raw_data, strategy, stage2_row)

        if len(tradable_trades) == 0:
            logger.warning(f"Strategy {strategy_idx}: No tradable trades")
            return create_empty_result(strategy)

        # Step 2: Generate folds
        fold_generator = FoldGenerator(config, logger)
        try:
            folds = fold_generator.generate_folds(tradable_trades)
        except InsufficientFoldsError:
            logger.warning(f"Strategy {strategy_idx}: Insufficient folds")
            return create_empty_result(strategy)

        # Step 3: Generate candidate values
        candidate_gen = CategoricalCandidateGenerator(config, logger)
        candidates = candidate_gen.generate_all_candidates(tradable_trades)

        if not candidates['industry_encoded'] and not candidates['weeks_after_earnings']:
            logger.warning(f"Strategy {strategy_idx}: No candidate values")
            return create_empty_result(strategy)

        # Step 4: Screen categorical values
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

        logger.info(
            f"Strategy {strategy_idx}: "
            f"{summary.n_excluded_industries} industries, "
            f"{summary.n_excluded_weeks} weeks excluded"
        )

        return {
            'summary': summary,
            'fold_results_industry': fold_results_industry,
            'fold_results_weeks': fold_results_weeks
        }

    except Exception as e:
        logger.error(f"Error processing strategy {strategy_idx}: {e}")
        return create_empty_result(strategy)


def main():
    """Main test function."""
    print("=" * 80)
    print("Stage 4 Test - Small Subset (10 Strategies)")
    print("=" * 80)

    # Load configuration
    print("\nLoading configuration...")
    config = load_config()
    logger = setup_logging(config)

    # Load data
    print("Loading data...")
    data_loader = DataLoader(config, logger)
    raw_data, stage2_strategies, stage3_strategies = data_loader.load_and_validate()

    # LIMIT TO 10 STRATEGIES FOR TESTING
    print("\n!!! LIMITING TO 10 STRATEGIES FOR TESTING !!!\n")
    stage3_strategies = stage3_strategies.head(10)
    stage2_strategies = stage2_strategies.head(10)

    logger.info(f"Testing with {len(stage3_strategies)} strategies")

    # Process strategies (serial mode for debugging)
    print("Processing strategies...")
    results = []
    for idx in range(len(stage3_strategies)):
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

    # Apply FDR correction
    print("\nApplying FDR correction...")
    strategy_summaries = [r['summary'] for r in results]
    fdr_controller = FDRController(config, logger)
    fdr_controller.apply_fdr_correction(strategy_summaries)

    # Generate outputs
    print("Generating output files...")
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

    # Summary
    n_industry = sum(1 for s in strategy_summaries if s.industry_exclusions_applied)
    n_weeks = sum(1 for s in strategy_summaries if s.weeks_exclusions_applied)

    # Validation checks to verify fixes worked
    print("\n" + "=" * 80)
    print("VALIDATION CHECKS (Verifying Fixes)")
    print("=" * 80)

    import pandas as pd
    import os

    # Check 1: Base expectancies should be non-zero
    df_main = pd.read_csv(config['output']['optimal_exclusions_file'])
    n_nonzero_exp = (df_main['expectancy_base_strategy'] != 0).sum()
    print(f"[CHECK 1] Non-zero base expectancies: {n_nonzero_exp}/{len(df_main)}")
    if n_nonzero_exp == 0:
        print("  [WARNING] All expectancies are zero - column name fix may not have worked!")
    else:
        print("  [OK] Column name fix working - expectancies calculated correctly")

    # Check 2: Diagnostics file should be populated
    diag_path = config['output']['diagnostics_file']
    diag_size = os.path.getsize(diag_path) if os.path.exists(diag_path) else 0
    print(f"[CHECK 2] Diagnostics file size: {diag_size:,} bytes")
    if diag_size <= 2:  # Empty or just header
        print("  [WARNING] Diagnostics file is empty - no fold results generated!")
        print("  This may indicate all strategies failed screening or had insufficient folds.")
    else:
        try:
            df_diag = pd.read_csv(diag_path)
            print(f"  [OK] Diagnostics populated with {len(df_diag):,} test results")
            print(f"       Variables tested: {df_diag['variable'].unique().tolist()}")
            print(f"       Unique strategies: {df_diag[['target_profit_pct', 'stop_loss_pct', 'holding_period_weeks']].drop_duplicates().shape[0]}")
        except pd.errors.EmptyDataError:
            print("  [WARNING] Diagnostics file has no data - no fold results generated!")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    print(f"Strategies processed: {len(strategy_summaries)}")
    print(f"Strategies with industry exclusions: {n_industry}")
    print(f"Strategies with weeks exclusions: {n_weeks}")
    print("\nOutput files generated:")
    print(f"  - {config['output']['optimal_exclusions_file']}")
    print(f"  - {config['output']['diagnostics_file']}")
    print("=" * 80)


if __name__ == '__main__':
    main()
