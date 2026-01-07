#!/usr/bin/env python3
"""
STAGE 2: STATISTICAL FEATURE SELECTION WITH TEMPORAL INTEGRITY

For each Stage 1 strategy, identify essential binary features using:
- Anchored walk-forward OOS validation
- Block bootstrap hypothesis testing
- BH-FDR correction within each fold
- Directed harm testing for interaction effects (simplified in this version)
- Average harm pruning for conflict resolution

Critical: Maintains strict temporal integrity (no data leakage)

Author: ME
Date: 2026-01-02
"""

import pandas as pd
import numpy as np
import yaml
import logging
import sys
from pathlib import Path
from typing import Dict, List
import warnings
from datetime import datetime
from tqdm import tqdm

# Add stage2_modules to path
sys.path.insert(0, str(Path(__file__).parent))

# Import all modules
from stage2_modules.data_loader import DataLoader
from stage2_modules.strategy_simulator import StrategySimulator
from stage2_modules.fold_generator import FoldGenerator, InsufficientFoldsError
from stage2_modules.univariate_screening import UnivariateScreener
from stage2_modules.cross_fold_aggregator import CrossFoldAggregator
from stage2_modules.harm_testing import HarmTester
from stage2_modules.conflict_resolver import ConflictResolver
from stage2_modules.final_validator import FinalValidator


def load_config(config_path: str = "stage2_config.yaml") -> dict:
    """Load configuration from YAML file."""
    config_file = Path(__file__).parent / config_path

    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Replace placeholders in paths
    output_dir = config['data']['output_dir']
    config['logging']['log_file'] = config['logging']['log_file'].replace('{output_dir}', output_dir)

    return config


def setup_logging(config: dict) -> logging.Logger:
    """Setup logging with file and console handlers."""
    logger = logging.getLogger('Stage2')
    logger.setLevel(config['logging']['log_level'])

    # Remove existing handlers
    logger.handlers = []

    # File handler
    log_file = Path(config['logging']['log_file'])
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(config['logging']['log_level'])

    # Console handler
    if config['logging']['log_to_console']:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(config['logging']['log_level'])
        logger.addHandler(console_handler)

    # Formatter
    formatter = logging.Formatter(
        config['logging']['log_format'],
        datefmt=config['logging']['date_format']
    )
    file_handler.setFormatter(formatter)

    if config['logging']['log_to_console']:
        console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def process_single_strategy(strategy_idx: int, strategy: pd.Series,
                            raw_data: pd.DataFrame, config: dict,
                            logger: logging.Logger) -> Dict:
    """
    Process a single strategy through all Stage 2 sub-stages.

    Parameters
    ----------
    strategy_idx : int
        Strategy index (for logging)
    strategy : pd.Series
        Strategy parameters from Stage 1
    raw_data : pd.DataFrame
        Raw ML-ready dataset
    config : dict
        Configuration dictionary
    logger : logging.Logger
        Logger instance

    Returns
    -------
    dict
        Results dictionary containing features and diagnostics
    """
    # Reduced logging - only log every 50 strategies or on errors
    if strategy_idx % 50 == 0 or strategy_idx == 0:
        logger.info(f"Processing strategy {strategy_idx + 1}/{1745}...")

    try:
        # Stage 2.1: Build strategy-specific working table
        simulator = StrategySimulator(config, logger)
        working_table = simulator.simulate_strategy(raw_data, strategy)

        if len(working_table) == 0:
            logger.warning("Strategy produced no valid trades, skipping")
            return create_empty_result(strategy)

        # Stage 2.2: Build anchored walk-forward folds
        fold_gen = FoldGenerator(config, logger)
        try:
            folds = fold_gen.generate_folds(working_table)
        except InsufficientFoldsError as e:
            # Enhanced logging with strategy context
            strat_id = f"TP={strategy['target_profit_pct']}, SL={strategy['stop_loss_pct']}, HP={strategy['holding_period_weeks']}"
            logger.warning(
                f"Strategy {strat_id}: Insufficient folds - {e}. "
                f"Working table has {len(working_table)} trades across "
                f"{working_table[config['time_integrity']['fold_group_col']].nunique()} unique quarters. "
                f"Skipping strategy."
            )
            return create_empty_result(strategy)

        # Stage 2.3: Univariate feature screening
        screener = UnivariateScreener(config, logger)
        fold_results = screener.screen_all_folds(working_table, folds)

        # Stage 2.4: Cross-fold aggregation
        aggregator = CrossFoldAggregator(config, logger)
        accepted_features = aggregator.aggregate(fold_results)

        if len(accepted_features) == 0:
            logger.info("No features accepted, outputting empty S_final")
            return create_empty_result(strategy)

        # Stage 2.5: Directed harm testing
        harm_tester = HarmTester(config, logger)
        harm_graph = harm_tester.test_all_pairs(working_table, folds, accepted_features)

        # Stage 2.6: Conflict resolution
        resolver = ConflictResolver(config, logger)
        pruned_features = resolver.resolve_conflicts(accepted_features, harm_graph)

        # Stage 2.7: Final viability check
        validator = FinalValidator(config, logger)
        final_features, is_viable, metrics = validator.validate_final_filter(
            working_table, pruned_features, strategy
        )

        # Store results
        return {
            'strategy': strategy,
            'features': final_features if is_viable else [],
            'is_viable': is_viable,
            'metrics': metrics,
            'fold_results': fold_results,
            'n_folds': len(folds)
        }

    except Exception as e:
        logger.error(f"Error processing strategy: {e}", exc_info=True)
        return create_empty_result(strategy)


def create_empty_result(strategy: pd.Series) -> Dict:
    """Create empty result for failed strategy."""
    return {
        'strategy': strategy,
        'features': [],
        'is_viable': False,
        'metrics': {},
        'fold_results': [],
        'n_folds': 0
    }


def save_results(all_results: List[Dict], config: dict, logger: logging.Logger):
    """Save Stage 2 output files."""
    logger.info("=" * 80)
    logger.info("SAVING RESULTS")
    logger.info("=" * 80)

    output_dir = Path(config['data']['output_dir'])

    # Save optimal features per strategy
    features_file = output_dir / config['output']['optimal_features_file']
    features_df = create_features_dataframe(all_results)
    features_df.to_csv(features_file, index=False, encoding='utf-8-sig')
    logger.info(f"Saved features file: {features_file} ({len(features_df)} strategies)")

    # Save diagnostics
    diagnostics_file = output_dir / config['output']['diagnostics_file']
    diagnostics_df = create_diagnostics_dataframe(all_results)
    diagnostics_df.to_csv(diagnostics_file, index=False, encoding='utf-8-sig')
    logger.info(f"Saved diagnostics file: {diagnostics_file} ({len(diagnostics_df)} rows)")


def create_features_dataframe(all_results: List[Dict]) -> pd.DataFrame:
    """Create output dataframe for optimal features per strategy."""
    rows = []

    for result in all_results:
        strategy = result['strategy']
        features = result['features']
        is_viable = result['is_viable']
        metrics = result['metrics']

        # Format features list
        feature_names = [f for f, c in features] if features else []
        features_str = ','.join(feature_names) if feature_names else ''

        row = {
            'target_profit_pct': strategy['target_profit_pct'],
            'stop_loss_pct': strategy['stop_loss_pct'],
            'holding_period_weeks': strategy['holding_period_weeks'],
            'n_features_selected': len(features),
            'features_selected': features_str,
            'is_viable': is_viable,
            'n_trades_final_filter': metrics.get('n_trades_final', 0),
            'expectancy_final_filter': metrics.get('expectancy_final', np.nan),
            'delta_vs_stage1': metrics.get('delta_vs_baseline', np.nan),
            'n_folds_eligible': result['n_folds']
        }

        rows.append(row)

    return pd.DataFrame(rows)


def create_diagnostics_dataframe(all_results: List[Dict]) -> pd.DataFrame:
    """Create diagnostic audit trail dataframe."""
    rows = []

    for result in all_results:
        strategy = result['strategy']
        fold_results = result['fold_results']

        for fold_result in fold_results:
            row = {
                'target_profit_pct': strategy['target_profit_pct'],
                'stop_loss_pct': strategy['stop_loss_pct'],
                'holding_period_weeks': strategy['holding_period_weeks'],
                'feature': fold_result.feature,
                'condition': fold_result.condition,
                'fold_id': fold_result.fold_id,
                'stage': 'univariate',
                'outcome': 'accepted' if fold_result.accepted_train else 'rejected',
                'n_train': fold_result.n_train,
                'delta_E_train': fold_result.delta_E_train,
                'p_value': fold_result.p_value,
                'ci_low': fold_result.ci_low,
                'ci_high': fold_result.ci_high,
                'n_test': fold_result.n_test,
                'delta_E_test': fold_result.delta_E_test if fold_result.oos_computed else np.nan,
                'oos_pass': fold_result.oos_passed if fold_result.oos_computed else np.nan
            }

            rows.append(row)

    return pd.DataFrame(rows)


def main():
    """Main pipeline execution."""
    print("=" * 80)
    print("STAGE 2: STATISTICAL FEATURE SELECTION PIPELINE")
    print("=" * 80)
    print("")

    # Load configuration
    config = load_config()
    logger = setup_logging(config)

    logger.info("Stage 2 pipeline started")
    start_time = datetime.now()

    # Stage 2.0: Load and validate data
    loader = DataLoader(config, logger)
    raw_data, stage1_candidates = loader.load_and_validate()

    # Process each strategy with progress bar
    all_results = []

    for idx, (_, strategy) in tqdm(
        enumerate(stage1_candidates.iterrows()),
        total=len(stage1_candidates),
        desc="Processing strategies",
        unit="strategy"
    ):
        result = process_single_strategy(idx, strategy, raw_data, config, logger)
        all_results.append(result)

    # Save results
    save_results(all_results, config, logger)

    # Summary
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    logger.info("=" * 80)
    logger.info("STAGE 2 COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Processed {len(all_results)} strategies")
    logger.info(f"Viable strategies: {sum(1 for r in all_results if r['is_viable'])}")
    logger.info(f"Execution time: {duration:.1f} seconds")
    logger.info("")

    print("\nStage 2 complete!")
    print(f"Results saved to: {config['data']['output_dir']}")


if __name__ == "__main__":
    main()
