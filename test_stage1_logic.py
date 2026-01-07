#!/usr/bin/env python3
"""
UNIT TESTS FOR STAGE 1 SIMULATION LOGIC

Tests the core simulation logic with known input/output pairs to validate:
1. Target profit triggering
2. Stop loss triggering
3. Both triggering (precedence handling)
4. Time exit (median calculation)
5. Edge cases (NaN handling, negative values, etc.)

Author: Professional Quant Trader
Date: 2025-12-26
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path to import stage_1_sl_tp_hp
sys.path.insert(0, str(Path(__file__).parent))

from stage_1_sl_tp_hp import (
    StrategyParams,
    simulate_strategy,
    calculate_metrics,
    load_config
)


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'


def print_test_header(test_name):
    """Print formatted test header"""
    print(f"\n{Colors.BLUE}{'='*80}{Colors.RESET}")
    print(f"{Colors.BLUE}TEST: {test_name}{Colors.RESET}")
    print(f"{Colors.BLUE}{'='*80}{Colors.RESET}")


def print_pass(message):
    """Print pass message"""
    print(f"{Colors.GREEN}PASS:{Colors.RESET} {message}")


def print_fail(message):
    """Print fail message"""
    print(f"{Colors.RED}FAIL:{Colors.RESET} {message}")


def print_info(message):
    """Print info message"""
    print(f"{Colors.YELLOW}INFO:{Colors.RESET} {message}")


def assert_equal(actual, expected, description):
    """Assert equality with detailed output"""
    if actual == expected:
        print_pass(f"{description}: {actual} == {expected}")
        return True
    else:
        print_fail(f"{description}: {actual} != {expected}")
        return False


def assert_close(actual, expected, description, tolerance=0.0001):
    """Assert numerical closeness"""
    if abs(actual - expected) < tolerance:
        print_pass(f"{description}: {actual} ~= {expected}")
        return True
    else:
        print_fail(f"{description}: {actual} != {expected} (diff: {abs(actual - expected)})")
        return False


# ============================================================================
# TEST 1: Target Profit Triggering
# ============================================================================

def test_target_profit_trigger():
    """Test that target profit triggers correctly and returns the TP value"""
    print_test_header("Target Profit Triggering")

    # Create test data: Week 2 hits 6% gain (above 5% target)
    test_data = pd.DataFrame({
        'max_gain_1week': [2.0],
        'max_loss_1week': [-1.0],
        'max_gain_before_max_loss_1week': [1],
        'max_gain_2week': [6.0],  # TRIGGERS 5% target
        'max_loss_2week': [-0.5],
        'max_gain_before_max_loss_2week': [1],
        'max_gain_3week': [7.0],
        'max_loss_3week': [-2.0],
        'max_gain_before_max_loss_3week': [1],
    })

    params = StrategyParams(
        target_profit_pct=5.0,
        stop_loss_pct=3.0,
        holding_period_weeks=3
    )

    returns = simulate_strategy(test_data, params, None)

    print_info(f"Strategy: TP={params.target_profit_pct}%, SL={params.stop_loss_pct}%, HP={params.holding_period_weeks}w")
    print_info(f"Data: Week 2 gain=6% (triggers TP=5%)")

    all_pass = True
    all_pass &= assert_equal(len(returns), 1, "Number of trades")
    all_pass &= assert_close(returns[0], 5.0, "Return equals TP value")

    return all_pass


# ============================================================================
# TEST 2: Stop Loss Triggering
# ============================================================================

def test_stop_loss_trigger():
    """Test that stop loss triggers correctly and returns the SL value"""
    print_test_header("Stop Loss Triggering")

    # Create test data: Week 1 hits -4% loss (below -3% stop)
    test_data = pd.DataFrame({
        'max_gain_1week': [1.0],
        'max_loss_1week': [-4.0],  # TRIGGERS -3% stop
        'max_gain_before_max_loss_1week': [1],
        'max_gain_2week': [2.0],
        'max_loss_2week': [-5.0],
        'max_gain_before_max_loss_2week': [0],
        'max_gain_3week': [3.0],
        'max_loss_3week': [-6.0],
        'max_gain_before_max_loss_3week': [0],
    })

    params = StrategyParams(
        target_profit_pct=10.0,
        stop_loss_pct=3.0,
        holding_period_weeks=3
    )

    returns = simulate_strategy(test_data, params, None)

    print_info(f"Strategy: TP={params.target_profit_pct}%, SL={params.stop_loss_pct}%, HP={params.holding_period_weeks}w")
    print_info(f"Data: Week 1 loss=-4% (triggers SL=-3%)")

    all_pass = True
    all_pass &= assert_equal(len(returns), 1, "Number of trades")
    all_pass &= assert_close(returns[0], -3.0, "Return equals -SL value")

    return all_pass


# ============================================================================
# TEST 3: Both Trigger - Gain Comes First
# ============================================================================

def test_both_trigger_gain_first():
    """Test that when both trigger, precedence determines which to use (gain first)"""
    print_test_header("Both Trigger - Gain Comes First")

    # Create test data: Week 3 both trigger, gain came first
    test_data = pd.DataFrame({
        'max_gain_1week': [2.0],
        'max_loss_1week': [-1.0],
        'max_gain_before_max_loss_1week': [1],
        'max_gain_2week': [3.0],
        'max_loss_2week': [-2.0],
        'max_gain_before_max_loss_2week': [1],
        'max_gain_3week': [8.0],  # TRIGGERS TP=5%
        'max_loss_3week': [-5.0],  # ALSO TRIGGERS SL=-4%
        'max_gain_before_max_loss_3week': [1],  # GAIN CAME FIRST
    })

    params = StrategyParams(
        target_profit_pct=5.0,
        stop_loss_pct=4.0,
        holding_period_weeks=3
    )

    returns = simulate_strategy(test_data, params, None)

    print_info(f"Strategy: TP={params.target_profit_pct}%, SL={params.stop_loss_pct}%, HP={params.holding_period_weeks}w")
    print_info(f"Data: Week 3 gain=8%, loss=-5% (both trigger, gain_first=1)")

    all_pass = True
    all_pass &= assert_equal(len(returns), 1, "Number of trades")
    all_pass &= assert_close(returns[0], 5.0, "Return equals TP (gain came first)")

    return all_pass


# ============================================================================
# TEST 4: Both Trigger - Loss Comes First
# ============================================================================

def test_both_trigger_loss_first():
    """Test that when both trigger, precedence determines which to use (loss first)"""
    print_test_header("Both Trigger - Loss Comes First")

    # Create test data: Week 2 both trigger, loss came first
    test_data = pd.DataFrame({
        'max_gain_1week': [1.0],
        'max_loss_1week': [-1.0],
        'max_gain_before_max_loss_1week': [1],
        'max_gain_2week': [10.0],  # TRIGGERS TP=5%
        'max_loss_2week': [-6.0],  # ALSO TRIGGERS SL=-4%
        'max_gain_before_max_loss_2week': [0],  # LOSS CAME FIRST
        'max_gain_3week': [12.0],
        'max_loss_3week': [-7.0],
        'max_gain_before_max_loss_3week': [0],
    })

    params = StrategyParams(
        target_profit_pct=5.0,
        stop_loss_pct=4.0,
        holding_period_weeks=3
    )

    returns = simulate_strategy(test_data, params, None)

    print_info(f"Strategy: TP={params.target_profit_pct}%, SL={params.stop_loss_pct}%, HP={params.holding_period_weeks}w")
    print_info(f"Data: Week 2 gain=10%, loss=-6% (both trigger, gain_first=0)")

    all_pass = True
    all_pass &= assert_equal(len(returns), 1, "Number of trades")
    all_pass &= assert_close(returns[0], -4.0, "Return equals -SL (loss came first)")

    return all_pass


# ============================================================================
# TEST 5: Time Exit - Positive Median
# ============================================================================

def test_time_exit_positive():
    """Test time exit with positive median return"""
    print_test_header("Time Exit - Positive Median")

    # Create test data: Neither triggers, final week has median > 0
    test_data = pd.DataFrame({
        'max_gain_1week': [1.0],
        'max_loss_1week': [-0.5],
        'max_gain_before_max_loss_1week': [1],
        'max_gain_2week': [2.0],
        'max_loss_2week': [-1.0],
        'max_gain_before_max_loss_2week': [1],
        'max_gain_3week': [3.0],  # Doesn't trigger TP=10%
        'max_loss_3week': [-1.5],  # Doesn't trigger SL=-5%
        'max_gain_before_max_loss_3week': [1],
    })

    params = StrategyParams(
        target_profit_pct=10.0,
        stop_loss_pct=5.0,
        holding_period_weeks=3
    )

    returns = simulate_strategy(test_data, params, None)

    expected_return = np.median([3.0, -1.5])  # = 0.75

    print_info(f"Strategy: TP={params.target_profit_pct}%, SL={params.stop_loss_pct}%, HP={params.holding_period_weeks}w")
    print_info(f"Data: Week 3 gain=3%, loss=-1.5% (neither triggers)")
    print_info(f"Expected: median(3.0, -1.5) = {expected_return}")

    all_pass = True
    all_pass &= assert_equal(len(returns), 1, "Number of trades")
    all_pass &= assert_close(returns[0], expected_return, "Return equals median")

    return all_pass


# ============================================================================
# TEST 6: Time Exit - Negative Median
# ============================================================================

def test_time_exit_negative():
    """Test time exit with negative median return"""
    print_test_header("Time Exit - Negative Median")

    # Create test data: Neither triggers, final week has median < 0
    test_data = pd.DataFrame({
        'max_gain_1week': [0.5],
        'max_loss_1week': [-1.0],
        'max_gain_before_max_loss_1week': [1],
        'max_gain_2week': [1.0],  # Doesn't trigger TP=10%
        'max_loss_2week': [-6.0],  # Doesn't trigger SL=-8%
        'max_gain_before_max_loss_2week': [1],
    })

    params = StrategyParams(
        target_profit_pct=10.0,
        stop_loss_pct=8.0,
        holding_period_weeks=2
    )

    returns = simulate_strategy(test_data, params, None)

    expected_return = np.median([1.0, -6.0])  # = -2.5

    print_info(f"Strategy: TP={params.target_profit_pct}%, SL={params.stop_loss_pct}%, HP={params.holding_period_weeks}w")
    print_info(f"Data: Week 2 gain=1%, loss=-6% (neither triggers)")
    print_info(f"Expected: median(1.0, -6.0) = {expected_return}")

    all_pass = True
    all_pass &= assert_equal(len(returns), 1, "Number of trades")
    all_pass &= assert_close(returns[0], expected_return, "Return equals median")

    return all_pass


# ============================================================================
# TEST 7: NaN Handling - Missing Final Week
# ============================================================================

def test_nan_handling_missing_final_week():
    """Test that rows with NaN in final week are skipped"""
    print_test_header("NaN Handling - Missing Final Week")

    # Create test data: Week 3 has NaN (sell happened before week 3)
    test_data = pd.DataFrame({
        'max_gain_1week': [1.0],
        'max_loss_1week': [-0.5],
        'max_gain_before_max_loss_1week': [1],
        'max_gain_2week': [2.0],
        'max_loss_2week': [-1.0],
        'max_gain_before_max_loss_2week': [1],
        'max_gain_3week': [np.nan],  # MISSING DATA
        'max_loss_3week': [np.nan],  # MISSING DATA
        'max_gain_before_max_loss_3week': [np.nan],
    })

    params = StrategyParams(
        target_profit_pct=5.0,
        stop_loss_pct=3.0,
        holding_period_weeks=3
    )

    returns = simulate_strategy(test_data, params, None)

    print_info(f"Strategy: TP={params.target_profit_pct}%, SL={params.stop_loss_pct}%, HP={params.holding_period_weeks}w")
    print_info(f"Data: Week 3 has NaN values (should skip row)")

    all_pass = True
    all_pass &= assert_equal(len(returns), 0, "No trades (row skipped)")

    return all_pass


# ============================================================================
# TEST 8: NaN Handling - Missing Mid-Week
# ============================================================================

def test_nan_handling_missing_mid_week():
    """Test that simulation stops when encountering NaN mid-holding period"""
    print_test_header("NaN Handling - Missing Mid-Week")

    # Create test data: Week 2 has NaN (should break and skip)
    test_data = pd.DataFrame({
        'max_gain_1week': [1.0],
        'max_loss_1week': [-0.5],
        'max_gain_before_max_loss_1week': [1],
        'max_gain_2week': [np.nan],  # MISSING DATA
        'max_loss_2week': [np.nan],  # MISSING DATA
        'max_gain_before_max_loss_2week': [np.nan],
        'max_gain_3week': [5.0],
        'max_loss_3week': [-2.0],
        'max_gain_before_max_loss_3week': [1],
    })

    params = StrategyParams(
        target_profit_pct=10.0,
        stop_loss_pct=5.0,
        holding_period_weeks=3
    )

    returns = simulate_strategy(test_data, params, None)

    print_info(f"Strategy: TP={params.target_profit_pct}%, SL={params.stop_loss_pct}%, HP={params.holding_period_weeks}w")
    print_info(f"Data: Week 2 has NaN (should break and skip)")

    all_pass = True
    all_pass &= assert_equal(len(returns), 0, "No trades (row skipped due to mid-week NaN)")

    return all_pass


# ============================================================================
# TEST 9: Edge Case - Both Values Negative
# ============================================================================

def test_edge_case_both_negative():
    """Test edge case where both max_gain and max_loss are negative (stock crash)"""
    print_test_header("Edge Case - Both Values Negative")

    # Create test data: Stock crashes, even max_gain is negative
    test_data = pd.DataFrame({
        'max_gain_1week': [-2.0],  # Best was still a loss
        'max_loss_1week': [-8.0],  # Worst was bigger loss
        'max_gain_before_max_loss_1week': [1],
        'max_gain_2week': [-3.0],  # Still doesn't trigger TP=10%
        'max_loss_2week': [-12.0],  # Still doesn't trigger SL=-15%
        'max_gain_before_max_loss_2week': [1],
    })

    params = StrategyParams(
        target_profit_pct=10.0,
        stop_loss_pct=15.0,
        holding_period_weeks=2
    )

    returns = simulate_strategy(test_data, params, None)

    expected_return = np.median([-3.0, -12.0])  # = -7.5

    print_info(f"Strategy: TP={params.target_profit_pct}%, SL={params.stop_loss_pct}%, HP={params.holding_period_weeks}w")
    print_info(f"Data: Week 2 gain=-3%, loss=-12% (both negative, neither triggers)")
    print_info(f"Expected: median(-3.0, -12.0) = {expected_return}")

    all_pass = True
    all_pass &= assert_equal(len(returns), 1, "Number of trades")
    all_pass &= assert_close(returns[0], expected_return, "Return equals median of negative values")

    return all_pass


# ============================================================================
# TEST 10: Multiple Rows
# ============================================================================

def test_multiple_rows():
    """Test processing multiple rows with different outcomes"""
    print_test_header("Multiple Rows Processing")

    # Create test data: 3 rows with different outcomes
    test_data = pd.DataFrame({
        # Row 0: Target profit triggers week 1
        'max_gain_1week': [6.0, 2.0, 1.0],
        'max_loss_1week': [-1.0, -5.0, -1.0],
        'max_gain_before_max_loss_1week': [1, 1, 1],
        # Row 1: Stop loss triggers week 2
        'max_gain_2week': [7.0, 3.0, 2.0],
        'max_loss_2week': [-2.0, -6.0, -2.0],
        'max_gain_before_max_loss_2week': [1, 0, 1],
        # Row 2: Time exit week 3
        'max_gain_3week': [8.0, 4.0, 3.0],
        'max_loss_3week': [-3.0, -7.0, -1.0],
        'max_gain_before_max_loss_3week': [1, 0, 1],
    })

    params = StrategyParams(
        target_profit_pct=5.0,
        stop_loss_pct=4.0,
        holding_period_weeks=3
    )

    returns = simulate_strategy(test_data, params, None)

    print_info(f"Strategy: TP={params.target_profit_pct}%, SL={params.stop_loss_pct}%, HP={params.holding_period_weeks}w")
    print_info(f"Row 0: Week 1 gain=6% -> TP triggers -> return=+5%")
    print_info(f"Row 1: Week 2 loss=-6% -> SL triggers -> return=-4%")
    print_info(f"Row 2: Week 3 neither triggers -> median(3, -1)=1% -> return=+1%")

    all_pass = True
    all_pass &= assert_equal(len(returns), 3, "Number of trades")
    all_pass &= assert_close(returns[0], 5.0, "Row 0: TP triggered")
    all_pass &= assert_close(returns[1], -4.0, "Row 1: SL triggered")
    all_pass &= assert_close(returns[2], 1.0, "Row 2: Time exit")

    return all_pass


# ============================================================================
# TEST 11: Metrics Calculation
# ============================================================================

def test_metrics_calculation():
    """Test that metrics are calculated correctly from returns"""
    print_test_header("Metrics Calculation")

    # Known returns: 3 wins, 2 losses
    returns = [5.0, -3.0, 2.0, -4.0, 1.0]

    # Expected metrics
    winning_trades = [5.0, 2.0, 1.0]
    losing_trades = [-3.0, -4.0]

    expected_win_rate = 3 / 5  # 60%
    expected_avg_win = np.mean(winning_trades)  # 2.667
    expected_avg_loss = abs(np.mean(losing_trades))  # 3.5
    expected_expectancy = (expected_win_rate * expected_avg_win) - ((1 - expected_win_rate) * expected_avg_loss)

    # Calculate using our function
    config = load_config(str(Path(__file__).parent / 'stage1_config.yaml'))
    params = StrategyParams(5.0, 3.0, 3)
    metrics = calculate_metrics(params, returns, config)

    print_info(f"Returns: {returns}")
    print_info(f"Wins: {winning_trades}, Losses: {losing_trades}")

    all_pass = True
    all_pass &= assert_equal(metrics.trade_count, 5, "Trade count")
    all_pass &= assert_close(metrics.win_rate, expected_win_rate, "Win rate", 0.001)
    all_pass &= assert_close(metrics.avg_win, expected_avg_win, "Average win", 0.001)
    all_pass &= assert_close(metrics.avg_loss, expected_avg_loss, "Average loss", 0.001)
    all_pass &= assert_close(metrics.expectancy, expected_expectancy, "Expectancy", 0.001)

    # Profit factor
    gross_profit = sum(winning_trades)  # 8.0
    gross_loss = abs(sum(losing_trades))  # 7.0
    expected_pf = gross_profit / gross_loss  # 1.143
    all_pass &= assert_close(metrics.profit_factor, expected_pf, "Profit factor", 0.001)

    return all_pass


# ============================================================================
# RUN ALL TESTS
# ============================================================================

def run_all_tests():
    """Run all test cases and report results"""
    print("\n" + "="*80)
    print("STAGE 1 UNIT TEST SUITE")
    print("="*80)

    tests = [
        ("Target Profit Triggering", test_target_profit_trigger),
        ("Stop Loss Triggering", test_stop_loss_trigger),
        ("Both Trigger - Gain First", test_both_trigger_gain_first),
        ("Both Trigger - Loss First", test_both_trigger_loss_first),
        ("Time Exit - Positive Median", test_time_exit_positive),
        ("Time Exit - Negative Median", test_time_exit_negative),
        ("NaN Handling - Missing Final Week", test_nan_handling_missing_final_week),
        ("NaN Handling - Missing Mid-Week", test_nan_handling_missing_mid_week),
        ("Edge Case - Both Negative", test_edge_case_both_negative),
        ("Multiple Rows Processing", test_multiple_rows),
        ("Metrics Calculation", test_metrics_calculation),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            passed = test_func()
            results.append((test_name, passed))
        except Exception as e:
            print_fail(f"EXCEPTION in {test_name}: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for test_name, passed in results:
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if passed else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"{status} - {test_name}")

    print("="*80)
    print(f"Results: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print(f"{Colors.GREEN}ALL TESTS PASSED!{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}SOME TESTS FAILED{Colors.RESET}")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
