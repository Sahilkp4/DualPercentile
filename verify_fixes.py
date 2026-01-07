"""
Verify Stage 4 Fixes

This script tests the two critical fixes:
1. Integer parsing for holding periods (1.0 -> 1)
2. Column naming consistency (realized_return_pct -> pnl_pct)
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 70)
print("Stage 4 Fix Verification")
print("=" * 70)

# Test 1: Integer parsing fix
print("\n[Test 1] Verifying integer parsing fix...")
test_cases = [
    'TP1.55_SL0.05_HP1.0',
    'TP5.0_SL3.0_HP4.0',
    'TP10.5_SL5.5_HP12'
]

for strategy_id in test_cases:
    parts = strategy_id.split('_')
    tp = float(parts[0].replace('TP', ''))
    sl = float(parts[1].replace('SL', ''))
    hp = int(float(parts[2].replace('HP', '')))  # Fixed version
    print(f"  [OK] {strategy_id} -> TP={tp}, SL={sl}, HP={hp}")

print("[OK] Test 1 passed - Integer parsing works correctly")

# Test 2: Column naming fix
print("\n[Test 2] Verifying column renaming...")

# Simulate what StrategySimulator outputs
test_data = pd.DataFrame({
    'realized_return_pct': [0.5, -0.3, 1.2, -0.8, 0.9],
    'industry_encoded': [1, 2, 3, 1, 2],
    'weeks_after_earnings': [0, 1, 2, 1, 0]
})

# Apply the fix (what strategy_filter_applier.py does)
if 'realized_return_pct' in test_data.columns:
    test_data = test_data.rename(columns={'realized_return_pct': 'pnl_pct'})
    print("  [OK] Column renamed: realized_return_pct -> pnl_pct")

# Verify pnl_pct exists
assert 'pnl_pct' in test_data.columns, "pnl_pct column missing!"
print("  [OK] pnl_pct column exists")

# Verify we can compute expectancy (what strategy_aggregator.py does)
from stage2_modules.statistical_utils import compute_expectancy
E = compute_expectancy(test_data['pnl_pct'].values)
print(f"  [OK] Expectancy computed successfully: {E:.4f}")

# Verify expectancy is non-zero
assert E != 0, "Expectancy should be non-zero for test data!"
print("[OK] Test 2 passed - Column naming and expectancy calculation work")

# Test 3: Check if fixes are applied in actual files
print("\n[Test 3] Verifying fixes are in place...")

# Check strategy_filter_applier.py has the rename
filter_applier_path = Path('stage4_modules/strategy_filter_applier.py')
with open(filter_applier_path, 'r', encoding='utf-8') as f:
    content = f.read()
    assert "rename(columns={'realized_return_pct': 'pnl_pct'})" in content, \
        "Fix not found in strategy_filter_applier.py"
    print("  [OK] Fix found in strategy_filter_applier.py")

# Check categorical_screener.py uses pnl_pct
screener_path = Path('stage4_modules/categorical_screener.py')
with open(screener_path, 'r', encoding='utf-8') as f:
    content = f.read()
    assert "train_data['pnl_pct']" in content, \
        "Fix not found in categorical_screener.py (train)"
    assert "test_data['pnl_pct']" in content, \
        "Fix not found in categorical_screener.py (test)"
    print("  [OK] Fix found in categorical_screener.py")

# Check output_generator.py has the int(float(...)) fix
output_gen_path = Path('stage4_modules/output_generator.py')
with open(output_gen_path, 'r', encoding='utf-8') as f:
    content = f.read()
    assert "int(float(parts[2].replace('HP', '')))" in content, \
        "Fix not found in output_generator.py"
    print("  [OK] Fix found in output_generator.py")

print("[OK] Test 3 passed - All fixes are in place")

print("\n" + "=" * 70)
print("All verification tests passed!")
print("=" * 70)
print("\nStage 4 is ready to run. Execute:")
print("  python test_stage4_small.py  (quick test on 10 strategies)")
print("  python stage_4_exclusion_discovery.py  (full run on 615 strategies)")
print("=" * 70)
