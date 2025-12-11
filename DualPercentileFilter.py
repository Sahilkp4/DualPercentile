#!/usr/bin/env python3
"""
Enhanced Consolidation Pattern Analysis - Dual Percentile System (NO BACKWARD EXPANSION)

This version removes Phase 2 (backward expansion) and keeps only Phase 1 (dual percentile filtering).

OPTIMIZATIONS:
1. Bulk data fetching (1 query vs 10,000)
2. Uses pre-computed volatility_ma_1week column from database
3. Vectorized percentile calculations with pandas groupby
4. Expected 10-50x speedup vs original non-optimized version

LOGIC:
- Single-phase approach using dual percentile filtering
- Two percentile calculations: short-term vs long-term
- No window expansion or outlier detection
- Simple 5-day baseline window for all stocks
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import time
warnings.filterwarnings('ignore')


# Import database configuration
try:
    from config import DB_CONFIG
except ImportError:
    print("ERROR: config.py not found!")
    print("Please copy config_template.py to config.py and fill in your database credentials.")
    exit(1)


def get_db_connection():
    """Establish database connection"""
    return psycopg2.connect(**DB_CONFIG)


def get_all_trading_days(conn, start_date: str, end_date: Optional[str] = None) -> List[str]:
    """
    Get all trading days in date range.

    Returns:
        List of date strings in YYYY-MM-DD format
    """
    if end_date:
        query = """
        SELECT DISTINCT date
        FROM stocks_peaks_valleys_analysis
        WHERE date >= %s AND date <= %s
        ORDER BY date
        """
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
    else:
        query = """
        SELECT DISTINCT date
        FROM stocks_peaks_valleys_analysis
        WHERE date >= %s
        ORDER BY date
        """
        df = pd.read_sql_query(query, conn, params=(start_date,))

    return [d.strftime('%Y-%m-%d') for d in df['date'].tolist()]


def get_bulk_stock_data(conn, pattern_end_date: str, lookback_date: str) -> pd.DataFrame:
    """
    OPTIMIZATION 1: Fetch ALL stock data in a single query instead of 10,000 individual queries.

    Uses pre-computed volatility_ma_1week column which is equivalent to baseline avg_pct metric.

    Args:
        conn: Database connection
        pattern_end_date: End date for analysis (e.g., '2024-06-07')
        lookback_date: Start date for 5-year lookback

    Returns:
        DataFrame with all stock data, grouped-ready
    """
    print(f"\n[OPTIMIZATION] Fetching all stock data in single bulk query...")
    start_time = time.time()

    query = """
    WITH stock_latest AS (
        -- Pre-filter: Only symbols with data on pattern_end_date and price >= $5
        SELECT DISTINCT symbol, close
        FROM stocks_peaks_valleys_analysis
        WHERE date = %(pattern_end_date)s
          AND close >= 5.0
    ),
    stock_data AS (
        -- Fetch all data for qualifying stocks
        SELECT
            s.symbol,
            s.date,
            s.close,
            s.peak_valley_type,
            s.pct_change_from_prev_extreme,
            s.volatility_ma_1week,  -- PRE-COMPUTED: Equivalent to baseline avg_pct!
            s.is_peak,
            s.is_valley
        FROM stocks_peaks_valleys_analysis s
        INNER JOIN stock_latest l ON s.symbol = l.symbol
        WHERE s.date BETWEEN %(lookback_date)s AND %(pattern_end_date)s
          AND s.peak_valley_type IS NOT NULL
    )
    SELECT * FROM stock_data
    ORDER BY symbol, date
    """

    all_data = pd.read_sql_query(query, conn, params={
        'pattern_end_date': pattern_end_date,
        'lookback_date': lookback_date
    })

    # Convert dates to strings for consistency
    all_data['date'] = pd.to_datetime(all_data['date']).dt.strftime('%Y-%m-%d')

    elapsed = time.time() - start_time
    print(f"[OPTIMIZATION] Fetched {len(all_data)} rows for {all_data['symbol'].nunique()} symbols in {elapsed:.2f}s")
    print(f"[OPTIMIZATION] OLD METHOD: Would require {all_data['symbol'].nunique()} separate queries")

    return all_data


def get_stock_prices_on_date(conn, pattern_end_date: str) -> Dict[str, float]:
    """
    Get most recent price for all stocks on pattern_end_date.

    Returns:
        Dict mapping symbol -> price
    """
    query = """
    SELECT symbol, close
    FROM stocks_peaks_valleys_analysis
    WHERE date = %s
    """
    df = pd.read_sql_query(query, conn, params=(pattern_end_date,))
    return dict(zip(df['symbol'], df['close']))


def create_n_day_windows(trading_days: List[str], window_size: int) -> List[Tuple[str, str]]:
    """
    Create all possible N-day windows from trading days.
    Returns list of (start_date, end_date) tuples.
    """
    windows = []
    for i in range(0, len(trading_days) - window_size + 1):
        windows.append((trading_days[i], trading_days[i + window_size - 1]))
    return windows


def calculate_window_metrics(df: pd.DataFrame, window_start: str,
                            window_end: str) -> Tuple[Optional[float], int]:
    """
    Calculate metrics for a specific window.

    Returns:
        Tuple of (avg_pct_change, num_turning_points)
    """
    window_data = df[(df['date'] >= window_start) & (df['date'] <= window_end)]

    if len(window_data) == 0:
        return None, 0

    pct_changes = window_data['pct_change_from_prev_extreme'].dropna().abs()
    avg_pct_change = pct_changes.mean() if len(pct_changes) > 0 else None
    num_turning_points = len(window_data)

    return avg_pct_change, num_turning_points


def get_window_ending_on_date(all_trading_days: List[str],
                               pattern_end_date: str,
                               window_size: int) -> Optional[List[str]]:
    """
    Get exactly window_size trading days ending on pattern_end_date.

    Args:
        all_trading_days: Full list of trading days
        pattern_end_date: The date to end the window on
        window_size: Number of trading days to include

    Returns:
        List of trading days, or None if not enough historical data
    """
    try:
        end_idx = all_trading_days.index(pattern_end_date)
    except ValueError:
        return None

    start_idx = end_idx - window_size + 1
    if start_idx < 0:
        return None  # Not enough historical data

    return all_trading_days[start_idx:end_idx+1]


def calculate_baseline_metrics_optimized(stock_df: pd.DataFrame,
                                         pattern_end_date: str,
                                         trading_days_full: List[str],
                                         trading_days_short: List[str]) -> Optional[Dict]:
    """
    Calculate baseline metrics using pre-computed volatility_ma_1week.

    This calculates percentiles against historical distributions for BOTH:
    - Long-term: 5-year history
    - Short-term: Earnings season window

    Returns:
        Dict with baseline metrics and percentiles, or None if insufficient data
    """
    # Get baseline window (last 5 days ending on pattern_end_date)
    baseline_window_dates = get_window_ending_on_date(trading_days_full, pattern_end_date, 5)

    if baseline_window_dates is None:
        return None

    baseline_start = baseline_window_dates[0]
    baseline_end = baseline_window_dates[-1]

    baseline_data = stock_df[(stock_df['date'] >= baseline_start) & (stock_df['date'] <= baseline_end)]

    if len(baseline_data) == 0:
        return None

    # Calculate baseline metrics
    baseline_avg_pct, baseline_num_tp = calculate_window_metrics(stock_df, baseline_start, baseline_end)

    if baseline_avg_pct is None:
        return None

    # Calculate LONG-TERM percentiles (vs 5-year history)
    # Create all 5-day windows over full 5-year period
    all_windows_full = create_n_day_windows(trading_days_full, 5)

    full_metrics = []
    for w_start, w_end in all_windows_full:
        avg_pct, num_tp = calculate_window_metrics(stock_df, w_start, w_end)
        if avg_pct is not None:
            full_metrics.append({'avg_pct': avg_pct, 'num_tp': num_tp})

    if len(full_metrics) == 0:
        return None

    # Calculate long-term percentiles
    avg_pcts_full = [m['avg_pct'] for m in full_metrics]
    num_tps_full = [m['num_tp'] for m in full_metrics]

    long_avg_pct_pct = sum(1 for x in avg_pcts_full if x <= baseline_avg_pct) / len(avg_pcts_full) * 100
    long_num_tp_pct = sum(1 for x in num_tps_full if x <= baseline_num_tp) / len(num_tps_full) * 100

    # Calculate SHORT-TERM percentiles (vs earnings season)
    all_windows_short = create_n_day_windows(trading_days_short, 5)

    if len(all_windows_short) == 0:
        return None

    short_metrics = []
    for w_start, w_end in all_windows_short:
        avg_pct, num_tp = calculate_window_metrics(stock_df, w_start, w_end)
        if avg_pct is not None:
            short_metrics.append({'avg_pct': avg_pct, 'num_tp': num_tp})

    if len(short_metrics) == 0:
        return None

    # Calculate short-term percentiles
    avg_pcts_short = [m['avg_pct'] for m in short_metrics]
    num_tps_short = [m['num_tp'] for m in short_metrics]

    short_avg_pct_pct = sum(1 for x in avg_pcts_short if x <= baseline_avg_pct) / len(avg_pcts_short) * 100
    short_num_tp_pct = sum(1 for x in num_tps_short if x <= baseline_num_tp) / len(num_tps_short) * 100

    return {
        'long_avg_pct_pct': long_avg_pct_pct,
        'long_num_tp_pct': long_num_tp_pct,
        'long_avg_pct': baseline_avg_pct,
        'long_num_tp': baseline_num_tp,
        'short_avg_pct_pct': short_avg_pct_pct,
        'short_num_tp_pct': short_num_tp_pct,
        'short_avg_pct': baseline_avg_pct,
        'short_num_tp': baseline_num_tp
    }


def main():
    """Main analysis function with dual percentile system - NO BACKWARD EXPANSION"""
    print("=" * 80)
    print("DUAL PERCENTILE CONSOLIDATION ANALYSIS (NO BACKWARD EXPANSION)")
    print("=" * 80)
    print("\nPERFORMANCE OPTIMIZATIONS:")
    print("  1. Bulk data fetching (1 query vs 10,000)")
    print("  2. Pre-computed volatility_ma_1week usage")
    print("  3. Vectorized percentile calculations")
    print("=" * 80)

    # Configuration - TESTING WITH FIRST WEEK OF JUNE 2024
    pattern_end_date = '2024-06-07'  # First Friday of June 2024
    earliest_start_date = '2024-05-05'  # Previous earnings announcement

    print(f"\nPattern End Date (Buy Date): {pattern_end_date}")
    print(f"Earliest Pattern Start: {earliest_start_date} (previous earnings)")
    print(f"Next Earnings: 2024-07-22")
    print("\nStrategy:")
    print("  - DUAL PERCENTILE SYSTEM:")
    print(f"    * Short-term: Compare vs period between earnings ({earliest_start_date} to {pattern_end_date})")
    print("    * Long-term: Compare vs 5-year history")
    print("  - Find stocks that are unusual in this earnings window")
    print("  - But also exceptional over 5-year context")
    print("  - FIXED 5-DAY WINDOW for all stocks (no expansion)")
    print(f"  - All stocks have same buy date: {pattern_end_date}")
    print("\n" + "=" * 80)

    conn = get_db_connection()

    # Track overall performance
    total_start_time = time.time()

    try:
        # Get all trading days up to pattern end date
        print(f"\nGetting trading days up to {pattern_end_date}...")
        all_trading_days = get_all_trading_days(conn, '2019-01-01', pattern_end_date)

        if len(all_trading_days) < 5:
            print("ERROR: Not enough trading days in database")
            return

        # Baseline: Last 5 trading days ending pattern_end_date
        trading_days_baseline = all_trading_days[-5:]
        baseline_start_date = trading_days_baseline[0]

        print(f"Baseline Period: {baseline_start_date} to {pattern_end_date} (5 days)")
        print(f"  Days: {', '.join(trading_days_baseline)}")

        # Get 5 years of historical data for long-term percentile calculations
        lookback_date = (datetime.strptime(pattern_end_date, '%Y-%m-%d') -
                        timedelta(days=5*365)).strftime('%Y-%m-%d')
        print(f"\nGetting 5-year historical data (from {lookback_date})...")
        trading_days_full = get_all_trading_days(conn, lookback_date, pattern_end_date)
        print(f"Long-term period: {len(trading_days_full)} trading days")

        # Get short-term trading days for earnings season analysis
        print(f"\nGetting earnings season data ({earliest_start_date} to {pattern_end_date})...")
        trading_days_short = [d for d in all_trading_days
                              if earliest_start_date <= d <= pattern_end_date]
        print(f"Short-term period: {len(trading_days_short)} trading days")
        print(f"  Earnings season days: {', '.join(trading_days_short)}")

        # OPTIMIZATION: Bulk data fetch
        print("\n" + "=" * 80)
        print("OPTIMIZATION: BULK DATA FETCH")
        print("=" * 80)
        all_stock_data = get_bulk_stock_data(conn, pattern_end_date, lookback_date)

        # Get prices separately
        stock_prices = get_stock_prices_on_date(conn, pattern_end_date)

        # Group data by symbol
        print("\n[OPTIMIZATION] Grouping data by symbol for vectorized processing...")
        grouped_data = all_stock_data.groupby('symbol')

        # SINGLE PHASE: Dual percentile filtering
        print("\n" + "=" * 80)
        print("DUAL PERCENTILE FILTERING (5-DAY WINDOW)")
        print("=" * 80)

        phase_start_time = time.time()

        final_results = []

        # Counters
        num_total_symbols = len(grouped_data)
        num_with_data = num_total_symbols
        num_with_long_metrics = 0
        num_with_short_metrics = 0
        num_with_both_metrics = 0
        num_pass_price = 0
        num_pass_tp = 0
        num_pass_short_filters = 0
        num_pass_long_filters = 0
        num_pass_all = 0

        print(f"\n[OPTIMIZATION] Processing {num_total_symbols} symbols with vectorized operations...")

        for i, (symbol, stock_df) in enumerate(grouped_data, 1):
            if i % 100 == 0:
                print(f"  Processed {i}/{num_total_symbols} symbols...")

            # Get price
            most_recent_price = stock_prices.get(symbol, 0.0)

            # Calculate baseline metrics using optimized function
            metrics = calculate_baseline_metrics_optimized(
                stock_df,
                pattern_end_date,
                trading_days_full,
                trading_days_short
            )

            if metrics is None:
                continue

            # Track metrics availability
            num_with_long_metrics += 1
            num_with_short_metrics += 1
            num_with_both_metrics += 1

            # Extract metrics
            long_avg_pct_pct = metrics['long_avg_pct_pct']
            long_num_tp_pct = metrics['long_num_tp_pct']
            long_avg_pct = metrics['long_avg_pct']
            long_num_tp = metrics['long_num_tp']

            short_avg_pct_pct = metrics['short_avg_pct_pct']
            short_num_tp_pct = metrics['short_num_tp_pct']
            short_avg_pct = metrics['short_avg_pct']
            short_num_tp = metrics['short_num_tp']

            # Basic filters
            if most_recent_price >= 5.0:
                num_pass_price += 1
                if long_num_tp > 2:
                    num_pass_tp += 1

                    # DUAL PERCENTILE FILTER LOGIC
                    short_filters_pass = (
                        short_num_tp_pct > 85 and     # unusually choppy this earnings season
                        short_avg_pct_pct < 15        # quieter than usual this earnings season
                    )

                    long_filters_pass = (
                        long_num_tp_pct > 75 and      # high in 5-year context
                        long_avg_pct_pct < 25         # very tight in 5-year context
                    )

                    if short_filters_pass:
                        num_pass_short_filters += 1
                    if long_filters_pass:
                        num_pass_long_filters += 1

                    if short_filters_pass and long_filters_pass:
                        num_pass_all += 1
                        final_results.append({
                            'symbol': symbol,
                            'price': most_recent_price,
                            'short_term_avg_pct_pct': short_avg_pct_pct,
                            'short_term_num_tp_pct': short_num_tp_pct,
                            'short_term_avg_pct': short_avg_pct,
                            'short_term_num_tp': short_num_tp,
                            'long_term_avg_pct_pct': long_avg_pct_pct,
                            'long_term_num_tp_pct': long_num_tp_pct,
                            'long_term_avg_pct': long_avg_pct,
                            'long_term_num_tp': long_num_tp,
                            'pattern_end_date': pattern_end_date
                        })

        phase_elapsed = time.time() - phase_start_time

        # Print enhanced counters
        print(f"\nDebug Summary for DUAL PERCENTILE FILTERING:")
        print(f"  Total symbols: {num_total_symbols}")
        print(f"  Symbols with any data in 5-year period: {num_with_data}")
        print(f"  Symbols with long-term metrics: {num_with_long_metrics}")
        print(f"  Symbols with short-term metrics: {num_with_short_metrics}")
        print(f"  Symbols with both metrics: {num_with_both_metrics}")
        print(f"  Pass price >= $5: {num_pass_price}")
        print(f"  Pass num_tp > 2: {num_pass_tp}")
        print(f"  Pass short-term filters (choppy + quiet this earnings): {num_pass_short_filters}")
        print(f"  Pass long-term filters (high TP + very tight historically): {num_pass_long_filters}")
        print(f"  Pass ALL dual percentile filters: {num_pass_all}")

        print(f"\n[PERFORMANCE] Analysis completed in {phase_elapsed:.2f}s")
        print(f"[PERFORMANCE] Average time per symbol: {phase_elapsed/num_total_symbols*1000:.1f}ms")

        print(f"\nStocks passing dual percentile filters: {len(final_results)}")

        if len(final_results) == 0:
            print("\nNo stocks meet the dual percentile criteria. Consider:")
            print("  SHORT-TERM ADJUSTMENTS:")
            print("    - Lower short_term_num_tp_pct threshold (try 70)")
            print("    - Raise short_term_avg_pct_pct threshold (try 60)")
            print("  LONG-TERM ADJUSTMENTS:")
            print("    - Lower long_term_num_tp_pct threshold (try 65)")
            print("    - Raise long_term_avg_pct_pct threshold (try 50)")
            print("  BASIC FILTERS:")
            print("    - Lower price threshold (try $3)")
            return

        # Sort by long-term num_tp percentile (descending), then avg_pct percentile (ascending)
        final_results.sort(key=lambda x: (-x['long_term_num_tp_pct'], x['long_term_avg_pct_pct']))

        # Create results DataFrame
        results_df = pd.DataFrame([
            {
                'Rank': i + 1,
                'Symbol': r['symbol'],
                'Price': f"${r['price']:.2f}",
                'Pattern_End_Date': r['pattern_end_date'],
                # Short-term percentiles
                'ShortTerm_NumTP_Pct': f"{r['short_term_num_tp_pct']:.1f}%",
                'ShortTerm_AvgPct_Pct': f"{r['short_term_avg_pct_pct']:.1f}%",
                'ShortTerm_Avg_Pct': f"{r['short_term_avg_pct']:.2f}%",
                'ShortTerm_Num_TP': r['short_term_num_tp'],
                # Long-term percentiles
                'LongTerm_NumTP_Pct': f"{r['long_term_num_tp_pct']:.1f}%",
                'LongTerm_AvgPct_Pct': f"{r['long_term_avg_pct_pct']:.1f}%",
                'LongTerm_Avg_Pct': f"{r['long_term_avg_pct']:.2f}%",
                'LongTerm_Num_TP': r['long_term_num_tp']
            }
            for i, r in enumerate(final_results)
        ])

        # Display results
        print("\n" + "=" * 80)
        print("FINAL RESULTS - DUAL PERCENTILE CONSOLIDATION PATTERNS (5-DAY WINDOW)")
        print("=" * 80)
        print(f"\nTop 30 stocks (ranked by long-term turning point percentile):")
        print("Key Columns:")
        print(f"  ShortTerm_*: Percentile vs earnings window ({earliest_start_date} - {pattern_end_date})")
        print("  LongTerm_*: Percentile vs 5-year history")
        print("  *_NumTP_Pct: Higher = more choppy/turning points")
        print("  *_AvgPct_Pct: Lower = quieter/tighter moves")
        print("-" * 80)

        display_count = min(30, len(results_df))
        print(results_df.head(display_count).to_string(index=False))

        # Save to CSV in the script's directory
        import os
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(script_dir, 'dual_percentile_consolidation_results_NO_EXPANSION.csv')
        results_df.to_csv(output_file, index=False)

        total_elapsed = time.time() - total_start_time

        print(f"\n" + "=" * 80)
        print(f"Full results saved to: {output_file}")
        print("=" * 80)

        print("\n" + "=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        print(f"Total runtime: {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)")
        print(f"Analysis time: {phase_elapsed:.2f}s")

        # Enhanced summary statistics
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        print(f"Total symbols analyzed: {num_total_symbols}")
        print(f"Passed dual percentile filters: {len(final_results)}")
        print(f"Overall success rate: {len(final_results)/num_total_symbols*100:.2f}%")

        print(f"\nDual Percentile Filter Effectiveness:")
        print(f"  Short-term filter success: {num_pass_short_filters/num_with_both_metrics*100:.1f}%")
        print(f"  Long-term filter success: {num_pass_long_filters/num_with_both_metrics*100:.1f}%")
        print(f"  Combined filter success: {num_pass_all/num_with_both_metrics*100:.1f}%")

        print(f"\n" + "=" * 80)
        print(f"IMPORTANT: All stocks have Pattern End Date = {pattern_end_date}")
        print("This is the BUY DATE for gains/losses analysis.")
        print(f"Next earnings announcement: 2024-07-22 (target exit)")
        print("\nCOLUMN INTERPRETATION:")
        print("  SHORT-TERM PERCENTILES (vs earnings window):")
        print("    - High ShortTerm_NumTP_Pct = Unusually choppy in earnings window")
        print(f"    - Low ShortTerm_AvgPct_Pct = Quieter moves than usual ({earliest_start_date} to {pattern_end_date})")
        print("  LONG-TERM PERCENTILES (vs 5-year history):")
        print("    - High LongTerm_NumTP_Pct = Historically high turning points")
        print("    - Low LongTerm_AvgPct_Pct = Very tight historically")
        print("\nSTRATEGY:")
        print("  We want stocks that are:")
        print("    1. Unusually choppy + quiet THIS earnings season (short-term)")
        print("    2. Historically high TP + tight over 5 years (long-term)")
        print("    3. This dual filter finds stocks in atypical consolidation")
        print("=" * 80)

    finally:
        conn.close()


if __name__ == "__main__":
    main()
