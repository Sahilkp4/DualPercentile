#!/usr/bin/env python3
"""
Machine Learning Dataset Generator - Dual Percentile Filtering (2020-2025)

This script generates a comprehensive ML dataset by running the dual percentile
stock filtering logic across a 5-year period (Jan 2020 - Jan 2025), tracking
all stocks that pass the filter and their subsequent performance.

Features:
- Analyzes all trading weeks between earnings windows (2020-2025)
- Replicates exact filtering logic from DualPercentileFilter.py
- Fetches moving averages and calculates performance metrics
- Outputs comprehensive CSV with ML-ready features
- Multi-core parallel processing for optimal performance (5-10x speedup)
- Thread-safe progress tracking and result collection
- Detailed logging and performance metrics

Optimizations:
- ProcessPoolExecutor with automatic worker count (min(cpu_count, 8))
- Each worker process maintains its own database connection
- Reduced logging overhead to minimize lock contention
- Parallel execution achieves near-linear speedup for database-bound operations

Output: 2020_2025_DualPercentileResults.csv
"""

import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count, Manager
import os
import sys
import threading

warnings.filterwarnings('ignore')

# Global cache for trading dates (shared across all calls in the same process)
_TRADING_DATES_CACHE = {}

# Database configuration
DB_CONFIG = {
    'dbname': 'daily_ohlcv_data',
    'user': 'postgres',
    'password': 'Moopeyman4!',
    'host': 'localhost',
    'port': 5432
}

# Earnings windows configuration (when NOT to check)
# Format: (month, start_day, end_day)
EARNINGS_WINDOWS = {
    'Q1': (1, 24, 31),  # Jan 24-31 (Q4 results) - Extended to month end
    'Q1_FEB': (2, 1, 7),  # Feb 1-7 (Q4 results continuation)
    'Q2': (4, 24, 30),  # Apr 24-30 (Q1 results) - Extended to month end
    'Q2_MAY': (5, 1, 8),  # May 1-8 (Q1 results continuation)
    'Q3': (7, 24, 31),  # Jul 24-31 (Q2 results) - Extended to month end
    'Q3_AUG': (8, 1, 7),  # Aug 1-7 (Q2 results continuation)
    'Q4': (10, 24, 31),  # Oct 24-31 (Q3 results) - Extended to month end
    'Q4_NOV': (11, 1, 7)   # Nov 1-7 (Q3 results continuation)
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_dataset_generation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def get_db_connection():
    """
    Establish database connection with optimized session-level settings.

    Sets PostgreSQL work_mem to 1GB and temp_buffers to 512MB to handle
    large sorting operations in memory rather than creating temp files on disk.
    """
    conn = psycopg2.connect(**DB_CONFIG)

    # Set session-level parameters to increase memory available for sorting
    # This prevents "No space left on device" errors from temp files
    with conn.cursor() as cursor:
        cursor.execute("SET work_mem = '1GB'")
        cursor.execute("SET temp_buffers = '512MB'")

    conn.commit()
    return conn


def is_in_earnings_window(date_obj: datetime) -> bool:
    """Check if a date falls within any earnings window"""
    month = date_obj.month
    day = date_obj.day

    for window_name, (win_month, start_day, end_day) in EARNINGS_WINDOWS.items():
        if month == win_month and start_day <= day <= end_day:
            return True
    return False


def get_quarter_period_label(date_obj: datetime) -> str:
    """
    Get the quarter period label for a given date.
    E.g., "after_q1_2020", "after_q2_2020", etc.
    """
    year = date_obj.year
    month = date_obj.month

    # Determine which post-earnings period we're in
    if 2 <= month <= 4 and not (month == 4 and date_obj.day >= 24):
        # After Q1 window (Feb 8 - Apr 23)
        return f"after_q1_{year}"
    elif ((month == 5 and date_obj.day >= 9) or (5 < month <= 7)) and not (month == 7 and date_obj.day >= 24):
        # After Q2 window (May 9 - Jul 23)
        return f"after_q2_{year}"
    elif ((month == 8 and date_obj.day >= 8) or (8 < month <= 10)) and not (month == 10 and date_obj.day >= 24):
        # After Q3 window (Aug 8 - Oct 23)
        return f"after_q3_{year}"
    elif ((month == 11 and date_obj.day >= 8) or month == 12 or month == 1) and not (month == 1 and date_obj.day >= 24):
        # After Q4 window (Nov 8 - Jan 23)
        # Handle year transition
        if month == 1:
            year -= 1
        return f"after_q4_{year}"
    elif month == 2 and date_obj.day >= 8:
        # Early Feb after Q1 window starts
        return f"after_q1_{year}"
    else:
        return None


def get_most_recent_earnings_end(date_obj: datetime) -> datetime:
    """
    Get the last day of the most recent earnings window before the given date.
    This is the earliest_start_date for pattern analysis.
    """
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day

    # Determine which earnings window just ended
    if month >= 2 and month < 4:
        # Most recent was Q1 (ended Feb 7)
        return datetime(year, 2, 7)
    elif month == 4 and day < 24:
        # Still in early April, most recent was Q1 (ended Feb 7)
        return datetime(year, 2, 7)
    elif month >= 5 and month < 7:
        # Most recent was Q2 (ended May 8)
        return datetime(year, 5, 8)
    elif month == 7 and day < 24:
        # Still in early July, most recent was Q2 (ended May 8)
        return datetime(year, 5, 8)
    elif month >= 8 and month < 10:
        # Most recent was Q3 (ended Aug 7)
        return datetime(year, 8, 7)
    elif month == 10 and day < 24:
        # Most recent was Q3 (ended Aug 7)
        return datetime(year, 8, 7)
    elif month >= 11 or month == 1:
        # Most recent was Q4 (ended Nov 7)
        if month == 1:
            # Early January, Q4 was in previous year
            return datetime(year - 1, 11, 7)
        else:
            return datetime(year, 11, 7)
    else:
        # In an earnings window, return None
        return None


def get_next_earnings_start(date_obj: datetime) -> datetime:
    """
    Get the first day of the next earnings window after the given date.
    This is the sell_date for performance calculations.
    """
    year = date_obj.year
    month = date_obj.month
    day = date_obj.day

    # Determine which earnings window comes next
    if month < 4 or (month == 4 and day < 24):
        # Next is Q2 (starts Apr 24)
        return datetime(year, 4, 24)
    elif month < 7 or (month == 7 and day < 24):
        # Next is Q3 (starts Jul 24)
        return datetime(year, 7, 24)
    elif month < 10 or (month == 10 and day < 24):
        # Next is Q4 (starts Oct 24)
        return datetime(year, 10, 24)
    else:
        # Next is Q1 of next year (starts Jan 24)
        return datetime(year + 1, 1, 24)


def get_all_trading_days(conn, start_date: str, end_date: str) -> List[str]:
    """
    Get all trading days in date range with intelligent caching.

    On the FIRST call, fetches ALL trading dates for 2020-2025 from the database
    and stores them in an in-memory cache. All subsequent calls use the cache
    for instant retrieval, eliminating the expensive DISTINCT query.

    This dramatically reduces database load and prevents temp file issues
    when running with parallel workers.

    Returns:
        List of date strings in YYYY-MM-DD format
    """
    global _TRADING_DATES_CACHE

    # Cache key for the entire dataset
    cache_key = 'all_dates_2020_2025'

    # Check if we need to fetch from database (first call only)
    if cache_key not in _TRADING_DATES_CACHE:
        logger.info("First call to get_all_trading_days - fetching ALL dates from database and caching...")

        # Fetch ALL dates for the entire analysis period (2020-2025)
        # Use GROUP BY instead of DISTINCT - much more efficient for large datasets
        query = """
        SELECT date::date AS date
        FROM stocks_peaks_valleys_analysis
        WHERE date >= '2020-01-01' AND date <= '2025-01-31'
        GROUP BY date::date
        ORDER BY date::date
        """

        start_time = time.time()
        df = pd.read_sql_query(query, conn)
        elapsed = time.time() - start_time

        # Convert to strings and store in cache
        all_dates = [d.strftime('%Y-%m-%d') for d in df['date'].tolist()]
        _TRADING_DATES_CACHE[cache_key] = all_dates

        logger.info(f"Cached {len(all_dates)} trading dates in {elapsed:.2f}s - all future calls will use cache")
    else:
        # Using cached data - super fast!
        all_dates = _TRADING_DATES_CACHE[cache_key]
        logger.debug(f"Using cached trading dates ({len(all_dates)} dates) - no database query needed")

    # Filter the cached data to the requested date range
    filtered_dates = [d for d in all_dates if start_date <= d <= end_date]

    return filtered_dates


def get_valid_analysis_weeks(conn, start_year: int = 2020, end_year: int = 2025) -> List[Dict]:
    """
    Get all valid trading weeks for analysis (between earnings windows).

    Returns:
        List of dicts with week info: {
            'buy_date': str,
            'quarter_period': str,
            'earliest_start_date': str,
            'sell_date': str,
            'lookback_date': str
        }
    """
    logger.info(f"Identifying valid analysis weeks from {start_year} to {end_year}...")

    # Get all trading days in the range
    all_trading_days = get_all_trading_days(
        conn,
        f"{start_year}-01-01",
        f"{end_year}-01-31"
    )

    logger.info(f"Found {len(all_trading_days)} total trading days")

    valid_weeks = []

    # Find all Fridays (or last trading day of week) between earnings windows
    current_week_end = None

    for i, date_str in enumerate(all_trading_days):
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')

        # Check if this is a Friday or the last trading day before a weekend/holiday
        is_friday = date_obj.weekday() == 4  # Friday = 4

        # Check if next trading day is Monday or later (indicating week boundary)
        is_week_end = False
        if i < len(all_trading_days) - 1:
            next_date = datetime.strptime(all_trading_days[i + 1], '%Y-%m-%d')
            days_gap = (next_date - date_obj).days
            is_week_end = days_gap >= 3 or is_friday
        else:
            is_week_end = True  # Last day in dataset

        # Only consider week-ending days that are NOT in earnings windows
        if is_week_end and not is_in_earnings_window(date_obj):
            quarter_period = get_quarter_period_label(date_obj)

            if quarter_period:
                earnings_end = get_most_recent_earnings_end(date_obj)
                next_earnings = get_next_earnings_start(date_obj)

                if earnings_end and next_earnings:
                    # Calculate lookback date (5 years before buy date)
                    lookback_date = date_obj - timedelta(days=5*365)

                    # Calculate weeks after earnings
                    weeks_after_earnings = (date_obj - earnings_end).days // 7

                    valid_weeks.append({
                        'buy_date': date_str,
                        'quarter_period': quarter_period,
                        'earliest_start_date': earnings_end.strftime('%Y-%m-%d'),
                        'sell_date': next_earnings.strftime('%Y-%m-%d'),
                        'lookback_date': lookback_date.strftime('%Y-%m-%d'),
                        'weeks_after_earnings': weeks_after_earnings
                    })

    logger.info(f"Found {len(valid_weeks)} valid analysis weeks")
    return valid_weeks


def get_bulk_stock_data(conn, pattern_end_date: str, lookback_date: str) -> pd.DataFrame:
    """
    Fetch ALL stock data in a single query (optimization).
    """
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
            s.volatility_ma_1week,
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

    return all_data


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


def get_moving_average_data(conn, symbol: str, date: str) -> Optional[Dict]:
    """
    Get price and volume moving average data for a stock on a specific date.

    Returns:
        Dict with MA data, or None if not available
    """
    query = """
    SELECT
        p.close,
        p.volume,
        p.price_ma_1month,
        p.price_ma_2month,
        p.price_ma_3month,
        p.price_ma_6month,
        p.price_ma_1year,
        p.price_ma_5year,
        v.volume_ma_1month,
        v.volume_ma_2month,
        v.volume_ma_3month,
        v.volume_ma_6month,
        v.volume_ma_1year,
        v.volume_ma_5year
    FROM stocks_price_rolling_averages p
    LEFT JOIN stocks_volume_rolling_averages v
        ON p.symbol = v.symbol AND p.date = v.date
    WHERE p.symbol = %s AND p.date = %s
    """

    df = pd.read_sql_query(query, conn, params=(symbol, date))

    if len(df) == 0:
        return None

    row = df.iloc[0]

    # Check if any critical fields are missing
    if pd.isna(row['close']) or pd.isna(row['volume']):
        return None

    return {
        'price': row['close'],
        'volume': row['volume'],
        'above_price_ma_1month': 1 if not pd.isna(row['price_ma_1month']) and row['close'] > row['price_ma_1month'] else 0,
        'above_price_ma_2month': 1 if not pd.isna(row['price_ma_2month']) and row['close'] > row['price_ma_2month'] else 0,
        'above_price_ma_3month': 1 if not pd.isna(row['price_ma_3month']) and row['close'] > row['price_ma_3month'] else 0,
        'above_price_ma_6month': 1 if not pd.isna(row['price_ma_6month']) and row['close'] > row['price_ma_6month'] else 0,
        'above_price_ma_1year': 1 if not pd.isna(row['price_ma_1year']) and row['close'] > row['price_ma_1year'] else 0,
        'above_price_ma_5year': 1 if not pd.isna(row['price_ma_5year']) and row['close'] > row['price_ma_5year'] else 0,
        'above_volume_ma_1month': 1 if not pd.isna(row['volume_ma_1month']) and row['volume'] > row['volume_ma_1month'] else 0,
        'above_volume_ma_2month': 1 if not pd.isna(row['volume_ma_2month']) and row['volume'] > row['volume_ma_2month'] else 0,
        'above_volume_ma_3month': 1 if not pd.isna(row['volume_ma_3month']) and row['volume'] > row['volume_ma_3month'] else 0,
        'above_volume_ma_6month': 1 if not pd.isna(row['volume_ma_6month']) and row['volume'] > row['volume_ma_6month'] else 0,
        'above_volume_ma_1year': 1 if not pd.isna(row['volume_ma_1year']) and row['volume'] > row['volume_ma_1year'] else 0,
        'above_volume_ma_5year': 1 if not pd.isna(row['volume_ma_5year']) and row['volume'] > row['volume_ma_5year'] else 0,
    }


def calculate_performance_metrics(conn, symbol: str, buy_date: str, sell_date: str) -> Optional[Dict]:
    """
    Calculate performance metrics from buy_date to sell_date.
    
    Note: sell_date may fall on a weekend/holiday, so we use the last 
    available trading day on or before sell_date.

    Returns:
        Dict with performance metrics, or None if data not available
    """
    query = """
    SELECT date, close
    FROM stocks_peaks_valleys_analysis
    WHERE symbol = %s
      AND date >= %s
      AND date <= %s
    ORDER BY date
    """

    df = pd.read_sql_query(query, conn, params=(symbol, buy_date, sell_date))

    if len(df) < 2:
        return None

    # CRITICAL: Convert date column to string format for comparison
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

    # Ensure we have the buy date (required - stock passed filter on this date)
    buy_row = df[df['date'] == buy_date]
    if len(buy_row) == 0:
        return None
    
    # Use the last available trading day as sell date
    # (handles weekends and holidays when sell_date isn't a trading day)
    buy_price = buy_row.iloc[0]['close']
    sell_price = df.iloc[-1]['close']

    # Calculate returns
    final_return_pct = ((sell_price - buy_price) / buy_price) * 100

    # Calculate max gain and max loss
    max_price = df['close'].max()
    min_price = df['close'].min()

    max_gain_pct = ((max_price - buy_price) / buy_price) * 100
    max_loss_pct = ((min_price - buy_price) / buy_price) * 100

    # Find when max gain and max loss occurred
    max_gain_date = df[df['close'] == max_price].iloc[0]['date']
    max_loss_date = df[df['close'] == min_price].iloc[0]['date']

    # Determine if max gain occurred before max loss
    max_gain_before_max_loss = 1 if max_gain_date < max_loss_date else 0

    # Calculate weeks to max gain/loss
    buy_date_obj = datetime.strptime(buy_date, '%Y-%m-%d')
    max_gain_date_obj = datetime.strptime(max_gain_date, '%Y-%m-%d')
    max_loss_date_obj = datetime.strptime(max_loss_date, '%Y-%m-%d')

    weeks_to_max_gain = round((max_gain_date_obj - buy_date_obj).days / 7, 2)
    weeks_to_max_loss = round((max_loss_date_obj - buy_date_obj).days / 7, 2)

    return {
        'final_return_pct': final_return_pct,
        'max_gain_pct': max_gain_pct,
        'max_loss_pct': max_loss_pct,
        'max_gain_before_max_loss': max_gain_before_max_loss,
        'weeks_to_max_gain': weeks_to_max_gain,
        'weeks_to_max_loss': weeks_to_max_loss
    }


def process_single_week_with_new_connection(week_info: Dict) -> List[Dict]:
    """
    Wrapper function for multiprocessing - creates its own database connection.
    Each process needs its own connection since they cannot be shared across processes.

    Args:
        week_info: Dictionary containing week analysis parameters

    Returns:
        List of result dicts for stocks that passed the filter
    """
    conn = None
    try:
        # Create a new database connection for this process
        conn = get_db_connection()

        # Call the actual processing function
        results = process_single_week(week_info, conn)

        return results

    except Exception as e:
        logger.error(f"Error in process_single_week_with_new_connection for {week_info.get('buy_date', 'unknown')}: {str(e)}")
        return []

    finally:
        # Always close the connection
        if conn is not None:
            conn.close()


def process_single_week(week_info: Dict, conn) -> List[Dict]:
    """
    Process a single week's analysis - run dual percentile filter and collect results.

    Returns:
        List of result dicts for stocks that passed the filter
    """
    buy_date = week_info['buy_date']
    quarter_period = week_info['quarter_period']
    earliest_start_date = week_info['earliest_start_date']
    sell_date = week_info['sell_date']
    lookback_date = week_info['lookback_date']
    weeks_after_earnings = week_info['weeks_after_earnings']

    results = []

    try:
        # Get trading days for this analysis period
        trading_days_full = get_all_trading_days(conn, lookback_date, buy_date)
        trading_days_short = get_all_trading_days(conn, earliest_start_date, buy_date)

        if len(trading_days_full) < 5 or len(trading_days_short) < 5:
            logger.warning(f"Insufficient trading days for {buy_date}")
            return results

        # Fetch bulk stock data
        all_stock_data = get_bulk_stock_data(conn, buy_date, lookback_date)

        if len(all_stock_data) == 0:
            logger.warning(f"No stock data for {buy_date}")
            return results

        # Group by symbol
        grouped_data = all_stock_data.groupby('symbol')

        # Process each stock
        for symbol, stock_df in grouped_data:
            # Calculate baseline metrics
            metrics = calculate_baseline_metrics_optimized(
                stock_df,
                buy_date,
                trading_days_full,
                trading_days_short
            )

            if metrics is None:
                continue

            # Extract metrics
            long_avg_pct_pct = metrics['long_avg_pct_pct']
            long_num_tp_pct = metrics['long_num_tp_pct']
            long_avg_pct = metrics['long_avg_pct']
            long_num_tp = metrics['long_num_tp']

            short_avg_pct_pct = metrics['short_avg_pct_pct']
            short_num_tp_pct = metrics['short_num_tp_pct']
            short_avg_pct = metrics['short_avg_pct']
            short_num_tp = metrics['short_num_tp']

            # Get current price
            current_price_data = stock_df[stock_df['date'] == buy_date]
            if len(current_price_data) == 0:
                continue

            current_price = current_price_data.iloc[0]['close']

            # Apply dual percentile filters
            if current_price >= 5.0 and long_num_tp > 2:
                short_filters_pass = (
                    short_num_tp_pct > 85 and     # unusually choppy this earnings season
                    short_avg_pct_pct < 15        # quieter than usual this earnings season
                )

                long_filters_pass = (
                    long_num_tp_pct > 75 and      # high in 5-year context
                    long_avg_pct_pct < 25         # very tight in 5-year context
                )

                if short_filters_pass and long_filters_pass:
                    # Stock passed the filter! Get additional data

                    # Get moving average data
                    ma_data = get_moving_average_data(conn, symbol, buy_date)
                    if ma_data is None:
                        logger.warning(f"Missing MA data for {symbol} on {buy_date}")
                        continue

                    # Get performance metrics
                    perf_data = calculate_performance_metrics(conn, symbol, buy_date, sell_date)
                    if perf_data is None:
                        logger.warning(f"Missing performance data for {symbol} from {buy_date} to {sell_date}")
                        continue

                    # Combine all data
                    result = {
                        # Identification
                        'symbol': symbol,
                        'quarter_period': quarter_period,
                        'buy_date': buy_date,
                        'sell_date': sell_date,
                        'weeks_after_earnings': weeks_after_earnings,

                        # Price and MA data
                        'price': ma_data['price'],
                        'above_price_ma_1month': ma_data['above_price_ma_1month'],
                        'above_price_ma_2month': ma_data['above_price_ma_2month'],
                        'above_price_ma_3month': ma_data['above_price_ma_3month'],
                        'above_price_ma_6month': ma_data['above_price_ma_6month'],
                        'above_price_ma_1year': ma_data['above_price_ma_1year'],
                        'above_price_ma_5year': ma_data['above_price_ma_5year'],

                        # Volume MA data
                        'above_volume_ma_1month': ma_data['above_volume_ma_1month'],
                        'above_volume_ma_2month': ma_data['above_volume_ma_2month'],
                        'above_volume_ma_3month': ma_data['above_volume_ma_3month'],
                        'above_volume_ma_6month': ma_data['above_volume_ma_6month'],
                        'above_volume_ma_1year': ma_data['above_volume_ma_1year'],
                        'above_volume_ma_5year': ma_data['above_volume_ma_5year'],

                        # Dual percentile metrics
                        'short_term_avg_pct_pct': short_avg_pct_pct,
                        'short_term_num_tp_pct': short_num_tp_pct,
                        'short_term_avg_pct': short_avg_pct,
                        'short_term_num_tp': short_num_tp,
                        'long_term_avg_pct_pct': long_avg_pct_pct,
                        'long_term_num_tp_pct': long_num_tp_pct,
                        'long_term_avg_pct': long_avg_pct,
                        'long_term_num_tp': long_num_tp,

                        # Performance metrics
                        'final_return_pct': perf_data['final_return_pct'],
                        'max_gain_pct': perf_data['max_gain_pct'],
                        'max_loss_pct': perf_data['max_loss_pct'],
                        'max_gain_before_max_loss': perf_data['max_gain_before_max_loss'],
                        'weeks_to_max_gain': perf_data['weeks_to_max_gain'],
                        'weeks_to_max_loss': perf_data['weeks_to_max_loss']
                    }

                    results.append(result)

    except Exception as e:
        logger.error(f"Error processing week {buy_date}: {str(e)}")
        return results

    return results


def main():
    """Main dataset generation function with multi-core parallel processing"""
    logger.info("=" * 80)
    logger.info("ML DATASET GENERATION - DUAL PERCENTILE FILTERING (2020-2025)")
    logger.info("=" * 80)

    overall_start_time = time.time()

    try:
        # Connect to database (only for initial setup)
        conn = get_db_connection()
        logger.info("Database connection established")

        # Get all valid analysis weeks
        valid_weeks = get_valid_analysis_weeks(conn, start_year=2020, end_year=2025)

        # Close the initial connection - each worker will create its own
        conn.close()

        if len(valid_weeks) == 0:
            logger.error("No valid analysis weeks found!")
            return

        logger.info(f"Total weeks to analyze: {len(valid_weeks)}")

        # Determine optimal number of workers
        # Use min(cpu_count(), 8) to avoid overwhelming the database
        max_workers = min(cpu_count(), 8)
        logger.info(f"Using {max_workers} parallel workers (CPU count: {cpu_count()})")

        # Estimate runtime with parallelization
        sequential_time_estimate = (len(valid_weeks) * 7.5) / 60
        parallel_time_estimate = sequential_time_estimate / max_workers
        logger.info(f"Estimated runtime (sequential): {sequential_time_estimate:.1f} minutes")
        logger.info(f"Estimated runtime (parallel): {parallel_time_estimate:.1f} minutes")
        logger.info(f"Expected speedup: ~{max_workers}x")

        # Process all weeks in parallel
        all_results = []
        processed_count = 0
        progress_lock = threading.Lock()

        logger.info("\nStarting parallel weekly analysis...")

        # Submit all weeks to the process pool executor
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks and create a mapping of future to week_info
            future_to_week = {
                executor.submit(process_single_week_with_new_connection, week_info): week_info
                for week_info in valid_weeks
            }

            # Process results as they complete
            for future in as_completed(future_to_week):
                week_info = future_to_week[future]
                buy_date = week_info['buy_date']
                quarter_period = week_info['quarter_period']

                try:
                    # Get the results from this week
                    week_results = future.result()

                    # Thread-safe result collection
                    with progress_lock:
                        processed_count += 1

                        if len(week_results) > 0:
                            all_results.extend(week_results)
                            logger.info(f"[{processed_count}/{len(valid_weeks)}] {buy_date} ({quarter_period}): Found {len(week_results)} stocks")

                        # Log progress every 20 weeks (reduced logging to avoid lock contention)
                        if processed_count % 20 == 0:
                            elapsed = time.time() - overall_start_time
                            avg_time_per_week = elapsed / processed_count
                            remaining_weeks = len(valid_weeks) - processed_count
                            estimated_remaining = remaining_weeks * avg_time_per_week / 60
                            logger.info(f"  Progress: {processed_count}/{len(valid_weeks)} weeks ({processed_count/len(valid_weeks)*100:.1f}%)")
                            logger.info(f"  Total results so far: {len(all_results)}")
                            logger.info(f"  Estimated time remaining: {estimated_remaining:.1f} minutes")

                except Exception as e:
                    logger.error(f"Error processing week {buy_date}: {str(e)}")
                    with progress_lock:
                        processed_count += 1

        # Convert results to DataFrame
        logger.info("\nConverting results to DataFrame...")
        results_df = pd.DataFrame(all_results)

        if len(results_df) == 0:
            logger.error("No results generated! Check filters and data availability.")
            return

        # Define column order
        column_order = [
            'symbol', 'quarter_period', 'buy_date', 'sell_date', 'weeks_after_earnings',
            'price',
            'above_price_ma_1month', 'above_price_ma_2month', 'above_price_ma_3month',
            'above_price_ma_6month', 'above_price_ma_1year', 'above_price_ma_5year',
            'above_volume_ma_1month', 'above_volume_ma_2month', 'above_volume_ma_3month',
            'above_volume_ma_6month', 'above_volume_ma_1year', 'above_volume_ma_5year',
            'short_term_avg_pct_pct', 'short_term_num_tp_pct', 'short_term_avg_pct', 'short_term_num_tp',
            'long_term_avg_pct_pct', 'long_term_num_tp_pct', 'long_term_avg_pct', 'long_term_num_tp',
            'final_return_pct', 'max_gain_pct', 'max_loss_pct', 'max_gain_before_max_loss',
            'weeks_to_max_gain', 'weeks_to_max_loss'
        ]

        results_df = results_df[column_order]

        # Save to CSV
        output_file = '2020_2025_DualPercentileResults.csv'
        results_df.to_csv(output_file, index=False)

        total_elapsed = time.time() - overall_start_time

        # Calculate actual speedup vs sequential estimate
        sequential_estimate_seconds = sequential_time_estimate * 60
        actual_speedup = sequential_estimate_seconds / total_elapsed if total_elapsed > 0 else 0

        # Print summary statistics
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY STATISTICS")
        logger.info("=" * 80)
        logger.info(f"Total runtime: {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)")
        logger.info(f"Sequential estimate: {sequential_estimate_seconds:.2f}s ({sequential_time_estimate:.1f} minutes)")
        logger.info(f"Actual speedup achieved: {actual_speedup:.2f}x")
        logger.info(f"Parallel efficiency: {(actual_speedup / max_workers * 100):.1f}%")
        logger.info(f"Average time per week: {total_elapsed / processed_count:.2f}s")
        logger.info(f"Total weeks analyzed: {processed_count}")
        logger.info(f"Total rows generated: {len(results_df)}")
        logger.info(f"Total unique stocks: {results_df['symbol'].nunique()}")
        logger.info(f"Date range: {results_df['buy_date'].min()} to {results_df['buy_date'].max()}")

        # Breakdown by quarter period
        logger.info("\nBreakdown by quarter period:")
        quarter_counts = results_df['quarter_period'].value_counts().sort_index()
        for period, count in quarter_counts.items():
            logger.info(f"  {period}: {count} entries")

        # Average stocks per week
        avg_stocks_per_week = len(results_df) / processed_count
        logger.info(f"\nAverage stocks passing per week: {avg_stocks_per_week:.2f}")

        # Performance statistics
        logger.info("\nPerformance Statistics:")
        logger.info(f"  Average final return: {results_df['final_return_pct'].mean():.2f}%")
        logger.info(f"  Median final return: {results_df['final_return_pct'].median():.2f}%")
        logger.info(f"  Average max gain: {results_df['max_gain_pct'].mean():.2f}%")
        logger.info(f"  Average max loss: {results_df['max_loss_pct'].mean():.2f}%")
        logger.info(f"  Win rate (positive returns): {(results_df['final_return_pct'] > 0).sum() / len(results_df) * 100:.1f}%")

        logger.info(f"\n" + "=" * 80)
        logger.info(f"Results saved to: {output_file}")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
