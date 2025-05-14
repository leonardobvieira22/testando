import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_timestamp(ts):
    """Parse timestamp string or float to datetime object."""
    if pd.isna(ts):
        logging.warning(f"Encountered NaN timestamp")
        return None
    try:
        if isinstance(ts, (float, int)):
            # Convert numeric timestamp (e.g., 20250502221355.241936) to string
            ts_str = str(ts)
            if '.' in ts_str:
                integer_part, decimal_part = ts_str.split('.')
                # Assume integer part is YYYYMMDDHHMMSS
                if len(integer_part) >= 14:
                    ts_str = f"{integer_part[:4]}-{integer_part[4:6]}-{integer_part[6:8]} " \
                             f"{integer_part[8:10]}:{integer_part[10:12]}:{integer_part[12:14]}.{decimal_part}"
                else:
                    logging.error(f"Invalid numeric timestamp format: {ts}")
                    return None
            else:
                logging.error(f"Numeric timestamp missing decimal: {ts}")
                return None
        elif isinstance(ts, str):
            ts_str = ts.strip()
        else:
            logging.error(f"Invalid timestamp type: {type(ts)} for value: {ts}")
            return None

        # Try parsing with microseconds
        try:
            return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S.%f')
        except ValueError:
            # Try parsing without microseconds
            return datetime.strptime(ts_str, '%Y-%m-%d %H:%M:%S')
    except Exception as e:
        logging.error(f"Failed to parse timestamp {ts}: {e}")
        return None

def find_closest_price_stat(signal_timestamp, price_stats):
    """Find the closest price stat within 1-2 seconds after signal timestamp."""
    signal_time = signal_timestamp
    target_time_min = signal_time + timedelta(seconds=1)
    target_time_max = signal_time + timedelta(seconds=2)
    price_stats['time_diff'] = (price_stats['timestamp'] - signal_time).abs()
    candidates = price_stats[
        (price_stats['timestamp'] >= target_time_min) &
        (price_stats['timestamp'] <= target_time_max)
    ]
    if candidates.empty:
        return None
    return candidates.loc[candidates['time_diff'].idxmin()]

def simulate_order_outcome(entry_price, direction, sl, tp, price_stats, signal_timestamp):
    """Simulate order outcome for a given SL/TP pair."""
    sl_price = entry_price - sl if direction == 'BUY' else entry_price + sl
    tp_price = entry_price + tp if direction == 'BUY' else entry_price - tp
    outcome = 'OPEN'

    for _, stat in price_stats[price_stats['timestamp'] > signal_timestamp].iterrows():
        price = stat['price']
        if direction == 'BUY':
            if price <= sl_price:
                return 'SL'
            if price >= tp_price:
                return 'TP'
        else:
            if price >= sl_price:
                return 'SL'
            if price <= tp_price:
                return 'TP'
    return outcome

def find_optimal_sl_tp(entry_price, direction, price_stats, signal_timestamp):
    """Find optimal SL/TP pair that hits TP with minimal SL."""
    sl_range = np.arange(0.0010, 0.0051, 0.0005)
    tp_range = np.arange(0.0010, 0.0101, 0.0005)
    optimal_sl = None
    optimal_tp = None
    min_sl = float('inf')

    for sl in sl_range:
        for tp in tp_range:
            outcome = simulate_order_outcome(entry_price, direction, sl, tp, price_stats, signal_timestamp)
            if outcome == 'TP' and sl < min_sl:
                min_sl = sl
                optimal_sl = sl
                optimal_tp = tp

    return optimal_sl, optimal_tp

def main():
    # Load CSV files
    try:
        signals = pd.read_csv('signals.csv', dtype={'timestamp': str})
        price_stats = pd.read_csv('prices_statistics.csv', dtype={'timestamp': str})
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        return
    except pd.errors.ParserError as e:
        logging.error(f"CSV parsing failed: {e}")
        return

    # Log column names
    logging.info(f"Signals CSV columns: {signals.columns.tolist()}")
    logging.info(f"Price Stats CSV columns: {price_stats.columns.tolist()}")

    # Check for timestamp column
    if 'timestamp' not in signals.columns:
        logging.error("No 'timestamp' column in signals.csv. Possible columns: %s", signals.columns.tolist())
        return
    if 'timestamp' not in price_stats.columns:
        logging.error("No 'timestamp' column in prices_statistics.csv. Possible columns: %s", price_stats.columns.tolist())
        return

    # Log raw data for signals
    logging.info(f"Raw Signals DataFrame head (before parsing):\n{signals[['signal_id', 'timestamp']].head().to_string()}")
    logging.info(f"Raw Price Stats DataFrame head:\n{price_stats[['id', 'timestamp']].head().to_string()}")

    # Parse timestamps
    signals['timestamp'] = signals['timestamp'].apply(parse_timestamp)
    price_stats['timestamp'] = price_stats['timestamp'].apply(parse_timestamp)

    # Log NaN counts
    signals_nan = signals['timestamp'].isna().sum()
    price_stats_nan = price_stats['timestamp'].isna().sum()
    logging.info(f"Signals with NaN timestamps: {signals_nan}/{len(signals)}")
    logging.info(f"Price stats with NaN timestamps: {price_stats_nan}/{len(price_stats)}")

    # Ensure numeric columns
    signals['rsi'] = pd.to_numeric(signals['rsi'], errors='coerce')
    price_stats['price'] = pd.to_numeric(price_stats['price'], errors='coerce')
    price_stats['rsi'] = pd.to_numeric(price_stats['rsi'], errors='coerce')

    # Drop rows with invalid data
    signals = signals.dropna(subset=['timestamp', 'rsi', 'direction'])
    price_stats = price_stats.dropna(subset=['timestamp', 'price'])

    # Log remaining rows
    logging.info(f"Valid signals after cleaning: {len(signals)}")
    logging.info(f"Valid price stats after cleaning: {len(price_stats)}")

    results = []
    successful_sl_tp_pairs = []

    print("Processing signals...\n")
    for _, signal in signals.iterrows():
        price_stat = find_closest_price_stat(signal['timestamp'], price_stats)
        if price_stat is None:
            print(f"No price stat found for signal {signal['signal_id']}")
            continue

        entry_price = price_stat['price']
        direction = signal['direction']
        optimal_sl, optimal_tp = find_optimal_sl_tp(entry_price, direction, price_stats, signal['timestamp'])

        result = {
            'signal_id': signal['signal_id'],
            'timestamp': signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S.%f') if signal['timestamp'] else 'N/A',
            'direction': direction,
            'entry_price': entry_price,
            'optimal_sl': optimal_sl,
            'optimal_tp': optimal_tp
        }
        results.append(result)

        print(f"Signal: {result['signal_id']}")
        print(f"Timestamp: {result['timestamp']}")
        print(f"Direction: {result['direction']}")
        print(f"Entry Price: {result['entry_price']:.5f}")
        print(f"Optimal SL: {result['optimal_sl']:.4f}" if result['optimal_sl'] else "Optimal SL: N/A")
        print(f"Optimal TP: {result['optimal_tp']:.4f}" if result['optimal_tp'] else "Optimal TP: N/A")
        print("-" * 50)

        if optimal_sl and optimal_tp:
            successful_sl_tp_pairs.append({'sl': optimal_sl, 'tp': optimal_tp})

    # Calculate global SL/TP range for 80% TP hit rate
    if successful_sl_tp_pairs:
        sl_values = sorted([pair['sl'] for pair in successful_sl_tp_pairs])
        tp_values = sorted([pair['tp'] for pair in successful_sl_tp_pairs])
        target_count = int(len(successful_sl_tp_pairs) * 0.8)
        min_sl, max_sl = sl_values[0], sl_values[-1]
        min_tp, max_tp = tp_values[0], tp_values[-1]
        best_area = float('inf')

        for i in range(len(sl_values)):
            for j in range(i, len(sl_values)):
                sl_min, sl_max = sl_values[i], sl_values[j]
                for k in range(len(tp_values)):
                    for m in range(k, len(tp_values)):
                        tp_min, tp_max = tp_values[k], tp_values[m]
                        count = sum(1 for pair in successful_sl_tp_pairs
                                    if sl_min <= pair['sl'] <= sl_max and tp_min <= pair['tp'] <= tp_max)
                        if count >= target_count:
                            area = (sl_max - sl_min) * (tp_max - tp_min)
                            if area < best_area:
                                best_area = area
                                min_sl, max_sl = sl_min, max_sl
                                min_tp, max_tp = tp_min, tp_max

        print("\nGlobal SL/TP Range (80% TP Hit Rate)")
        print(f"SL Range: {min_sl:.4f} - {max_sl:.4f}")
        print(f"TP Range: {min_tp:.4f} - {max_tp:.4f}")
        print(f"Orders with TP Hit: {len(successful_sl_tp_pairs)} / {len(results)}")
    else:
        print("No successful SL/TP pairs found.")

if __name__ == "__main__":
    main()