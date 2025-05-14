Binance Futures Trading Bot
Overview
This Binance futures trading bot automates trading strategies using RSI and Bollinger Bands indicators, with support for real and simulated trading modes. It includes advanced features like SL Streak Pausing (pauses trading after consecutive Stop Losses) and Inverted Mode (inverts order directions after a configurable number of SLs). The bot operates in a three-terminal setup for real trading, PnL monitoring/Telegram notifications, and simulated trading, ensuring complete isolation between modes.

Key Features:

Dual signal engines: RSI (buy ≤ 40, sell ≥ 68) and Bollinger Bands (20-period, 2.0 deviation).
Configurable trading parameters: 100 USDT, 30x leverage, 0.2% TP/SL, pairs (XRPUSDT, DOGEUSDT, TRXUSDT, BTCUSDT), timeframes (1m, 5m).
SL Streak Pausing: Pauses trading after 3 SLs (7-minute pause for XRPUSDT/DOGEUSDT/TRXUSDT, 5-minute for BTCUSDT).
Inverted Mode: Inverts order directions (e.g., LONG → SHORT) after 3 SLs for the next 2 signals.
Telegram notifications for order closures, SL streaks, pauses, and Inverted Mode events.
Comprehensive logging to bot.log and data storage in CSVs (signals.csv, orders.csv, etc.).
Simulated trading mode for testing strategies without Binance API calls.


Architecture:

Real Trading: main.py runs RSI (sinal_engine.py) and Bollinger (sinal_engine_bollinger.py) engines.
PnL Monitoring: monitor_pnl.py calculates PnL and sends Telegram notifications.
Simulated Trading: simulated_order_engine.py simulates trades using signals.csv and prices_statistics.csv.
Utilities: binance_utils.py for Binance API, utils.py for shared functions, initialize_csvs.py for CSV setup, analyze_inverted_mode.py for performance analysis.



Prerequisites

Python: 3.12 or higher.
Dependencies: Listed in requirements.txt.
Binance Account: API key and secret for real trading (stored in config.py).
Telegram: Bot token and chat IDs for notifications (configured in config.py).

Installation

Clone the Repository:
git clone <repository_url>
cd novo_teste


Install Dependencies:
pip install -r requirements.txt

Contents of requirements.txt:
pandas==2.2.2
numpy==1.26.4
python-binance==1.0.28
filelock==3.15.4
requests==2.32.3


Configure config.py:

Set REAL_API_KEY and REAL_API_SECRET with your Binance API credentials.
Update TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID (real), and SIMULATED_TELEGRAM_CHAT_ID (simulated) with your Telegram bot details.
Adjust trading parameters (e.g., PAIRS, TIMEFRAMES, INVERTED_MODE_CONFIG) as needed.


Initialize CSVs:
python initialize_csvs.py

This creates all necessary CSVs (signals.csv, orders.csv, orders_pnl.csv, prices_statistics.csv, pause_log.csv, simulated_orders.csv, simulated_pause_log.csv, inverted_state.csv, simulated_inverted_state.csv) with correct schemas.


File Structure

Core Files:

main.py: Runs real trading engines (RSI, Bollinger).
sinal_engine.py: RSI-based signal generation and order placement for real trading.
sinal_engine_bollinger.py: Bollinger Bands-based signal generation and order placement for real trading.
simulated_order_engine.py: Simulates trading using signals and price data.
monitor_pnl.py: Calculates PnL and sends Telegram notifications for real orders.
binance_utils.py: Binance API utilities for real trading.
utils.py: Shared functions (RSI calculation, CSV management, logging, Inverted Mode state).
config.py: Configuration for trading parameters, API keys, and Telegram settings.
initialize_csvs.py: Initializes all CSVs with correct schemas.
test_api_keys.py: Tests Binance API connectivity.
analyze_inverted_mode.py: Analyzes Inverted Mode performance.


Data Files:

signals.csv: Stores generated signals (signal ID, pair, timeframe, direction, etc.).
orders.csv: Tracks real orders (order ID, direction, entry price, status, order_mode).
orders_pnl.csv: Tracks real orders with PnL data.
prices_statistics.csv: Stores price, RSI, and Bollinger Bands data.
pause_log.csv: Logs SL Streak Pausing events for real trading.
simulated_orders.csv: Tracks simulated orders.
simulated_pause_log.csv: Logs SL Streak Pausing events for simulated trading.
inverted_state.csv: Tracks Inverted Mode state for real trading (SL streak, inversion count).
simulated_inverted_state.csv: Tracks Inverted Mode state for simulated trading.
bot.log: Logs all system events.
pnl.csv: Stores detailed PnL calculations for real orders.
pnl_total.txt: Stores total PnL for closed real orders.
inverted_mode_analysis.txt: Stores Inverted Mode performance analysis.



Usage

Run the Real Trading Bot:

First terminal:cd C:\Users\User\Desktop\novo_teste
C:\Python312\python.exe main.py


Executes RSI and Bollinger engines, placing real orders and writing to signals.csv, orders.csv, orders_pnl.csv, prices_statistics.csv, pause_log.csv, inverted_state.csv.


Run the Monitor:

Second terminal:C:\Python312\python.exe monitor_pnl.py


Updates pnl.csv every 60 seconds and sends Telegram notifications for real order closures, SL streaks, pauses, and Inverted Mode events to TELEGRAM_CHAT_ID.


Run the Simulated Order Terminal:

Third terminal:C:\Python312\python.exe simulated_order_engine.py


Simulates trades using signals.csv and prices_statistics.csv, writing to simulated_orders.csv, simulated_pause_log.csv, simulated_inverted_state.csv, and sending Telegram notifications to SIMULATED_TELEGRAM_CHAT_ID.


Test API Keys:

Verify Binance API connectivity:C:\Python312\python.exe test_api_keys.py




Analyze Inverted Mode:

Run the analysis script:C:\Python312\python.exe analyze_inverted_mode.py


Outputs performance metrics to inverted_mode_analysis.txt and bot.log, comparing original vs. inverted orders.



Configuration
Edit config.py to customize the bot:

Trading Parameters:
PAIRS, TIMEFRAMES: Enable/disable trading pairs and timeframes.
ORDER_VALUE_USDT, LEVERAGE, TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT: Set order size, leverage, and TP/SL levels.
RSI_PERIOD, RSI_BUY_THRESHOLD, RSI_SELL_THRESHOLD: Configure RSI settings.
BOLLINGER_PERIOD, BOLLINGER_DEVIATION: Configure Bollinger Bands.


SL Streak Pausing:
SL_STREAK_CONFIG, SIMULATED_SL_STREAK_CONFIG: Set SL thresholds and pause durations per pair.
SL_STREAK_NOTIFICATIONS_ENABLED, SIMULATED_SL_STREAK_NOTIFICATIONS_ENABLED: Toggle Telegram notifications.


Inverted Mode:
INVERTED_MODE_ENABLED, SIMULATED_INVERTED_MODE_ENABLED: Enable/disable Inverted Mode.
INVERTED_MODE_CONFIG, SIMULATED_INVERTED_MODE_CONFIG: Set SL thresholds and inversion counts per pair.
INVERTED_MODE_NOTIFICATIONS_ENABLED, SIMULATED_INVERTED_MODE_NOTIFICATIONS_ENABLED: Toggle Telegram notifications.


Simulated Trading:
SIMULATED_MODE, SIMULATED_PAIRS, SIMULATED_TIMEFRAMES, SIMULATED_ORDER_VALUE_USDT, SIMULATED_LEVERAGE, SIMULATED_TP_PERCENT, SIMULATED_SL_PERCENT, SIMULATED_FEE_PERCENT: Configure simulated trading.


Telegram:
TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SIMULATED_TELEGRAM_CHAT_ID: Set Telegram bot and chat IDs.


API Keys:
REAL_API_KEY, REAL_API_SECRET: Binance API credentials for real trading.



Example Outputs
bot.log
2025-04-23 13:05:32,123 [WARNING] [RSI][SL STREAK] ⚠️ XRPUSDT (1m) [3/3]: Mais 0 SL(s) ativará uma pausa de 7.0 minutos.
2025-04-23 13:05:32,124 [INFO] [RSI][INVERTED MODE] 🔄 Inverted Mode ativado para XRPUSDT (1m) após atingir o limite de SLs. Próximas 2 ordens serão invertidas.
2025-04-23 13:06:15,456 [INFO] [RSI][INVERTED MODE] 🔄 Ordem invertida para XRPUSDT (1m): Original BUY → Invertida SELL. Ordens invertidas restantes: 1.
2025-04-23 13:06:30,789 [INFO] [SIMULATED][INVERTED MODE] 🔄 Ordem invertida para XRPUSDT (1m): Original BUY → Invertida SELL. Ordens invertidas restantes: 1.

Telegram Notification (Real, Chat ID: 6097421181)
Inverted Mode Activation:
🔄 Inverted Mode Ativado
Par: XRPUSDT
Timeframe: 1m
Engine: RSI
Motivo: 3 SLs consecutivos
Próximas Ordens Invertidas: 2

Inverted Order:
🔄 Ordem Invertida
Par: XRPUSDT
Timeframe: 1m
Engine: RSI
Original: BUY
Invertida: SELL
Ordens Invertidas Restantes: 1

Telegram Notification (Simulated, Chat ID: 6097421182)
Closed Inverted Order:
🔔 Ordem Simulada Fechada
Par: XRPUSDT
Timeframe: 1m
Engine: Bollinger
Sinal: Preço abaixo da banda inferior: 2.2300 <= 2.2100
Direção: SELL
Modo: inverted
Preço de Entrada: 2.2300
Preço de Fechamento: 2.2256
Motivo: TP
Net PnL: 0.56 USDT
Taxa de Entrada: 0.0120 USDT
Taxa de Saída: 0.0120 USDT
Horário de Fechamento: 2025-04-23 13:06:30
Tempo aberta: 0:00:45
Sinal ID: SIG_BOLLINGER_20250423_130545_789012
💰 PnL Simulado Total do Dia: 0.56 USDT

inverted_mode_analysis.txt
Resumo de Ordens Reais:
 pair   timeframe signal_engine order_mode       sum  count      mean
XRPUSDT        1m          RSI   original   -1.2000      5   -0.2400
XRPUSDT        1m          RSI   inverted    0.5600      2    0.2800

PnL Total Inverted (Real):
 pair   timeframe signal_engine    sum
XRPUSDT        1m          RSI 0.5600

Resumo de Ordens Simuladas:
 pair   timeframe signal_engine order_mode       sum  count      mean
XRPUSDT        1m    Bollinger   original   -1.2000      5   -0.2400
XRPUSDT        1m    Bollinger   inverted    0.5600      2    0.2800

PnL Total Inverted (Simulado):
 pair   timeframe signal_engine    sum
XRPUSDT        1m    Bollinger 0.5600

Troubleshooting

No Orders Placed: Verify PAIRS, TIMEFRAMES, and SIGNAL_ENGINES in config.py. Check bot.log for API errors.
No Inverted Orders: Ensure INVERTED_MODE_ENABLED/SIMULATED_INVERTED_MODE_ENABLED is True and SL thresholds are reached (inverted_state.csv/simulated_inverted_state.csv).
No Telegram Notifications: Verify TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, SIMULATED_TELEGRAM_CHAT_ID, and notification toggles (INVERTED_MODE_NOTIFICATIONS_ENABLED, SL_STREAK_NOTIFICATIONS_ENABLED).
CSV Errors: Run initialize_csvs.py to reset schemas. Check bot.log for file access issues (ensure filelock is installed).
API Rate Limits: Monitor bot.log for [BinanceUtils] Limite de peso da API próximo. Adjust PRICE_STATISTICS_UPDATE_INTERVAL if needed.

Notes

Isolation: Simulated mode (simulated_order_engine.py) uses no Binance API calls, ensuring real trading (main.py, monitor_pnl.py) is unaffected.
Logging: All events (signals, orders, pauses, inversions) are logged to bot.log with prefixes ([RSI], [Bollinger], [SIMULATED], [INVERTED MODE]).
Extensibility: Add new signal engines by extending main.py or enhance analysis in analyze_inverted_mode.py.

For further assistance or additional features (e.g., backtesting, dashboard), please specify your requirements.
