import logging
import os
import pandas as pd
from datetime import datetime
from filelock import FileLock
from config import *
import requests
from typing import Dict, Optional
import csv
import json
import time

# Define telegram endpoint
telegram_api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"

class SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            super().emit(record)
        except UnicodeEncodeError:
            pass

# Configuração do logger
logging.basicConfig(
    level=logging.DEBUG,  # Alterado para DEBUG para maior detalhamento nos logs
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        SafeStreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Função config_logger adicionada para resolver o erro de importação
def config_logger():
    """Configure or reconfigure the logger with appropriate settings."""
    # You can customize logger settings here if needed
    # For example, set different log levels for different handlers
    logger.setLevel(logging.DEBUG)
    
    # If you need to add new handlers at runtime:
    # file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    # file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    # logger.addHandler(file_handler)
    
    logger.info("Logger configured/reconfigured")
    return logger

def calculate_rsi(data, period=13):
    """
    Calcula o Relative Strength Index (RSI) usando a média móvel suavizada de Wilder (SMMA).
    Temporariamente removida a suavização SMA de 13 períodos para teste.

    Args:
        data (pd.DataFrame): DataFrame com a coluna 'close' contendo dados de preço.
        period (int): Número de períodos para o cálculo do RSI (padrão é 13).

    Returns:
        pd.Series: Valores do RSI (base, sem suavização adicional).
    """
    delta = data['close'].diff()
    
    # Separa ganhos e perdas
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calcula as médias de ganhos e perdas usando SMMA de Wilder
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    # Calcula o Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    # Calcula o RSI padrão
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def save_to_csv(data, filename):
    try:
        with FileLock(f"{filename}.lock"):
            df = pd.DataFrame([data])
            header = not os.path.exists(filename)
            df.to_csv(filename, mode='a', index=False, header=header)
            if filename != PRICES_STATISTICS_CSV:
                # logger.info(f"Dados salvos em {filename}: {data.get('id', data.get('signal_id', 'N/A'))}")
                pass
    except Exception as e:
        logger.error(f"Erro ao salvar em {filename}: {e}")

def generate_id(prefix="ID"):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

def log_sl_streak_progress(pair, timeframe, current_streak, threshold, pause_minutes, mode, engine):
    prefix = "[SIMULATED]" if mode == "simulated" else f"[{engine}]"
    remaining = threshold - current_streak
    logger.warning(
        f"{prefix}[SL STREAK] ⚠️ {pair} ({timeframe}) [{current_streak}/{threshold}]: "
        f"Mais {remaining} SL(s) ativará uma pausa de {pause_minutes:.1f} minutos."
    )

def send_sl_streak_telegram(message, mode):
    chat_id = SIMULATED_TELEGRAM_CHAT_ID if mode == "simulated" else TELEGRAM_CHAT_ID
    enabled = SIMULATED_SL_STREAK_NOTIFICATIONS_ENABLED if mode == "simulated" else SL_STREAK_NOTIFICATIONS_ENABLED
    if not enabled:
        logger.debug(f"[{'SIMULATED' if mode == 'simulated' else 'REAL'}][TELEGRAM] Notificações de SL Streak desativadas")
        return
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        requests.post(telegram_api_url, data=payload, timeout=10)
        logger.info(f"[{'SIMULATED' if mode == 'simulated' else 'REAL'}][TELEGRAM] Mensagem de SL Streak enviada: {message[:50]}...")
    except Exception as e:
        logger.error(f"[{'SIMULATED' if mode == 'simulated' else 'REAL'}][TELEGRAM] Erro ao enviar mensagem de SL Streak: {e}")

def manage_inverted_state(pair, timeframe, sl_streak, invert_count_remaining, is_inverted, mode, reset=False):
    filename = SIMULATED_INVERTED_STATE_CSV if mode == "simulated" else INVERTED_STATE_CSV
    try:
        with FileLock(f"{filename}.lock"):
            if os.path.exists(filename):
                df = pd.read_csv(filename)
            else:
                df = pd.DataFrame(columns=['pair', 'timeframe', 'sl_streak', 'invert_count_remaining', 'is_inverted'])
            key = (pair, timeframe)
            if reset:
                df = df[~((df['pair'] == pair) & (df['timeframe'] == timeframe))]
                sl_streak = 0
                invert_count_remaining = 0
                is_inverted = False
            else:
                df = df[~((df['pair'] == pair) & (df['timeframe'] == timeframe))]
                df = pd.concat([df, pd.DataFrame([{
                    'pair': pair,
                    'timeframe': timeframe,
                    'sl_streak': sl_streak,
                    'invert_count_remaining': invert_count_remaining,
                    'is_inverted': is_inverted
                }])], ignore_index=True)
            df.to_csv(filename, index=False)
            logger.info(f"[{'SIMULATED' if mode == 'simulated' else 'REAL'}] Estado de Inverted Mode salvo em {filename} para {pair} ({timeframe})")
    except Exception as e:
        logger.error(f"[{'SIMULATED' if mode == 'simulated' else 'REAL'}] Erro ao salvar estado de Inverted Mode em {filename}: {e}")

def load_inverted_state(pair, timeframe, mode):
    filename = SIMULATED_INVERTED_STATE_CSV if mode == "simulated" else INVERTED_STATE_CSV
    try:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            row = df[(df['pair'] == pair) & (df['timeframe'] == timeframe)]
            if not row.empty:
                return {
                    'sl_streak': row.iloc[0]['sl_streak'],
                    'invert_count_remaining': row.iloc[0]['invert_count_remaining'],
                    'is_inverted': row.iloc[0]['is_inverted']
                }
        return {'sl_streak': 0, 'invert_count_remaining': 0, 'is_inverted': False}
    except Exception as e:
        logger.error(f"[{'SIMULATED' if mode == 'simulated' else 'REAL'}] Erro ao carregar estado de Inverted Mode de {filename}: {e}")
        return {'sl_streak': 0, 'invert_count_remaining': 0, 'is_inverted': False}

def log_inverted_mode(pair, timeframe, action, original_direction, inverted_direction, invert_count_remaining, mode, engine):
    prefix = "[SIMULATED]" if mode == "simulated" else f"[{engine}]"
    if action == "activated":
        logger.info(
            f"{prefix}[INVERTED MODE] 🔄 Inverted Mode ativado para {pair} ({timeframe}) após atingir o limite de SLs. "
            f"Próximas {invert_count_remaining} ordens serão invertidas."
        )
    elif action == "applied":
        logger.info(
            f"{prefix}[INVERTED MODE] 🔄 Ordem invertida para {pair} ({timeframe}): "
            f"Original {original_direction} → Invertida {inverted_direction}. "
            f"Ordens invertidas restantes: {invert_count_remaining}."
        )
    elif action == "deactivated":
        logger.info(
            f"{prefix}[INVERTED MODE] 🔄 Inverted Mode desativado para {pair} ({timeframe}). "
            f"Ordens voltarão ao modo original."
        )

def send_inverted_mode_telegram(message, mode):
    chat_id = SIMULATED_TELEGRAM_CHAT_ID if mode == "simulated" else TELEGRAM_CHAT_ID
    enabled = SIMULATED_INVERTED_MODE_NOTIFICATIONS_ENABLED if mode == "simulated" else INVERTED_MODE_NOTIFICATIONS_ENABLED
    if not enabled:
        logger.debug(f"[{'SIMULATED' if mode == 'simulated' else 'REAL'}][TELEGRAM] Notificações de Inverted Mode desativadas")
        return
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        requests.post(telegram_api_url, data=payload, timeout=10)
        logger.info(f"[{'SIMULATED' if mode == 'simulated' else 'REAL'}][TELEGRAM] Mensagem de Inverted Mode enviada: {message[:50]}...")
    except Exception as e:
        logger.error(f"[{'SIMULATED' if mode == 'simulated' else 'REAL'}][TELEGRAM] Erro ao enviar mensagem de Inverted Mode: {e}")

def log_configurations(config):
    logger.info("=== Configurações Selecionadas ===")
    logger.info(f"Timeframes Ativos (Real): {[tf for tf, active in config.TIMEFRAMES.items() if active]}")
    logger.info(f"Pares Ativos (Real): {[pair for pair, active in config.PAIRS.items() if active]}")
    logger.info(f"Valor da Ordem (Real): {config.ORDER_VALUE_USDT} USDT")
    logger.info(f"Alavancagem (Real): {config.LEVERAGE}x")
    logger.info(f"Stop Loss (Real): {config.STOP_LOSS_PERCENT*100:.2f}%")
    logger.info(f"RSI Período: {config.RSI_PERIOD}")
    logger.info(f"RSI Compra Threshold: {config.RSI_BUY_THRESHOLD}")
    logger.info(f"RSI Venda Threshold: {config.RSI_SELL_THRESHOLD}")
    logger.info(f"RSI Delta Mínimo: {config.RSI_DELTA_MIN}")
    logger.info(f"Motores de Sinais Ativos: {[engine for engine, active in config.SIGNAL_ENGINES.items() if active]}")
    logger.info(f"Modo Simulado: {getattr(config, 'SIMULATED_MODE', 'N/A')}")
    logger.info(f"Timeframes Ativos (Simulado): {[tf for tf, active in getattr(config, 'SIMULATED_TIMEFRAMES', {}).items() if active]}")
    logger.info(f"Pares Ativos (Simulado): {[pair for pair, active in getattr(config, 'SIMULATED_PAIRS', {}).items() if active]}")
    logger.info(f"Valor da Ordem (Simulado): {getattr(config, 'SIMULATED_ORDER_VALUE_USDT', 'N/A')} USDT")
    logger.info(f"Alavancagem (Simulado): {getattr(config, 'SIMULATED_LEVERAGE', 'N/A')}x")
    logger.info(f"Stop Loss (Simulado): {getattr(config, 'SIMULATED_SL_PERCENT', 0)*100:.2f}%")
    logger.info(f"Taxa Simulada: {getattr(config, 'SIMULATED_FEE_PERCENT', 0)*100:.4f}%")
    logger.info(f"SL Streak Simulado Ativado: {getattr(config, 'SIMULATED_SL_STREAK_ENABLED', 'N/A')}")
    logger.info(f"Notificações Telegram Simuladas: {getattr(config, 'SIMULATED_TELEGRAM_ENABLED', 'N/A')}")
    logger.info(f"Notificações SL Streak (Real): {getattr(config, 'SL_STREAK_NOTIFICATIONS_ENABLED', 'N/A')}")
    logger.info(f"Notificações SL Streak (Simulado): {getattr(config, 'SIMULATED_SL_STREAK_NOTIFICATIONS_ENABLED', 'N/A')}")
    logger.info(f"Inverted Mode (Real): {getattr(config, 'INVERTED_MODE_ENABLED', 'N/A')}")
    logger.info(f"Notificações Inverted Mode (Real): {getattr(config, 'INVERTED_MODE_NOTIFICATIONS_ENABLED', 'N/A')}")
    logger.info(f"Inverted Mode (Simulado): {getattr(config, 'SIMULATED_INVERTED_MODE_ENABLED', 'N/A')}")
    logger.info(f"Notificações Inverted Mode (Simulado): {getattr(config, 'SIMULATED_INVERTED_MODE_NOTIFICATIONS_ENABLED', 'N/A')}")
    logger.info(f"Arquivos: Signals={config.SIGNALS_CSV}, Orders={config.ORDERS_CSV}, Prices/Statistics={getattr(config, 'PRICES_STATISTICS_CSV', 'N/A')}, Simulated Orders={getattr(config, 'SIMULATED_ORDERS_CSV', 'N/A')}, Log={config.LOG_FILE}")
    logger.info(f"Intervalos: PnL={config.PNL_UPDATE_INTERVAL}s, Prices/Statistics={getattr(config, 'PRICE_STATISTICS_UPDATE_INTERVAL', 'N/A')}s")
    logger.info(f"SL Streak (Real): Config={getattr(config, 'SL_STREAK_CONFIG', 'N/A')}")
    logger.info(f"SL Streak (Simulado): Config={getattr(config, 'SIMULATED_SL_STREAK_CONFIG', 'N/A')}")
    logger.info(f"Inverted Mode (Real): Config={getattr(config, 'INVERTED_MODE_CONFIG', 'N/A')}")
    logger.info(f"Inverted Mode (Simulado): Config={getattr(config, 'SIMULATED_INVERTED_MODE_CONFIG', 'N/A')}")
    logger.info("================================")

def ensure_orders_pnl_csv_exists(orders_csv='orders.csv', pnl_csv='orders_pnl.csv'):
    if not os.path.exists(pnl_csv):
        if os.path.exists(orders_csv):
            df = pd.read_csv(orders_csv)
            df.to_csv(pnl_csv, index=False)
        else:
            cols = [
                'order_id','signal_id','pair','timeframe','direction','entry_price',
                'quantity','tp_price','sl_price','tp_order_id','sl_order_id',
                'status','signal_engine','order_mode','timestamp',
                'close_reason','close_order_id','pnl','close_timestamp','close_price'
            ]
            pd.DataFrame(columns=cols).to_csv(pnl_csv, index=False)

def ensure_simulated_orders_csv_exists(orders_csv='simulated_orders.csv'):
    if not os.path.exists(orders_csv):
        cols = ['order_id','signal_id','pair','timeframe','direction','entry_price','quantity','tp_price','sl_price','status','timestamp','close_reason','pnl','entry_fee','exit_fee','close_timestamp','close_price','signal_engine','order_mode']
        pd.DataFrame(columns=cols).to_csv(orders_csv, index=False)

def ensure_simulated_pause_log_csv_exists(pause_csv='simulated_pause_log.csv'):
    if not os.path.exists(pause_csv):
        cols = ['id','pair','timeframe','trigger_time','pause_end','sl_streak','signal_engine']
        pd.DataFrame(columns=cols).to_csv(pause_csv, index=False)

def save_streak_pause(sl_streak: Dict[tuple, int], pause_until: Dict[tuple, Optional[pd.Timestamp]], filename: str = "streak_pause.csv") -> None:
    try:
        with FileLock(f"{filename}.lock"):
            data = []
            for (pair, tf), streak in sl_streak.items():
                pause_time = pause_until.get((pair, tf), None)
                data.append({"pair": pair, "timeframe": tf, "streak": streak, "pause_until": pause_time})
            pd.DataFrame(data).to_csv(filename, index=False)
            logger.info(f"Streaks e pausas salvos em {filename}")
    except Exception as e:
        logger.error(f"Erro ao salvar streaks/pausas em {filename}: {e}")

def load_streak_pause(filename: str = "streak_pause.csv") -> tuple[Dict[tuple, int], Dict[tuple, Optional[pd.Timestamp]]]:
    sl_streak: Dict[tuple, int] = {}
    pause_until: Dict[tuple, Optional[pd.Timestamp]] = {}
    try:
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            for _, row in df.iterrows():
                key = (row['pair'], row['timeframe'])
                sl_streak[key] = row['streak']
                if pd.notna(row['pause_until']):
                    pause_until[key] = pd.to_datetime(row['pause_until'])
            logger.info(f"Streaks e pausas carregados de {filename}")
    except Exception as e:
        logger.error(f"Erro ao carregar streaks/pausas de {filename}: {e}")
    return sl_streak, pause_until

def log_order_open(order):
    logger.info(f"[LOG] Ordem aberta anotada: {order.get('pair', '')} ({order.get('timeframe', '')}), Sinal ID: {order.get('signal_id', '')}, Ordem ID: {order.get('order_id', '')}, Engine: {order.get('signal_engine', 'N/A')}, Modo: {order.get('order_mode', 'N/A')}")

def log_order_close(order):
    logger.info(f"[LOG] Ordem fechada anotada: {order.get('pair', '')} ({order.get('timeframe', '')}), Sinal ID: {order.get('signal_id', '')}, Ordem ID: {order.get('order_id', '')}, Motivo: {order.get('close_reason', '')}, Engine: {order.get('signal_engine', 'N/A')}, Modo: {order.get('order_mode', 'N/A')}")

def salvar_soma_pnl_total(_df=None):
    try:
        if os.path.exists('pnl.csv'):
            df = pd.read_csv('pnl.csv')
            soma = df['pnl_liquido'].astype(float).sum() if 'pnl_liquido' in df.columns else 0.0
        else:
            soma = 0.0
        with open('pnl_total.txt', 'w') as f:
            f.write(f"PnL total de ordens fechadas: {soma:.2f} USDT\n")
        logger.info(f"[TELEGRAM] Soma total de ordens fechadas: {soma:.2f} USDT (salvo em pnl_total.txt)")
    except Exception as e:
        logger.error(f"Erro ao calcular / salvar pnl_total.txt: {e}")

def send_orphan_position_alert(pair, order_id, quantity, direction):
    """
    Envia um alerta via Telegram para posições órfãs detectadas sem pernas TP/SL.

    Args:
        pair (str): Par de negociação (ex.: AVAAIUSDT).
        order_id (str): ID da ordem de mercado.
        quantity (float): Quantidade da posição.
        direction (str): Direção da ordem (BUY/SELL).
    """
    message = (
        f"🚨 <b>Posição Órfã Detectada</b>\n"
        f"Par: <b>{pair}</b>\n"
        f"Ordem ID: <b>{order_id}</b>\n"
        f"Quantidade: <b>{quantity}</b>\n"
        f"Direção: <b>{direction}</b>\n"
        f"Motivo: <b>Posição aberta sem pernas TP/SL</b>\n"
        f"Ação: <b>Fechar manualmente ou verificar o sistema</b>"
    )
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        requests.post(telegram_api_url, data=payload, timeout=10)
        logger.info(f"[TELEGRAM] Alerta de posição órfã enviado para {pair}: {message[:50]}...")
    except Exception as e:
        logger.error(f"[TELEGRAM] Erro ao enviar alerta de posição órfã para {pair}: {e}")

def update_order_csv(csv_path, symbol, side, quantity, close_price, realized_pnl, close_reason):
    """
    Atualiza o arquivo orders.csv com os detalhes da ordem fechada.
    """
    import csv
    from datetime import datetime

    with open(csv_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            datetime.now().isoformat(),
            symbol,
            side,
            quantity,
            close_price,
            realized_pnl,
            close_reason
        ])

def log_accepted_signal(symbol, timeframe, direction, entry_price, position_size, stop_loss, take_profit, rsi_value, additional_data=None):
    file_path = "data/accepted_signals.csv"
    headers = ["timestamp", "date_time", "symbol", "timeframe", "signal_direction", "entry_price", "position_size", "stop_loss", "take_profit", "rsi_value", "additional_data"]

    if not os.path.exists(file_path):
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            int(time.time()),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            symbol,
            timeframe,
            direction,
            entry_price,
            position_size,
            stop_loss,
            take_profit,
            rsi_value,
            json.dumps(additional_data, ensure_ascii=False)
        ])

from filelock import FileLock

def acquire_csv_lock(file_path):
    """
    Retorna um gerenciador de contexto para bloquear arquivos CSV.
    """
    lock_path = f"{file_path}.lock"
    return FileLock(lock_path)