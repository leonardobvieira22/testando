import os, re
import pandas as pd
import threading
import time
import requests
import csv
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import REAL_API_KEY, REAL_API_SECRET, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, ORDERS_CSV, ORDERS_PNL_CSV, SIGNALS_CSV, PNL_CSV
from utils import logger, ensure_orders_pnl_csv_exists, salvar_soma_pnl_total
from filelock import FileLock

client = Client(REAL_API_KEY, REAL_API_SECRET)
CHECK_INTERVAL = 30
PNL_CALC_INTERVAL = 60
telegram_api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
chat_id = TELEGRAM_CHAT_ID

def get_order_info(symbol: str, order_id: int, cache={}):
    cache_key = f"{symbol}_{order_id}"
    if cache_key in cache:
        return cache[cache_key]
    try:
        trades = client.futures_account_trades(symbol=symbol, orderId=order_id)
        total_qty = sum(float(trade['qty']) for trade in trades)
        avg_price = sum(float(trade['price']) * float(trade['qty']) for trade in trades) / total_qty if total_qty else 0.0
        commission = sum(float(trade['commission']) for trade in trades)
        cache[cache_key] = (total_qty, avg_price, commission)
        logger.info(f"[PnL] Obteve trades para {symbol} orderId={order_id}")
        return total_qty, avg_price, commission
    except BinanceAPIException as e:
        logger.error(f"[PnL] Erro ao buscar ordem {order_id}: {e}")
        return 0.0, 0.0, 0.0

def parse_time(time_str):
    try:
        return pd.to_datetime(time_str)
    except Exception:
        return datetime.now()

def calculate_pnl():
    while True:
        try:
            with open(ORDERS_CSV, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                orders = [row for row in reader if row.get('status') == 'CLOSED']
                if not orders:
                    logger.info("[PnL] Nenhuma ordem fechada encontrada em orders.csv")
                    time.sleep(PNL_CALC_INTERVAL)
                    continue
                fieldnames = reader.fieldnames if reader.fieldnames else []
            extra_fields = [
                'real_entry_price', 'real_close_price', 'real_quantity',
                'fee_open', 'fee_close', 'pnl_bruto', 'pnl_liquido', 'duration_seconds'
            ]
            all_fields = fieldnames + [f for f in extra_fields if f not in fieldnames]
            with FileLock("pnl.csv.lock"):
                with open('pnl.csv', 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=all_fields)
                    writer.writeheader()
                    for row in orders:
                        symbol = row.get('pair')
                        direction = row.get('direction')
                        close_reason = row.get('close_reason')
                        open_id = int(float(row.get('order_id', 0)))
                        close_id = int(float(row.get('close_order_id', 0)))
                        open_time_str = row.get('timestamp')
                        close_time_str = row.get('close_timestamp')
                        qty_open, entry_price, fee_open = get_order_info(symbol, open_id)
                        qty_close, close_price, fee_close = get_order_info(symbol, close_id)
                        quantity = min(qty_open, qty_close)
                        pnl_bruto = (close_price - entry_price) * quantity
                        if direction == 'SELL':
                            pnl_bruto *= -1
                        pnl_liquido = pnl_bruto - (fee_open + fee_close)
                        open_time = parse_time(open_time_str)
                        close_time = parse_time(close_time_str)
                        duration_seconds = (close_time - open_time).total_seconds()
                        row['real_entry_price'] = f"{entry_price:.8f}"
                        row['real_close_price'] = f"{close_price:.8f}"
                        row['real_quantity'] = f"{quantity:.8f}"
                        row['fee_open'] = f"{fee_open:.8f}"
                        row['fee_close'] = f"{fee_close:.8f}"
                        row['pnl_bruto'] = f"{pnl_bruto:.8f}"
                        row['pnl_liquido'] = f"{pnl_liquido:.8f}"
                        row['duration_seconds'] = int(duration_seconds)
                        logger.info(f"[PnL] Escrevendo ordem {row.get('order_id')} em pnl.csv")
                        writer.writerow(row)
            logger.info("[PnL] Processamento concluído. Verifique o arquivo pnl.csv.")
            # atualiza pnl_total.txt com base em ORDERS_PNL_CSV
            try:
                salvar_soma_pnl_total()
            except Exception as e:
                logger.error(f"[PnL] Erro ao salvar pnl_total.txt: {e}")
        except Exception as e:
            logger.error(f"[PnL] Erro no loop de cálculo: {e}")
        time.sleep(PNL_CALC_INTERVAL)

def send_telegram_message(message):
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML"
    }
    try:
        requests.post(telegram_api_url, data=payload, timeout=10)
        logger.info(f"[TELEGRAM] Mensagem enviada: {message[:50]}...")
    except Exception as e:
        logger.error(f"[TELEGRAM] Erro ao enviar mensagem: {e}")

def get_signal_reason(signal_id):
    try:
        df_signals = pd.read_csv(SIGNALS_CSV)
        row = df_signals[df_signals['signal_id'] == signal_id]
        if not row.empty:
            return row.iloc[0]['reason']
    except Exception as e:
        logger.error(f"[TELEGRAM] Erro ao buscar reason do sinal: {e}")
    return "N/A"

def get_order_open_time(order_row):
    try:
        return pd.to_datetime(order_row['timestamp'])
    except Exception:
        return None

def get_order_close_time(order_row):
    try:
        return pd.to_datetime(order_row['close_timestamp'])
    except Exception:
        return None

def get_realized_pnl(client, symbol, open_order_id, close_order_id):
    realized_pnl = 0.0
    for oid in [open_order_id, close_order_id]:
        if pd.isna(oid):
            continue
        try:
            trades = client.futures_account_trades(symbol=symbol, orderId=int(float(oid)))
            logger.debug(f"[TELEGRAM] Trades para {symbol} orderId={oid}: {trades}")
            for trade in trades:
                realized_pnl += float(trade['realizedPnl'])
        except Exception as e:
            logger.error(f"[TELEGRAM] Erro ao buscar trades para {symbol} orderId={oid}: {e}")
    logger.debug(f"[TELEGRAM] RealizedPnL calculado para {symbol} ({open_order_id}, {close_order_id}): {realized_pnl}")
    return realized_pnl

def get_daily_realized_pnl(df):
    today = datetime.now().date()
    mask = (pd.to_datetime(df['close_timestamp'], errors='coerce').dt.date == today) & (df['status'] == "CLOSED")
    return df.loc[mask, 'pnl'].fillna(0).sum()

def notify_telegram():
    ensure_orders_pnl_csv_exists()
    last_notified = set()
    while True:
        try:
            # load realized PnL from pnl.csv
            if os.path.exists(PNL_CSV):
                df_pnl = pd.read_csv(PNL_CSV)
                pnl_map = {str(int(float(r['order_id']))): float(r['pnl_liquido']) for _, r in df_pnl.iterrows()}
            else:
                pnl_map = {}
            # load daily total from pnl_total.txt
            if os.path.exists('pnl_total.txt'):
                text = open('pnl_total.txt').read()
                m = re.search(r": ([\d\-.]+)", text)
                daily_total = float(m.group(1)) if m else 0.0
            else:
                daily_total = 0.0
            # notify for closed orders
            df_orders = pd.read_csv(ORDERS_PNL_CSV)
            for _, row in df_orders.iterrows():
                if row['status'] == 'CLOSED' and row['order_id'] not in last_notified:
                    order_id_str = str(int(float(row['order_id'])))
                    pnl_real = pnl_map.get(order_id_str, 0.0)
                    signal_id = row['signal_id']
                    reason = get_signal_reason(signal_id)
                    open_time = get_order_open_time(row)
                    close_time = get_order_close_time(row)
                    tempo_aberta = 'N/A'
                    if open_time and close_time:
                        tempo_aberta = str(close_time - open_time).split('.')[0]
                    close_time_str = pd.to_datetime(row['close_timestamp']).strftime('%Y-%m-%d %H:%M:%S') if not pd.isna(row['close_timestamp']) else 'N/A'
                    message = (
                        f"🔔 <b>Ordem Fechada</b>\n"
                        f"Par: <b>{row['pair']}</b>\n"
                        f"Timeframe: <b>{row['timeframe']}</b>\n"
                        f"Engine: <b>{row.get('signal_engine','N/A')}</b>\n"
                        f"Modo: <b>{row.get('order_mode','original')}</b>\n"
                        f"Sinal: <b>{reason}</b>\n"
                        f"Direção: <b>{row['direction']}</b>\n"
                        f"Preço de Entrada: <b>{row['entry_price']}</b>\n"
                        f"Preço de Fechamento: <b>{row.get('close_price','N/A')}</b>\n"
                        f"Motivo: <b>{row.get('close_reason','N/A')}</b>\n"
                        f"Realized PnL: <b>{pnl_real:.2f}</b> USDT\n"
                        f"Horário de Fechamento: <b>{close_time_str}</b>\n"
                        f"Tempo aberta: <b>{tempo_aberta}</b>\n"
                        f"Sinal ID: <code>{signal_id}</code>\n\n"
                        f"💰 <b>PnL realizado total do dia: {daily_total:.2f} USDT</b>"
                    )
                    send_telegram_message(message)
                    last_notified.add(row['order_id'])
        except Exception as e:
            logger.error(f"[TELEGRAM] Erro no loop de notificação: {e}")
        logger.info("[TELEGRAM] Heartbeat: script ativo e monitorando ordens...")
        time.sleep(CHECK_INTERVAL)

def main():
    ensure_orders_pnl_csv_exists()
    logger.info("[MONITOR] Iniciando monitor de PnL e notificações Telegram...")
    pnl_thread = threading.Thread(target=calculate_pnl, daemon=True)
    telegram_thread = threading.Thread(target=notify_telegram, daemon=True)
    pnl_thread.start()
    telegram_thread.start()
    try:
        while True:
            time.sleep(60)
            logger.info("[MONITOR] Monitor de PnL e Telegram ativo...")
    except KeyboardInterrupt:
        logger.info("[MONITOR] Monitor interrompido pelo usuário")

if __name__ == "__main__":
    main()