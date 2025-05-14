import os
import pandas as pd
import threading
import time
from binance.enums import SIDE_BUY, SIDE_SELL
from utils import logger, save_to_csv, generate_id, send_orphan_position_alert
import config
from binance_utils import BinanceUtils
from config import *
from datetime import datetime, timedelta
from filelock import FileLock
from decimal import Decimal, ROUND_DOWN

class CandleSignalEngine:
    """Motor de sinais de candle para trading automatizado na Binance Futures."""
    def __init__(self, api_key, api_secret, shared_state, ws=None):
        """Inicializa o motor de sinais de candle."""
        self.binance = BinanceUtils(api_key, api_secret)
        self.ws = ws
        self.active_pairs = [pair for pair, active in PAIRS.items() if active]
        self.active_timeframes = [tf for tf, active in CANDLE_TIMEFRAMES.items() if active]
        self.open_positions = shared_state['open_positions']
        self.active_monitors = shared_state['active_monitors']
        self.lock = shared_state['lock']
        self.running = True
        self.pause_until = shared_state['pause_until']
        logger.info(f"[Candle] Pares ativos configurados: {self.active_pairs}")
        logger.info(f"[Candle] Timeframes ativos configurados: {self.active_timeframes}")
        
        # Validar pares na inicialização
        self.validate_pairs()

        # Inicializar threads
        self.price_thread = threading.Thread(target=self.save_price_statistics_loop, daemon=True)
        self.price_thread.start()
        
        # Limpar ordens pendentes da sessão anterior
        for p in self.active_pairs:
            try:
                self.binance.cancel_open_orders(p)
                logger.info(f"[Candle] Open orders limpas em startup para {p}")
            except Exception as e:
                logger.warning(f"[Candle] Erro ao limpar open orders para {p}: {e}")

        # Garantir existência e cabeçalho de ORDERS_CSV e ORDERS_PNL_CSV
        if not os.path.exists(ORDERS_CSV):
            cols = [
                'order_id', 'signal_id', 'pair', 'timeframe', 'direction', 'entry_price',
                'quantity', 'tp_price', 'sl_price', 'tp_order_id', 'sl_order_id',
                'status', 'signal_engine', 'order_mode', 'timestamp', 'close_reason',
                'close_order_id', 'pnl', 'close_price', 'close_timestamp'
            ]
            pd.DataFrame(columns=cols).to_csv(ORDERS_CSV, index=False)
            logger.info(f"[Candle] Arquivo {ORDERS_CSV} criado com colunas padrão.")
        if not os.path.exists(ORDERS_PNL_CSV):
            pd.read_csv(ORDERS_CSV, nrows=0).to_csv(ORDERS_PNL_CSV, index=False)
            logger.info(f"[Candle] Arquivo {ORDERS_PNL_CSV} criado com colunas padrão.")

        # Recuperar ordens abertas
        self.recover_open_orders_monitoring()

    def validate_pairs(self):
        """Valida se todos os pares ativos são suportados na Binance Futures."""
        try:
            invalid_pairs = []
            for pair in self.active_pairs:
                if not self.binance.validate_futures_symbol(pair):
                    logger.error(f"[Candle] Par {pair} inválido para futuros. Desativando.")
                    invalid_pairs.append(pair)
            if invalid_pairs:
                self.active_pairs = [p for p in self.active_pairs if p not in invalid_pairs]
                logger.warning(f"[Candle] Pares inválidos desativados: {invalid_pairs}")
                message = (
                    f"🚨 <b>Pares Inválidos Detectados</b>\n"
                    f"Pares: <b>{', '.join(invalid_pairs)}</b>\n"
                    f"Motivo: <b>Não suportados na Binance Futures</b>\n"
                    f"Ação: <b>Desativados do trading</b>"
                )
                send_orphan_position_alert(invalid_pairs[0], "N/A", 0, "N/A")
        except Exception as e:
            logger.error(f"[Candle] Erro ao validar pares: {e}")

    def recover_open_orders_monitoring(self):
        """Recupera ordens abertas do orders.csv e verifica posições órfãs na Binance."""
        try:
            if not os.path.exists(ORDERS_CSV):
                logger.warning(f"[Candle][RECOVERY] Arquivo {ORDERS_CSV} não existe. Verificando posições órfãs.")
            else:
                df = pd.read_csv(ORDERS_CSV)
                df = df[(df['status'] == 'OPEN') & (df['signal_engine'] == 'Candle')]
                if df.empty:
                    logger.info("[Candle][RECOVERY] Nenhuma ordem aberta registrada em orders.csv.")
                else:
                    df = df.sort_values('timestamp', ascending=False)
                    df = df.groupby(['pair', 'timeframe']).first().reset_index()

                    for _, row in df.iterrows():
                        order_id = row['order_id']
                        pair = row['pair']
                        timeframe = row['timeframe']
                        position_key = (pair, timeframe)
                        entry_price = float(row['entry_price'])
                        quantity = float(row['quantity'])
                        direction = row['direction']
                        signal_id = row['signal_id']
                        tp_price = float(row.get('tp_price', 0)) if pd.notna(row.get('tp_price')) else 0
                        sl_price = float(row.get('sl_price', 0)) if pd.notna(row.get('sl_price')) else 0

                        with self.lock:
                            if position_key in self.open_positions:
                                logger.warning(f"[Candle][RECOVERY] Ordem duplicada encontrada para {pair} ({timeframe}). Fechando ordem antiga: {self.open_positions[position_key]['order_id']}")
                                old_order_id = self.open_positions[position_key]['order_id']
                                old_direction = self.open_positions[position_key]['side']
                                old_quantity = self.open_positions[position_key]['quantity']
                                close_side = SIDE_SELL if old_direction == SIDE_BUY else SIDE_BUY
                                close_order = self.binance.close_position(pair, close_side, old_quantity)
                                if close_order:
                                    close_order_id = close_order['orderId']
                                    close_price = self.binance.get_current_price(pair) or entry_price
                                    pnl = (close_price - entry_price) * quantity if old_direction == SIDE_BUY else (entry_price - close_price) * quantity
                                    self.update_order_status(old_order_id, 'CLOSED', 'Duplicate_Order', close_order_id, pnl, close_price)
                                    logger.info(f"[Candle][RECOVERY] Ordem duplicada fechada: {old_order_id}")
                                if old_order_id in self.active_monitors:
                                    del self.active_monitors[old_order_id]

                            if quantity <= 0 or entry_price <= 0:
                                logger.error(f"[Candle][RECOVERY] Dados inválidos para ordem {order_id}: quantity={quantity}, entry_price={entry_price}")
                                continue

                            if not self.binance.check_open_position(pair):
                                logger.warning(f"[Candle][RECOVERY] Posição não encontrada na Binance para {pair} ({timeframe}). Marcando como fechada.")
                                self.update_order_status(order_id, 'CLOSED', 'Position_Not_Found', None, 0, entry_price)
                                continue

                            try:
                                tp_id = row['tp_order_id'] if pd.notna(row['tp_order_id']) else None
                                sl_id = row['sl_order_id'] if pd.notna(row['sl_order_id']) else None
                            except KeyError as e:
                                logger.error(f"[Candle][RECOVERY] Colunas tp_order_id/sl_order_id ausentes para ordem {order_id}: {e}")
                                tp_id = None
                                sl_id = None

                            if not tp_id or not sl_id:
                                logger.warning(f"[Candle][RECOVERY] Ordem {order_id} sem pernas TP/SL. Tentando recriar.")
                                if direction == SIDE_BUY:
                                    tp_price = entry_price * (1 + TAKE_PROFIT_PERCENT)
                                    sl_price = entry_price * (1 - STOP_LOSS_PERCENT)
                                else:
                                    tp_price = entry_price * (1 - TAKE_PROFIT_PERCENT)
                                    sl_price = entry_price * (1 + STOP_LOSS_PERCENT)
                                prec = self.binance.get_price_precision(pair)
                                tp_price = round(tp_price, prec)
                                sl_price = round(sl_price, prec)
                                current_price = self.binance.get_current_price(pair)
                                if current_price:
                                    sl_price = self.adjust_sl_price(pair, direction, sl_price, current_price)
                                tp_order, sl_order = self.binance.place_tp_sl_orders(
                                    symbol=pair,
                                    side=direction,
                                    entry_price=entry_price,
                                    tp_percent=TAKE_PROFIT_PERCENT,
                                    sl_percent=STOP_LOSS_PERCENT,
                                    quantity=quantity
                                )
                                if tp_order and sl_order:
                                    tp_id = tp_order.get('orderId')
                                    sl_id = sl_order.get('orderId')
                                    with FileLock(f"{ORDERS_CSV}.lock"):
                                        df = pd.read_csv(ORDERS_CSV)
                                        df.loc[df['order_id'] == order_id, 'tp_order_id'] = tp_id
                                        df.loc[df['order_id'] == order_id, 'sl_order_id'] = sl_id
                                        df.loc[df['order_id'] == order_id, 'tp_price'] = tp_price
                                        df.loc[df['order_id'] == order_id, 'sl_price'] = sl_price
                                        df.to_csv(ORDERS_CSV, index=False)
                                        logger.info(f"[Candle][RECOVERY] Pernas TP/SL recriadas para ordem {order_id}: TP_ID={tp_id}, SL_ID={sl_id}")
                                else:
                                    logger.error(f"[Candle][RECOVERY] Falha ao recriar pernas TP/SL para ordem {order_id}. Fechando posição.")
                                    close_side = SIDE_SELL if direction == SIDE_BUY else SIDE_BUY
                                    close_order = self.binance.close_position(pair, close_side, quantity)
                                    if close_order:
                                        close_order_id = close_order['orderId']
                                        close_price = self.binance.get_current_price(pair) or entry_price
                                        pnl = (close_price - entry_price) * quantity if direction == SIDE_BUY else (entry_price - close_price) * quantity
                                        self.update_order_status(order_id, 'CLOSED', 'Failed_TP_SL', close_order_id, pnl, close_price)
                                    continue

                            t = threading.Thread(
                                target=self.monitor_and_close_position,
                                args=(pair, entry_price, tp_price, sl_price, signal_id, order_id, direction, quantity, timeframe),
                                daemon=True
                            )
                            with self.lock:
                                self.open_positions[position_key] = {
                                    'signal_id': signal_id,
                                    'order_id': order_id,
                                    'tp_order_id': tp_id,
                                    'sl_order_id': sl_id,
                                    'entry_price': entry_price,
                                    'quantity': quantity,
                                    'side': direction
                                }
                                self.active_monitors[order_id] = t
                            t.start()
                            logger.info(f"[Candle][RECOVERY] Retomando monitor para ordem {order_id} em {position_key}")

            self.check_orphan_positions()

        except Exception as e:
            logger.error(f"[Candle][RECOVERY] Erro geral ao recuperar ordens abertas: {e}")

    def check_orphan_positions(self):
        """Verifica posições abertas na Binance que não estão registradas em orders.csv."""
        try:
            positions = self.binance.client.futures_position_information()
            self.binance.track_weight(5)
            registered_orders = pd.read_csv(ORDERS_CSV) if os.path.exists(ORDERS_CSV) else pd.DataFrame(columns=['order_id', 'pair', 'timeframe', 'status'])
            for pos in positions:
                pair = pos['symbol']
                position_amt = float(pos.get('positionAmt', 0))
                if position_amt == 0 or pair not in self.active_pairs:
                    continue
                if not registered_orders[(registered_orders['pair'] == pair) & (registered_orders['status'] == 'OPEN')].empty:
                    continue
                logger.warning(f"[Candle][RECOVERY] Posição órfã encontrada para {pair}: {position_amt}")
                
                if not self.binance.validate_futures_symbol(pair):
                    logger.error(f"[Candle][RECOVERY] Símbolo {pair} inválido para futuros. Fechando posição órfã.")
                    quantity = abs(position_amt)
                    direction = SIDE_BUY if position_amt > 0 else SIDE_SELL
                    close_side = SIDE_SELL if direction == SIDE_BUY else SIDE_BUY
                    close_order = self.binance.close_position(pair, close_side, quantity)
                    if close_order:
                        close_order_id = close_order['orderId']
                        close_price = self.binance.get_current_price(pair) or 0
                        order_id = generate_id("ORPHAN")
                        signal_id = generate_id("ORPHAN_SIG")
                        order_data = {
                            'order_id': order_id,
                            'signal_id': signal_id,
                            'pair': pair,
                            'timeframe': self.active_timeframes[0],
                            'direction': direction,
                            'entry_price': float(pos.get('entryPrice', 0)),
                            'quantity': quantity,
                            'tp_price': None,
                            'sl_price': None,
                            'tp_order_id': None,
                            'sl_order_id': None,
                            'status': 'CLOSED',
                            'signal_engine': 'Candle',
                            'order_mode': 'original',
                            'timestamp': pd.Timestamp.now(),
                            'close_reason': 'Invalid_Symbol_Orphan',
                            'close_order_id': close_order_id,
                            'pnl': 0,
                            'close_price': close_price,
                            'close_timestamp': pd.Timestamp.now()
                        }
                        save_to_csv(order_data, ORDERS_CSV)
                        save_to_csv(order_data, ORDERS_PNL_CSV)
                        send_orphan_position_alert(pair, order_id, quantity, direction)
                    continue

                quantity = abs(position_amt)
                direction = SIDE_BUY if position_amt > 0 else SIDE_SELL
                entry_price = float(pos.get('entryPrice', 0))
                signal_id = generate_id("ORPHAN_SIG")
                order_id = generate_id("ORPHAN")
                timeframe = self.active_timeframes[0]
                position_key = (pair, timeframe)

                if direction == SIDE_BUY:
                    tp_price = entry_price * (1 + TAKE_PROFIT_PERCENT)
                    sl_price = entry_price * (1 - STOP_LOSS_PERCENT)
                else:
                    tp_price = entry_price * (1 - TAKE_PROFIT_PERCENT)
                    sl_price = entry_price * (1 + STOP_LOSS_PERCENT)
                prec = self.binance.get_price_precision(pair)
                tp_price = round(tp_price, prec)
                sl_price = round(sl_price, prec)
                current_price = self.binance.get_current_price(pair)
                if current_price:
                    sl_price = self.adjust_sl_price(pair, direction, sl_price, current_price)
                tp_order, sl_order = self.binance.place_tp_sl_orders(
                    symbol=pair,
                    side=direction,
                    entry_price=entry_price,
                    tp_percent=TAKE_PROFIT_PERCENT,
                    sl_percent=STOP_LOSS_PERCENT,
                    quantity=quantity
                )
                if tp_order and sl_order:
                    tp_id = tp_order.get('orderId')
                    sl_id = sl_order.get('orderId')
                    logger.info(f"[Candle][RECOVERY] Pernas TP/SL criadas para posição órfã {order_id}: TP_ID={tp_id}, SL_ID={sl_id}")
                else:
                    logger.error(f"[Candle][RECOVERY] Falha ao criar pernas TP/SL para posição órfã {order_id}. Fechando posição.")
                    close_side = SIDE_SELL if direction == SIDE_BUY else SIDE_BUY
                    close_order = self.binance.close_position(pair, close_side, quantity)
                    if close_order:
                        close_order_id = close_order['orderId']
                        close_price = self.binance.get_current_price(pair) or entry_price
                        pnl = (close_price - entry_price) * quantity if direction == SIDE_BUY else (entry_price - close_price) * quantity
                        order_data = {
                            'order_id': order_id,
                            'signal_id': signal_id,
                            'pair': pair,
                            'timeframe': timeframe,
                            'direction': direction,
                            'entry_price': entry_price,
                            'quantity': quantity,
                            'tp_price': None,
                            'sl_price': None,
                            'tp_order_id': None,
                            'sl_order_id': None,
                            'status': 'CLOSED',
                            'signal_engine': 'Candle',
                            'order_mode': 'original',
                            'timestamp': pd.Timestamp.now(),
                            'close_reason': 'Orphan_Position',
                            'close_order_id': close_order_id,
                            'pnl': pnl,
                            'close_price': close_price,
                            'close_timestamp': pd.Timestamp.now()
                        }
                        save_to_csv(order_data, ORDERS_CSV)
                        save_to_csv(order_data, ORDERS_PNL_CSV)
                        send_orphan_position_alert(pair, order_id, quantity, direction)
                        continue

                order_data = {
                    'order_id': order_id,
                    'signal_id': signal_id,
                    'pair': pair,
                    'timeframe': timeframe,
                    'direction': direction,
                    'entry_price': entry_price,
                    'quantity': quantity,
                    'tp_price': tp_price,
                    'sl_price': sl_price,
                    'tp_order_id': tp_id,
                    'sl_order_id': sl_id,
                    'status': 'OPEN',
                    'signal_engine': 'Candle',
                    'order_mode': 'original',
                    'timestamp': pd.Timestamp.now(),
                    'close_reason': None,
                    'close_order_id': None,
                    'pnl': None,
                    'close_price': None,
                    'close_timestamp': None
                }
                save_to_csv(order_data, ORDERS_CSV)
                save_to_csv(order_data, ORDERS_PNL_CSV)
                logger.info(f"[Candle][RECOVERY] Posição órfã registrada para {pair}: order_id={order_id}")

                t = threading.Thread(
                    target=self.monitor_and_close_position,
                    args=(pair, entry_price, tp_price, sl_price, signal_id, order_id, direction, quantity, timeframe),
                    daemon=True
                )
                with self.lock:
                    self.open_positions[position_key] = {
                        'signal_id': signal_id,
                        'order_id': order_id,
                        'tp_order_id': tp_id,
                        'sl_order_id': sl_id,
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'side': direction,
                        'signal_engine': 'Candle',
                        'order_mode': 'original'
                    }
                    self.active_monitors[order_id] = t
                t.start()
                logger.info(f"[Candle][RECOVERY] Monitoramento iniciado para posição órfã {order_id} em {position_key}")
                send_orphan_position_alert(pair, order_id, quantity, direction)

        except Exception as e:
            logger.error(f"[Candle][RECOVERY] Erro ao verificar posições órfãs: {e}")

    def save_price_statistics_loop(self):
        """Salva estatísticas de preço em intervalos regulares."""
        while self.running:
            try:
                for pair in self.active_pairs:
                    if pair not in [p for p, active in PAIRS.items() if active]:
                        logger.warning(f"[Candle] Par {pair} não está ativo em PAIRS. Ignorando.")
                        continue
                    for timeframe in self.active_timeframes:
                        data = BinanceUtils.get_historical_data(self.binance, pair, timeframe, limit=1)
                        if data is None or len(data) < 1 or 'close' not in data.columns:
                            logger.warning(f"[Candle] Dados insuficientes para {pair} ({timeframe})")
                            continue
                        price = float(data['close'].iloc[-1])
                        statistics_data = {
                            'id': generate_id("PRICE_STAT"),
                            'pair': pair,
                            'timeframe': timeframe,
                            'price': price,
                            'rsi': None,
                            'bb_upper': None,
                            'bb_middle': None,
                            'bb_lower': None,
                            'signal_engine': 'Candle',
                            'timestamp': pd.Timestamp.now()
                        }
                        save_to_csv(statistics_data, PRICES_STATISTICS_CSV)
                        logger.info(f"[Candle] Estatísticas salvas para {pair} ({timeframe}): id={statistics_data['id']}")
                time.sleep(PRICE_STATISTICS_UPDATE_INTERVAL)
            except Exception as e:
                logger.error(f"[Candle] Erro ao salvar estatísticas de preço: {e}")
                time.sleep(1)

    def adjust_sl_price(self, pair, direction, sl_price, current_price):
        """Ajusta o preço de stop-loss para evitar acionamento imediato."""
        try:
            volatility = self.binance.get_volatility(pair, '1m', limit=100)
            min_sl_percent = max(STOP_LOSS_PERCENT, volatility * 0.5)
            price_precision = self.binance.get_price_precision(pair)
            
            if direction == SIDE_BUY:
                if sl_price >= current_price:
                    sl_price = current_price * (1 - min_sl_percent)
            else:
                if sl_price <= current_price:
                    sl_price = current_price * (1 + min_sl_percent)
            
            return round(sl_price, price_precision)
        except Exception as e:
            logger.error(f"[Candle] Erro ao ajustar preço de SL para {pair}: {e}")
            return sl_price

    def generate_signals(self):
        """Gera sinais de trading com base em padrões de candles."""
        logger.info("[Candle] Iniciando loop de geração de sinais")
        try:
            if not os.path.exists(ORDERS_CSV):
                columns = [
                    'order_id', 'signal_id', 'pair', 'timeframe', 'direction', 'entry_price',
                    'quantity', 'tp_price', 'sl_price', 'tp_order_id', 'sl_order_id',
                    'status', 'signal_engine', 'order_mode', 'timestamp', 'close_reason',
                    'close_order_id', 'pnl', 'close_price', 'close_timestamp'
                ]
                pd.DataFrame(columns=columns).to_csv(ORDERS_CSV, index=False)
                logger.info(f"[Candle] Arquivo {ORDERS_CSV} criado com colunas padrão.")

            for pair in self.active_pairs:
                pu = self.pause_until.get(pair)
                if pu and datetime.now() < pu:
                    logger.info(f"[Candle] Pausa ativa para {pair} até {pu}, pulando geração de sinais.")
                    continue
                if pair not in [p for p, active in PAIRS.items() if active]:
                    logger.warning(f"[Candle] Par {pair} não está ativo em PAIRS. Ignorando.")
                    continue
                for timeframe in self.active_timeframes:
                    if not self._can_open_new(pair, timeframe):
                        logger.debug(f"[Candle] Não abrir sinal para {pair}({timeframe}): ordem anterior não fechada completamente.")
                        continue
                    try:
                        position_key = (pair, timeframe)
                        with self.lock:
                            if position_key in self.open_positions:
                                logger.info(f"[Candle] Posição aberta existente para {pair} ({timeframe}), sinal ID: {self.open_positions[position_key]['signal_id']}. Ignorando sinal.")
                                continue

                        if not self.binance.validate_futures_symbol(pair):
                            logger.error(f"[Candle] Par {pair} inválido para futuros. Ignorando sinal.")
                            continue

                        if self.binance.check_open_position(pair):
                            logger.warning(f"[Candle] Posição aberta detectada na Binance para {pair}, mas não registrada localmente. Fechando para evitar duplicatas.")
                            positions = self.binance.client.futures_position_information(symbol=pair)
                            self.binance.track_weight(5)
                            position_amt = 0.0
                            for pos in positions:
                                amt = float(pos.get('positionAmt', 0.0))
                                if amt != 0.0:
                                    position_amt = amt
                                    break
                            quantity = abs(position_amt)
                            if quantity <= 0:
                                logger.error(f"[Candle] Quantidade inválida na Binance para {pair}: {quantity}")
                                continue
                            close_side = SIDE_SELL if position_amt > 0 else SIDE_BUY
                            close_order = self.binance.close_position(pair, close_side, quantity)
                            if close_order:
                                logger.info(f"[Candle] Posição não registrada fechada na Binance: {pair}")
                            continue

                        period = CANDLE_PERIODS.get(timeframe, 1)
                        data = BinanceUtils.get_historical_data(self.binance, pair, timeframe, limit=period + 1)
                        if data is None or len(data) < period + 1 or not all(col in data.columns for col in ['open', 'close', 'close_time']):
                            logger.warning(f"[Candle] Dados insuficientes para {pair} ({timeframe})")
                            continue
                        if data[['open', 'close']].isnull().any().any():
                            logger.warning(f"[Candle] Dados históricos contêm valores nulos: {pair} ({timeframe})")
                            continue

                        data['open'] = data['open'].astype(float)
                        data['close'] = data['close'].astype(float)
                        last_closed = data.iloc[-2]
                        logger.debug(f"[Candle] Último candle fechado: open={last_closed['open']:.6f}, close={last_closed['close']:.6f}, close_time={pd.Timestamp(last_closed['close_time'], unit='ms')}")

                        is_green = last_closed['close'] > last_closed['open']
                        is_red = last_closed['close'] < last_closed['open']
                        if not (is_green or is_red):
                            logger.debug(f"[Candle] Nenhum padrão de candle consistente para {pair} ({timeframe})")
                            continue

                        direction = SIDE_BUY if is_green else SIDE_SELL
                        reason = f"Último candle fechado {'verde' if is_green else 'vermelho'}"
                        signal_id = generate_id("SIG")
                        logger.info(f"[Candle] Sinal gerado: {signal_id}, {pair} ({timeframe}), Direção: {direction}, Motivo: {reason}")

                        signal_data = {
                            'signal_id': signal_id,
                            'pair': pair,
                            'timeframe': timeframe,
                            'direction': direction,
                            'reason': reason,
                            'signal_engine': 'Candle',
                            'timestamp': pd.Timestamp.now()
                        }
                        save_to_csv(signal_data, SIGNALS_CSV)
                        logger.info(f"[Candle] Sinal salvo em {SIGNALS_CSV}: signal_id={signal_id}")

                        exchange = self.binance.client.futures_exchange_info()
                        symbol_data = next(s for s in exchange['symbols'] if s['symbol'] == pair)
                        lot_filter = next(f for f in symbol_data['filters'] if f['filterType'] == 'LOT_SIZE')
                        min_qty = Decimal(lot_filter['minQty'])
                        step_size = Decimal(lot_filter['stepSize'])
                        price = self.binance.get_current_price(pair)
                        if price is None or price <= 0:
                            logger.error(f"[Candle] Preço atual inválido para {pair} ({timeframe}): {price}")
                            continue
                        qty = (Decimal(str(ORDER_VALUE_USDT)) * Decimal(str(LEVERAGE))) / Decimal(str(price))
                        quantity = qty.quantize(step_size, rounding=ROUND_DOWN)
                        if quantity < min_qty or quantity <= 0:
                            logger.error(f"[Candle] Quantidade inválida para {pair} ({timeframe}): {quantity}, mínimo: {min_qty}")
                            continue

                        entry_price = last_closed['close']
                        if direction == SIDE_BUY:
                            tp_price = entry_price * (1 + TAKE_PROFIT_PERCENT)
                            sl_price = entry_price * (1 - STOP_LOSS_PERCENT)
                        else:
                            tp_price = entry_price * (1 - TAKE_PROFIT_PERCENT)
                            sl_price = entry_price * (1 + STOP_LOSS_PERCENT)
                        prec = self.binance.get_price_precision(pair)
                        tp_price = round(tp_price, prec)
                        sl_price = round(sl_price, prec)
                        if price:
                            sl_price = self.adjust_sl_price(pair, direction, sl_price, price)
                        logger.info(f"[Candle] Calculado TP={tp_price:.6f}, SL={sl_price:.6f} para {pair} ({timeframe})")

                        # Configurar o modo de margem como "isolated"
                        try:
                            self.binance.client.futures_change_margin_type(symbol=pair, marginType='ISOLATED')
                            logger.info(f"[Candle] Modo de margem alterado para ISOLATED para {pair}")
                        except Exception as e:
                            if "No need to change margin type" in str(e):
                                logger.debug(f"[Candle] {pair} já está em modo ISOLATED")
                            else:
                                logger.error(f"[Candle] Erro ao configurar modo ISOLATED para {pair}: {e}")
                                continue

                        # Configurar a alavancagem
                        try:
                            self.binance.client.futures_change_leverage(symbol=pair, leverage=LEVERAGE)
                            logger.info(f"[Candle] Alavancagem configurada para {LEVERAGE}x para {pair}")
                        except Exception as e:
                            logger.error(f"[Candle] Erro ao configurar alavancagem para {pair}: {e}")
                            continue

                        # Colocar a ordem de mercado
                        logger.info(f"[Candle] Colocando ordem de mercado: {pair}, Direção: {direction}, Quantidade: {quantity}, Valor USDT: {ORDER_VALUE_USDT}, Alavancagem: {LEVERAGE}")
                        order = self.place_market_order_with_retry(symbol=pair, side=direction, usdt_value=ORDER_VALUE_USDT, leverage=LEVERAGE)
                        if not order or 'orderId' not in order:
                            logger.error(f"[Candle] Falha ao colocar ordem de mercado para {pair} ({timeframe})")
                            continue

                        executed_qty = float(order.get('executedQty', 0))
                        qty = executed_qty if executed_qty > 0 else float(quantity)
                        logger.info(f"[Candle] Ordem executada: ID={order['orderId']}, Quantidade={qty}")

                        qty_precision = self.binance.get_quantity_precision(pair)
                        qty = round(qty, qty_precision)
                        if qty < min_qty:
                            logger.error(f"[Candle] Quantidade ajustada {qty} para {pair} menor que o mínimo {min_qty}. Cancelando ordem.")
                            self.cancel_or_close_order(pair, order['orderId'], direction, qty)
                            continue

                        max_retries = 3
                        tp_id = None
                        sl_id = None
                        for attempt in range(max_retries):
                            self.binance.cancel_open_orders(pair)
                            try:
                                tp_order, sl_order = self.binance.place_tp_sl_orders(
                                    symbol=pair,
                                    side=direction,
                                    entry_price=entry_price,
                                    tp_percent=TAKE_PROFIT_PERCENT,
                                    sl_percent=STOP_LOSS_PERCENT,
                                    quantity=qty
                                )
                                if not tp_order or not sl_order:
                                    raise Exception("Resposta de TP/SL inválida")
                                tp_id = tp_order.get('orderId')
                                sl_id = sl_order.get('orderId')
                                logger.info(f"[Candle] TP/SL criados: TP_ID={tp_id}, SL_ID={sl_id}, Quantidade={qty}")
                                break
                            except Exception as e:
                                logger.error(f"[Candle] Falha ao criar TP/SL para {pair} ({timeframe}), tentativa {attempt + 1}: {e}")
                                if attempt == max_retries - 1:
                                    logger.error(f"[Candle] Falha persistente ao criar TP/SL. Cancelando/fechando ordem {order['orderId']}")
                                    self.cancel_or_close_order(pair, order['orderId'], direction, qty)
                                    return
                                time.sleep(1)

                        order_data = {
                            'order_id': order['orderId'],
                            'signal_id': signal_id,
                            'pair': pair,
                            'timeframe': timeframe,
                            'direction': direction,
                            'entry_price': entry_price,
                            'quantity': qty,
                            'tp_price': tp_price if tp_id else None,
                            'sl_price': sl_price if sl_id else None,
                            'tp_order_id': tp_id,
                            'sl_order_id': sl_id,
                            'status': 'OPEN',
                            'signal_engine': 'Candle',
                            'order_mode': 'original',
                            'timestamp': pd.Timestamp.now(),
                            'close_reason': None,
                            'close_order_id': None,
                            'pnl': None,
                            'close_price': None,
                            'close_timestamp': None
                        }
                        with FileLock(f"{ORDERS_CSV}.lock"):
                            df = pd.read_csv(ORDERS_CSV) if os.path.exists(ORDERS_CSV) else pd.DataFrame(columns=order_data.keys())
                            df = df[df['order_id'] != order['orderId']]
                            df = pd.concat([df, pd.DataFrame([order_data])], ignore_index=True)
                            df.to_csv(ORDERS_CSV, index=False)
                        with FileLock(f"{ORDERS_PNL_CSV}.lock"):
                            df_pnl = pd.read_csv(ORDERS_PNL_CSV) if os.path.exists(ORDERS_PNL_CSV) else pd.DataFrame(columns=order_data.keys())
                            df_pnl = df_pnl[df_pnl['order_id'] != order['orderId']]
                            df_pnl = pd.concat([df_pnl, pd.DataFrame([order_data])], ignore_index=True)
                            df_pnl.to_csv(ORDERS_PNL_CSV, index=False)
                        logger.info(f"[Candle] Ordem salva em {ORDERS_CSV}: order_id={order['orderId']}")

                        if tp_id and sl_id:
                            t = threading.Thread(
                                target=self.monitor_and_close_position,
                                args=(pair, entry_price, tp_price, sl_price, signal_id, order['orderId'], direction, qty, timeframe),
                                daemon=True
                            )
                            t.start()
                            with self.lock:
                                self.open_positions[position_key] = {
                                    'signal_id': signal_id,
                                    'order_id': order['orderId'],
                                    'tp_order_id': tp_id,
                                    'sl_order_id': sl_id,
                                    'entry_price': entry_price,
                                    'quantity': qty,
                                    'side': direction,
                                    'signal_engine': 'Candle',
                                    'order_mode': 'original'
                                }
                                self.active_monitors[order['orderId']] = t
                    except Exception as e:
                        logger.error(f"[Candle] Erro ao processar {pair} ({timeframe}): {e}")
        except Exception as e:
            logger.error(f"[Candle] Erro geral na geração de sinais: {e}")

    def cancel_or_close_order(self, pair, order_id, direction, quantity):
        """Tenta cancelar uma ordem de mercado ou fechar a posição."""
        try:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    order_status = self.binance.client.futures_get_order(symbol=pair, orderId=int(order_id))
                    if order_status['status'] in ['NEW', 'PARTIALLY_FILLED']:
                        self.cancel_order_with_retry(symbol=pair, orderId=int(order_id))
                        logger.info(f"[Candle] Ordem de mercado {order_id} cancelada")
                        return True
                    elif order_status['status'] == 'FILLED':
                        logger.warning(f"[Candle] Ordem {order_id} já preenchida. Fechando posição.")
                        close_side = SIDE_SELL if direction == SIDE_BUY else SIDE_BUY
                        close_order = self.binance.close_position(pair, close_side, quantity)
                        if close_order:
                            logger.info(f"[Candle] Posição para ordem {order_id} fechada")
                            return True
                        else:
                            logger.error(f"[Candle] Falha ao fechar posição para ordem {order_id}")
                    else:
                        logger.info(f"[Candle] Ordem {order_id} já está {order_status['status']}. Nenhuma ação necessária.")
                        return True
                except Exception as e:
                    logger.error(f"[Candle] Erro ao cancelar/fechar ordem {order_id}, tentativa {attempt + 1}: {e}")
                    if attempt == max_retries - 1:
                        logger.critical(f"[Candle] Falha persistente ao cancelar/fechar ordem {order_id}. Enviando alerta.")
                        send_orphan_position_alert(pair, order_id, quantity, direction)
                    time.sleep(0.5 * (attempt + 1))
            return False
        except Exception as e:
            logger.error(f"[Candle] Erro geral ao cancelar/fechar ordem {order_id}: {e}")
            return False

    def get_current_price(self, pair):
        # Robust fallback: tenta WebSocket, se falhar usa HTTP
        price = None
        if self.ws and hasattr(self.ws, 'is_active') and self.ws.is_active():
            try:
                price = self.ws.get_price(pair)
            except Exception as e:
                logger.warning(f"[Candle] Erro ao obter preço via WebSocket para {pair}: {e}")
        if price is None:
            price = self.binance.get_current_price(pair)
            logger.warning(f"[Candle] Fallback para HTTP para {pair}")
        return price

    def monitor_and_close_position(self, symbol, entry_price, tp_price, sl_price, signal_id, order_id, side, quantity, timeframe):
        """Monitora e fecha posições quando TP/SL são atingidos."""
        try:
            logger.info(f"[Candle][MONITOR] Iniciando monitoramento para {symbol} ({timeframe}), Ordem ID: {order_id}")
            position_key = (symbol, timeframe)
            max_retries = 3
            last_log_time = 0
            while self.running and order_id in self.active_monitors:
                try:
                    with self.lock:
                        tp_id = self.open_positions.get(position_key, {}).get('tp_order_id')
                        sl_id = self.open_positions.get(position_key, {}).get('sl_order_id')
                    if not tp_id or not sl_id:
                        logger.error(f"[Candle][MONITOR] IDs de TP/SL ausentes para {symbol} ({timeframe}): TP_ID={tp_id}, SL_ID={sl_id}")
                        close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
                        close_order = self.binance.close_position(symbol, close_side, quantity)
                        if close_order:
                            close_order_id = close_order['orderId']
                            close_price = self.binance.get_current_price(symbol) or entry_price
                            pnl = (close_price - entry_price) * quantity if side == SIDE_BUY else (entry_price - close_price) * quantity
                            self.update_order_status(order_id, 'CLOSED', 'Missing_TP_SL', close_order_id, pnl, close_price)
                        with self.lock:
                            self.open_positions.pop(position_key, None)
                            self.active_monitors.pop(order_id, None)
                        send_orphan_position_alert(symbol, order_id, quantity, side)
                        return

                    current_price = self.get_current_price(symbol)
                    if current_price is None:
                        logger.warning(f"[Candle][MONITOR] Preço atual indisponível para {symbol} ({timeframe})")
                        time.sleep(1)
                        continue

                    position_quantity = None
                    for attempt in range(max_retries):
                        position_quantity = self.binance.get_position_quantity(symbol)
                        if position_quantity is not None and position_quantity > 0:
                            break
                        logger.warning(f"[Candle][MONITOR] Falha ao obter quantidade da posição para {symbol} ({timeframe}), tentativa {attempt + 1}")
                        time.sleep(0.5)
                    if position_quantity is None or position_quantity <= 0:
                        logger.warning(f"[Candle][MONITOR] Posição não encontrada para {symbol} ({timeframe}). Verificando status da ordem.")
                        try:
                            order_status = self.binance.client.futures_get_order(symbol=symbol, orderId=int(order_id))
                            if order_status['status'] in ['CANCELLED', 'FILLED', 'EXPIRED']:
                                logger.info(f"[Candle][MONITOR] Ordem {order_id} está {order_status['status']}. Verificando pernas TP/SL.")
                                tp_status = self.binance.client.futures_get_order(symbol=symbol, orderId=int(tp_id))['status']
                                sl_status = self.binance.client.futures_get_order(symbol=symbol, orderId=int(sl_id))['status']
                                if tp_status == 'FILLED':
                                    close_reason = 'TP'
                                    close_price = float(self.binance.client.futures_get_order(symbol=symbol, orderId=int(tp_id))['avgPrice'])
                                elif sl_status == 'FILLED':
                                    close_reason = 'SL'
                                    close_price = float(self.binance.client.futures_get_order(symbol=symbol, orderId=int(sl_id))['avgPrice'])
                                else:
                                    close_reason = f"Order_{order_status['status']}"
                                    close_price = entry_price
                                pnl = (close_price - entry_price) * quantity if side == SIDE_BUY else (entry_price - close_price) * quantity
                                self.update_order_status(order_id, 'CLOSED', close_reason, None, pnl, close_price)
                                with self.lock:
                                    self.open_positions.pop(position_key, None)
                                    self.active_monitors.pop(order_id, None)
                                return
                        except Exception as e:
                            logger.error(f"[Candle][MONITOR] Erro ao verificar status da ordem {order_id}: {e}")
                        self.update_order_status(order_id, 'CLOSED', 'Position_Not_Found', None, 0, entry_price)
                        with self.lock:
                            self.open_positions.pop(position_key, None)
                            self.active_monitors.pop(order_id, None)
                        return

                    qty_precision = self.binance.get_quantity_precision(symbol)
                    adjusted_quantity = round(position_quantity, qty_precision)
                    if adjusted_quantity <= 0:
                        logger.error(f"[Candle][MONITOR] Quantidade ajustada inválida para {symbol} ({timeframe}): {adjusted_quantity}")
                        return

                    if time.time() - last_log_time >= 2:
                        logger.info(
                            f"[Candle][MONITOR] {symbol} ({timeframe}) | Signal ID: {signal_id} | "
                            f"Preço atual: {current_price:.6f} | TP: {tp_price:.6f} | SL: {sl_price:.6f} | "
                            f"TP_ID: {tp_id} | SL_ID: {sl_id} | Quantidade: {adjusted_quantity}"
                        )
                        last_log_time = time.time()

                    tp_status = None
                    sl_status = None
                    for attempt in range(max_retries):
                        try:
                            tp_status = self.binance.client.futures_get_order(symbol=symbol, orderId=int(tp_id))['status']
                            sl_status = self.binance.client.futures_get_order(symbol=symbol, orderId=int(sl_id))['status']
                            break
                        except Exception as e:
                            logger.error(f"[Candle][MONITOR] Erro ao verificar status de TP/SL para {symbol} ({timeframe}), tentativa {attempt + 1}: {e}")
                            if attempt == max_retries - 1:
                                logger.warning(f"[Candle][MONITOR] Usando verificação de preço como fallback para {symbol} ({timeframe})")
                                if side == SIDE_BUY:
                                    if current_price >= tp_price:
                                        tp_status = 'FILLED'
                                        sl_status = 'NEW'
                                    elif current_price <= sl_price:
                                        sl_status = 'FILLED'
                                        tp_status = 'NEW'
                                else:
                                    if current_price <= tp_price:
                                        tp_status = 'FILLED'
                                        sl_status = 'NEW'
                                    elif current_price >= sl_price:
                                        sl_status = 'FILLED'
                                        tp_status = 'NEW'
                                break
                            time.sleep(0.5)

                    close_reason = None
                    filled = None
                    if tp_status == 'FILLED':
                        filled = tp_id
                        close_reason = 'TP'
                    elif sl_status == 'FILLED':
                        filled = sl_id
                        close_reason = 'SL'

                    if filled or close_reason:
                        close_price = float(self.binance.client.futures_get_order(symbol=symbol, orderId=int(filled))['avgPrice']) if filled else current_price
                        close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
                        logger.info(f"[Candle][MONITOR] Fechando posição: {symbol} ({timeframe}), Quantidade: {adjusted_quantity}, Motivo: {close_reason}")

                        if not filled:
                            close_order = self.binance.close_position(symbol, close_side, adjusted_quantity)
                            if not close_order:
                                logger.error(f"[Candle][MONITOR] Falha ao fechar posição para {symbol} ({timeframe})")
                                time.sleep(1)
                                continue
                            close_order_id = close_order['orderId']
                        else:
                            close_order_id = filled

                        for leg_id in (tp_id, sl_id):
                            if leg_id and leg_id != close_order_id:
                                self.binance.cancel_open_orders(symbol)
                                logger.info(f"[Candle][MONITOR] Leg pendente cancelada: ID={leg_id}")

                        # Fechar possíveis fragmentos remanescentes de posição
                        leftover = self.binance.get_position_quantity(symbol)
                        if leftover and leftover > 0:
                            fragment_close = self.binance.close_position(symbol, close_side, leftover)
                            if fragment_close:
                                logger.info(f"[Candle][MONITOR] Fragmento de posição fechado: {leftover}")

                        if close_reason == 'SL':
                            cfg = config.SL_STREAK_CONFIG.get(symbol, {}) if not config.SIMULATED_MODE else config.SIMULATED_SL_STREAK_CONFIG.get(symbol, {})
                            mins = cfg.get('pause_minutes', 3)
                            if mins > 0:
                                self.pause_until[symbol] = datetime.now() + timedelta(minutes=mins)
                                logger.info(f"[Candle][MONITOR] Pausando {symbol} por {mins} minutos após SL.")

                        pnl = (close_price - entry_price) * adjusted_quantity if side == SIDE_BUY else (entry_price - close_price) * adjusted_quantity
                        self.update_order_status(order_id, 'CLOSED', close_reason, close_order_id, pnl, close_price)

                        with self.lock:
                            self.open_positions.pop(position_key, None)
                            self.active_monitors.pop(order_id, None)
                        return
                    time.sleep(0.06)
                except Exception as e:
                    logger.error(f"[Candle][MONITOR] Erro durante monitoramento para {symbol} ({timeframe}), Sinal ID: {signal_id}: {e}")
                    time.sleep(1)
        except Exception as e:
            logger.error(f"[Candle][MONITOR] Erro geral no monitoramento para {symbol} ({timeframe}), Sinal ID: {signal_id}: {e}")

    def update_order_status(self, order_id, status, close_reason, close_order_id, pnl, close_price):
        """Atualiza o status de uma ordem no orders.csv."""
        try:
            logger.info(
                f"[Candle] Atualizando ordem: ID={order_id}, Status={status}, "
                f"Motivo={close_reason}, Close_Order_ID={close_order_id}, PnL={pnl:.2f} USDT"
            )
            with FileLock(f"{ORDERS_CSV}.lock"):
                df = pd.read_csv(ORDERS_CSV)
                df.loc[df['order_id'] == order_id, 'status'] = status
                df.loc[df['order_id'] == order_id, 'close_reason'] = close_reason
                df.loc[df['order_id'] == order_id, 'close_order_id'] = close_order_id
                df.loc[df['order_id'] == order_id, 'pnl'] = pnl
                df.loc[df['order_id'] == order_id, 'close_timestamp'] = pd.Timestamp.now()
                df.loc[df['order_id'] == order_id, 'close_price'] = close_price
                df.to_csv(ORDERS_CSV, index=False)
                if status == 'CLOSED':
                    with FileLock(f"{ORDERS_PNL_CSV}.lock"):
                        df_pnl = pd.read_csv(ORDERS_PNL_CSV) if os.path.exists(ORDERS_PNL_CSV) else pd.DataFrame(columns=df.columns)
                        closed_row = df[df['order_id'] == order_id]
                        df_pnl = df_pnl[df_pnl['order_id'] != order_id]
                        df_pnl = pd.concat([df_pnl, closed_row], ignore_index=True)
                        df_pnl.to_csv(ORDERS_PNL_CSV, index=False)
        except Exception as e:
            logger.error(f"[Candle] Erro ao atualizar status da ordem {order_id}: {e}")

    def stop(self):
        """Para o motor de sinais."""
        try:
            self.running = False
            logger.info("[Candle] Motor de sinais parado.")
        except Exception as e:
            logger.error(f"[Candle] Erro ao parar motor de sinais: {e}")

    def _can_open_new(self, pair, timeframe):
        """Verifica se é seguro abrir nova ordem para (pair, timeframe)."""
        try:
            key = (pair, timeframe)
            if key in self.open_positions:
                return False
            df = pd.read_csv(ORDERS_CSV)
            df2 = df[(df['pair'] == pair) & (df['timeframe'] == timeframe)].sort_values('timestamp', ascending=False)
            if not df2.empty and df2.iloc[0]['status'] != 'CLOSED':
                logger.warning(f"[Candle] Última ordem para {pair}({timeframe}) não fechada: {df2.iloc[0]['status']}")
                return False
            open_orders = self.binance.client.futures_get_open_orders(symbol=pair)
            if open_orders:
                logger.warning(f"[Candle] Ordens pendentes na Binance para {pair}: {len(open_orders)} - cancelando para limpeza")
                try:
                    self.binance.cancel_open_orders(pair)
                except Exception as e:
                    logger.error(f"[Candle] Falha ao cancelar ordens pendentes para {pair}: {e}")
                return False
            if self.binance.check_open_position(pair):
                logger.warning(f"[Candle] Posição ainda aberta na Binance para {pair}")
                return False
            return True
        except Exception as e:
            logger.warning(f"[Candle] Falha ao verificar se pode abrir nova ordem para {pair} ({timeframe}): {e}")
            return False

    def place_market_order_with_retry(self, *args, **kwargs):
        for attempt in range(3):
            try:
                order = self.binance.place_market_order(*args, **kwargs)
                if order and float(order.get('executedQty', 0.0)) > 0:
                    return order
                logger.warning(f"[Candle] Tentativa {attempt + 1}/3: Falha ao colocar ordem.")
            except Exception as e:
                logger.error(f"[Candle] Erro ao colocar ordem (tentativa {attempt+1}): {e}")
            time.sleep(1)
        logger.critical("[Candle] Falha persistente ao colocar ordem!")
        return None

    def cancel_order_with_retry(self, *args, **kwargs):
        for attempt in range(3):
            try:
                result = self.binance.cancel_open_orders(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"[Candle] Erro ao cancelar ordem (tentativa {attempt+1}): {e}")
                time.sleep(1)
        logger.critical("[Candle] Falha persistente ao cancelar ordem!")
        return None