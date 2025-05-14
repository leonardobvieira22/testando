import pandas as pd
import threading
import time
from binance.enums import SIDE_BUY, SIDE_SELL
from utils import logger, save_to_csv, generate_id, save_streak_pause, load_streak_pause, log_sl_streak_progress, send_sl_streak_telegram, manage_inverted_state, load_inverted_state, log_inverted_mode, send_inverted_mode_telegram
from binance_utils import BinanceUtils
from config import *
import os
from datetime import datetime, timedelta
from filelock import FileLock

class BollingerSignalEngine:
    def __init__(self, api_key, api_secret, shared_state):
        self.binance = BinanceUtils(api_key, api_secret)
        # stream de preços compartilhado via WebSocket principal
        self.prices = shared_state['prices']
        self.active_pairs = [pair for pair, active in PAIRS.items() if active]
        logger.info(f"[Bollinger] Pares ativos configurados: {self.active_pairs}")
        self.active_timeframes = [tf for tf, active in TIMEFRAMES.items() if active]
        logger.info(f"[Bollinger] Timeframes ativos configurados: {self.active_timeframes}")
        self.open_positions = shared_state['open_positions']
        self.running = True
        self.active_monitors = shared_state['active_monitors']
        self.sl_streak = shared_state['sl_streak']
        self.pause_until = shared_state['pause_until']
        self.lock = shared_state['lock']
        self.price_thread = threading.Thread(target=self.save_price_statistics_loop)
        self.price_thread.daemon = True
        self.price_thread.start()
        self.recover_open_orders_monitoring()

    def recover_open_orders_monitoring(self):
        try:
            df = pd.read_csv(ORDERS_CSV)
            for idx, row in df.iterrows():
                if row['status'] == 'OPEN' and row.get('signal_engine', '') == 'Bollinger':
                    order_id = row['order_id']
                    with self.lock:
                        if order_id not in self.active_monitors or not self.active_monitors[order_id].is_alive():
                            t = threading.Thread(
                                target=self.monitor_and_close_position,
                                args=(row['pair'], row['entry_price'], row['tp_price'], row['sl_price'], row['signal_id'], row['order_id'], row['direction'], row['quantity'], row['timeframe']),
                                daemon=True
                            )
                            t.start()
                            self.active_monitors[order_id] = t
                            logger.info(f"[Bollinger][RECOVERY] Monitoramento reativado para ordem aberta: {order_id} ({row['pair']} {row['timeframe']})")
                        else:
                            logger.info(f"[Bollinger][RECOVERY] Monitoramento já ativo para ordem: {order_id}")
        except Exception as e:
            logger.error(f"[Bollinger][RECOVERY] Erro ao recuperar monitoramento de ordens abertas: {e}")

    def save_price_statistics_loop(self):
        while self.running:
            try:
                for pair in self.active_pairs:
                    if pair not in [p for p, active in PAIRS.items() if active]:
                        logger.warning(f"[Bollinger] Par {pair} não está ativo em PAIRS. Ignorando.")
                        continue
                    for timeframe in self.active_timeframes:
                        data = self.binance.get_historical_data(pair, timeframe, limit=BOLLINGER_PERIOD + 1)
                        if data is None or len(data) < BOLLINGER_PERIOD + 1:
                            logger.warning(f"[Bollinger] Dados insuficientes para salvar preço/estatísticas: {pair} ({timeframe})")
                            continue
                        price = data['close'].iloc[-1]
                        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data)
                        if any(pd.isna(x) for x in [bb_upper, bb_middle, bb_lower]):
                            logger.warning(f"[Bollinger] Dados de Bollinger inválidos para salvar: {pair} ({timeframe})")
                            continue
                            
                        # Calcula a volatilidade atual
                        volatility = self.binance.get_volatility(pair, timeframe)
                        
                        statistics_data = {
                            'id': generate_id("PRICE_STAT"),
                            'pair': pair,
                            'timeframe': timeframe,
                            'price': price,
                            'rsi': None,
                            'bb_upper': bb_upper,
                            'bb_middle': bb_middle,
                            'bb_lower': bb_lower,
                            'signal_engine': 'Bollinger',
                            'timestamp': pd.Timestamp.now(),
                            'volatility': volatility  # Adiciona volatilidade às estatísticas
                        }
                        save_to_csv(statistics_data, PRICES_STATISTICS_CSV)
                time.sleep(PRICE_STATISTICS_UPDATE_INTERVAL)
            except Exception as e:
                logger.error(f"[Bollinger] Erro ao salvar preço/estatísticas: {e}")
                time.sleep(1)

    def calculate_bollinger_bands(self, data):
        from utils import calculate_bollinger_bands
        upper, middle, lower = calculate_bollinger_bands(data, period=BOLLINGER_PERIOD, deviation=BOLLINGER_DEVIATION)
        return upper.iloc[-1], middle.iloc[-1], lower.iloc[-1]

    def generate_signals(self):
        logger.info("[Bollinger] Iniciando loop de geração de sinais")
        for pair in self.active_pairs:
            if pair not in [p for p, active in PAIRS.items() if active]:
                logger.warning(f"[Bollinger] Par {pair} não está ativo em PAIRS. Ignorando.")
                continue
            for timeframe in self.active_timeframes:
                self.process_pair_timeframe(pair, timeframe)
        logger.info("[Bollinger] Loop de geração de sinais concluído")

    def process_pair_timeframe(self, pair, timeframe):
        try:
            position_key = (pair, timeframe)
            key = (pair, timeframe)
            with self.lock:
                if key in self.pause_until:
                    now = datetime.now()
                    if now < self.pause_until[key]:
                        minutos = int((self.pause_until[key] - now).total_seconds() // 60) + 1
                        logger.info(f"[Bollinger][PAUSE] Pausa ativa para {key}, faltam {minutos} minutos para retomar operações.")
                        return
                    else:
                        logger.info(f"[Bollinger][PAUSE] Pausa encerrada para {key}, retomando operações.")
                        del self.pause_until[key]
                        self.sl_streak[key] = 0
                        save_streak_pause(self.sl_streak, self.pause_until)
            if position_key in self.open_positions:
                logger.info(f"[Bollinger] Posição aberta existente para {pair} ({timeframe}), sinal ID: {self.open_positions[position_key]['signal_id']}. Ignorando novo sinal.")
                return
            if self.binance.check_open_position(pair):
                logger.info(f"[Bollinger] Posição aberta detectada na Binance para {pair}. Ignorando novo sinal.")
                return
            logger.info(f"[Bollinger] Processando {pair} ({timeframe})")
            data = self.binance.get_historical_data(pair, timeframe, limit=BOLLINGER_PERIOD + 1)
            if data is None or len(data) < BOLLINGER_PERIOD + 1:
                logger.warning(f"[Bollinger] Dados insuficientes para {pair} ({timeframe})")
                return
            price = data['close'].iloc[-1]
            bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data)
            if any(pd.isna(x) for x in [bb_upper, bb_middle, bb_lower]):
                logger.warning(f"[Bollinger] Bandas de Bollinger inválidas para {pair} ({timeframe})")
                return
                
            # Calcula a volatilidade atual
            volatility = self.binance.get_volatility(pair, timeframe)
            logger.info(f"[Bollinger] Volatilidade atual para {pair} ({timeframe}): {volatility:.4f}")
            
            signal_id = generate_id("SIG_BOLLINGER")
            direction = None
            reason = None
            if price <= bb_lower:
                direction = SIDE_BUY
                reason = f"Preço abaixo da banda inferior: {price:.4f} <= {bb_lower:.4f}"
            elif price >= bb_upper:
                direction = SIDE_SELL
                reason = f"Preço acima da banda superior: {price:.4f} >= {bb_upper:.4f}"
            if direction:
                logger.info(f"[Bollinger] Sinal gerado: {signal_id}, {pair} ({timeframe}), Direção: {direction}, Motivo: {reason}, Volatilidade: {volatility:.4f}")
                signal_data = {
                    'signal_id': signal_id,
                    'pair': pair,
                    'timeframe': timeframe,
                    'direction': direction,
                    'price': price,
                    'bb_upper': bb_upper,
                    'bb_middle': bb_middle,
                    'bb_lower': bb_lower,
                    'reason': reason,
                    'signal_engine': 'Bollinger',
                    'timestamp': pd.Timestamp.now(),
                    'volatility': volatility  # Adiciona volatilidade ao sinal
                }
                save_to_csv(signal_data, SIGNALS_CSV)
                
                # Verificar Inverted Mode
                invert_state = load_inverted_state(pair, timeframe, mode="real")
                order_direction = direction
                order_mode = "original"
                if INVERTED_MODE_ENABLED and invert_state['is_inverted']:
                    order_direction = SIDE_SELL if direction == SIDE_BUY else SIDE_BUY
                    order_mode = "inverted"
                    log_inverted_mode(
                        pair=pair,
                        timeframe=timeframe,
                        action="applied",
                        original_direction=direction,
                        inverted_direction=order_direction,
                        invert_count_remaining=invert_state['invert_count_remaining'],
                        mode="real",
                        engine="Bollinger"
                    )
                    if INVERTED_MODE_NOTIFICATIONS_ENABLED:
                        message = (
                            f"🔄 <b>Ordem Invertida</b>\n"
                            f"Par: <b>{pair}</b>\n"
                            f"Timeframe: <b>{timeframe}</b>\n"
                            f"Engine: <b>Bollinger</b>\n"
                            f"Original: <b>{direction}</b>\n"
                            f"Invertida: <b>{order_direction}</b>\n"
                            f"Ordens Invertidas Restantes: <b>{invert_state['invert_count_remaining']}</b>"
                        )
                        send_inverted_mode_telegram(message, mode="real")
                
                # Configurar o modo de margem como "isolated"
                try:
                    self.binance.client.futures_change_margin_type(symbol=pair, marginType='ISOLATED')
                    logger.info(f"[Bollinger] Modo de margem alterado para ISOLATED para {pair}")
                except Exception as e:
                    if "No need to change margin type" in str(e):
                        logger.debug(f"[Bollinger] {pair} já está em modo ISOLATED")
                    else:
                        logger.error(f"[Bollinger] Erro ao configurar modo ISOLATED para {pair}: {e}")
                        return

                # Configurar a alavancagem
                try:
                    self.binance.client.futures_change_leverage(symbol=pair, leverage=LEVERAGE)
                    logger.info(f"[Bollinger] Alavancagem configurada para {LEVERAGE}x para {pair}")
                except Exception as e:
                    logger.error(f"[Bollinger] Erro ao configurar alavancagem para {pair}: {e}")
                    return

                # Colocar a ordem de mercado
                order = self.place_market_order_with_retry(
                    symbol=pair,
                    side=order_direction,
                    usdt_value=ORDER_VALUE_USDT,
                    leverage=LEVERAGE
                )
                if order:
                    quantity = float(order['executedQty']) or 0.0
                    entry_price = price
                    
                    # Ajusta SL/TP baseado na volatilidade do par
                    if VOLATILITY_DYNAMIC_ADJUSTMENT:
                        # Normaliza a volatilidade em relação aos níveis de referência
                        volatility_ratio = volatility / VOLATILITY_MEDIUM
                        
                        # Ajusta os multiplicadores com base na volatilidade
                        sl_multiplier = min(VOLATILITY_SL_MAX_MULTIPLIER, 
                                         max(VOLATILITY_SL_MIN_MULTIPLIER, 
                                            1.0 + (volatility_ratio - 1.0) * 0.3))
                        tp_multiplier = min(VOLATILITY_TP_MAX_MULTIPLIER,
                                         max(VOLATILITY_TP_MIN_MULTIPLIER,
                                            1.0 + (volatility_ratio - 1.0) * 0.2))
                        
                        # Aplica os multiplicadores aos percentuais base
                        adjusted_stop_loss = STOP_LOSS_PERCENT * sl_multiplier
                        adjusted_take_profit = TAKE_PROFIT_PERCENT * tp_multiplier
                        
                        logger.info(f"[Bollinger] SL/TP ajustados por volatilidade ({volatility:.4f}): "
                                    f"SL: {STOP_LOSS_PERCENT:.4f} → {adjusted_stop_loss:.4f}, "
                                    f"TP: {TAKE_PROFIT_PERCENT:.4f} → {adjusted_take_profit:.4f}")
                    else:
                        adjusted_stop_loss = STOP_LOSS_PERCENT
                        adjusted_take_profit = TAKE_PROFIT_PERCENT
                    
                    if order_direction == SIDE_BUY:
                        tp_price = entry_price * (1 + adjusted_take_profit)
                        sl_price = entry_price * (1 - adjusted_stop_loss)
                    else:
                        tp_price = entry_price * (1 - adjusted_take_profit)
                        sl_price = entry_price * (1 + adjusted_stop_loss)
                    
                    price_precision = self.binance.get_price_precision(pair)
                    tp_price = round(tp_price, price_precision)
                    sl_price = round(sl_price, price_precision)
                    
                    tp_order, sl_order = self.binance.place_tp_sl_orders(
                        symbol=pair,
                        side=order_direction,
                        entry_price=entry_price,
                        tp_percent=adjusted_take_profit,
                        sl_percent=adjusted_stop_loss,
                        quantity=quantity
                    )
                    
                    with self.lock:
                        self.open_positions[position_key] = {
                            'signal_id': signal_id,
                            'order_id': order['orderId'],
                            'entry_price': entry_price,
                            'quantity': quantity,
                            'side': order_direction,
                            'tp_price': tp_price,
                            'sl_price': sl_price,
                            'signal_engine': 'Bollinger',
                            'order_mode': order_mode,
                            'volatility': volatility  # Armazena volatilidade na posição
                        }
                    logger.info(f"[Bollinger] Posição aberta registrada: {pair} ({timeframe}), Sinal ID: {signal_id}")
                    order_data = {
                        'order_id': order['orderId'],
                        'signal_id': signal_id,
                        'pair': pair,
                        'timeframe': timeframe,
                        'direction': order_direction,
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'status': 'OPEN',
                        'signal_engine': 'Bollinger',
                        'order_mode': order_mode,
                        'timestamp': pd.Timestamp.now(),
                        'volatility': volatility  # Armazena volatilidade na ordem
                    }
                    save_to_csv(order_data, ORDERS_CSV)
                    save_to_csv(order_data, ORDERS_PNL_CSV)
                    if INVERTED_MODE_ENABLED and invert_state['is_inverted']:
                        new_invert_count = max(0, invert_state['invert_count_remaining'] - 1)
                        manage_inverted_state(
                            pair=pair,
                            timeframe=timeframe,
                            sl_streak=invert_state['sl_streak'],
                            invert_count_remaining=new_invert_count,
                            is_inverted=new_invert_count > 0,
                            mode="real"
                        )
                    t = threading.Thread(
                        target=self.monitor_and_close_position,
                        args=(pair, entry_price, tp_price, sl_price, signal_id, order['orderId'], order_direction, quantity, timeframe),
                        daemon=True
                    )
                    t.start()
                    with self.lock:
                        self.active_monitors[order['orderId']] = t
                else:
                    logger.error(f"[Bollinger] Falha ao colocar ordem para {pair} ({timeframe}), Sinal ID: {signal_id}")
            else:
                logger.debug(f"[Bollinger] Nenhum sinal gerado para {pair} ({timeframe}), Preço: {price:.4f}, BB: {bb_lower:.4f}/{bb_middle:.4f}/{bb_upper:.4f}")
        except Exception as e:
            logger.error(f"[Bollinger] Erro ao processar {pair} ({timeframe}): {e}")

    def get_current_price(self, pair):
        # Robust fallback: tenta WebSocket, se falhar usa HTTP
        price = None
        if self.ws and hasattr(self.ws, 'is_active') and self.ws.is_active():
            try:
                price = self.ws.get_price(pair)
            except Exception as e:
                logger.warning(f"[Bollinger] Erro ao obter preço via WebSocket para {pair}: {e}")
        if price is None:
            price = self.binance.get_current_price(pair)
            logger.warning(f"[Bollinger] Fallback para HTTP para {pair}")
        return price

    def place_market_order_with_retry(self, *args, **kwargs):
        for attempt in range(3):
            try:
                order = self.binance.place_market_order(*args, **kwargs)
                if order and float(order.get('executedQty', 0.0)) > 0:
                    return order
                logger.warning(f"[Bollinger] Tentativa {attempt + 1}/3: Falha ao colocar ordem.")
            except Exception as e:
                logger.error(f"[Bollinger] Erro ao colocar ordem (tentativa {attempt+1}): {e}")
            time.sleep(1)
        logger.critical("[Bollinger] Falha persistente ao colocar ordem!")
        return None

    def cancel_order_with_retry(self, *args, **kwargs):
        for attempt in range(3):
            try:
                result = self.binance.cancel_open_orders(*args, **kwargs)
                return result
            except Exception as e:
                logger.error(f"[Bollinger] Erro ao cancelar ordem (tentativa {attempt+1}): {e}")
                time.sleep(1)
        logger.critical("[Bollinger] Falha persistente ao cancelar ordem!")
        return None

    def monitor_and_close_position(self, symbol, entry_price, tp_price, sl_price, signal_id, order_id, side, quantity, timeframe):
        try:
            logger.info(f"[Bollinger] Iniciando monitoramento de preço para {symbol} ({timeframe}), Sinal ID: {signal_id}, Ordem ID: {order_id}")
            position_key = (symbol, timeframe)
            closed = False
            last_log_time = 0
            while not closed and self.running:
                try:
                    current_price = self.get_current_price(symbol)
                    if current_price is None:
                        logger.warning(f"[Bollinger] Preço atual indisponível para {symbol} ({timeframe})")
                        time.sleep(1)
                        continue
                    if time.time() - last_log_time >= 5:
                        logger.info(f"[Bollinger][MONITOR] {symbol} ({timeframe}) | Sinal ID: {signal_id} | Preço atual: {current_price} | TP: {tp_price} | SL: {sl_price}")
                        last_log_time = time.time()
                    close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
                    close_reason = None
                    if side == SIDE_BUY:
                        if current_price >= tp_price:
                            close_reason = "TP"
                            logger.info(f"[Bollinger][MONITOR] TP atingido para {symbol} ({timeframe}), Sinal ID: {signal_id}")
                        elif current_price <= sl_price:
                            close_reason = "SL"
                            logger.info(f"[Bollinger][MONITOR] SL atingido para {symbol} ({timeframe}), Sinal ID: {signal_id}")
                    else:
                        if current_price <= tp_price:
                            close_reason = "TP"
                            logger.info(f"[Bollinger][MONITOR] TP atingido para {symbol} ({timeframe}), Sinal ID: {signal_id}")
                        elif current_price >= sl_price:
                            close_reason = "SL"
                            logger.info(f"[Bollinger][MONITOR] SL atingido para {symbol} ({timeframe}), Sinal ID: {signal_id}")
                    if close_reason:
                        logger.info(f"[Bollinger][MONITOR] Tentando fechar posição para {symbol} ({timeframe}), Sinal ID: {signal_id}, Motivo: {close_reason}")
                        close_order = self.binance.close_position(symbol, close_side, quantity)
                        if close_order:
                            close_price = current_price
                            if side == SIDE_BUY:
                                pnl = (close_price - entry_price) * quantity
                            else:
                                pnl = (entry_price - close_price) * quantity
                            logger.info(
                                f"[Bollinger] Posição fechada: {symbol} ({timeframe}), Sinal ID: {signal_id}, "
                                f"Ordem ID: {close_order['orderId']}, Motivo: {close_reason}, PnL: {pnl:.2f} USDT"
                            )
                            self.update_order_status(order_id, 'CLOSED', close_reason, close_order['orderId'], pnl, close_price)
                            closed = True
                            with self.lock:
                                if position_key in self.open_positions:
                                    del self.open_positions[position_key]
                                    logger.info(f"[Bollinger] Posição removida do rastreamento: {symbol} ({timeframe}), Sinal ID: {signal_id}")
                                if order_id in self.active_monitors:
                                    del self.active_monitors[order_id]
                        else:
                            logger.error(f"[Bollinger] Falha ao fechar posição para {symbol} ({timeframe}), Sinal ID: {signal_id}")
                    time.sleep(0.1)
                except Exception as e:
                    logger.error(f"[Bollinger] Erro durante monitoramento de preço para {symbol} ({timeframe}), Sinal ID: {signal_id}: {e}")
                    time.sleep(1)
        except Exception as e:
            logger.error(f"[Bollinger] Erro geral no monitoramento para {symbol} ({timeframe}), Sinal ID: {signal_id}: {e}")

    def update_order_status(self, order_id, status, close_reason, close_order_id, pnl, close_price):
        try:
            logger.info(
                f"[Bollinger] Atualizando status da ordem: {order_id}, Status: {status}, "
                f"Motivo: {close_reason}, Ordem de Fechamento ID: {close_order_id}, PnL: {pnl:.2f} USDT"
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
                logger.info(f"[Bollinger] Status da ordem atualizado em {ORDERS_CSV}: {order_id}")
            if status == 'CLOSED':
                with FileLock(f"{ORDERS_PNL_CSV}.lock"):
                    df_pnl = pd.read_csv(ORDERS_PNL_CSV) if os.path.exists(ORDERS_PNL_CSV) else pd.DataFrame(columns=df.columns)
                    closed_row = df[df['order_id'] == order_id]
                    if not closed_row.empty:
                        df_pnl = df_pnl[df_pnl['order_id'] != order_id]
                        df_pnl = pd.concat([df_pnl, closed_row], ignore_index=True)
                        df_pnl.to_csv(ORDERS_PNL_CSV, index=False)
                        logger.info(f"[Bollinger] Ordem {order_id} sincronizada em {ORDERS_PNL_CSV}")
            row = None
            try:
                df = pd.read_csv(ORDERS_CSV)
                row = df[df['order_id'] == order_id].iloc[0]
            except Exception:
                pass
            if row is not None:
                key = (row['pair'], row['timeframe'])
                with self.lock:
                    # Atualizar SL Streak Pausing
                    threshold = SL_STREAK_CONFIG.get(row['pair'], {"threshold": 3})["threshold"]
                    base_pause_minutes = SL_STREAK_CONFIG.get(row['pair'], {"pause_minutes": 7})["pause_minutes"]
                    
                    # Usa a volatilidade da ordem ou calcula uma nova
                    volatility = row.get('volatility')
                    if volatility is None or pd.isna(volatility):
                        volatility = self.binance.get_volatility(row['pair'], row['timeframe'])
                    
                    pause_minutes = min(60, base_pause_minutes * (volatility / 0.01))
                    
                    # Atualizar Inverted Mode
                    invert_state = load_inverted_state(row['pair'], row['timeframe'], mode="real")
                    invert_threshold = INVERTED_MODE_CONFIG.get(row['pair'], {"sl_threshold": 3})["sl_threshold"]
                    invert_count = INVERTED_MODE_CONFIG.get(row['pair'], {"invert_count": 2})["invert_count"]
                    if close_reason == "SL":
                        self.sl_streak[key] = self.sl_streak.get(key, 0) + 1
                        log_sl_streak_progress(
                            pair=row['pair'],
                            timeframe=row['timeframe'],
                            current_streak=self.sl_streak[key],
                            threshold=threshold,
                            pause_minutes=pause_minutes,
                            mode="real",
                            engine="Bollinger"
                        )
                        # Atualizar Inverted Mode state
                        if INVERTED_MODE_ENABLED:
                            invert_state['sl_streak'] = invert_state['sl_streak'] + 1
                            if invert_state['sl_streak'] >= invert_threshold and not invert_state['is_inverted']:
                                invert_state['is_inverted'] = True
                                invert_state['invert_count_remaining'] = invert_count
                                log_inverted_mode(
                                    pair=row['pair'],
                                    timeframe=row['timeframe'],
                                    action="activated",
                                    original_direction=None,
                                    inverted_direction=None,
                                    invert_count_remaining=invert_count,
                                    mode="real",
                                    engine="Bollinger"
                                )
                                if INVERTED_MODE_NOTIFICATIONS_ENABLED:
                                    message = (
                                        f"🔄 <b>Inverted Mode Ativado</b>\n"
                                        f"Par: <b>{row['pair']}</b>\n"
                                        f"Timeframe: <b>{row['timeframe']}</b>\n"
                                        f"Engine: <b>Bollinger</b>\n"
                                        f"Motivo: <b>{invert_threshold} SLs consecutivos</b>\n"
                                        f"Próximas Ordens Invertidas: <b>{invert_count}</b>"
                                    )
                                    send_inverted_mode_telegram(message, mode="real")
                            manage_inverted_state(
                                pair=row['pair'],
                                timeframe=row['timeframe'],
                                sl_streak=invert_state['sl_streak'],
                                invert_count_remaining=invert_state['invert_count_remaining'],
                                is_inverted=invert_state['is_inverted'],
                                mode="real"
                            )
                        if self.sl_streak[key] >= threshold:
                            self.pause_until[key] = datetime.now() + timedelta(minutes=pause_minutes)
                            logger.warning(f"[Bollinger][PAUSE] ⏸️ Pausa de {pause_minutes:.1f} minutos ativada para {key} após {threshold} SLs consecutivos.")
                            pause_data = {
                                'id': generate_id("PAUSE"),
                                'pair': row['pair'],
                                'timeframe': row['timeframe'],
                                'trigger_time': pd.Timestamp.now(),
                                'pause_end': self.pause_until[key],
                                'sl_streak': self.sl_streak[key],
                                'signal_engine': 'Bollinger',
                                'volatility': volatility  # Adiciona volatilidade aos dados de pausa
                            }
                            save_to_csv(pause_data, PAUSE_LOG_CSV)
                            if SL_STREAK_NOTIFICATIONS_ENABLED:
                                message = (
                                    f"⏸️ <b>Pausa Ativada</b>\n"
                                    f"Par: <b>{row['pair']}</b>\n"
                                    f"Timeframe: <b>{row['timeframe']}</b>\n"
                                    f"Engine: <b>Bollinger</b>\n"
                                    f"Motivo: <b>{threshold} SLs consecutivos</b>\n"
                                    f"Duração: <b>{pause_minutes:.1f} minutos</b>\n"
                                    f"Volatilidade: <b>{volatility:.4f}</b>\n"
                                    f"Horário de Início: <b>{pause_data['trigger_time'].strftime('%Y-%m-%d %H:%M:%S')}</b>"
                                )
                                send_sl_streak_telegram(message, mode="real")
                        else:
                            if SL_STREAK_NOTIFICATIONS_ENABLED:
                                remaining = threshold - self.sl_streak[key]
                                message = (
                                    f"⚠️ <b>SL Streak Aviso</b>\n"
                                    f"Par: <b>{row['pair']}</b>\n"
                                    f"Timeframe: <b>{row['timeframe']}</b>\n"
                                    f"Engine: <b>Bollinger</b>\n"
                                    f"SL Streak: <b>{self.sl_streak[key]}/{threshold}</b>\n"
                                    f"Volatilidade: <b>{volatility:.4f}</b>\n"
                                    f"Aviso: <b>Mais {remaining} SL(s) ativará uma pausa de {pause_minutes:.1f} minutos</b>"
                                )
                                send_sl_streak_telegram(message, mode="real")
                    else:
                        self.sl_streak[key] = 0
                        if INVERTED_MODE_ENABLED:
                            manage_inverted_state(
                                pair=row['pair'],
                                timeframe=row['timeframe'],
                                sl_streak=0,
                                invert_count_remaining=0,
                                is_inverted=False,
                                mode="real",
                                reset=True
                            )
                            if invert_state['is_inverted']:
                                log_inverted_mode(
                                    pair=row['pair'],
                                    timeframe=row['timeframe'],
                                    action="deactivated",
                                    original_direction=None,
                                    inverted_direction=None,
                                    invert_count_remaining=0,
                                    mode="real",
                                    engine="Bollinger"
                                )
                                if INVERTED_MODE_NOTIFICATIONS_ENABLED:
                                    message = (
                                        f"🔄 <b>Inverted Mode Desativado</b>\n"
                                        f"Par: <b>{row['pair']}</b>\n"
                                        f"Timeframe: <b>{row['timeframe']}</b>\n"
                                        f"Engine: <b>Bollinger</b>\n"
                                        f"Motivo: <b>Ordem fechada por TP</b>\n"
                                        f"Ordens voltarão ao modo original."
                                    )
                                    send_inverted_mode_telegram(message, mode="real")
                    save_streak_pause(self.sl_streak, self.pause_until)
        except Exception as e:
            logger.error(f"[Bollinger] Erro ao atualizar status da ordem {order_id}: {e}")

    def stop(self):
        self.running = False