import pandas as pd
import numpy as np
import time
import os
from binance.enums import SIDE_BUY, SIDE_SELL
from datetime import datetime, timedelta
from utils import logger, save_to_csv, generate_id, save_streak_pause, load_streak_pause, log_sl_streak_progress, send_sl_streak_telegram, manage_inverted_state, load_inverted_state, log_inverted_mode, send_inverted_mode_telegram
from config import *
from filelock import FileLock
import logging
logging.getLogger("filelock").setLevel(logging.WARNING)
import threading

class SimulatedOrderEngine:
    """Motor de ordens simuladas para ambiente de teste."""

    def __init__(self, binance_utils, shared_state=None):
        """Inicializa o motor de ordens simuladas."""
        self.binance = binance_utils
        self.shared_state = shared_state or {}
        self.prices = self.shared_state.get('prices', {})
        # Para simulações sem WebSocket
        if not self.prices:
            self.prices = {}
        self.running = True
        self.active_orders = {}
        self.lock = threading.Lock() if 'lock' not in self.shared_state else self.shared_state['lock']
        self.sl_streak = {} if 'sl_streak' not in self.shared_state else self.shared_state['sl_streak']
        self.pause_until = {} if 'pause_until' not in self.shared_state else self.shared_state['pause_until']
        self.monitor_thread = threading.Thread(target=self.monitor_orders_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("[SIM] Motor de ordens simuladas inicializado")

    def process_signals(self):
        """Processa sinais e gera ordens simuladas."""
        logger.info("[SIM] Processando sinais")
        if not os.path.exists(SIGNALS_CSV):
            logger.warning(f"[SIM] Arquivo de sinais {SIGNALS_CSV} não encontrado")
            return

        try:
            df_signals = pd.read_csv(SIGNALS_CSV)
            if df_signals.empty:
                logger.info("[SIM] Nenhum sinal encontrado")
                return
            
            # Tratamento seguro para a nova coluna de volatilidade
            if 'volatility' not in df_signals.columns:
                logger.warning("[SIM] Coluna de volatilidade não encontrada nos sinais. Usando valor padrão.")
                df_signals['volatility'] = 0.05  # Valor default moderado
            
            # Converte timestamp para datetime
            if 'timestamp' in df_signals.columns:
                df_signals['timestamp'] = pd.to_datetime(df_signals['timestamp'])
            
            # Carrega ordens existentes para evitar duplicação
            existing_orders = set()
            if os.path.exists(SIM_ORDERS_CSV):
                df_orders = pd.read_csv(SIM_ORDERS_CSV)
                existing_orders = set(df_orders['signal_id'])
            
            # Processa apenas sinais das últimas 24h e não processados anteriormente
            recent_signals = df_signals[
                (df_signals['timestamp'] >= (pd.Timestamp.now() - pd.Timedelta(days=1))) & 
                (~df_signals['signal_id'].isin(existing_orders))
            ]
            
            if recent_signals.empty:
                logger.info("[SIM] Nenhum sinal novo encontrado")
                return
            
            logger.info(f"[SIM] Encontrados {len(recent_signals)} sinais novos para processar")
            
            for _, signal in recent_signals.iterrows():
                try:
                    if self._should_skip_signal(signal):
                        continue
                    
                    # Obtém preço atual via Binance com fallback para cache
                    current_price = self._get_current_price(signal['pair'])
                    if current_price is None:
                        logger.warning(f"[SIM] Não foi possível obter preço para {signal['pair']}. Ignorando sinal.")
                        continue
                    
                    # Volatilidade atual do par e timeframe (usa valor do sinal ou calcula se não disponível)
                    volatility = signal.get('volatility')
                    if volatility is None or pd.isna(volatility):
                        volatility = self.binance.get_volatility(signal['pair'], signal['timeframe'])
                        logger.info(f"[SIM] Volatilidade calculada para {signal['pair']} ({signal['timeframe']}): {volatility:.4f}")
                    else:
                        logger.info(f"[SIM] Usando volatilidade do sinal para {signal['pair']} ({signal['timeframe']}): {volatility:.4f}")
                    
                    # Aplica estado de inversão se configurado
                    direction = signal['direction']
                    order_mode = "original"
                    if SIM_INVERTED_MODE_ENABLED:
                        invert_state = load_inverted_state(signal['pair'], signal['timeframe'], mode="sim")
                        if invert_state['is_inverted']:
                            direction = SIDE_SELL if signal['direction'] == SIDE_BUY else SIDE_BUY
                            order_mode = "inverted"
                            log_inverted_mode(
                                pair=signal['pair'],
                                timeframe=signal['timeframe'],
                                action="applied",
                                original_direction=signal['direction'],
                                inverted_direction=direction,
                                invert_count_remaining=invert_state['invert_count_remaining'],
                                mode="sim",
                                engine=signal.get('signal_engine', 'unknown')
                            )
                    
                    # Calcula quantidade com base na alavancagem e valor padrão
                    leverage = SIM_LEVERAGE
                    usdt_value = SIM_ORDER_VALUE_USDT
                    entry_price = current_price
                    quantity = (usdt_value * leverage) / entry_price
                    
                    # Ajusta SL/TP com base na volatilidade
                    sl_percent = self._adjust_sl_percent(signal['pair'], volatility)
                    tp_percent = self._adjust_tp_percent(signal['pair'], volatility)
                    
                    if direction == SIDE_BUY:
                        sl_price = entry_price * (1 - sl_percent)
                        tp_price = entry_price * (1 + tp_percent)
                    else:
                        sl_price = entry_price * (1 + sl_percent)
                        tp_price = entry_price * (1 - tp_percent)
                    
                    # Cria ordem simulada
                    order_id = generate_id("SIM_ORD")
                    entry_fee = (entry_price * quantity) * 0.0004  # Taxa Taker padrão 0.04%
                    
                    order = {
                        'order_id': order_id,
                        'signal_id': signal['signal_id'],
                        'pair': signal['pair'],
                        'timeframe': signal['timeframe'],
                        'direction': direction,
                        'entry_price': entry_price,
                        'quantity': quantity,
                        'tp_price': tp_price,
                        'sl_price': sl_price,
                        'sl_percent': sl_percent,
                        'tp_percent': tp_percent,
                        'status': 'OPEN',
                        'signal_engine': signal.get('signal_engine', 'unknown'),
                        'order_mode': order_mode,
                        'entry_fee': entry_fee,
                        'exit_fee': 0.0,
                        'timestamp': pd.Timestamp.now(),
                        'volatility': volatility,  # Armazena volatilidade na ordem
                        'close_reason': None,
                        'close_order_id': None,
                        'pnl': None,
                        'close_timestamp': None,
                        'close_price': None
                    }
                    
                    # Registra ordem no CSV e na memória
                    save_to_csv(order, SIM_ORDERS_CSV)
                    
                    with self.lock:
                        self.active_orders[order_id] = order
                    
                    logger.info(f"[SIM] Ordem criada: {order_id} para {signal['pair']} ({signal['timeframe']}), Direção: {direction}, Entrada: {entry_price:.6f}, SL: {sl_price:.6f}, TP: {tp_price:.6f}, Volatilidade: {volatility:.4f}")
                    
                    # Atualiza estado de inversão se aplicável
                    if SIM_INVERTED_MODE_ENABLED and invert_state['is_inverted']:
                        new_invert_count = max(0, invert_state['invert_count_remaining'] - 1)
                        manage_inverted_state(
                            pair=signal['pair'],
                            timeframe=signal['timeframe'],
                            sl_streak=invert_state['sl_streak'],
                            invert_count_remaining=new_invert_count,
                            is_inverted=new_invert_count > 0,
                            mode="sim"
                        )
                    
                except Exception as e:
                    logger.error(f"[SIM] Erro ao processar sinal {signal['signal_id']}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"[SIM] Erro ao processar sinais: {e}")

    def _should_skip_signal(self, signal):
        """Verifica se um sinal deve ser ignorado com base em regras de negócio."""
        pair = signal['pair']
        timeframe = signal['timeframe']
        key = (pair, timeframe)
        
        # Verifica se o par está em pausa devido a SL streak
        with self.lock:
            if key in self.pause_until:
                now = datetime.now()
                if now < self.pause_until[key]:
                    minutes_left = int((self.pause_until[key] - now).total_seconds() / 60) + 1
                    logger.info(f"[SIM] Par {pair} ({timeframe}) em pausa por mais {minutes_left} minutos. Ignorando sinal.")
                    return True
                else:
                    # Pausa encerrada
                    logger.info(f"[SIM] Pausa encerrada para {pair} ({timeframe}). Retomando operações.")
                    del self.pause_until[key]
                    self.sl_streak[key] = 0
                    save_streak_pause(self.sl_streak, self.pause_until, mode="sim")
        
        # Verifica se já existe uma posição aberta para este par/timeframe
        active_positions = self._get_active_positions_for_pair(pair, timeframe)
        if active_positions:
            logger.info(f"[SIM] Já existe posição aberta para {pair} ({timeframe}). Ignorando sinal.")
            return True
        
        return False

    def _get_active_positions_for_pair(self, pair, timeframe=None):
        """Retorna ordens ativas para um par/timeframe específico."""
        active = []
        with self.lock:
            for order_id, order in self.active_orders.items():
                if order['pair'] == pair and (timeframe is None or order['timeframe'] == timeframe):
                    if order['status'] == 'OPEN':
                        active.append(order)
        return active

    def _get_current_price(self, symbol):
        """Obtém o preço atual, primeiro do cache de WebSocket, depois via API."""
        price = self.prices.get(symbol)
        if price is not None:
            return price
        return self.binance.get_current_price(symbol)

    def _adjust_sl_percent(self, pair, volatility):
        """Ajusta o percentual de stop loss com base na volatilidade."""
        # Stop loss padrão da configuração
        base_sl_percent = SIM_STOP_LOSS_PERCENT
        
        # Fator de ajuste baseado na volatilidade
        # Para pares mais voláteis, aumentamos o SL para evitar stop loss prematuros
        volatility_factor = min(3.0, max(0.5, volatility / 0.05))  # 0.05 é considerada volatilidade "normal"
        
        # Aplicamos um ajuste mais suave à configuração base
        # O multiplicador varia entre 0.8x e 1.5x dependendo da volatilidade
        multiplier = 0.8 + (volatility_factor - 0.5) * 0.7 / 2.5
        
        adjusted_sl = base_sl_percent * multiplier
        
        # Limites de segurança para o SL
        min_sl = 0.005  # 0.5% no mínimo
        max_sl = 0.05   # 5% no máximo
        
        final_sl = min(max_sl, max(min_sl, adjusted_sl))
        
        logger.debug(f"[SIM] SL ajustado para {pair}: Base={base_sl_percent:.4f}, Volatilidade={volatility:.4f}, Fator={volatility_factor:.2f}, Ajustado={final_sl:.4f}")
        return final_sl

    def _adjust_tp_percent(self, pair, volatility):
        """Ajusta o percentual de take profit com base na volatilidade."""
        # Take profit padrão da configuração
        base_tp_percent = SIM_TAKE_PROFIT_PERCENT
        
        # Fator de ajuste baseado na volatilidade
        # Para pares mais voláteis, aumentamos o TP para capturar mais dos movimentos
        volatility_factor = min(3.0, max(0.5, volatility / 0.05))
        
        # Aumentamos o TP para volatilidades maiores, mas com um teto
        multiplier = 0.9 + (volatility_factor - 0.5) * 0.6 / 2.5
        
        adjusted_tp = base_tp_percent * multiplier
        
        # Limites de segurança para o TP
        min_tp = 0.01   # 1% no mínimo
        max_tp = 0.10   # 10% no máximo
        
        final_tp = min(max_tp, max(min_tp, adjusted_tp))
        
        logger.debug(f"[SIM] TP ajustado para {pair}: Base={base_tp_percent:.4f}, Volatilidade={volatility:.4f}, Fator={volatility_factor:.2f}, Ajustado={final_tp:.4f}")
        return final_tp

    def monitor_orders_loop(self):
        """Loop para monitorar ordens abertas e verificar condições de fechamento."""
        while self.running:
            try:
                now = datetime.now()
                with self.lock:
                    orders_to_monitor = list(self.active_orders.values())
                
                for order in orders_to_monitor:
                    if order['status'] != 'OPEN':
                        continue
                    
                    try:
                        pair = order['pair']
                        current_price = self._get_current_price(pair)
                        if current_price is None:
                            continue
                        
                        # Verifica condições de TP/SL
                        close_reason = None
                        if order['direction'] == SIDE_BUY:
                            if current_price <= order['sl_price']:
                                close_reason = "SL"
                            elif current_price >= order['tp_price']:
                                close_reason = "TP"
                        else:  # SIDE_SELL
                            if current_price >= order['sl_price']:
                                close_reason = "SL"
                            elif current_price <= order['tp_price']:
                                close_reason = "TP"
                        
                        if close_reason:
                            self._close_order(order, current_price, close_reason)
                    
                    except Exception as e:
                        logger.error(f"[SIM] Erro ao monitorar ordem {order['order_id']}: {e}")
                
                time.sleep(0.5)  # Reduz carga de CPU
            
            except Exception as e:
                logger.error(f"[SIM] Erro no loop de monitoramento: {e}")
                time.sleep(1)

    def _close_order(self, order, close_price, close_reason):
        """Fecha uma ordem simulada e calcula o PnL."""
        try:
            close_order_id = generate_id("SIM_CLOSE")
            close_fee = (close_price * order['quantity']) * 0.0004  # Taxa Taker padrão 0.04%
            
            if order['direction'] == SIDE_BUY:
                gross_pnl = (close_price - order['entry_price']) * order['quantity']
            else:
                gross_pnl = (order['entry_price'] - close_price) * order['quantity']
                
            net_pnl = gross_pnl - order['entry_fee'] - close_fee
            
            closed_order = order.copy()
            closed_order.update({
                'status': 'CLOSED',
                'close_reason': close_reason,
                'close_order_id': close_order_id,
                'pnl': net_pnl,
                'close_timestamp': pd.Timestamp.now(),
                'close_price': close_price,
                'exit_fee': close_fee
            })
            
            # Atualiza no dicionário interno
            with self.lock:
                self.active_orders[order['order_id']] = closed_order
            
            # Atualiza o arquivo CSV
            self._update_order_in_csv(closed_order)
            
            logger.info(f"[SIM] Ordem {order['order_id']} fechada: Par={order['pair']}, Motivo={close_reason}, PnL={net_pnl:.4f} USDT")
            
            # Atualização de SL Streak se aplicável
            if close_reason == "SL":
                key = (order['pair'], order['timeframe'])
                with self.lock:
                    threshold = SIM_SL_STREAK_CONFIG.get(order['pair'], {"threshold": 1})["threshold"]
                    base_pause_minutes = SIM_SL_STREAK_CONFIG.get(order['pair'], {"pause_minutes": 7})["pause_minutes"]
                    volatility = order.get('volatility', 0.05)  # Use a volatilidade armazenada ou valor padrão
                    pause_minutes = min(20, base_pause_minutes * (volatility / 0.01))
                    
                    # Gerencia estado de inversão para modo simulado
                    invert_state = load_inverted_state(order['pair'], order['timeframe'], mode="sim")
                    invert_threshold = SIM_INVERTED_MODE_CONFIG.get(order['pair'], {"sl_threshold": 1})["sl_threshold"]
                    invert_count = SIM_INVERTED_MODE_CONFIG.get(order['pair'], {"invert_count": 2})["invert_count"]
                    
                    self.sl_streak[key] = self.sl_streak.get(key, 0) + 1
                    
                    log_sl_streak_progress(
                        pair=order['pair'],
                        timeframe=order['timeframe'],
                        current_streak=self.sl_streak[key],
                        threshold=threshold,
                        pause_minutes=pause_minutes,
                        mode="sim",
                        engine=order.get('signal_engine', 'unknown')
                    )
                    
                    # Gerencia modo invertido
                    if SIM_INVERTED_MODE_ENABLED:
                        invert_state['sl_streak'] = invert_state['sl_streak'] + 1
                        if invert_state['sl_streak'] >= invert_threshold and not invert_state['is_inverted']:
                            invert_state['is_inverted'] = True
                            invert_state['invert_count_remaining'] = invert_count
                            log_inverted_mode(
                                pair=order['pair'],
                                timeframe=order['timeframe'],
                                action="activated",
                                original_direction=None,
                                inverted_direction=None,
                                invert_count_remaining=invert_count,
                                mode="sim",
                                engine=order.get('signal_engine', 'unknown')
                            )
                        
                        manage_inverted_state(
                            pair=order['pair'],
                            timeframe=order['timeframe'],
                            sl_streak=invert_state['sl_streak'],
                            invert_count_remaining=invert_state['invert_count_remaining'],
                            is_inverted=invert_state['is_inverted'],
                            mode="sim"
                        )
                    
                    # Ativa pausa se atingir o limiar
                    if self.sl_streak[key] >= threshold:
                        self.pause_until[key] = datetime.now() + timedelta(minutes=pause_minutes)
                        save_streak_pause(self.sl_streak, self.pause_until, mode="sim")
                        
                        logger.warning(f"[SIM] Pausa de {pause_minutes:.1f} minutos ativada para {key} após {threshold} SLs consecutivos")
                        
                        pause_data = {
                            'id': generate_id("SIM_PAUSE"),
                            'pair': order['pair'],
                            'timeframe': order['timeframe'],
                            'trigger_time': pd.Timestamp.now(),
                            'pause_end': self.pause_until[key],
                            'sl_streak': self.sl_streak[key],
                            'signal_engine': order.get('signal_engine', 'unknown')
                        }
                        save_to_csv(pause_data, SIM_PAUSE_LOG_CSV)
            else:
                # Se não for SL, resetamos o contador
                key = (order['pair'], order['timeframe'])
                with self.lock:
                    self.sl_streak[key] = 0
                    save_streak_pause(self.sl_streak, self.pause_until, mode="sim")
                    
                    # Reseta o modo invertido
                    if SIM_INVERTED_MODE_ENABLED:
                        manage_inverted_state(
                            pair=order['pair'],
                            timeframe=order['timeframe'],
                            sl_streak=0,
                            invert_count_remaining=0,
                            is_inverted=False,
                            mode="sim",
                            reset=True
                        )
            
            return True
        
        except Exception as e:
            logger.error(f"[SIM] Erro ao fechar ordem {order['order_id']}: {e}")
            return False

    def _update_order_in_csv(self, order):
        """Atualiza uma ordem no arquivo CSV."""
        try:
            # Carrega o arquivo
            if os.path.exists(SIM_ORDERS_CSV):
                with FileLock(f"{SIM_ORDERS_CSV}.lock"):
                    df = pd.read_csv(SIM_ORDERS_CSV)
                    
                    # Atualiza a linha correspondente
                    mask = df['order_id'] == order['order_id']
                    if any(mask):
                        for key, value in order.items():
                            df.loc[mask, key] = value
                    else:
                        df = pd.concat([df, pd.DataFrame([order])], ignore_index=True)
                    
                    # Salva o arquivo
                    df.to_csv(SIM_ORDERS_CSV, index=False)
            else:
                save_to_csv(order, SIM_ORDERS_CSV)
                
            # Atualiza também o arquivo de PnL para ordens fechadas
            if order['status'] == 'CLOSED':
                with FileLock(f"{SIM_ORDERS_PNL_CSV}.lock"):
                    if os.path.exists(SIM_ORDERS_PNL_CSV):
                        df_pnl = pd.read_csv(SIM_ORDERS_PNL_CSV)
                        df_pnl = df_pnl[df_pnl['order_id'] != order['order_id']]
                        df_pnl = pd.concat([df_pnl, pd.DataFrame([order])], ignore_index=True)
                    else:
                        df_pnl = pd.DataFrame([order])
                    
                    df_pnl.to_csv(SIM_ORDERS_PNL_CSV, index=False)
            
            return True
        
        except Exception as e:
            logger.error(f"[SIM] Erro ao atualizar ordem {order['order_id']} no CSV: {e}")
            return False

    def clear_old_orders(self, days=7):
        """Remove ordens simuladas antigas do dicionário interno."""
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        with self.lock:
            for order_id in list(self.active_orders.keys()):
                order = self.active_orders[order_id]
                if pd.to_datetime(order['timestamp']) < cutoff:
                    if order['status'] == 'OPEN':
                        logger.warning(f"[SIM] Fechando ordem antiga {order_id} (> {days} dias) sem execução")
                        self._close_order(order, order['entry_price'], "EXPIRED")
                    else:
                        logger.debug(f"[SIM] Removendo ordem fechada {order_id} (> {days} dias) da memória")
                        self.active_orders.pop(order_id, None)
        logger.info(f"[SIM] Limpeza de ordens antigas concluída. {len(self.active_orders)} ordens em memória.")

    def load_existing_orders(self):
        """Carrega ordens existentes do arquivo CSV para a memória."""
        try:
            if os.path.exists(SIM_ORDERS_CSV):
                df = pd.read_csv(SIM_ORDERS_CSV)
                count = 0
                with self.lock:
                    for _, row in df.iterrows():
                        order_dict = row.to_dict()
                        self.active_orders[order_dict['order_id']] = order_dict
                        count += 1
                logger.info(f"[SIM] {count} ordens carregadas do arquivo CSV")
            else:
                logger.warning(f"[SIM] Arquivo de ordens {SIM_ORDERS_CSV} não encontrado")
        except Exception as e:
            logger.error(f"[SIM] Erro ao carregar ordens existentes: {e}")

    def calculate_pnl_statistics(self):
        """Calcula estatísticas de PnL para ordens fechadas."""
        try:
            if os.path.exists(SIM_ORDERS_PNL_CSV):
                df = pd.read_csv(SIM_ORDERS_PNL_CSV)
                df_closed = df[df['status'] == 'CLOSED']
                
                if df_closed.empty:
                    logger.info("[SIM] Não há ordens fechadas para calcular estatísticas")
                    return {}
                
                # Estatísticas gerais
                total_trades = len(df_closed)
                total_pnl = df_closed['pnl'].sum()
                win_trades = len(df_closed[df_closed['pnl'] > 0])
                loss_trades = len(df_closed[df_closed['pnl'] <= 0])
                win_rate = win_trades / total_trades if total_trades > 0 else 0
                avg_win = df_closed[df_closed['pnl'] > 0]['pnl'].mean() if win_trades > 0 else 0
                avg_loss = df_closed[df_closed['pnl'] <= 0]['pnl'].mean() if loss_trades > 0 else 0
                profit_factor = abs(df_closed[df_closed['pnl'] > 0]['pnl'].sum() / df_closed[df_closed['pnl'] < 0]['pnl'].sum()) if df_closed[df_closed['pnl'] < 0]['pnl'].sum() != 0 else float('inf')
                
                # Estatísticas por volatilidade
                if 'volatility' in df_closed.columns:
                    # Criar faixas de volatilidade
                    df_closed['vol_range'] = pd.cut(
                        df_closed['volatility'], 
                        bins=[0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 2.0], 
                        labels=['0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '30-50%', '50-100%', '100%+']
                    )
                    
                    vol_stats = df_closed.groupby('vol_range').agg(
                        count=('pnl', 'count'),
                        win_rate=('pnl', lambda x: (x > 0).mean()),
                        avg_pnl=('pnl', 'mean'),
                        total_pnl=('pnl', 'sum')
                    ).reset_index()
                    
                    # Encontra a faixa de volatilidade mais lucrativa
                    best_vol_range = vol_stats.loc[vol_stats['avg_pnl'].idxmax(), 'vol_range'] if not vol_stats.empty else None
                    best_vol_win_rate = vol_stats.loc[vol_stats['win_rate'].idxmax(), 'vol_range'] if not vol_stats.empty else None
                else:
                    vol_stats = pd.DataFrame()
                    best_vol_range = None
                    best_vol_win_rate = None
                
                stats = {
                    'total_trades': total_trades,
                    'total_pnl': total_pnl,
                    'win_trades': win_trades,
                    'loss_trades': loss_trades,
                    'win_rate': win_rate,
                    'avg_win': avg_win,
                    'avg_loss': avg_loss,
                    'profit_factor': profit_factor,
                    'best_volatility_range': best_vol_range,
                    'best_volatility_win_rate': best_vol_win_rate,
                    'volatility_stats': vol_stats.to_dict('records') if not vol_stats.empty else []
                }
                
                logger.info(f"""[SIM] Estatísticas de PnL:
                    Total de trades: {total_trades}
                    PnL total: {total_pnl:.4f} USDT
                    Win rate: {win_rate*100:.2f}%
                    Profit factor: {profit_factor:.2f}
                    Melhor faixa de volatilidade (PnL): {best_vol_range}
                    Melhor faixa de volatilidade (Win Rate): {best_vol_win_rate}
                """)
                
                return stats
            else:
                logger.warning(f"[SIM] Arquivo de PnL {SIM_ORDERS_PNL_CSV} não encontrado")
                return {}
        except Exception as e:
            logger.error(f"[SIM] Erro ao calcular estatísticas de PnL: {e}")
            return {}

    def shutdown(self):
        """Encerra o motor de ordens simuladas."""
        logger.info("[SIM] Desligando motor de ordens simuladas")
        self.running = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2)