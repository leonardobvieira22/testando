"""
Sistema de Trading - Motor de Sinais v2.0
Classe compatível com o sistema de trading existente
Implementa todas as funcionalidades da versão 2.0
"""
import os
import time
import json
import csv
import pandas as pd
import numpy as np
import threading
import traceback
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from filelock import FileLock

import config
from binance_utils import BinanceUtils
from utils import logger  # Usando o mesmo logger do sistema
from rejected_signals_tracker import RejectedSignalsTracker  # Importação para tracking de rejeições

class SignalEngine:
    """
    Motor de sinais v2.0 para o sistema de trading.
    Implementação compatível com o main.py e com todas as funcionalidades
    """
    
    def __init__(self, api_key, api_secret, pairs):
        """
        Inicializa o motor de sinais.
        
        Args:
            api_key: Chave API da Binance
            api_secret: Secret da API da Binance
            pairs: Lista de pares de trading
        """
        logger.info("Inicializando SignalEngine v2.0...")
        
        self.api_key = api_key
        self.api_secret = api_secret
        self.pairs = pairs
        
        # Usar timeframes ativos da configuração
        self.timeframes = [tf for tf, enabled in config.TIMEFRAMES.items() if enabled]
        logger.info(f"Usando timeframes da configuração: {self.timeframes}")
        
        # Inicializar a API da Binance
        self.binance = BinanceUtils(self.api_key, self.api_secret)
        
        # Inicializar o rastreador de sinais rejeitados
        self.rejected_tracker = RejectedSignalsTracker()
        
        # Flag para controle do loop principal
        self.running = False
        self.signal_thread = None
        
        # Contador de ciclos para logging periódico
        self.cycle_counter = 0
        
        # Diretórios para armazenamento de dados
        self.DATA_DIR = "data"
        os.makedirs(self.DATA_DIR, exist_ok=True)
        
        # Arquivos para controle de estado
        self.POSITIONS_FILE = os.path.join(self.DATA_DIR, config.ORDERS_CSV)
        self.ATTENTIVE_MODE_FILE = os.path.join(self.DATA_DIR, "attentive_mode.csv")
        self.SL_STREAK_FILE = os.path.join(self.DATA_DIR, config.SL_STREAK_LOG_CSV)
        self.PAUSED_PAIRS_FILE = os.path.join(self.DATA_DIR, config.PAUSE_LOG_CSV)
        self.INVERTED_MODE_FILE = os.path.join(self.DATA_DIR, config.INVERTED_MODE_LOG_CSV)
        
        # Inicializar arquivos de dados
        self.initialize_data_files()
        
        logger.info("SignalEngine v2.0 inicializado com sucesso")
    
    # MÉTODO ADICIONADO PARA COMPATIBILIDADE COM MAIN.PY
    def generate_signals(self):
        """Método compatibilidade com main.py - Gera sinais para todos os pares configurados."""
        logger.info("Gerando sinais para pares configurados...")
        return self.process_pairs()
    
    def initialize_data_files(self):
        """Inicializa os arquivos CSV para controle de estado se eles não existirem."""
        files_and_headers = {
            self.ATTENTIVE_MODE_FILE: ["symbol", "timeframe", "mode", "direction", "timestamp", "extreme_value"],
            self.SL_STREAK_FILE: ["symbol", "streak_count", "last_update"],
            self.PAUSED_PAIRS_FILE: ["symbol", "until_timestamp", "reason"],
            self.INVERTED_MODE_FILE: ["symbol", "remaining_signals", "last_update"],
        }
        
        for file_path, headers in files_and_headers.items():
            if not os.path.exists(file_path):
                try:
                    with FileLock(f"{file_path}.lock"):
                        with open(file_path, 'w', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow(headers)
                    logger.info(f"Arquivo {file_path} criado com sucesso")
                except Exception as e:
                    logger.error(f"Erro ao criar {file_path}: {e}")

    def get_open_positions(self):
        """Recupera posições abertas do arquivo CSV."""
        positions = []
        if not os.path.exists(self.POSITIONS_FILE):
            return positions
        
        try:
            with FileLock(f"{self.POSITIONS_FILE}.lock"):
                with open(self.POSITIONS_FILE, 'r', newline='') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        positions.append(row)
        except Exception as e:
            logger.error(f"Erro ao ler arquivo de posições: {e}")
        
        return positions

    def save_position(self, symbol, entry_price, position_size, direction, take_profit, stop_loss):
        """Salva uma nova posição no arquivo CSV."""
        try:
            with FileLock(f"{self.POSITIONS_FILE}.lock"):
                with open(self.POSITIONS_FILE, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        symbol, entry_price, position_size, direction, 
                        int(time.time()), take_profit, stop_loss
                    ])
                logger.info(f"Posição salva para {symbol} {direction} a {entry_price}")
        except Exception as e:
            logger.error(f"Erro ao salvar posição: {e}")

    def remove_position(self, symbol):
        """Remove uma posição do arquivo CSV."""
        positions = self.get_open_positions()
        try:
            with FileLock(f"{self.POSITIONS_FILE}.lock"):
                with open(self.POSITIONS_FILE, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["symbol", "entry_price", "position_size", "direction", "timestamp", "take_profit", "stop_loss"])
                    for position in positions:
                        if position["symbol"] != symbol:
                            writer.writerow([
                                position["symbol"], position["entry_price"], 
                                position["position_size"], position["direction"], 
                                position["timestamp"], position["take_profit"], position["stop_loss"]
                            ])
                logger.info(f"Posição removida para {symbol}")
        except Exception as e:
            logger.error(f"Erro ao remover posição: {e}")

    def get_attentive_mode_status(self, symbol, timeframe):
        """Verifica se o par está em modo atento."""
        if not os.path.exists(self.ATTENTIVE_MODE_FILE):
            return None
        
        try:
            with FileLock(f"{self.ATTENTIVE_MODE_FILE}.lock"):
                with open(self.ATTENTIVE_MODE_FILE, 'r', newline='') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if row["symbol"] == symbol and row["timeframe"] == timeframe:
                            # Verificar se o modo atento expirou
                            timeout = config.ATTENTIVE_MODE_TIMEOUT_MINUTES * 60
                            current_time = int(time.time())
                            if current_time - int(row["timestamp"]) > timeout:
                                logger.info(f"{symbol} {timeframe} modo atento expirou após {timeout}s")
                                return None
                            return {
                                "mode": row["mode"],
                                "direction": row["direction"],
                                "extreme_value": float(row["extreme_value"])
                            }
        except Exception as e:
            logger.error(f"Erro ao ler arquivo de modo atento: {e}")
        
        return None

    def save_attentive_mode(self, symbol, timeframe, mode, direction, extreme_value):
        """Salva o estado do modo atento no arquivo CSV."""
        statuses = []
        if os.path.exists(self.ATTENTIVE_MODE_FILE):
            try:
                with FileLock(f"{self.ATTENTIVE_MODE_FILE}.lock"):
                    with open(self.ATTENTIVE_MODE_FILE, 'r', newline='') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            if row["symbol"] != symbol or row["timeframe"] != timeframe:
                                statuses.append(row)
            except Exception as e:
                logger.error(f"Erro ao ler arquivo de modo atento: {e}")
        
        try:
            with FileLock(f"{self.ATTENTIVE_MODE_FILE}.lock"):
                with open(self.ATTENTIVE_MODE_FILE, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["symbol", "timeframe", "mode", "direction", "timestamp", "extreme_value"])
                    for status in statuses:
                        writer.writerow([
                            status["symbol"], status["timeframe"], status["mode"], 
                            status["direction"], status["timestamp"], status["extreme_value"]
                        ])
                    if mode:  # Adiciona apenas se mode não for None
                        writer.writerow([
                            symbol, timeframe, mode, direction, int(time.time()), extreme_value
                        ])
                if mode:
                    logger.info(f"{symbol} {timeframe} modo atento definido: {direction} com extremo {extreme_value}")
                else:
                    logger.info(f"{symbol} {timeframe} modo atento limpo")
        except Exception as e:
            logger.error(f"Erro ao salvar modo atento: {e}")

    def get_sl_streak(self, symbol):
        """Obtém o número atual de stop losses consecutivos para um par."""
        if not os.path.exists(self.SL_STREAK_FILE):
            return 0
        
        try:
            with FileLock(f"{self.SL_STREAK_FILE}.lock"):
                with open(self.SL_STREAK_FILE, 'r', newline='') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if row["symbol"] == symbol:
                            return int(row["streak_count"])
        except Exception as e:
            logger.error(f"Erro ao ler arquivo de SL streaks: {e}")
        
        return 0

    def update_sl_streak(self, symbol, increment=True):
        """Atualiza o contador de stop losses consecutivos."""
        if not config.SL_STREAK_PAUSE_ENABLED:
            return 0
            
        streaks = []
        found = False
        current_streak = 0
        
        if os.path.exists(self.SL_STREAK_FILE):
            try:
                with FileLock(f"{self.SL_STREAK_FILE}.lock"):
                    with open(self.SL_STREAK_FILE, 'r', newline='') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            if row["symbol"] == symbol:
                                found = True
                                current_streak = int(row["streak_count"])
                                if increment:
                                    current_streak += 1
                                else:
                                    current_streak = 0
                                streaks.append({
                                    "symbol": symbol,
                                    "streak_count": current_streak,
                                    "last_update": int(time.time())
                                })
                            else:
                                streaks.append(row)
            except Exception as e:
                logger.error(f"Erro ao ler arquivo de SL streaks: {e}")
        
        if not found:
            streaks.append({
                "symbol": symbol,
                "streak_count": 1 if increment else 0,
                "last_update": int(time.time())
            })
            current_streak = 1 if increment else 0
        
        try:
            with FileLock(f"{self.SL_STREAK_FILE}.lock"):
                with open(self.SL_STREAK_FILE, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["symbol", "streak_count", "last_update"])
                    for streak in streaks:
                        writer.writerow([
                            streak["symbol"], streak["streak_count"], streak["last_update"]
                        ])
            if increment:
                logger.info(f"{symbol} SL streak atualizado para {current_streak}")
            else:
                logger.info(f"{symbol} SL streak resetado para 0")
        except Exception as e:
            logger.error(f"Erro ao atualizar arquivo de SL streaks: {e}")
        
        return current_streak

    def check_pair_paused(self, symbol):
        """Verifica se o par está em pausa devido a stop losses consecutivos."""
        if not os.path.exists(self.PAUSED_PAIRS_FILE) or not config.SL_STREAK_PAUSE_ENABLED:
            return False
        
        try:
            with FileLock(f"{self.PAUSED_PAIRS_FILE}.lock"):
                with open(self.PAUSED_PAIRS_FILE, 'r', newline='') as file:
                    reader = csv.DictReader(file)
                    current_time = int(time.time())
                    for row in reader:
                        if row["symbol"] == symbol and int(row["until_timestamp"]) > current_time:
                            remaining_time = int(row["until_timestamp"]) - current_time
                            logger.debug(f"{symbol} está em pausa por mais {remaining_time}s. Razão: {row['reason']}")
                            return True
        except Exception as e:
            logger.error(f"Erro ao verificar arquivo de pares em pausa: {e}")
        
        return False

    def pause_pair(self, symbol, duration_seconds, reason="SL_STREAK"):
        """Pausa um par por um período específico."""
        if not config.SL_STREAK_PAUSE_ENABLED:
            return
            
        paused_pairs = []
        until_timestamp = int(time.time()) + duration_seconds
        
        if os.path.exists(self.PAUSED_PAIRS_FILE):
            try:
                with FileLock(f"{self.PAUSED_PAIRS_FILE}.lock"):
                    with open(self.PAUSED_PAIRS_FILE, 'r', newline='') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            if row["symbol"] != symbol:
                                paused_pairs.append(row)
            except Exception as e:
                logger.error(f"Erro ao ler arquivo de pares em pausa: {e}")
        
        try:
            with FileLock(f"{self.PAUSED_PAIRS_FILE}.lock"):
                with open(self.PAUSED_PAIRS_FILE, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["symbol", "until_timestamp", "reason"])
                    for pair in paused_pairs:
                        writer.writerow([
                            pair["symbol"], pair["until_timestamp"], pair["reason"]
                        ])
                    writer.writerow([symbol, until_timestamp, reason])
            logger.info(f"Par {symbol} pausado até {datetime.fromtimestamp(until_timestamp)} devido a {reason}")
        except Exception as e:
            logger.error(f"Erro ao atualizar arquivo de pares em pausa: {e}")

    def check_inverted_mode(self, symbol):
        """Verifica se o par está em modo invertido."""
        if not os.path.exists(self.INVERTED_MODE_FILE) or not config.INVERTED_MODE_ENABLED:
            return False
        
        try:
            with FileLock(f"{self.INVERTED_MODE_FILE}.lock"):
                with open(self.INVERTED_MODE_FILE, 'r', newline='') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if row["symbol"] == symbol and int(row["remaining_signals"]) > 0:
                            return True
        except Exception as e:
            logger.error(f"Erro ao verificar arquivo de modo invertido: {e}")
        
        return False

    def update_inverted_mode(self, symbol, set_inverted=True, signals_count=None):
        """Atualiza o modo invertido para um par."""
        if not config.INVERTED_MODE_ENABLED:
            return
            
        pairs = []
        found = False
        
        # Obter configuração específica para o par ou usar padrão
        if symbol in config.INVERTED_MODE_CONFIG:
            invert_config = config.INVERTED_MODE_CONFIG[symbol]
        else:
            invert_config = config.INVERTED_MODE_CONFIG["default"]
        
        if signals_count is None:
            signals_count = invert_config["invert_count"]
        
        if os.path.exists(self.INVERTED_MODE_FILE):
            try:
                with FileLock(f"{self.INVERTED_MODE_FILE}.lock"):
                    with open(self.INVERTED_MODE_FILE, 'r', newline='') as file:
                        reader = csv.DictReader(file)
                        for row in reader:
                            if row["symbol"] == symbol:
                                found = True
                                if set_inverted:
                                    pairs.append({
                                        "symbol": symbol,
                                        "remaining_signals": signals_count,
                                        "last_update": int(time.time())
                                    })
                                    logger.info(f"{symbol} modo invertido ativado para os próximos {signals_count} sinais")
                                else:
                                    count = int(row["remaining_signals"]) - 1
                                    if count > 0:
                                        pairs.append({
                                            "symbol": symbol,
                                            "remaining_signals": count,
                                            "last_update": int(time.time())
                                        })
                                        logger.info(f"{symbol} modo invertido: {count} sinais restantes")
                            else:
                                pairs.append(row)
            except Exception as e:
                logger.error(f"Erro ao ler arquivo de modo invertido: {e}")
        
        if not found and set_inverted:
            pairs.append({
                "symbol": symbol,
                "remaining_signals": signals_count,
                "last_update": int(time.time())
            })
            logger.info(f"{symbol} modo invertido ativado para os próximos {signals_count} sinais")
        
        try:
            with FileLock(f"{self.INVERTED_MODE_FILE}.lock"):
                with open(self.INVERTED_MODE_FILE, 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(["symbol", "remaining_signals", "last_update"])
                    for pair in pairs:
                        writer.writerow([
                            pair["symbol"], pair["remaining_signals"], pair["last_update"]
                        ])
        except Exception as e:
            logger.error(f"Erro ao atualizar arquivo de modo invertido: {e}")

    def check_multi_timeframe_filter(self, symbol, direction, timeframe, rsi):
        """Verifica o filtro multi-timeframe para validar o sinal."""
        if not config.MTF_FILTER_ENABLED:
            return True
        
        try:
            # Obter timeframe de referência
            if timeframe in config.MTF_REFERENCE_TIMEFRAMES:
                reference_tf = config.MTF_REFERENCE_TIMEFRAMES[timeframe]
            else:
                logger.warning(f"Nenhum timeframe de referência definido para {timeframe}, filtro MTF ignorado")
                return True
            
            # Verificar RSI no timeframe de referência
            if config.MTF_RSI_FILTER_ENABLED:
                try:
                    # Obter dados do timeframe de referência e calcular RSI
                    df = self.binance.get_historical_data(symbol, reference_tf, include_rsi_period=config.MTF_RSI_PERIOD)
                    if df.empty:
                        logger.warning(f"Sem dados históricos para {symbol} em {reference_tf} para filtro MTF RSI")
                        return True
                        
                    ref_rsi = df['rsi'].iloc[-1]
                    current_price = self.binance.get_current_price(symbol)
                    
                    # Verificar se o RSI está em zona extrema oposta
                    if direction == "BUY" and ref_rsi > config.MTF_RSI_MAX_FOR_BUY:
                        reject_reason = f"RSI em {reference_tf} ({ref_rsi:.2f}) acima do limite para compra ({config.MTF_RSI_MAX_FOR_BUY})"
                        self.rejected_tracker.log_rejected_signal(
                            symbol, timeframe, direction, "MTF_RSI", reject_reason, 
                            rsi_value=rsi, current_price=current_price,
                            additional_data={"ref_timeframe": reference_tf, "ref_rsi": ref_rsi}
                        )
                        return False
                    elif direction == "SELL" and ref_rsi < config.MTF_RSI_MIN_FOR_SELL:
                        reject_reason = f"RSI em {reference_tf} ({ref_rsi:.2f}) abaixo do limite para venda ({config.MTF_RSI_MIN_FOR_SELL})"
                        self.rejected_tracker.log_rejected_signal(
                            symbol, timeframe, direction, "MTF_RSI", reject_reason, 
                            rsi_value=rsi, current_price=current_price,
                            additional_data={"ref_timeframe": reference_tf, "ref_rsi": ref_rsi}
                        )
                        return False
                    
                    logger.info(f"{symbol} filtro MTF RSI passou: RSI do TF ({reference_tf}) = {ref_rsi:.2f}")
                except Exception as e:
                    logger.error(f"Erro no filtro MTF RSI para {symbol}: {e}")
                    return True  # Em caso de erro, permitir o sinal (fail-safe)
            
            # Verificar tendência EMA no timeframe de referência
            if config.MTF_TREND_FILTER_ENABLED:
                try:
                    # Obter dados do timeframe de referência
                    df = self.binance.get_historical_data(symbol, reference_tf)
                    if df.empty:
                        logger.warning(f"Sem dados históricos para {symbol} em {reference_tf} para filtro MTF EMA")
                        return True
                        
                    # Calcular EMA no DataFrame
                    df['ema'] = self.binance.calculate_ema(df['close'], period=config.MTF_EMA_PERIOD)
                    ema = df['ema'].iloc[-1]
                    current_price = df['close'].iloc[-1]
                    
                    # Verificar tendência com base na EMA longa
                    if direction == "BUY" and current_price < ema:
                        reject_reason = f"Preço ({current_price:.6f}) abaixo da EMA {config.MTF_EMA_PERIOD} ({ema:.6f}) em {reference_tf}"
                        self.rejected_tracker.log_rejected_signal(
                            symbol, timeframe, direction, "MTF_TREND", reject_reason, 
                            rsi_value=rsi, current_price=current_price,
                            additional_data={"ref_timeframe": reference_tf, "ema": ema}
                        )
                        return False
                    elif direction == "SELL" and current_price > ema:
                        reject_reason = f"Preço ({current_price:.6f}) acima da EMA {config.MTF_EMA_PERIOD} ({ema:.6f}) em {reference_tf}"
                        self.rejected_tracker.log_rejected_signal(
                            symbol, timeframe, direction, "MTF_TREND", reject_reason, 
                            rsi_value=rsi, current_price=current_price,
                            additional_data={"ref_timeframe": reference_tf, "ema": ema}
                        )
                        return False
                    
                    logger.info(f"{symbol} filtro MTF tendência passou: Preço {current_price:.6f} vs EMA {ema:.6f} em {reference_tf}")
                except Exception as e:
                    logger.error(f"Erro no filtro MTF tendência para {symbol}: {e}")
                    return True  # Em caso de erro, permitir o sinal (fail-safe)
            
            return True
        except Exception as e:
            logger.error(f"Erro no filtro multi-timeframe para {symbol}: {e}")
            return True  # Em caso de erro, permitir o sinal (fail-safe)

    def check_sr_filter(self, symbol, direction, timeframe):
        """Verifica o filtro de suporte e resistência para validar o sinal."""
        if not config.SR_FILTER_ENABLED:
            return True
        
        try:
            # Obter preço atual
            current_price = self.binance.get_current_price(symbol)
            if not current_price:
                logger.warning(f"Não foi possível obter preço atual para {symbol} para filtro S/R")
                return True
            
            # Verificar pontos de suporte/resistência dinâmicos (swing points)
            if config.SR_SWING_POINTS_ENABLED:
                try:
                    # Obter dados históricos
                    df = self.binance.get_historical_data(symbol, timeframe, limit=config.SR_SWING_LOOKBACK_PERIOD + config.SR_SWING_WINDOW + 1)
                    if df.empty:
                        logger.warning(f"Sem dados históricos suficientes para {symbol} em {timeframe} para filtro Swing Points")
                        return True
                        
                    # Obter swing points
                    swing_lows, swing_highs = self.binance.get_swing_points(df, config.SR_SWING_WINDOW, config.SR_SWING_LOOKBACK_PERIOD)
                    
                    # Verificar se o preço está próximo de algum swing point
                    found_support_resistance = False
                    
                    # Verificar níveis de suporte (lows) para compra
                    if direction == "BUY":
                        for level in swing_lows:
                            distance_pct = abs(current_price - level) / level * 100
                            if distance_pct <= config.SR_SWING_PRICE_PROXIMITY_PERCENT * 100:
                                logger.info(f"{symbol} sinal COMPRA confirmado por swing low em {level:.6f} (distância: {distance_pct:.2f}%)")
                                found_support_resistance = True
                                break
                    
                    # Verificar níveis de resistência (highs) para venda
                    elif direction == "SELL":
                        for level in swing_highs:
                            distance_pct = abs(current_price - level) / level * 100
                            if distance_pct <= config.SR_SWING_PRICE_PROXIMITY_PERCENT * 100:
                                logger.info(f"{symbol} sinal VENDA confirmado por swing high em {level:.6f} (distância: {distance_pct:.2f}%)")
                                found_support_resistance = True
                                break
                    
                    if not found_support_resistance:
                        reject_reason = f"Nenhum nível de {'suporte' if direction == 'BUY' else 'resistência'} próximo encontrado"
                        self.rejected_tracker.log_rejected_signal(
                            symbol, timeframe, direction, "SR_SWING_POINTS", reject_reason,
                            current_price=current_price,
                            additional_data={"swing_lows": swing_lows[:3], "swing_highs": swing_highs[:3]} if len(swing_lows) > 0 or len(swing_highs) > 0 else None
                        )
                        return False
                        
                except Exception as e:
                    logger.error(f"Erro no filtro Swing Points para {symbol}: {e}")
                    return True  # Em caso de erro, permitir o sinal (fail-safe)
            
            # Verificar níveis de Pivot Points
            if config.SR_PIVOT_POINTS_ENABLED:
                try:
                    # Obter níveis de pivot
                    pivot_levels = self.binance.get_pivot_points(symbol, config.SR_PIVOT_TIMEFRAME)
                    
                    if not pivot_levels:
                        logger.warning(f"Não foi possível calcular Pivot Points para {symbol}")
                        return True
                        
                    # Verificar se o preço está próximo de algum nível de Pivot
                    found_pivot = False
                    
                    for level_name in config.SR_PIVOT_LEVELS_TO_USE:
                        if level_name not in pivot_levels:
                            continue
                            
                        level_price = pivot_levels[level_name]
                        distance_pct = abs(current_price - level_price) / level_price * 100
                        
                        if distance_pct <= config.SR_PIVOT_PRICE_PROXIMITY_PERCENT * 100:
                            # Para compras, verificar níveis de suporte (S1, S2, S3)
                            if direction == "BUY" and level_name.startswith("S"):
                                logger.info(f"{symbol} sinal COMPRA confirmado por pivot {level_name} em {level_price:.6f} (distância: {distance_pct:.2f}%)")
                                found_pivot = True
                                break
                            # Para vendas, verificar níveis de resistência (R1, R2, R3)
                            elif direction == "SELL" and level_name.startswith("R"):
                                logger.info(f"{symbol} sinal VENDA confirmado por pivot {level_name} em {level_price:.6f} (distância: {distance_pct:.2f}%)")
                                found_pivot = True
                                break
                    
                    if not found_pivot:
                        reject_reason = f"Nenhum nível de pivot {'suporte' if direction == 'BUY' else 'resistência'} próximo encontrado"
                        self.rejected_tracker.log_rejected_signal(
                            symbol, timeframe, direction, "SR_PIVOT_POINTS", reject_reason,
                            current_price=current_price,
                            additional_data={"pivot_levels": pivot_levels}
                        )
                        return False
                        
                except Exception as e:
                    logger.error(f"Erro no filtro Pivot Points para {symbol}: {e}")
                    return True  # Em caso de erro, permitir o sinal (fail-safe)
            
            return True
        except Exception as e:
            logger.error(f"Erro no filtro de S/R para {symbol}: {e}")
            return True  # Em caso de erro, permitir o sinal (fail-safe)

    def check_volume_filter(self, symbol, timeframe):
        """Verifica o filtro de volume para validar o sinal."""
        if not config.VOLUME_FILTER_ENABLED:
            return True
        
        try:
            # Obter dados históricos para análise de volume
            df = self.binance.get_historical_data(symbol, timeframe, limit=config.VOLUME_LOOKBACK_PERIOD + 10)
            if df.empty:
                logger.warning(f"Sem dados históricos para {symbol} em {timeframe} para filtro de Volume")
                return True
                
            # Calcular média de volume
            avg_volume = df['volume'].iloc[-(config.VOLUME_LOOKBACK_PERIOD+1):-1].mean()
            current_volume = df['volume'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Verificar se o volume atual é maior que a média por uma porcentagem mínima
            volume_threshold = avg_volume * (1 + config.VOLUME_INCREASE_PERCENT / 100)
            if current_volume >= volume_threshold:
                logger.info(f"{symbol} filtro Volume passou: Volume atual {current_volume:.2f} > Média {avg_volume:.2f} * {1 + config.VOLUME_INCREASE_PERCENT/100}")
                return True
            else:
                reject_reason = f"Volume atual ({current_volume:.2f}) abaixo do mínimo requerido ({volume_threshold:.2f})"
                self.rejected_tracker.log_rejected_signal(
                    symbol, timeframe, "BOTH", "VOLUME", reject_reason,
                    current_price=current_price,
                    additional_data={"avg_volume": avg_volume, "required_threshold": volume_threshold}
                )
                return False
        except Exception as e:
            logger.error(f"Erro no filtro de volume para {symbol}: {e}")
            return True  # Em caso de erro, permitir o sinal (fail-safe)

    def check_trend_filter(self, symbol, direction, timeframe):
        """Verifica o filtro de tendência no timeframe operacional."""
        if not config.TREND_FILTER_ENABLED:
            return True
        
        try:
            # Obter dados para a EMA longa
            df = self.binance.get_historical_data(symbol, timeframe, limit=config.TREND_EMA_PERIOD + 10)
            if df.empty:
                logger.warning(f"Sem dados históricos para {symbol} em {timeframe} para filtro de Tendência")
                return True
                
            # Calcular EMA
            df['ema'] = self.binance.calculate_ema(df['close'], period=config.TREND_EMA_PERIOD)
            ema = df['ema'].iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Verificar se o preço está acima da EMA para compras ou abaixo para vendas
            if direction == "BUY" and current_price < ema:
                reject_reason = f"Preço ({current_price:.6f}) abaixo da EMA {config.TREND_EMA_PERIOD} ({ema:.6f})"
                self.rejected_tracker.log_rejected_signal(
                    symbol, timeframe, direction, "TREND", reject_reason,
                    current_price=current_price,
                    additional_data={"ema_period": config.TREND_EMA_PERIOD, "ema_value": ema}
                )
                return False
            elif direction == "SELL" and current_price > ema:
                reject_reason = f"Preço ({current_price:.6f}) acima da EMA {config.TREND_EMA_PERIOD} ({ema:.6f})"
                self.rejected_tracker.log_rejected_signal(
                    symbol, timeframe, direction, "TREND", reject_reason,
                    current_price=current_price,
                    additional_data={"ema_period": config.TREND_EMA_PERIOD, "ema_value": ema}
                )
                return False
            
            logger.info(f"{symbol} filtro Tendência passou para {direction}: Preço {current_price:.6f} vs EMA {ema:.6f}")
            return True
        except Exception as e:
            logger.error(f"Erro no filtro de tendência para {symbol}: {e}")
            return True  # Em caso de erro, permitir o sinal (fail-safe)

    def generate_signal_for_pair(self, symbol, timeframe):
        """Gera sinais para um par e timeframe específicos."""
        try:
            # Verificar se o RSI está habilitado
            if not config.SIGNAL_ENGINES["RSI"]:
                return
                
            # Verificar se o par está pausado por SL streak
            if self.check_pair_paused(symbol):
                logger.info(f"{symbol} está atualmente pausado, ignorando geração de sinal")
                return
            
            # Obter dados do par com RSI já calculado
            df = self.binance.get_historical_data(symbol, timeframe, limit=config.RSI_PERIOD + 10, include_rsi_period=config.RSI_PERIOD)
            if df.empty or len(df) < config.RSI_PERIOD + 2:
                logger.warning(f"Dados insuficientes para {symbol} {timeframe}: {len(df) if not df.empty else 0} candles")
                return
            
            # Extrai valores RSI
            if 'rsi' not in df.columns:
                logger.warning(f"RSI não calculado para {symbol} {timeframe}")
                return
                
            current_rsi = df['rsi'].iloc[-1]
            previous_rsi = df['rsi'].iloc[-2]
            
            # Verificar estado do modo atento
            attentive_status = self.get_attentive_mode_status(symbol, timeframe)
            inverted_mode = self.check_inverted_mode(symbol)
            
            # Verificar posições abertas
            has_position = self.binance.check_open_position(symbol)
            
            if has_position:
                logger.info(f"{symbol} tem uma posição aberta, ignorando geração de sinal")
                return
            
            # Lógica de geração de sinais
            signal = None
            
            # Log do RSI atual
            logger.info(f"{symbol} {timeframe} RSI atual: {current_rsi:.2f} (anterior: {previous_rsi:.2f}), modo invertido: {inverted_mode}")
            
            # Modo atento não ativado - verificar condições iniciais
            if not attentive_status:
                # No modo normal
                if not inverted_mode:
                    # Verificar condições de entrada em modo atento para compra
                    if current_rsi <= config.RSI_BUY_THRESHOLD and current_rsi <= config.MAX_RSI_FOR_BUY:  
                        logger.info(f"{symbol} entrou em modo atento para COMPRA com RSI {current_rsi:.2f}")
                        self.save_attentive_mode(symbol, timeframe, "attentive", "BUY", current_rsi)
                        
                    # Verificar condições de entrada em modo atento para venda
                    elif current_rsi >= config.RSI_SELL_THRESHOLD and current_rsi >= config.MIN_RSI_FOR_SELL:
                        logger.info(f"{symbol} entrou em modo atento para VENDA com RSI {current_rsi:.2f}")
                        self.save_attentive_mode(symbol, timeframe, "attentive", "SELL", current_rsi)
                # No modo invertido
                else:
                    # Condições invertidas: compra quando RSI alto, venda quando RSI baixo
                    if current_rsi <= config.RSI_BUY_THRESHOLD and current_rsi <= config.MAX_RSI_FOR_BUY:
                        logger.info(f"{symbol} entrou em modo atento para VENDA (invertido) com RSI {current_rsi:.2f}")
                        self.save_attentive_mode(symbol, timeframe, "attentive", "SELL", current_rsi)
                    elif current_rsi >= config.RSI_SELL_THRESHOLD and current_rsi >= config.MIN_RSI_FOR_SELL:
                        logger.info(f"{symbol} entrou em modo atento para COMPRA (invertido) com RSI {current_rsi:.2f}")
                        self.save_attentive_mode(symbol, timeframe, "attentive", "BUY", current_rsi)
            else:
                # Já está em modo atento - verificar condições para sinal
                direction = attentive_status["direction"]
                extreme_value = attentive_status["extreme_value"]
                
                if direction == "BUY":
                    # Verificar delta para compra
                    if current_rsi - extreme_value >= config.RSI_DELTA_MIN:
                        logger.info(f"{symbol} gerou sinal COMPRA com RSI {current_rsi:.2f} (delta: {current_rsi - extreme_value:.2f})")
                        signal = "BUY"
                        self.save_attentive_mode(symbol, timeframe, None, None, None)  # Limpar modo atento
                    # Atualizar valor extremo se o RSI atual for menor
                    elif current_rsi < extreme_value:
                        logger.info(f"{symbol} atualizando RSI extremo para COMPRA: {current_rsi:.2f} < {extreme_value:.2f}")
                        self.save_attentive_mode(symbol, timeframe, "attentive", "BUY", current_rsi)
                else:  # direction == "SELL"
                    # Verificar delta para venda
                    if extreme_value - current_rsi >= config.RSI_DELTA_MIN:
                        logger.info(f"{symbol} gerou sinal VENDA com RSI {current_rsi:.2f} (delta: {extreme_value - current_rsi:.2f})")
                        signal = "SELL"
                        self.save_attentive_mode(symbol, timeframe, None, None, None)  # Limpar modo atento
                    # Atualizar valor extremo se o RSI atual for maior
                    elif current_rsi > extreme_value:
                        logger.info(f"{symbol} atualizando RSI extremo para VENDA: {current_rsi:.2f} > {extreme_value:.2f}")
                        self.save_attentive_mode(symbol, timeframe, "attentive", "SELL", current_rsi)
            
            # Se não gerou sinal, encerra
            if not signal:
                return
            
            # Verificar RSI mínimo para compra e máximo para venda (proteção adicional)
            if signal == "BUY" and current_rsi < config.RSI_MIN_LONG:
                reject_reason = f"RSI atual ({current_rsi:.2f}) abaixo do mínimo requerido ({config.RSI_MIN_LONG})"
                self.rejected_tracker.log_rejected_signal(
                    symbol, timeframe, signal, "RSI_MIN", reject_reason, 
                    rsi_value=current_rsi, current_price=df['close'].iloc[-1]
                )
                return
            elif signal == "SELL" and current_rsi > config.RSI_MAX_SHORT:
                reject_reason = f"RSI atual ({current_rsi:.2f}) acima do máximo permitido ({config.RSI_MAX_SHORT})"
                self.rejected_tracker.log_rejected_signal(
                    symbol, timeframe, signal, "RSI_MAX", reject_reason, 
                    rsi_value=current_rsi, current_price=df['close'].iloc[-1]
                )
                return
            
            # Aplicar filtros avançados da v2.0
            if not self.check_multi_timeframe_filter(symbol, signal, timeframe, current_rsi):
                logger.info(f"{symbol} sinal {signal} rejeitado pelo filtro multi-timeframe")
                return
            
            if not self.check_sr_filter(symbol, signal, timeframe):
                logger.info(f"{symbol} sinal {signal} rejeitado pelo filtro S/R")
                return
            
            if not self.check_volume_filter(symbol, timeframe):
                logger.info(f"{symbol} sinal {signal} rejeitado pelo filtro de volume")
                return
            
            if not self.check_trend_filter(symbol, signal, timeframe):
                logger.info(f"{symbol} sinal {signal} rejeitado pelo filtro de tendência")
                return
            
            # Sinal aprovado por todos os filtros, executar ordem
            logger.info(f"{symbol} {timeframe} - Sinal {signal} APROVADO por todos os filtros - Executando ordem")
            self.execute_signal(symbol, signal, timeframe, current_rsi)
            
            # Se está em modo invertido, atualizar contador
            if inverted_mode:
                logger.info(f"{symbol} usando sinal invertido")
                self.update_inverted_mode(symbol, False)  # Reduzir contador
            
        except Exception as e:
            logger.error(f"Erro gerando sinal para {symbol} {timeframe}: {str(e)}")
            logger.error(traceback.format_exc())

    def execute_signal(self, symbol, direction, timeframe, current_rsi, additional_data=None):
        """Executa um sinal de trading, colocando uma ordem."""
        try:
            side = "BUY" if direction == "BUY" else "SELL"
            sl_percent = config.STOP_LOSS_PERCENT
            tp_percent = config.TAKE_PROFIT_PERCENT

            entry_price = self.binance.get_current_price(symbol)
            position_size = config.ORDER_VALUE_USDT / entry_price * config.LEVERAGE

            stop_loss = entry_price * (1 - sl_percent) if direction == "BUY" else entry_price * (1 + sl_percent)
            take_profit = entry_price * (1 + tp_percent) if direction == "BUY" else entry_price * (1 - tp_percent)

            order_result = self.binance.place_market_order(symbol, side, config.ORDER_VALUE_USDT, config.LEVERAGE)

            if order_result:
                self.save_position(symbol, entry_price, position_size, direction, take_profit, stop_loss)

                # Registrar sinal aceito detalhadamente
                additional_data = {
                    "filters_passed": {
                        "MTF": config.MTF_FILTER_ENABLED,
                        "SR": config.SR_FILTER_ENABLED,
                        "VOLUME": config.VOLUME_FILTER_ENABLED,
                        "TREND": config.TREND_FILTER_ENABLED
                    },
                    "volatility_dynamic_sl_tp": config.VOLATILITY_DYNAMIC_SL_TP_RSI_ENABLED,
                    "inverted_mode": self.check_inverted_mode(symbol)
                }

                from utils import log_accepted_signal
                log_accepted_signal(
                    symbol=symbol,
                    timeframe=timeframe,
                    signal_direction=direction,
                    entry_price=entry_price,
                    position_size=position_size,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    rsi_value=current_rsi,
                    additional_data=additional_data
                )

                logger.info(f"Ordem executada para {symbol} {direction} a {entry_price:.6f}, tamanho: {position_size}, SL: {stop_loss:.6f}, TP: {take_profit:.6f}")

            else:
                logger.error(f"Falha ao executar ordem {direction} para {symbol}")
        except Exception as e:
            logger.error(f"Erro ao executar sinal para {symbol}: {str(e)}")
            logger.error(traceback.format_exc())

    def monitor_and_close_positions(self):
        """Monitora posições abertas e fecha-as quando necessário."""
        try:
            positions = self.get_open_positions()
            
            for position in positions:
                symbol = position.get("symbol")
                direction = position.get("direction")
                entry_price = float(position.get("entry_price"))
                stop_loss = float(position.get("stop_loss"))
                take_profit = float(position.get("take_profit"))
                position_size = float(position.get("position_size"))

                if not all([symbol, direction, entry_price, stop_loss, take_profit, position_size]):
                    logger.warning(f"Entrada inválida encontrada em posições abertas: {position}")
                    continue

                position_exists = self.binance.check_open_position(symbol)
                if not position_exists:
                    logger.info(f"Posição para {symbol} não existe mais na Binance, removendo dos registros")
                    self.remove_position(symbol)
                    continue

                current_price = self.binance.get_current_price(symbol)
                if not current_price:
                    logger.warning(f"Não foi possível obter preço atual para {symbol}, pulando monitoramento")
                    continue

                hit_sl = (direction == "BUY" and current_price <= stop_loss) or \
                         (direction == "SELL" and current_price >= stop_loss)
                hit_tp = (direction == "BUY" and current_price >= take_profit) or \
                         (direction == "SELL" and current_price <= take_profit)

                if hit_sl or hit_tp:
                    result = "SL" if hit_sl else "TP"
                    close_side = "SELL" if direction == "BUY" else "BUY"
                    close_result = self.binance.close_position(symbol, close_side, str(position_size))

                    if close_result:
                        self.remove_position(symbol)
                        logger.info(f"Posição {direction} para {symbol} fechada com sucesso em {current_price:.6f} (Hit {result})")
                    else:
                        logger.error(f"Falha ao fechar posição para {symbol}")

        except Exception as e:
            logger.error(f"Erro ao monitorar posições: {str(e)}")
            logger.error(traceback.format_exc())

    def recover_open_positions_from_binance(self):
        """Recupera posições abertas diretamente da Binance e sincroniza com o arquivo local."""
        try:
            binance_positions = self.binance.get_all_open_positions()
            local_positions = self.get_open_positions()
            local_symbols = {pos["symbol"] for pos in local_positions}

            for position in binance_positions:
                symbol = position["symbol"]
                if symbol not in local_symbols:
                    entry_price = float(position["entryPrice"])
                    position_size = float(position["positionAmt"])
                    direction = "BUY" if position_size > 0 else "SELL"
                    sl_percent = config.STOP_LOSS_PERCENT
                    tp_percent = config.TAKE_PROFIT_PERCENT

                    stop_loss = entry_price * (1 - sl_percent) if direction == "BUY" else entry_price * (1 + sl_percent)
                    take_profit = entry_price * (1 + tp_percent) if direction == "BUY" else entry_price * (1 - tp_percent)

                    self.save_position(symbol, entry_price, position_size, direction, take_profit, stop_loss)
                    logger.info(f"Posição recuperada da Binance e salva localmente: {symbol} {direction}")

            logger.info("Sincronização de posições abertas com Binance concluída.")
        except Exception as e:
            logger.error(f"Erro ao recuperar posições da Binance: {e}")
            logger.error(traceback.format_exc())

    def process_pairs(self):
        """Processa todos os pares configurados."""
        try:
            active_pairs = self.pairs
            active_timeframes = self.timeframes
            
            if not active_pairs:
                logger.warning("Nenhum par de trading ativo definido.")
                return
                
            if not active_timeframes:
                logger.warning("Nenhum timeframe ativo definido.")
                return
            
            # Incrementar contador de ciclos    
            self.cycle_counter += 1
            
            # Log periódico (a cada 10 ciclos)
            if self.cycle_counter % 10 == 0:
                logger.info(f"Processando {len(active_pairs)} pares em {len(active_timeframes)} timeframes")
                self.cycle_counter = 0
            
            for symbol in active_pairs:
                for timeframe in active_timeframes:
                    self.generate_signal_for_pair(symbol, timeframe)
                    # Pequena pausa para evitar rate limits da API
                    time.sleep(0.1)
            
            # Monitorar e fechar posições
            self.monitor_and_close_positions()
        except Exception as e:
            logger.error(f"Erro ao processar pares: {str(e)}")
            logger.error(traceback.format_exc())
    
    def start(self):
        """Inicia o motor de sinais em uma thread separada."""
        if self.running:
            logger.warning("Motor de sinais já está em execução")
            return False
        
        logger.info("==================================================")
        logger.info("Iniciando motor de sinais v2.0")
        logger.info("==================================================")
        
        # Recuperar posições abertas da Binance ao iniciar
        if config.RECOVER_POSITIONS_ON_STARTUP:
            self.recover_positions_on_startup()

        # Verificar configuração
        if not config.SIGNAL_ENGINES["RSI"]:
            logger.warning("Motor de sinais RSI desativado. Verifique config.SIGNAL_ENGINES.")
            
        # Mostrar configuração ativa
        logger.info("Configuração ativa:")
        logger.info(f"- Motores de sinais: {[engine for engine, enabled in config.SIGNAL_ENGINES.items() if enabled]}")
        logger.info(f"- Pares de trading: {self.pairs}")
        logger.info(f"- Timeframes: {self.timeframes}")
        logger.info(f"- RSI settings: Period={config.RSI_PERIOD}, Buy={config.RSI_BUY_THRESHOLD}, Sell={config.RSI_SELL_THRESHOLD}")
        
        # Definir flag de execução
        self.running = True
        
        # Iniciar thread
        self.signal_thread = threading.Thread(target=self._run_loop)
        self.signal_thread.daemon = True
        self.signal_thread.start()
        
        return True
    
    def _run_loop(self):
        """Loop interno para execução contínua do motor de sinais."""
        try:
            while self.running:
                start_time = time.time()
                
                # Processar pares
                self.process_pairs()
                
                # Calcular tempo para próxima execução
                elapsed = time.time() - start_time
                sleep_time = max(1, config.MAIN_LOOP_INTERVAL - elapsed)
                
                logger.info(f"Ciclo completo em {elapsed:.2f}s, aguardando {sleep_time:.2f}s")
                time.sleep(sleep_time)
        except Exception as e:
            logger.critical(f"Erro crítico no loop principal do motor de sinais: {str(e)}")
            logger.critical(traceback.format_exc())
            
            # Enviar notificação Telegram em caso de erro crítico
            if config.TELEGRAM_NOTIFICATIONS_ENABLED and config.TELEGRAM_NOTIFY_ON_ERROR:
                from utils import send_telegram_message
                send_telegram_message(f"❗ ERRO CRÍTICO NO MOTOR DE SINAIS: {str(e)}")
                
            self.running = False
    
    def stop(self):
        """Para o motor de sinais."""
        if not self.running:
            logger.warning("Motor de sinais já está parado")
            return False
        
        logger.info("Parando motor de sinais...")
        self.running = False
        
        if self.signal_thread and self.signal_thread.is_alive():
            self.signal_thread.join(timeout=5)
        
        logger.info("Motor de sinais parado")
        return True

    def recover_positions_on_startup(self):
        """Recupera posições abertas diretamente da Binance e sincroniza com o arquivo local."""
        try:
            binance_positions = self.binance.get_all_open_positions()
            local_positions = self.get_open_positions()
            local_symbols = {pos["symbol"] for pos in local_positions}

            for position in binance_positions:
                symbol = position["symbol"]
                position_amt = float(position["positionAmt"])
                if position_amt == 0:
                    continue  # Ignorar posições zeradas

                if symbol not in local_symbols:
                    entry_price = float(position["entryPrice"])
                    position_size = abs(position_amt)
                    direction = "BUY" if position_amt > 0 else "SELL"
                    sl_percent = config.STOP_LOSS_PERCENT
                    tp_percent = config.TAKE_PROFIT_PERCENT

                    stop_loss = entry_price * (1 - sl_percent) if direction == "BUY" else entry_price * (1 + sl_percent)
                    take_profit = entry_price * (1 + tp_percent) if direction == "BUY" else entry_price * (1 - tp_percent)

                    self.save_position(symbol, entry_price, position_size, direction, take_profit, stop_loss)
                    logger.info(f"Posição recuperada da Binance e salva localmente: {symbol} {direction}")

            logger.info("Sincronização de posições abertas com Binance concluída.")
        except Exception as e:
            logger.error(f"Erro ao recuperar posições da Binance: {e}")
            logger.error(traceback.format_exc())

# Função main para execução direta do arquivo
if __name__ == "__main__":
    logger.info("Iniciando SignalEngine em modo standalone")
    
    # Usar valores padrão de configuração para teste
    engine = SignalEngine(
        api_key=config.REAL_API_KEY, 
        api_secret=config.REAL_API_SECRET,
        pairs=[pair for pair, enabled in config.PAIRS.items() if enabled]
    )
    
    try:
        engine.start()
        # Manter o processo principal vivo
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Interrupção do usuário detectada")
        engine.stop()
    except Exception as e:
        logger.critical(f"Erro crítico: {str(e)}")
        logger.critical(traceback.format_exc())
        engine.stop()