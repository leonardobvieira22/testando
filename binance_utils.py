# Atualizado em 12/05/2025 para incluir novas funcionalidades e robustez
from binance.client import Client
from binance.enums import SIDE_BUY, SIDE_SELL, ORDER_TYPE_MARKET
from binance.exceptions import BinanceAPIException
from utils import logger # calculate_rsi foi movido para dentro desta classe ou será uma nova implementação
import pandas as pd
import time
import numpy as np
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta

class BinanceUtils:
    def __init__(self, api_key, api_secret):
        self.client = Client(api_key, api_secret)
        self.sync_time()
        self.weight = 0
        self.futures_symbol_cache = None
        self.last_market_order_time = {}
        self.volatility_cache = {}
        self.volatility_cache_time = {}
        try:
            self.futures_symbol_cache = self.client.futures_exchange_info()["symbols"]
            logger.info("[BinanceUtils] Cache de símbolos de futuros carregado com sucesso")
        except Exception as e:
            logger.error(f"[BinanceUtils] Erro ao carregar cache de símbolos de futuros: {e}")
            self.futures_symbol_cache = []
        logger.info("BinanceUtils inicializado")

    def sync_time(self):
        try:
            server_time = self.client.get_server_time()["serverTime"]
            local_time = int(time.time() * 1000)
            self.client.time_offset = server_time - local_time
            logger.info(f"[BinanceUtils] Time offset ajustado: {self.client.time_offset} ms")
        except Exception as e:
            logger.error(f"[BinanceUtils] Erro ao sincronizar tempo: {e}")

    def track_weight(self, weight_to_add):
        self.weight += weight_to_add
        # logger.debug(f"[BinanceUtils] Peso da API: {self.weight}") # Log de peso pode ser muito verboso
        if self.weight >= 1180: # Limite é 1200, margem de segurança
            logger.warning(f"[BinanceUtils] Limite de peso da API próximo (atual: {self.weight}/1200). Aguardando 5 segundos...")
            time.sleep(5)
            self.weight = 0
        elif self.weight >= 600 and self.weight % 100 < weight_to_add : # Loga a cada 100 pontos a partir de 600
             logger.info(f"[BinanceUtils] Peso da API: {self.weight}/1200")

    def get_current_price(self, symbol):
        """
        Obtém o preço atual de mercado para o símbolo fornecido.
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            raise RuntimeError(f"Erro ao obter preço atual para {symbol}: {e}")

    def get_entry_price(self, symbol):
        """
        Obtém o preço de entrada da posição aberta para o símbolo fornecido.
        """
        try:
            positions = self.client.futures_account()['positions']
            for position in positions:
                if position['symbol'] == symbol:
                    return float(position['entryPrice'])
            raise ValueError(f"Nenhuma posição aberta encontrada para {symbol}")
        except Exception as e:
            raise RuntimeError(f"Erro ao obter preço de entrada para {symbol}: {e}")

    def get_current_price(self, symbol):
        for attempt in range(3):
            try:
                ticker = self.client.futures_ticker(symbol=symbol)
                self.track_weight(1)
                price = float(ticker["lastPrice"])
                logger.debug(f"[BinanceUtils] Preço obtido via ticker para {symbol}: {price:.6f}")
                return price
            except Exception as e:
                logger.warning(f"[BinanceUtils] Falha ao obter preço via ticker para {symbol}, tentativa {attempt + 1}/3: {e}")
                try:
                    klines = self.client.futures_klines(symbol=symbol, interval="1m", limit=1)
                    self.track_weight(2)
                    if klines:
                        price = float(klines[0][4])
                        logger.info(f"[BinanceUtils] Preço obtido via klines para {symbol}: {price:.6f}")
                        return price
                except Exception as e2:
                    logger.warning(f"[BinanceUtils] Falha ao obter preço via klines para {symbol}, tentativa {attempt + 1}/3: {e2}")
                time.sleep(0.2 * (attempt + 1))
        logger.error(f"[BinanceUtils] Falha persistente ao obter preço atual para {symbol} após 3 tentativas")
        return None

    def get_historical_data(self, symbol, interval, limit=250, include_rsi_period=None):
        try:
            klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)
            self.track_weight(2) # Ajustar peso conforme documentação da API para klines
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "quote_asset_volume", "number_of_trades",
                "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
            ])
            # Converter colunas numéricas para float
            numeric_cols = ["open", "high", "low", "close", "volume", "quote_asset_volume", 
                            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume"]
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

            if include_rsi_period and isinstance(include_rsi_period, int) and include_rsi_period > 0:
                 df["rsi"] = self.calculate_rsi(df["close"], period=include_rsi_period)
            
            # logger.debug(f"[BinanceUtils] Dados históricos obtidos para {symbol} ({interval}), {len(df)} candles.")
            return df
        except BinanceAPIException as e:
            logger.error(f"[BinanceUtils] Erro API ao obter dados históricos para {symbol} ({interval}): {e.status_code} - {e.message}")
            return pd.DataFrame() # Retorna DataFrame vazio em caso de erro
        except Exception as e:
            logger.error(f"[BinanceUtils] Erro geral ao obter dados históricos para {symbol} ({interval}): {e}")
            return pd.DataFrame()

    def calculate_rsi(self, prices_series: pd.Series, period: int = 14) -> pd.Series:
        if not isinstance(prices_series, pd.Series):
            raise TypeError("Input prices_series must be a pandas Series.")
        if prices_series.empty or len(prices_series) < period:
            # logger.warning(f"[BinanceUtils] Dados insuficientes para calcular RSI com período {period}. Got {len(prices_series)}.")
            return pd.Series([np.nan] * len(prices_series), index=prices_series.index)
        
        delta = prices_series.diff()
        gain = delta.where(delta > 0, 0.0).fillna(0.0)  # Preenche NaNs com 0 após where
        loss = -delta.where(delta < 0, 0.0).fillna(0.0) # Preenche NaNs com 0 após where

        avg_gain = gain.rolling(window=period, min_periods=1).mean()
        avg_loss = loss.rolling(window=period, min_periods=1).mean()
        
        # Para o primeiro cálculo, use a média simples. Para os seguintes, use a média móvel ponderada (semelhante à EMA)
        # Esta é uma forma comum de calcular RSI, mas a Binance pode usar Wilder's smoothing.
        # Para simplificar e alinhar com muitas bibliotecas, usaremos a média móvel simples para gain/loss.

        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi.replace([np.inf, -np.inf], np.nan, inplace=True) # Lida com divisão por zero em avg_loss
        rsi.fillna(50, inplace=True) # Preenche NaNs iniciais ou por divisão por zero com 50 (neutro)
        return rsi

    def calculate_ema(self, prices_series: pd.Series, period: int = 20) -> pd.Series:
        if not isinstance(prices_series, pd.Series):
            raise TypeError("Input prices_series must be a pandas Series.")
        if prices_series.empty or len(prices_series) < period:
            # logger.warning(f"[BinanceUtils] Dados insuficientes para calcular EMA com período {period}. Got {len(prices_series)}.")
            return pd.Series([np.nan] * len(prices_series), index=prices_series.index)
        
        ema = prices_series.ewm(span=period, adjust=False, min_periods=period).mean()
        return ema

    def get_swing_points(self, historical_data_df: pd.DataFrame, window: int, lookback_period: int):
        if not isinstance(historical_data_df, pd.DataFrame) or historical_data_df.empty:
            logger.warning("[BinanceUtils-SWING] DataFrame de dados históricos vazio ou inválido.")
            return [], []
        if not all(col in historical_data_df.columns for col in ["high", "low"]):
            logger.warning("[BinanceUtils-SWING] DataFrame não contém colunas 'high' e 'low'.")
            return [], []

        data_to_scan = historical_data_df.iloc[-lookback_period:]
        if len(data_to_scan) < (2 * window + 1):
            # logger.warning(f"[BinanceUtils-SWING] Dados insuficientes ({len(data_to_scan)}) para a janela de swing ({window}) e lookback ({lookback_period}).")
            return [], []

        swing_highs_prices = []
        swing_lows_prices = []

        for i in range(window, len(data_to_scan) - window):
            is_swing_high = True
            current_high = data_to_scan["high"].iloc[i]
            for j in range(1, window + 1):
                if current_high <= data_to_scan["high"].iloc[i-j] or \
                   current_high <= data_to_scan["high"].iloc[i+j]:
                    is_swing_high = False
                    break
            if is_swing_high:
                swing_highs_prices.append(current_high)

            is_swing_low = True
            current_low = data_to_scan["low"].iloc[i]
            for j in range(1, window + 1):
                if current_low >= data_to_scan["low"].iloc[i-j] or \
                   current_low >= data_to_scan["low"].iloc[i+j]:
                    is_swing_low = False
                    break
            if is_swing_low:
                swing_lows_prices.append(current_low)
        
        # Retorna únicos e ordenados (lows decrescente, highs crescente para facilitar a busca por proximidade)
        unique_lows = sorted(list(set(swing_lows_prices)), reverse=True)
        unique_highs = sorted(list(set(swing_highs_prices)))
        # logger.debug(f"[BinanceUtils-SWING] Swings encontrados: Lows={unique_lows}, Highs={unique_highs}")
        return unique_lows, unique_highs

    def get_pivot_points(self, pair: str, pivot_timeframe: str = "1d"):
        # Tenta obter os 2 últimos candles do timeframe de pivot para garantir o fechado.
        # A lógica para garantir o candle *anterior fechado* pode ser complexa.
        # Se pivot_timeframe é '1d', e chamamos às 00:05 UTC, limit=2 pode pegar o de D-1 e D-2.
        # Se chamamos às 23:55 UTC, limit=2 pode pegar o de D (em formação) e D-1.
        # Uma abordagem mais robusta envolveria checar o timestamp do candle.
        
        # Para simplificar, pegamos 3 candles e assumimos que o [-2] é o D-1 completo se o atual é D.
        # Ou, se o bot roda após o fechamento do candle diário, limit=1 para o dia anterior seria suficiente.
        # Vamos assumir que precisamos do candle fechado do dia anterior.
        # A API da Binance para klines retorna [open_time, ..., close_time, ...]
        # Se pedirmos klines com endTime, podemos garantir que pegamos candles fechados.
        
        # Abordagem: Pegar os últimos 3 candles do pivot_timeframe e usar o penúltimo (índice -2)
        # como o candle de referência (D-1, H-1, etc., completo).
        prev_period_data = self.get_historical_data(pair, pivot_timeframe, limit=3)
        if prev_period_data.empty or len(prev_period_data) < 2: # Precisa de pelo menos 2 para ter um [-2]
            logger.warning(f"[BinanceUtils-PIVOT] Dados insuficientes para calcular Pivot Points para {pair} em {pivot_timeframe}. Recebidos {len(prev_period_data)} candles.")
            return {}
        
        # Usar o candle em [-2] como o período anterior completo
        # Se len=2, [-2] é o primeiro. Se len=3, [-2] é o do meio.
        ref_candle_index = -2
        if len(prev_period_data) == 2:
             ref_candle_index = -2 # O mais antigo dos dois
        elif len(prev_period_data) >=3:
             ref_candle_index = -2 # O do meio dos três mais recentes

        prev_high = prev_period_data["high"].iloc[ref_candle_index]
        prev_low = prev_period_data["low"].iloc[ref_candle_index]
        prev_close = prev_period_data["close"].iloc[ref_candle_index]

        pp = (prev_high + prev_low + prev_close) / 3
        s1 = (2 * pp) - prev_high
        r1 = (2 * pp) - prev_low
        s2 = pp - (prev_high - prev_low)
        r2 = pp + (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pp) # Também pode ser: s3 = pp - 2 * (prev_high - pp)
        r3 = prev_high + 2 * (pp - prev_low) # Também pode ser: r3 = pp + 2 * (pp - prev_low)

        pivots = {"PP": pp, "S1": s1, "R1": r1, "S2": s2, "R2": r2, "S3": s3, "R3": r3}
        # logger.debug(f"[BinanceUtils-PIVOT] Pivots para {pair} ({pivot_timeframe}) usando H:{prev_high}, L:{prev_low}, C:{prev_close}: {pivots}")
        return pivots

    def validate_futures_symbol(self, symbol):
        try:
            if self.futures_symbol_cache is None:
                logger.info("[BinanceUtils] Cache de símbolos de futuros não carregado, tentando carregar agora.")
                self.futures_symbol_cache = self.client.futures_exchange_info()["symbols"]
                self.track_weight(40) # exchangeInfo é pesado
            valid = any(s["symbol"] == symbol for s in self.futures_symbol_cache)
            if not valid:
                logger.warning(f"[BinanceUtils] Símbolo {symbol} não encontrado na lista de símbolos de futuros")
            return valid
        except Exception as e:
            logger.error(f"[BinanceUtils] Erro ao validar símbolo {symbol}: {e}")
            return False

    def _retry_on_timestamp(func):
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except BinanceAPIException as e:
                if e.code == -1021 or "Timestamp for this request was" in e.message:
                    logger.warning("[BinanceUtils] Timestamp desincronizado, ressincronizando...")
                    self.sync_time()
                    return func(self, *args, **kwargs) # Tenta novamente
                raise # Re-levanta outras BinanceAPIExceptions
            except Exception as e:
                raise # Re-levanta exceções não-API
        return wrapper

    @_retry_on_timestamp
    def place_market_order(self, symbol, side, usdt_value, leverage):
        try:
            if not self.validate_futures_symbol(symbol):
                logger.error(f"[BinanceUtils] Símbolo {symbol} inválido para futuros")
                return None

            # Cancelar ordens abertas (TP/SL) antes de abrir uma nova posição pode ser perigoso
            # se a intenção é aumentar uma posição ou se houver múltiplas estratégias.
            # Removido o cancel_open_orders automático daqui. Deve ser gerenciado pela estratégia.
            # try:
            #     self.cancel_open_orders(symbol)
            # except Exception:
            #     logger.warning(f"[BinanceUtils] Falha ao cancelar ordens pendentes para {symbol}")

            last_order_time = self.last_market_order_time.get(symbol)
            current_time = time.time()
            min_interval = 5 # Intervalo mínimo de 5 segundos entre ordens para o mesmo par
            if last_order_time and (current_time - last_order_time) < min_interval:
                wait_time = min_interval - (current_time - last_order_time)
                logger.info(f"[BinanceUtils] Aguardando {wait_time:.2f}s antes de nova ordem para {symbol} (rate limit local)")
                time.sleep(wait_time)

            exchange_info = self.client.futures_exchange_info()
            self.track_weight(40)
            symbol_info = next((s for s in exchange_info["symbols"] if s["symbol"] == symbol), None)
            if not symbol_info:
                logger.error(f"[BinanceUtils] Informações não encontradas para o símbolo {symbol}")
                return None

            lot_filter = next((f for f in symbol_info["filters"] if f["filterType"] == "LOT_SIZE"), None)
            if not lot_filter:
                logger.error(f"[BinanceUtils] Filtro LOT_SIZE não encontrado para {symbol}")
                return None
            step_size = Decimal(lot_filter["stepSize"])
            min_qty = Decimal(lot_filter["minQty"])

            # Mudar alavancagem e modo de margem (se necessário)
            try:
                current_leverage = self.client.futures_leverage_bracket(symbol=symbol)[0]["brackets"][0]["initialLeverage"]
                self.track_weight(1)
                if int(current_leverage) != int(leverage):
                    self.client.futures_change_leverage(symbol=symbol, leverage=leverage)
                    logger.info(f"[BinanceUtils] Alavancagem alterada para {leverage}x para {symbol}")
                    self.track_weight(1)
            except Exception as e:
                 logger.error(f"[BinanceUtils] Erro ao verificar/alterar alavancagem para {symbol}: {e}. Tentando prosseguir.")
            
            # Alterar para margem ISOLADA (se não estiver já)
            # try:
            #     # Não há uma forma direta de checar o tipo de margem sem tentar mudar e tratar o erro.
            #     self.client.futures_change_margin_type(symbol=symbol, marginType='ISOLATED')
            #     logger.info(f"[BinanceUtils] Modo de margem alterado para ISOLATED para {symbol}")
            #     self.track_weight(1)
            # except BinanceAPIException as e:
            #     if "No need to change margin type" in str(e.message):
            #         pass # Já está ISOLATED
            #     else:
            #         logger.error(f"[BinanceUtils] Erro ao configurar modo ISOLATED para {symbol}: {e}. Tentando prosseguir.")
            # except Exception as e:
            #     logger.error(f"[BinanceUtils] Erro desconhecido ao configurar modo ISOLATED para {symbol}: {e}. Tentando prosseguir.")

            price = self.get_current_price(symbol)
            if price is None:
                logger.error(f"[BinanceUtils] Preço atual indisponível para {symbol} ao calcular quantidade.")
                return None

            qty_decimal = (Decimal(str(usdt_value)) * Decimal(str(leverage))) / Decimal(str(price))
            quantity_adjusted = qty_decimal.quantize(step_size, rounding=ROUND_DOWN)

            if quantity_adjusted < min_qty:
                logger.error(f"[BinanceUtils] Quantidade calculada {quantity_adjusted} para {symbol} menor que o mínimo {min_qty}. USDT: {usdt_value}, Lev: {leverage}, Price: {price}")
                return None
            if quantity_adjusted <= 0:
                logger.error(f"[BinanceUtils] Quantidade calculada inválida para {symbol}: {quantity_adjusted}")
                return None

            order = self.client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=str(quantity_adjusted)
            )
            self.track_weight(1) # Peso para criar ordem

            order_id = order.get("orderId")
            executed_qty_final = Decimal("0.0")
            avg_price_final = Decimal("0.0")
            commission_final = Decimal("0.0")
            commission_asset_final = "USDT"

            # Esperar e verificar preenchimento da ordem
            for check_attempt in range(5): # Tentar por até ~2.5 segundos
                time.sleep(0.5)
                try:
                    order_details = self.client.futures_get_order(symbol=symbol, orderId=order_id)
                    self.track_weight(1)
                    status = order_details.get("status")
                    if status == "FILLED":
                        executed_qty_final = Decimal(order_details.get("executedQty", "0.0"))
                        avg_price_final = Decimal(order_details.get("avgPrice", "0.0"))
                        # Obter dados de comissão dos trades associados à ordem
                        trades = self.client.futures_account_trades(symbol=symbol, orderId=order_id)
                        self.track_weight(5)
                        for trade in trades:
                            commission_final += Decimal(trade.get("commission", "0.0"))
                            commission_asset_final = trade.get("commissionAsset", "USDT")
                        break # Sai do loop de verificação
                    elif status in ["CANCELED", "EXPIRED", "REJECTED"]:
                        logger.error(f"[BinanceUtils] Ordem {order_id} para {symbol} falhou com status: {status}")
                        return None # Ordem não foi preenchida
                except Exception as e_check:
                    logger.warning(f"[BinanceUtils] Erro ao verificar ordem {order_id} para {symbol} (tentativa {check_attempt+1}): {e_check}")
            
            if executed_qty_final <= Decimal("0.0"):
                logger.error(f"[BinanceUtils] Ordem {order_id} para {symbol} não foi preenchida (ou quantidade zero) após verificações.")
                # Tentar cancelar se não foi preenchida para evitar problemas
                try: self.client.futures_cancel_order(symbol=symbol, orderId=order_id); self.track_weight(1) 
                except: pass
                return None

            order["executedQty"] = str(executed_qty_final) # Atualiza com o valor Decimal convertido para string
            order["avgPrice"] = str(avg_price_final)
            order["commission"] = str(commission_final)
            order["commissionAsset"] = commission_asset_final
            
            self.last_market_order_time[symbol] = time.time()
            logger.info(f"[BinanceUtils] Ordem de mercado {order_id} para {symbol} ({side}) Qtd: {executed_qty_final}, Preço Médio: {avg_price_final}, Comissão: {commission_final} {commission_asset_final}")
            return order

        except BinanceAPIException as e:
            logger.error(f"[BinanceUtils] Erro API ao colocar ordem de mercado para {symbol}: {e.status_code} - {e.message}")
            return None
        except Exception as e:
            logger.error(f"[BinanceUtils] Erro inesperado ao colocar ordem de mercado para {symbol}: {e}")
            return None

    def place_tp_sl_orders(self, symbol, side, entry_price_str, tp_percent_str, sl_percent_str, quantity_str):
        # Esta função precisa ser revista ou removida, pois o monitoramento TP/SL será feito por polling no sinal_engine.
        # A criação de ordens TP/SL diretamente na Binance pode conflitar com a lógica de fechamento do bot.
        logger.warning("[BinanceUtils] place_tp_sl_orders não é recomendado com a lógica atual do bot. TP/SL são monitorados internamente.")
        return None, None

    @_retry_on_timestamp
    def close_position(self, symbol, side_to_open_to_close, quantity_to_close_str):
        try:
            if not self.validate_futures_symbol(symbol):
                logger.error(f"[BinanceUtils] Símbolo {symbol} inválido para fechar posição.")
                return None

            # Obter informações do símbolo para precisão da quantidade
            exchange_info = self.client.futures_exchange_info()
            self.track_weight(40)
            symbol_info = next((s for s in exchange_info["symbols"] if s["symbol"] == symbol), None)
            if not symbol_info:
                logger.error(f"[BinanceUtils] Informações não encontradas para o símbolo {symbol} ao fechar posição.")
                return None
            lot_filter = next((f for f in symbol_info["filters"] if f["filterType"] == "LOT_SIZE"), None)
            if not lot_filter:
                logger.error(f"[BinanceUtils] Filtro LOT_SIZE não encontrado para {symbol} ao fechar posição.")
                return None
            step_size = Decimal(lot_filter["stepSize"])

            # Verificar posição atual
            current_positions = self.client.futures_position_information(symbol=symbol)
            self.track_weight(5)
            position_amount_dec = Decimal("0.0")
            for pos in current_positions:
                if pos["symbol"] == symbol:
                    position_amount_dec = Decimal(pos.get("positionAmt", "0.0"))
                    break
            
            if position_amount_dec == Decimal("0.0"):
                logger.warning(f"[BinanceUtils] Nenhuma posição aberta encontrada para {symbol} para fechar.")
                return {"status": "NO_POSITION"} # Indica que não havia posição

            quantity_to_close_dec = Decimal(quantity_to_close_str).quantize(step_size, rounding=ROUND_DOWN)
            
            # Garantir que não estamos tentando fechar mais do que a posição atual
            if quantity_to_close_dec > abs(position_amount_dec):
                logger.warning(f"[BinanceUtils] Tentando fechar {quantity_to_close_dec} de {symbol}, mas posição é {abs(position_amount_dec)}. Ajustando para fechar total.")
                quantity_to_close_dec = abs(position_amount_dec)
            
            if quantity_to_close_dec <= Decimal("0.0"):
                logger.warning(f"[BinanceUtils] Quantidade zero ou negativa para fechar posição em {symbol}: {quantity_to_close_dec}")
                return None

            logger.info(f"[BinanceUtils] Tentando fechar {quantity_to_close_dec} de {symbol}. Lado da ordem de fechamento: {side_to_open_to_close}")
            
            order = None
            for attempt in range(3):
                try:
                    order = self.client.futures_create_order(
                        symbol=symbol,
                        side=side_to_open_to_close, # Lado OPOSTO da posição para fechar
                        type=ORDER_TYPE_MARKET,
                        quantity=str(quantity_to_close_dec),
                        reduceOnly=True # Importante para garantir que apenas feche ou reduza
                    )
                    self.track_weight(1)
                    logger.info(f"[BinanceUtils] Ordem de fechamento {order.get('orderId')} para {symbol} ({side_to_open_to_close}), Qtd: {quantity_to_close_dec} enviada (reduceOnly=True).")
                    
                    # Verificar se a posição foi realmente reduzida/fechada
                    time.sleep(1.0) # Dar tempo para a ordem ser processada
                    new_positions = self.client.futures_position_information(symbol=symbol)
                    self.track_weight(5)
                    new_pos_amt_dec = Decimal("0.0")
                    for n_pos in new_positions:
                        if n_pos["symbol"] == symbol:
                            new_pos_amt_dec = Decimal(n_pos.get("positionAmt", "0.0"))
                            break
                    
                    if (position_amount_dec > 0 and new_pos_amt_dec < position_amount_dec) or \
                       (position_amount_dec < 0 and new_pos_amt_dec > position_amount_dec) or \
                       new_pos_amt_dec == Decimal("0.0"):
                        logger.info(f"[BinanceUtils] Posição para {symbol} reduzida/fechada. Antiga: {position_amount_dec}, Nova: {new_pos_amt_dec}")
                        return order # Sucesso
                    else:
                        logger.warning(f"[BinanceUtils] Tentativa {attempt+1}: Posição para {symbol} não parece ter sido reduzida. Antiga: {position_amount_dec}, Nova: {new_pos_amt_dec}. Ordem: {order}")

                except BinanceAPIException as e:
                    logger.error(f"[BinanceUtils] Erro API na tentativa {attempt+1} de fechar {symbol} com reduceOnly: {e.status_code} - {e.message}")
                    if e.code == -2022: # ReduceOnly Order is rejected
                        logger.warning(f"[BinanceUtils] ReduceOnly rejeitado. Tentando fechar com closePosition=True para {symbol}")
                        try:
                            order = self.client.futures_create_order(
                                symbol=symbol, 
                                side=side_to_open_to_close, 
                                type=ORDER_TYPE_MARKET, 
                                closePosition=True # Param para fechar toda a posição para o par e lado
                            )
                            self.track_weight(1)
                            logger.info(f"[BinanceUtils] Ordem de fechamento (closePosition=True) {order.get('orderId')} para {symbol} enviada.")
                            # Verificar fechamento
                            time.sleep(1.0)
                            final_positions = self.client.futures_position_information(symbol=symbol)
                            self.track_weight(5)
                            final_pos_amt_dec = Decimal("0.0")
                            for f_pos in final_positions:
                                if f_pos["symbol"] == symbol:
                                    final_pos_amt_dec = Decimal(f_pos.get("positionAmt", "0.0"))
                                    break
                            if final_pos_amt_dec == Decimal("0.0"):
                                logger.info(f"[BinanceUtils] Posição para {symbol} fechada com sucesso via closePosition=True.")
                                return order # Sucesso
                            else:
                                logger.error(f"[BinanceUtils] Falha ao fechar {symbol} mesmo com closePosition=True. Posição restante: {final_pos_amt_dec}")
                                return None # Falha mesmo com fallback
                        except Exception as e_cp:
                            logger.error(f"[BinanceUtils] Erro ao tentar fechar {symbol} com closePosition=True: {e_cp}")
                            # Não retorna aqui, deixa o loop de retry principal continuar se houver mais tentativas
                    # Outros erros podem precisar de tratamento específico ou apenas retry
                except Exception as e:
                    logger.error(f"[BinanceUtils] Erro geral na tentativa {attempt+1} de fechar {symbol}: {e}")
                time.sleep(0.5 * (attempt + 1)) # Backoff exponencial
            
            logger.error(f"[BinanceUtils] Falha persistente ao fechar posição para {symbol} após múltiplas tentativas.")
            return None

        except BinanceAPIException as e:
            logger.error(f"[BinanceUtils] Erro API (externo ao loop de retry) ao fechar posição para {symbol}: {e.status_code} - {e.message}")
            return None
        except Exception as e:
            logger.error(f"[BinanceUtils] Erro inesperado (externo ao loop de retry) ao fechar posição para {symbol}: {e}")
            return None

    def check_open_position(self, symbol):
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            self.track_weight(5)
            for pos in positions:
                if pos["symbol"] == symbol and Decimal(pos.get("positionAmt", "0.0")) != Decimal("0.0"):
                    # logger.debug(f"[BinanceUtils] Posição aberta encontrada para {symbol}: {pos['positionAmt']}")
                    return True # Retorna True se qualquer posição para o símbolo for diferente de zero
            return False
        except Exception as e:
            logger.error(f"[BinanceUtils] Erro ao verificar posição aberta para {symbol}: {e}")
            return False # Assume que não há posição em caso de erro para evitar trades conflitantes

    def get_price_precision(self, symbol):
        try:
            if self.futures_symbol_cache is None: self.validate_futures_symbol(symbol) # Carrega cache se não existir
            symbol_info = next((s for s in self.futures_symbol_cache if s["symbol"] == symbol), None)
            if symbol_info:
                return int(symbol_info.get("pricePrecision", 8)) # 'pricePrecision' é direto
            logger.warning(f"[BinanceUtils] Não foi possível obter pricePrecision para {symbol} do cache. Usando default 8.")
            return 8
        except Exception as e:
            logger.error(f"[BinanceUtils] Erro ao obter precisão de preço para {symbol}: {e}. Usando default 8.")
            return 8

    def get_quantity_precision(self, symbol):
        try:
            if self.futures_symbol_cache is None: self.validate_futures_symbol(symbol)
            symbol_info = next((s for s in self.futures_symbol_cache if s["symbol"] == symbol), None)
            if symbol_info:
                return int(symbol_info.get("quantityPrecision", 8)) # 'quantityPrecision' é direto
            logger.warning(f"[BinanceUtils] Não foi possível obter quantityPrecision para {symbol} do cache. Usando default 8.")
            return 8
        except Exception as e:
            logger.error(f"[BinanceUtils] Erro ao obter precisão de quantidade para {symbol}: {e}. Usando default 8.")
            return 8

    def get_commission_rate(self, symbol: str) -> dict:
        try:
            fees = self.client.futures_commission_rate(symbol=symbol)
            self.track_weight(20) # Checar peso para este endpoint
            return {
                "maker": Decimal(fees.get("makerCommissionRate", "0.0002")), # Default 0.02%
                "taker": Decimal(fees.get("takerCommissionRate", "0.0005"))  # Default 0.05%
            }
        except Exception as e:
            logger.error(f"[BinanceUtils] Erro ao obter taxas de comissão para {symbol}: {e}. Usando defaults.")
            return {"maker": Decimal("0.0002"), "taker": Decimal("0.0005")}

    def get_volatility(self, symbol, interval, lookback_periods=20, cache_ttl_seconds=60):
        cache_key = f"{symbol}_{interval}_volatility"
        current_time = time.time()
        
        cached_value = self.volatility_cache.get(cache_key)
        cached_time = self.volatility_cache_time.get(cache_key, 0)

        if cached_value is not None and (current_time - cached_time) < cache_ttl_seconds:
            # logger.debug(f"[BinanceUtils] Usando volatilidade em cache para {symbol} ({interval}): {cached_value:.4f}")
            return cached_value
        
        try:
            # +10 para ter margem para cálculo de retornos e std
            data_df = self.get_historical_data(symbol, interval, limit=lookback_periods + 10)
            if data_df.empty or len(data_df) < lookback_periods:
                logger.warning(f"[BinanceUtils] Dados insuficientes ({len(data_df)}) para calcular volatilidade de {symbol} ({interval}) com lookback {lookback_periods}. Usando default.")
                return Decimal("0.05")
            
            data_df["returns"] = np.log(data_df["close"] / data_df["close"].shift(1))
            returns_series = data_df["returns"].dropna().tail(lookback_periods)
            
            if returns_series.empty or len(returns_series) < max(2, lookback_periods // 2): # Precisa de pelo menos alguns retornos
                logger.warning(f"[BinanceUtils] Sem retornos válidos ({len(returns_series)}) para calcular volatilidade de {symbol} ({interval}). Usando default.")
                return Decimal("0.05")
            
            std_dev = returns_series.std()
            if pd.isna(std_dev) or std_dev == 0:
                logger.warning(f"[BinanceUtils] Std dev inválido ({std_dev}) para {symbol} ({interval}). Usando default.")
                return Decimal("0.05")

            intervals_per_year = {
                "1m": 365*24*60, "3m": 365*24*20, "5m": 365*24*12, "15m": 365*24*4,
                "30m": 365*24*2, "1h": 365*24, "2h": 365*12, "4h": 365*6,
                "6h": 365*4, "8h": 365*3, "12h": 365*2, "1d": 365
            }
            annualization_factor = np.sqrt(intervals_per_year.get(interval, 365*24*60)) # Default para 1m se não encontrado
            volatility = Decimal(str(std_dev)) * Decimal(str(annualization_factor))
            
            # Limitar volatilidade a um range razoável (1% a 200% anualizada)
            volatility_clamped = max(Decimal("0.01"), min(Decimal("2.0"), volatility))
            
            self.volatility_cache[cache_key] = volatility_clamped
            self.volatility_cache_time[cache_key] = current_time
            
            # logger.info(f"[BinanceUtils] Volatilidade calculada para {symbol} ({interval}): {volatility_clamped:.4f}")
            return volatility_clamped
            
        except Exception as e:
            logger.error(f"[BinanceUtils] Erro geral ao calcular volatilidade para {symbol} ({interval}): {e}. Usando default.")
            return Decimal("0.05")

    def get_open_position_amount_and_side(self, symbol: str) -> tuple[Decimal, str | None]:
        try:
            positions = self.client.futures_position_information(symbol=symbol)
            self.track_weight(5)
            for pos in positions:
                if pos["symbol"] == symbol:
                    amount = Decimal(pos.get("positionAmt", "0.0"))
                    if amount != Decimal("0.0"):
                        side = SIDE_BUY if amount > 0 else SIDE_SELL
                        return abs(amount), side
            return Decimal("0.0"), None
        except Exception as e:
            logger.error(f"[BinanceUtils] Erro ao obter quantidade e lado da posição para {symbol}: {e}")
            return Decimal("0.0"), None

    @_retry_on_timestamp
    def cancel_open_orders(self, symbol: str, max_retries: int = 3) -> bool:
        try:
            if not self.validate_futures_symbol(symbol):
                logger.error(f"[BinanceUtils] Símbolo {symbol} inválido para cancelar ordens.")
                return False
            
            open_orders = self.client.futures_get_open_orders(symbol=symbol)
            self.track_weight(3) # get_open_orders tem peso 3 com symbol
            if not open_orders:
                # logger.info(f"[BinanceUtils] Nenhuma ordem aberta para cancelar em {symbol}.")
                return True

            logger.info(f"[BinanceUtils] Tentando cancelar {len(open_orders)} ordens abertas para {symbol}...")
            success_all = True
            for order_to_cancel in open_orders:
                order_id = order_to_cancel.get("orderId")
                for attempt in range(1, max_retries + 1):
                    try:
                        self.client.futures_cancel_order(symbol=symbol, orderId=order_id)
                        self.track_weight(1)
                        logger.info(f"[BinanceUtils] Ordem {order_id} cancelada para {symbol}.")
                        break # Sucesso, sai do loop de tentativas para esta ordem
                    except BinanceAPIException as e_cancel:
                        logger.warning(f"[BinanceUtils] Tentativa {attempt}/{max_retries} de cancelar ordem {order_id} para {symbol} falhou: {e_cancel.message} (Code: {e_cancel.code})")
                        if e_cancel.code == -2011: # Order does not exist or already filled/canceled
                            logger.info(f"[BinanceUtils] Ordem {order_id} para {symbol} já não existe ou foi processada.")
                            break # Considera como sucesso para esta ordem
                        if attempt == max_retries:
                            logger.error(f"[BinanceUtils] Falha persistente ao cancelar ordem {order_id} para {symbol}.")
                            success_all = False
                        time.sleep(0.3 * attempt) # Pequeno backoff
                    except Exception as e_gen_cancel:
                        logger.error(f"[BinanceUtils] Erro geral na tentativa {attempt}/{max_retries} de cancelar ordem {order_id} para {symbol}: {e_gen_cancel}")
                        if attempt == max_retries:
                            success_all = False
                        time.sleep(0.3 * attempt)
            return success_all
        except BinanceAPIException as e:
            logger.error(f"[BinanceUtils] Erro API ao listar/cancelar ordens abertas para {symbol}: {e.message}")
            return False
        except Exception as e:
            logger.error(f"[BinanceUtils] Erro geral ao cancelar ordens abertas para {symbol}: {e}")
            return False

    def get_all_open_positions(self):
        """Retorna todas as posições abertas na conta de futuros."""
        try:
            positions = self.client.futures_position_information()
            open_positions = [pos for pos in positions if float(pos['positionAmt']) != 0]
            return open_positions
        except Exception as e:
            logger.error(f"Erro ao obter posições abertas da Binance: {e}")
            return []

# Exportar as funções para uso externo
def get_current_price(api_key, api_secret, symbol):
    """
    Wrapper para obter o preço atual usando BinanceUtils.
    """
    binance_utils = BinanceUtils(api_key, api_secret)
    return binance_utils.get_current_price(symbol)

def get_entry_price(api_key, api_secret, symbol):
    """
    Wrapper para obter o preço de entrada usando BinanceUtils.
    """
    binance_utils = BinanceUtils(api_key, api_secret)
    return binance_utils.get_entry_price(symbol)