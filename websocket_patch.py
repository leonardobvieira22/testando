"""
Patch para resolver problemas com o gerenciador de WebSocket
"""
import time
import logging

logger = logging.getLogger(__name__)

class SafeWebSocketManager:
    """
    Versão segura do gerenciador de WebSocket que não trata variáveis internas como símbolos
    """
    
    def __init__(self, api_key, api_secret):
        """Inicializa o gerenciador de WebSocket seguro"""
        self.api_key = api_key
        self.api_secret = api_secret
        self.socket_connections = {}
        self.messages_received = 0
        self.price_log_time = int(time.time())
        self.metrics_history = []
    
    def is_valid_trading_symbol(self, symbol):
        """
        Verifica se um símbolo é um par de trading válido
        Previne tentativas de obter dados para variáveis internas
        """
        # Lista de símbolos inválidos conhecidos
        invalid_symbols = ['price_log_time', 'websocket_manager', 'socket_connections']
        
        # Verificação básica
        if symbol in invalid_symbols:
            logger.warning(f"Tentativa de usar variável interna '{symbol}' como símbolo de trading")
            return False
        
        # Verificações adicionais (opcional)
        if not symbol.endswith('USDT') and not symbol.endswith('BTC'):
            logger.warning(f"Símbolo potencialmente inválido: '{symbol}'")
            return False
            
        return True
    
    def get_historical_data_safe(self, binance_utils, symbol, interval, limit=250, include_rsi_period=None):
        """
        Versão segura do método get_historical_data que verifica se o símbolo é válido
        
        Args:
            binance_utils: Instância da classe BinanceUtils
            symbol: Símbolo para obter dados
            interval: Intervalo de tempo
            limit: Limite de candles
            include_rsi_period: Período RSI (opcional)
            
        Returns:
            DataFrame de dados ou DataFrame vazio se símbolo inválido
        """
        if not self.is_valid_trading_symbol(symbol):
            logger.info(f"Ignorando busca de dados para símbolo inválido: '{symbol}'")
            import pandas as pd
            return pd.DataFrame()  # Retorna DataFrame vazio em vez de chamar a API
            
        # Chamada segura para o método original
        return binance_utils.get_historical_data(symbol, interval, limit, include_rsi_period)
    
    def process_metrics(self):
        """
        Processa métricas de desempenho e conexão sem chamar a API da Binance
        """
        try:
            # Obter timestamp atual
            current_time = int(time.time())
            
            # Registrar métricas sem tentar obter dados históricos
            metrics = {
                'timestamp': current_time,
                'active_connections': len(self.socket_connections),
                'messages_received': self.messages_received,
                'messages_per_second': self.calculate_messages_per_second() if hasattr(self, 'calculate_messages_per_second') else 0,
                'last_price_log_time': self.price_log_time
            }
            
            # Armazenar métricas para análise posterior
            self.metrics_history.append(metrics)
            
            # Limitar tamanho do histórico
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
                
            # Log das métricas
            if current_time % 60 == 0:  # Log a cada minuto
                logger.info(f"WebSocket Metrics: {metrics}")
                
            return metrics
        except Exception as e:
            logger.error(f"Erro ao processar métricas: {e}")
            return {}