
# Patch para corrigir problemas com websocket_manager.py
# Substitua a função problemática existente por esta versão corrigida

def process_metrics(self):
    """
    Processa métricas de desempenho e conexão do WebSocket sem chamar a API da Binance
    """
    try:
        # Obter timestamp atual
        current_time = int(time.time())
        
        # Registrar métricas sem tentar obter dados históricos
        metrics = {
            'timestamp': current_time,
            'active_connections': len(self.socket_connections),
            'messages_received': self.messages_received,
            'messages_per_second': self.calculate_messages_per_second(),
            'last_price_log_time': getattr(self, 'price_log_time', 0)
        }
        
        # Log das métricas
        if current_time % 60 == 0:  # Log a cada minuto
            logger.info(f"WebSocket Metrics: {metrics}")
            
        return metrics
    except Exception as e:
        logger.error(f"Erro ao processar métricas: {e}")
        return {}
