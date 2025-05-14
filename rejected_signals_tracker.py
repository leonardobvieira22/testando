"""
Rastreador de sinais rejeitados para o motor de sinais v2.0
Implementa logging detalhado e persistência em CSV para análise posterior
"""
import os
import csv
import time
from datetime import datetime
from filelock import FileLock
from utils import logger

class RejectedSignalsTracker:
    """
    Rastreador de sinais rejeitados pelos filtros avançados
    """
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)
        self.rejected_signals_file = os.path.join(self.data_dir, "rejected_signals.csv")
        self.ensure_csv_exists()
        
    def ensure_csv_exists(self):
        """Garante que o arquivo CSV para sinais rejeitados existe com os cabeçalhos corretos"""
        if not os.path.exists(self.rejected_signals_file):
            try:
                with FileLock(f"{self.rejected_signals_file}.lock"):
                    with open(self.rejected_signals_file, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([
                            "timestamp", "date_time", "symbol", "timeframe", "signal_direction",
                            "filter_type", "rejection_reason", "rsi_value", "price", 
                            "additional_data"
                        ])
                logger.info(f"Arquivo CSV para rastreamento de sinais rejeitados criado: {self.rejected_signals_file}")
            except Exception as e:
                logger.error(f"Erro ao criar arquivo de rastreamento de sinais rejeitados: {e}")
    
    def log_rejected_signal(self, symbol, timeframe, direction, filter_type, reason, rsi_value=None, current_price=None, additional_data=None):
        """
        Registra um sinal rejeitado no CSV e gera log detalhado
        
        Args:
            symbol: Par de trading
            timeframe: Timeframe do sinal
            direction: Direção do sinal (BUY/SELL)
            filter_type: Tipo de filtro que rejeitou o sinal
            reason: Motivo detalhado da rejeição
            rsi_value: Valor do RSI no momento da rejeição (opcional)
            current_price: Preço atual no momento da rejeição (opcional)
            additional_data: Dados adicionais relevantes para a rejeição (opcional)
        """
        timestamp = int(time.time())
        date_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
        
        # Log detalhado
        message = f"SINAL REJEITADO: {symbol} {timeframe} {direction} - Filtro: {filter_type} - Motivo: {reason}"
        if rsi_value is not None:
            message += f" - RSI: {rsi_value:.2f}"
        if current_price is not None:
            message += f" - Preço: {current_price:.8f}"
            
        logger.info(message)
        
        # Salvar no CSV
        try:
            with FileLock(f"{self.rejected_signals_file}.lock"):
                with open(self.rejected_signals_file, 'a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([
                        timestamp, date_time, symbol, timeframe, direction, 
                        filter_type, reason, 
                        f"{rsi_value:.2f}" if rsi_value is not None else "N/A",
                        f"{current_price:.8f}" if current_price is not None else "N/A",
                        str(additional_data) if additional_data else "N/A"
                    ])
        except Exception as e:
            logger.error(f"Erro ao salvar sinal rejeitado no CSV: {e}")
            
    def get_rejection_stats(self, days_back=7):
        """
        Retorna estatísticas sobre sinais rejeitados nos últimos dias
        
        Args:
            days_back: Número de dias para olhar para trás
            
        Returns:
            dict: Estatísticas de rejeições por filtro
        """
        stats = {
            "total_rejections": 0,
            "rejections_by_filter": {},
            "rejections_by_symbol": {},
            "rejections_by_direction": {"BUY": 0, "SELL": 0}
        }
        
        cutoff_time = int(time.time()) - (days_back * 24 * 60 * 60)
        
        try:
            with FileLock(f"{self.rejected_signals_file}.lock"):
                with open(self.rejected_signals_file, 'r', newline='') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        if int(row["timestamp"]) < cutoff_time:
                            continue
                            
                        stats["total_rejections"] += 1
                        
                        filter_type = row["filter_type"]
                        if filter_type not in stats["rejections_by_filter"]:
                            stats["rejections_by_filter"][filter_type] = 0
                        stats["rejections_by_filter"][filter_type] += 1
                        
                        symbol = row["symbol"]
                        if symbol not in stats["rejections_by_symbol"]:
                            stats["rejections_by_symbol"][symbol] = 0
                        stats["rejections_by_symbol"][symbol] += 1
                        
                        direction = row["signal_direction"]
                        if direction in stats["rejections_by_direction"]:
                            stats["rejections_by_direction"][direction] += 1
        except Exception as e:
            logger.error(f"Erro ao gerar estatísticas de rejeições: {e}")
            
        return stats