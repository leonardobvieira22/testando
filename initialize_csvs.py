import os
import pandas as pd
from utils import logger
from config import *

def initialize_csvs():
    """Inicializa todos os arquivos CSV com esquemas corretos, incluindo a coluna de volatilidade."""
    
    # signals.csv: Armazena sinais gerados pelos motores
    if not os.path.exists(SIGNALS_CSV) or os.path.getsize(SIGNALS_CSV) == 0:
        cols = [
            'signal_id', 'pair', 'timeframe', 'direction', 'rsi', 
            'bb_upper', 'bb_middle', 'bb_lower', 'reason', 
            'signal_engine', 'timestamp', 'volatility'  # Adicionado volatility
        ]
        pd.DataFrame(columns=cols).to_csv(SIGNALS_CSV, index=False)
        logger.info(f"Arquivo {SIGNALS_CSV} inicializado com esquema atualizado (inclui volatilidade).")
    else:
        # Verifica se é necessário migrar dados existentes para o novo esquema
        try:
            df = pd.read_csv(SIGNALS_CSV)
            if 'volatility' not in df.columns:
                logger.info(f"Migrando {SIGNALS_CSV} para incluir coluna de volatilidade.")
                df['volatility'] = None  # Coluna com valores nulos para dados históricos
                df.to_csv(SIGNALS_CSV, index=False)
        except Exception as e:
            logger.error(f"Erro ao verificar/migrar {SIGNALS_CSV}: {e}")

    # orders.csv: Armazena ordens abertas e fechadas
    if not os.path.exists(ORDERS_CSV) or os.path.getsize(ORDERS_CSV) == 0:
        cols = [
            'order_id', 'signal_id', 'pair', 'timeframe', 'direction',
            'entry_price', 'quantity', 'tp_price', 'sl_price', 'status',
            'close_reason', 'close_order_id', 'pnl', 'close_timestamp',
            'close_price', 'signal_engine', 'order_mode', 'entry_fee',
            'exit_fee', 'timestamp', 'volatility'  # Adicionado volatility
        ]
        pd.DataFrame(columns=cols).to_csv(ORDERS_CSV, index=False)
        logger.info(f"Arquivo {ORDERS_CSV} inicializado com esquema atualizado (inclui volatilidade).")
    else:
        # Verifica se é necessário migrar dados existentes para o novo esquema
        try:
            df = pd.read_csv(ORDERS_CSV)
            if 'volatility' not in df.columns:
                logger.info(f"Migrando {ORDERS_CSV} para incluir coluna de volatilidade.")
                df['volatility'] = None
                df.to_csv(ORDERS_CSV, index=False)
        except Exception as e:
            logger.error(f"Erro ao verificar/migrar {ORDERS_CSV}: {e}")

    # orders_pnl.csv: Armazena ordens finalizadas para análise de PnL
    if not os.path.exists(ORDERS_PNL_CSV) or os.path.getsize(ORDERS_PNL_CSV) == 0:
        # Usa o mesmo esquema de orders.csv
        if os.path.exists(ORDERS_CSV) and os.path.getsize(ORDERS_CSV) > 0:
            try:
                df = pd.read_csv(ORDERS_CSV)
                empty_df = pd.DataFrame(columns=df.columns)
                empty_df.to_csv(ORDERS_PNL_CSV, index=False)
            except:
                cols = [
                    'order_id', 'signal_id', 'pair', 'timeframe', 'direction',
                    'entry_price', 'quantity', 'tp_price', 'sl_price', 'status',
                    'close_reason', 'close_order_id', 'pnl', 'close_timestamp',
                    'close_price', 'signal_engine', 'order_mode', 'entry_fee',
                    'exit_fee', 'timestamp', 'volatility'  # Adicionado volatility
                ]
                pd.DataFrame(columns=cols).to_csv(ORDERS_PNL_CSV, index=False)
        else:
            cols = [
                'order_id', 'signal_id', 'pair', 'timeframe', 'direction',
                'entry_price', 'quantity', 'tp_price', 'sl_price', 'status',
                'close_reason', 'close_order_id', 'pnl', 'close_timestamp',
                'close_price', 'signal_engine', 'order_mode', 'entry_fee',
                'exit_fee', 'timestamp', 'volatility'  # Adicionado volatility
            ]
            pd.DataFrame(columns=cols).to_csv(ORDERS_PNL_CSV, index=False)
        logger.info(f"Arquivo {ORDERS_PNL_CSV} inicializado com esquema atualizado (inclui volatilidade).")
    else:
        # Verifica se é necessário migrar dados existentes para o novo esquema
        try:
            df = pd.read_csv(ORDERS_PNL_CSV)
            if 'volatility' not in df.columns:
                logger.info(f"Migrando {ORDERS_PNL_CSV} para incluir coluna de volatilidade.")
                df['volatility'] = None
                df.to_csv(ORDERS_PNL_CSV, index=False)
        except Exception as e:
            logger.error(f"Erro ao verificar/migrar {ORDERS_PNL_CSV}: {e}")

    # prices_statistics.csv: Armazena estatísticas de preço e RSI
    if not os.path.exists(PRICES_STATISTICS_CSV) or os.path.getsize(PRICES_STATISTICS_CSV) == 0:
        cols = [
            'id', 'pair', 'timeframe', 'price', 'rsi',
            'bb_upper', 'bb_middle', 'bb_lower',
            'signal_engine', 'timestamp', 'volatility'  # Adicionado volatility
        ]
        pd.DataFrame(columns=cols).to_csv(PRICES_STATISTICS_CSV, index=False)
        logger.info(f"Arquivo {PRICES_STATISTICS_CSV} inicializado com esquema atualizado (inclui volatilidade).")
    else:
        # Verifica se é necessário migrar dados existentes para o novo esquema
        try:
            df = pd.read_csv(PRICES_STATISTICS_CSV)
            if 'volatility' not in df.columns:
                logger.info(f"Migrando {PRICES_STATISTICS_CSV} para incluir coluna de volatilidade.")
                df['volatility'] = None
                df.to_csv(PRICES_STATISTICS_CSV, index=False)
        except Exception as e:
            logger.error(f"Erro ao verificar/migrar {PRICES_STATISTICS_CSV}: {e}")

    # outros arquivos que não precisam da coluna de volatilidade
    if not os.path.exists(PAUSE_LOG_CSV) or os.path.getsize(PAUSE_LOG_CSV) == 0:
        cols = [
            'id', 'pair', 'timeframe', 'trigger_time', 
            'pause_end', 'sl_streak', 'signal_engine'
        ]
        pd.DataFrame(columns=cols).to_csv(PAUSE_LOG_CSV, index=False)
        logger.info(f"Arquivo {PAUSE_LOG_CSV} inicializado com esquema.")

    # SL Streak para logs
    if not os.path.exists(SL_STREAK_LOG_CSV) or os.path.getsize(SL_STREAK_LOG_CSV) == 0:
        cols = [
            'id', 'pair', 'timeframe', 'timestamp', 'current_streak',
            'threshold', 'pause_minutes', 'signal_engine', 'mode'
        ]
        pd.DataFrame(columns=cols).to_csv(SL_STREAK_LOG_CSV, index=False)
        logger.info(f"Arquivo {SL_STREAK_LOG_CSV} inicializado com esquema.")

    # Inverted mode para logs
    if not os.path.exists(INVERTED_MODE_LOG_CSV) or os.path.getsize(INVERTED_MODE_LOG_CSV) == 0:
        cols = [
            'id', 'pair', 'timeframe', 'timestamp', 'action',
            'original_direction', 'inverted_direction', 'invert_count_remaining',
            'signal_engine', 'mode'
        ]
        pd.DataFrame(columns=cols).to_csv(INVERTED_MODE_LOG_CSV, index=False)
        logger.info(f"Arquivo {INVERTED_MODE_LOG_CSV} inicializado com esquema.")

    logger.info("Inicialização ou verificação de todos os arquivos CSV concluída.")

if __name__ == "__main__":
    initialize_csvs()
    logger.info("Arquivos CSV inicializados com sucesso!")