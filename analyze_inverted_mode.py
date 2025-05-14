import pandas as pd
from utils import logger
from config import ORDERS_PNL_CSV, SIMULATED_ORDERS_CSV, INVERTED_STATE_CSV, SIMULATED_INVERTED_STATE_CSV

def analyze_inverted_mode():
    """Analisa o desempenho do Inverted Mode em ordens reais e simuladas."""
    try:
        # Carregar ordens reais
        df_real = pd.read_csv(ORDERS_PNL_CSV) if pd.io.common.file_exists(ORDERS_PNL_CSV) else pd.DataFrame()
        df_sim = pd.read_csv(SIMULATED_ORDERS_CSV) if pd.io.common.file_exists(SIMULATED_ORDERS_CSV) else pd.DataFrame()
        
        # Carregar estados de inversão
        df_real_invert = pd.read_csv(INVERTED_STATE_CSV) if pd.io.common.file_exists(INVERTED_STATE_CSV) else pd.DataFrame()
        df_sim_invert = pd.read_csv(SIMULATED_INVERTED_STATE_CSV) if pd.io.common.file_exists(SIMULATED_INVERTED_STATE_CSV) else pd.DataFrame()

        # Análise de ordens reais
        if not df_real.empty:
            real_summary = df_real[df_real['status'] == 'CLOSED'].groupby(['pair', 'timeframe', 'signal_engine', 'order_mode'])['pnl'].agg(['sum', 'count', 'mean']).reset_index()
            real_inverted = real_summary[real_summary['order_mode'] == 'inverted']
            real_original = real_summary[real_summary['order_mode'] == 'original']
            logger.info("Resumo de Ordens Reais:")
            logger.info(f"\n{real_summary.to_string(index=False)}")
            if not real_inverted.empty:
                logger.info("PnL Total Inverted (Real):")
                logger.info(f"\n{real_inverted[['pair', 'timeframe', 'signal_engine', 'sum']].to_string(index=False)}")
            if not real_original.empty:
                logger.info("PnL Total Original (Real):")
                logger.info(f"\n{real_original[['pair', 'timeframe', 'signal_engine', 'sum']].to_string(index=False)}")

        # Análise de ordens simuladas
        if not df_sim.empty:
            sim_summary = df_sim[df_sim['status'] == 'CLOSED'].groupby(['pair', 'timeframe', 'signal_engine', 'order_mode'])['pnl'].agg(['sum', 'count', 'mean']).reset_index()
            sim_inverted = sim_summary[sim_summary['order_mode'] == 'inverted']
            sim_original = sim_summary[sim_summary['order_mode'] == 'original']
            logger.info("Resumo de Ordens Simuladas:")
            logger.info(f"\n{sim_summary.to_string(index=False)}")
            if not sim_inverted.empty:
                logger.info("PnL Total Inverted (Simulado):")
                logger.info(f"\n{sim_inverted[['pair', 'timeframe', 'signal_engine', 'sum']].to_string(index=False)}")
            if not sim_original.empty:
                logger.info("PnL Total Original (Simulado):")
                logger.info(f"\n{sim_original[['pair', 'timeframe', 'signal_engine', 'sum']].to_string(index=False)}")

        # Análise de estados de inversão
        if not df_real_invert.empty:
            logger.info("Estado de Inverted Mode (Real):")
            logger.info(f"\n{df_real_invert.to_string(index=False)}")
        if not df_sim_invert.empty:
            logger.info("Estado de Inverted Mode (Simulado):")
            logger.info(f"\n{df_sim_invert.to_string(index=False)}")

        # Salvar análise em um arquivo
        with open('inverted_mode_analysis.txt', 'w') as f:
            if not df_real.empty:
                f.write("Resumo de Ordens Reais:\n")
                f.write(real_summary.to_string(index=False) + "\n\n")
                if not real_inverted.empty:
                    f.write("PnL Total Inverted (Real):\n")
                    f.write(real_inverted[['pair', 'timeframe', 'signal_engine', 'sum']].to_string(index=False) + "\n\n")
                if not real_original.empty:
                    f.write("PnL Total Original (Real):\n")
                    f.write(real_original[['pair', 'timeframe', 'signal_engine', 'sum']].to_string(index=False) + "\n\n")
            if not df_sim.empty:
                f.write("Resumo de Ordens Simuladas:\n")
                f.write(sim_summary.to_string(index=False) + "\n\n")
                if not sim_inverted.empty:
                    f.write("PnL Total Inverted (Simulado):\n")
                    f.write(sim_inverted[['pair', 'timeframe', 'signal_engine', 'sum']].to_string(index=False) + "\n\n")
                if not sim_original.empty:
                    f.write("PnL Total Original (Simulado):\n")
                    f.write(sim_original[['pair', 'timeframe', 'signal_engine', 'sum']].to_string(index=False) + "\n\n")
            if not df_real_invert.empty:
                f.write("Estado de Inverted Mode (Real):\n")
                f.write(df_real_invert.to_string(index=False) + "\n\n")
            if not df_sim_invert.empty:
                f.write("Estado de Inverted Mode (Simulado):\n")
                f.write(df_sim_invert.to_string(index=False) + "\n")
        logger.info("Análise de Inverted Mode salva em inverted_mode_analysis.txt")

    except Exception as e:
        logger.error(f"Erro ao analisar Inverted Mode: {e}")

if __name__ == "__main__":
    analyze_inverted_mode()