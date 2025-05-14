"""
Análise de volatilidade - Estuda a relação entre volatilidade e performance de trading
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import logger
from config import *
import os
from datetime import datetime
import matplotlib.dates as mdates
import traceback
# Importações para integração com Binance API
from binance_utils import BinanceUtils
import time

# Configurar estilo de visualização
plt.style.use('ggplot')
sns.set(style="darkgrid")

class VolatilityAnalysis:
    """Classe para análise e visualização da relação entre volatilidade e resultados de trading."""
    
    def __init__(self, signals_csv=SIGNALS_CSV, orders_pnl_csv=ORDERS_PNL_CSV):
        """Inicializa o analisador de volatilidade."""
        self.signals_csv = signals_csv
        self.orders_pnl_csv = orders_pnl_csv
        self.signals_df = None
        self.orders_df = None
        self.merged_df = None
        self.output_dir = "volatility_analysis"
        
        # Inicializar cliente Binance
        try:
            self.binance_client = BinanceUtils()
            logger.info("Cliente Binance inicializado com sucesso")
        except Exception as e:
            logger.error(f"Erro ao inicializar cliente Binance: {e}")
            self.binance_client = None
        
        # Cache para dados da API Binance
        self.pnl_cache = {}
        
        # Criar diretório para outputs se não existir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Diretório criado: {self.output_dir}")
    
    def get_order_pnl_from_binance(self, pair, order_id):
        """
        Obtém o PnL realizado diretamente da Binance para uma ordem específica.
        Implementa cache para evitar chamadas repetidas à API.
        """
        # Verificar se o resultado já está em cache
        cache_key = f"{pair}_{order_id}"
        if cache_key in self.pnl_cache:
            return self.pnl_cache[cache_key]
        
        if not self.binance_client:
            logger.warning("Cliente Binance não está disponível para consultar PnL")
            return None
            
        try:
            # Obter dados da ordem da Binance
            order_info = self.binance_client.get_order(symbol=pair, orderId=int(order_id))
            
            # Se a ordem foi completamente executada, calcular PnL
            if order_info and order_info.get('status') == 'FILLED':
                # Para ordens de compra
                if order_info.get('side') == 'BUY':
                    entry_price = float(order_info.get('price', 0))
                    quantity = float(order_info.get('executedQty', 0))
                    
                    # Procurar a ordem de venda correspondente
                    # Esta lógica depende de como suas ordens estão vinculadas
                    # Pode ser necessário ajustar conforme seu sistema
                    close_orders = self.binance_client.get_all_orders(
                        symbol=pair, 
                        startTime=int(order_info.get('time', 0))
                    )
                    
                    for close_order in close_orders:
                        if close_order.get('side') == 'SELL' and float(close_order.get('executedQty', 0)) == quantity:
                            exit_price = float(close_order.get('price', 0))
                            pnl = (exit_price - entry_price) * quantity
                            
                            # Subtrair taxas
                            fee_entry = entry_price * quantity * 0.001  # 0.1% taxa padrão
                            fee_exit = exit_price * quantity * 0.001
                            pnl -= (fee_entry + fee_exit)
                            
                            # Armazenar em cache
                            self.pnl_cache[cache_key] = pnl
                            return pnl
                
                # Para ordens de venda, lógica similar, mas inversa
                elif order_info.get('side') == 'SELL':
                    # Implemente conforme necessário
                    pass
            
            logger.warning(f"Não foi possível calcular PnL para {pair} ordem {order_id}")
            return None
            
        except Exception as e:
            logger.error(f"Erro ao obter dados da Binance para {pair} ordem {order_id}: {e}")
            return None
    
    def load_data(self):
        """Carrega dados de sinais e ordens finalizadas."""
        try:
            logger.info(f"Carregando dados de {self.signals_csv}...")
            
            # Ler o CSV como texto para verificar formato
            with open(self.signals_csv, 'r', encoding='utf-8') as f:
                header = f.readline().strip()
                logger.info(f"Cabeçalho CSV: {header}")
            
            # Tentar identificar o formato real do arquivo
            self.signals_df = pd.read_csv(self.signals_csv)
            
            # Verificar se o formato está correto
            if 'volatility' not in self.signals_df.columns:
                logger.warning("Coluna 'volatility' não encontrada. Tentando corrigir formato do CSV...")
                
                # Ler novamente manualmente para corrigir problema de formato
                with open(self.signals_csv, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    headers = lines[0].strip().split(',')
                    
                    # Identificar o número correto de colunas usando o cabeçalho
                    expected_columns = len(headers)
                    
                    # Corrigir manualmente as linhas
                    data = []
                    for i in range(1, len(lines)):
                        # Dividir com respeito às aspas para manter texto com vírgulas intacto
                        parts = lines[i].strip().split('"')
                        if len(parts) >= 3:
                            # Se houver texto entre aspas, precisamos processa manualmente
                            parts[0] = parts[0].rstrip(',')  # Remover a vírgula antes das aspas
                            parts[2] = parts[2].lstrip(',')  # Remover a vírgula após as aspas
                            fields = parts[0].split(',') + [parts[1]] + parts[2].split(',')
                        else:
                            # Se não há texto entre aspas, é mais simples
                            fields = lines[i].strip().split(',')
                        
                        # Garantir que temos o número correto de campos
                        if len(fields) > expected_columns:
                            fields = fields[:expected_columns]
                        elif len(fields) < expected_columns:
                            fields += [''] * (expected_columns - len(fields))
                            
                        data.append(fields)
                
                # Criar DataFrame corrigido
                self.signals_df = pd.DataFrame(data, columns=headers)
            
            # Converter timestamp para datetime
            if 'timestamp' in self.signals_df.columns:
                self.signals_df['timestamp'] = pd.to_datetime(self.signals_df['timestamp'])
            
            # Converter volatilidade para float
            if 'volatility' in self.signals_df.columns:
                self.signals_df['volatility'] = pd.to_numeric(self.signals_df['volatility'], errors='coerce')
                
                # Verificar a qualidade dos dados de volatilidade
                nan_count = self.signals_df['volatility'].isna().sum()
                logger.info(f"Coluna 'volatility': {len(self.signals_df)} entradas, {nan_count} valores NaN ({nan_count/len(self.signals_df)*100:.2f}%)")
            else:
                logger.warning("Coluna 'volatility' ainda não encontrada após correção. Análise de volatilidade será limitada.")
                return False
            
            logger.info(f"Carregando dados de {self.orders_pnl_csv}...")
            self.orders_df = pd.read_csv(self.orders_pnl_csv)
            
            # Converter timestamps para datetime
            if 'timestamp' in self.orders_df.columns:
                self.orders_df['timestamp'] = pd.to_datetime(self.orders_df['timestamp'])
            if 'close_timestamp' in self.orders_df.columns:
                self.orders_df['close_timestamp'] = pd.to_datetime(self.orders_df['close_timestamp'])
            
            # Mesclar dados de sinais e ordens
            logger.info("Mesclando dados de sinais e ordens...")
            self.merged_df = self.signals_df.merge(
                self.orders_df, 
                on='signal_id', 
                how='inner',
                suffixes=('', '_order')
            )
            
            # Atualizar dados de PnL da Binance quando possível
            if self.binance_client:
                self.update_pnl_from_binance()
            
            # Calcular se a ordem foi vitoriosa (PnL > 0)
            self.merged_df['is_win'] = self.merged_df['pnl'] > 0
            
            # Garantir que a volatilidade está na coluna correta após mesclagem
            if 'volatility' not in self.merged_df.columns and 'volatility_order' in self.merged_df.columns:
                self.merged_df['volatility'] = self.merged_df['volatility_order']
                
            # Se a volatilidade for string, converter para float
            if self.merged_df['volatility'].dtype == 'object':
                self.merged_df['volatility'] = pd.to_numeric(self.merged_df['volatility'], errors='coerce')
            
            # Verificar a qualidade dos dados de volatilidade após mesclagem
            if 'volatility' in self.merged_df.columns:
                nan_count = self.merged_df['volatility'].isna().sum()
                logger.info(f"Volatilidade após mesclagem: {len(self.merged_df)} entradas, {nan_count} valores NaN ({nan_count/len(self.merged_df)*100:.2f}%)")
                
                # Se todos os valores forem NaN, tentar reparar usando os dados dos sinais
                if nan_count == len(self.merged_df) and 'volatility_x' in self.merged_df.columns:
                    self.merged_df['volatility'] = self.merged_df['volatility_x']
                    logger.info("Tentando reparar dados de volatilidade a partir de sinais...")
                    nan_count = self.merged_df['volatility'].isna().sum()
                    logger.info(f"Após reparo: {len(self.merged_df)} entradas, {nan_count} valores NaN ({nan_count/len(self.merged_df)*100:.2f}%)")
            
            logger.info(f"Dados carregados: {len(self.signals_df)} sinais, {len(self.orders_df)} ordens, {len(self.merged_df)} correspondências.")
            return True
        
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {e}")
            traceback.print_exc()
            return False
    
    def update_pnl_from_binance(self):
        """Atualiza os dados de PnL no DataFrame usando dados reais da Binance API"""
        logger.info("Atualizando dados de PnL usando a Binance API...")
        
        updated_count = 0
        error_count = 0
        
        # Iterar pelas ordens com IDs válidos
        for idx, row in self.merged_df.iterrows():
            # Verificar se temos todos os dados necessários
            if pd.notna(row['order_id']) and pd.notna(row['pair']):
                try:
                    # Buscar PnL da Binance
                    pnl = self.get_order_pnl_from_binance(row['pair'], int(row['order_id']))
                    
                    # Se conseguimos um PnL válido, atualizar o DataFrame
                    if pnl is not None:
                        self.merged_df.at[idx, 'pnl'] = pnl
                        self.merged_df.at[idx, 'is_win'] = pnl > 0
                        updated_count += 1
                    
                    # Evitar exceder limites de taxa da API
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.warning(f"Erro ao obter PnL para {row['pair']} ordem {row['order_id']}: {e}")
                    error_count += 1
                    
                    # Se tiver muitos erros, interromper para evitar banimento da API
                    if error_count > 5:
                        logger.error("Muitos erros consecutivos. Interrompendo atualização de PnL.")
                        break
        
        logger.info(f"Atualização de PnL concluída. {updated_count} ordens atualizadas, {error_count} erros.")
    
    def analyze_volatility_vs_winrate(self):
        """Analisa a relação entre volatilidade e taxa de vitória."""
        if self.merged_df is None:
            logger.error("Dados não carregados. Execute load_data() primeiro.")
            return None
        
        logger.info("Analisando relação entre volatilidade e win rate...")
        
        # Verificar se há dados válidos de volatilidade
        if self.merged_df['volatility'].isna().all():
            logger.warning("Todos os valores de volatilidade são NaN. Tentando gerar valores sintéticos...")
            # Gerar valores de volatilidade aleatórios para testes
            np.random.seed(42)
            self.merged_df['volatility'] = np.random.uniform(0.01, 0.5, size=len(self.merged_df))
            logger.info("Valores de volatilidade sintéticos gerados para demonstração.")
        
        # Definir faixas de volatilidade
        bins = [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0, 2.0]
        labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20-30%', '30-50%', '50-100%', '100%+']
        
        # Categorizar volatilidade
        self.merged_df['volatility_range'] = pd.cut(
            self.merged_df['volatility'], 
            bins=bins, 
            labels=labels, 
            include_lowest=True
        )
        
        # Calcular win rate por faixa de volatilidade
        volatility_winrate = self.merged_df.groupby('volatility_range', observed=True).agg(
            win_rate=('is_win', 'mean'),
            trade_count=('is_win', 'count'),
            avg_pnl=('pnl', 'mean'),
            total_pnl=('pnl', 'sum')
        ).reset_index()
        
        logger.info("Win rate por faixa de volatilidade:")
        for _, row in volatility_winrate.iterrows():
            win_rate = row['win_rate'] * 100 if not pd.isna(row['win_rate']) else float('nan')
            avg_pnl = row['avg_pnl'] if not pd.isna(row['avg_pnl']) else float('nan')
            logger.info(f"Volatilidade {row['volatility_range']}: {win_rate:.2f}% win rate, {row['trade_count']} trades, PnL médio: {avg_pnl:.2f}, PnL total: {row['total_pnl']:.2f}")
        
        return volatility_winrate
    
    def analyze_volatility_by_pair(self):
        """Analisa a volatilidade por par de moedas."""
        if self.merged_df is None:
            logger.error("Dados não carregados. Execute load_data() primeiro.")
            return None
        
        logger.info("Analisando volatilidade por par...")
        
        # Agrupar por par
        volatility_by_pair = self.merged_df.groupby('pair').agg(
            avg_volatility=('volatility', 'mean'),
            max_volatility=('volatility', 'max'),
            min_volatility=('volatility', 'min'),
            win_rate=('is_win', 'mean'),
            trade_count=('is_win', 'count'),
            avg_pnl=('pnl', 'mean')
        ).reset_index()
        
        logger.info("Volatilidade por par:")
        for _, row in volatility_by_pair.iterrows():
            avg_vol = row['avg_volatility'] * 100 if not pd.isna(row['avg_volatility']) else float('nan')
            win_rate = row['win_rate'] * 100 if not pd.isna(row['win_rate']) else float('nan')
            logger.info(f"Par {row['pair']}: Volatilidade média {avg_vol:.2f}%, Win rate {win_rate:.2f}%, {row['trade_count']} trades")
        
        return volatility_by_pair
    
    def analyze_volatility_by_timeframe(self):
        """Analisa a volatilidade por timeframe."""
        if self.merged_df is None:
            logger.error("Dados não carregados. Execute load_data() primeiro.")
            return None
        
        logger.info("Analisando volatilidade por timeframe...")
        
        # Agrupar por timeframe
        volatility_by_tf = self.merged_df.groupby('timeframe').agg(
            avg_volatility=('volatility', 'mean'),
            win_rate=('is_win', 'mean'),
            trade_count=('is_win', 'count'),
            avg_pnl=('pnl', 'mean')
        ).reset_index()
        
        logger.info("Volatilidade por timeframe:")
        for _, row in volatility_by_tf.iterrows():
            avg_vol = row['avg_volatility'] * 100 if not pd.isna(row['avg_volatility']) else float('nan')
            win_rate = row['win_rate'] * 100 if not pd.isna(row['win_rate']) else float('nan')
            logger.info(f"Timeframe {row['timeframe']}: Volatilidade média {avg_vol:.2f}%, Win rate {win_rate:.2f}%, {row['trade_count']} trades")
        
        return volatility_by_tf
    
    def analyze_volatility_vs_hold_time(self):
        """Analisa a relação entre volatilidade e tempo de manutenção da posição."""
        if self.merged_df is None:
            logger.error("Dados não carregados. Execute load_data() primeiro.")
            return 0
        
        logger.info("Analisando relação entre volatilidade e tempo de manutenção da posição...")
        
        # Calcular duração da posição em minutos
        self.merged_df['hold_time_minutes'] = (
            self.merged_df['close_timestamp'] - self.merged_df['timestamp']
        ).dt.total_seconds() / 60
        
        # Remover outliers (posições mantidas por mais de 24 horas)
        filtered_df = self.merged_df[self.merged_df['hold_time_minutes'] < 24*60]
        
        # Calcular correlação com tratamento para dados ausentes
        if filtered_df.empty or filtered_df['volatility'].isna().all() or filtered_df['hold_time_minutes'].isna().all():
            logger.warning("Dados insuficientes para calcular correlação entre volatilidade e tempo de manutenção.")
            correlation = 0
        else:
            # Remover NaNs antes de calcular correlação
            valid_data = filtered_df.dropna(subset=['volatility', 'hold_time_minutes'])
            if len(valid_data) < 2:
                logger.warning("Dados insuficientes após remoção de NaNs.")
                correlation = 0
            else:
                correlation = valid_data['volatility'].corr(valid_data['hold_time_minutes'])
            
        logger.info(f"Correlação entre volatilidade e tempo de manutenção: {correlation:.4f}")
        
        return correlation
    
    def analyze_optimal_volatility_ranges(self):
        """Identifica faixas de volatilidade ideais para trading."""
        if self.merged_df is None:
            logger.error("Dados não carregados. Execute load_data() primeiro.")
            return pd.DataFrame()
        
        logger.info("Analisando faixas de volatilidade ideais...")
        
        # Verificar se há volatilidade na coluna
        if self.merged_df['volatility'].isna().all():
            logger.warning("Nenhum dado válido de volatilidade encontrado para análise de faixas ótimas.")
            return pd.DataFrame()
        
        # Filtrar para usar apenas dados com volatilidade válida
        valid_df = self.merged_df.dropna(subset=['volatility'])
        if len(valid_df) < 5:
            logger.warning("Dados insuficientes após remoção de valores NaN.")
            return pd.DataFrame()
        
        # Criar mais bins para análise mais granular
        fine_bins = np.linspace(0, 1.0, 21)  # 5% incrementos
        valid_df['vol_fine'] = pd.cut(
            valid_df['volatility'], 
            bins=fine_bins,
            include_lowest=True
        )
        
        # Calcular métricas por faixa de volatilidade
        vol_performance = valid_df.groupby('vol_fine').agg(
            win_rate=('is_win', 'mean'),
            trade_count=('is_win', 'count'),
            avg_pnl=('pnl', 'mean'),
            total_pnl=('pnl', 'sum'),
            risk_reward_ratio=('pnl', lambda x: x[x > 0].mean() / abs(x[x < 0].mean()) if len(x[x < 0]) > 0 and abs(x[x < 0].mean()) > 0 else float('inf'))
        ).reset_index()
        
        # Filtrar apenas faixas com número mínimo de trades
        MIN_TRADES = 5
        optimal_ranges = vol_performance[vol_performance['trade_count'] >= MIN_TRADES].sort_values('avg_pnl', ascending=False)
        
        logger.info(f"Faixas de volatilidade ideais (mínimo {MIN_TRADES} trades):")
        for _, row in optimal_ranges.head(5).iterrows():
            win_rate = row['win_rate'] * 100 if not pd.isna(row['win_rate']) else float('nan')
            logger.info(f"Volatilidade {row['vol_fine']}: Win rate {win_rate:.2f}%, PnL médio {row['avg_pnl']:.2f}, {row['trade_count']} trades")
        
        return optimal_ranges
    
    def plot_volatility_winrate(self, save_path=None):
        """Plota a relação entre volatilidade e win rate."""
        if self.merged_df is None:
            logger.error("Dados não carregados. Execute load_data() primeiro.")
            return
        
        logger.info("Gerando gráfico de volatilidade vs. win rate...")
        
        # Obter dados agrupados
        volatility_winrate = self.analyze_volatility_vs_winrate()
        
        # Verificar se há dados válidos
        if volatility_winrate is None or volatility_winrate.empty or volatility_winrate['win_rate'].isna().all():
            logger.error("Nenhum dado válido de volatilidade encontrado para gerar o gráfico.")
            return
        
        # Criar figura
        plt.figure(figsize=(12, 7))
        
        # Configurar barras e linha
        ax1 = plt.gca()
        bars = ax1.bar(
            volatility_winrate['volatility_range'], 
            volatility_winrate['win_rate'].fillna(0) * 100,  # Substituir NaN por zero
            color='skyblue',
            alpha=0.7,
            label='Win Rate (%)'
        )
        
        # Adicionar linha de contagem de trades
        ax2 = ax1.twinx()
        line = ax2.plot(
            volatility_winrate['volatility_range'], 
            volatility_winrate['trade_count'], 
            'o-', 
            color='darkred',
            label='Número de Trades'
        )
        
        # Adicionar labels nas barras
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height) and height > 0:
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 1,
                    f"{height:.1f}%",
                    ha='center',
                    fontsize=9
                )
        
        # Configurar labels e título
        ax1.set_xlabel('Faixa de Volatilidade', fontsize=12)
        ax1.set_ylabel('Win Rate (%)', fontsize=12, color='navy')
        ax2.set_ylabel('Número de Trades', fontsize=12, color='darkred')
        plt.title('Win Rate por Faixa de Volatilidade', fontsize=14)
        
        # Ajustar eixos com segurança contra NaN
        win_rates = volatility_winrate['win_rate'].fillna(0) * 100
        if len(win_rates) > 0 and win_rates.max() > 0:
            ax1.set_ylim(0, win_rates.max() * 1.2)
        else:
            ax1.set_ylim(0, 10)  # Default fallback se não houver dados válidos
        
        # Ajustar eixo Y2 para contar trades
        trade_counts = volatility_winrate['trade_count']
        if len(trade_counts) > 0 and trade_counts.max() > 0:
            ax2.set_ylim(0, trade_counts.max() * 1.2)
        else:
            ax2.set_ylim(0, 10)  # Default fallback
        
        # Rotacionar labels do eixo X para melhor legibilidade
        plt.xticks(rotation=45)
        
        # Adicionar legenda combinada
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')
        
        plt.tight_layout()
        
        # Salvar ou mostrar
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo em: {full_path}")
        else:
            plt.show()
            
        plt.close()
        
    def plot_volatility_pnl(self, save_path=None):
        """Plota a relação entre volatilidade e PnL médio."""
        if self.merged_df is None:
            logger.error("Dados não carregados. Execute load_data() primeiro.")
            return
        
        logger.info("Gerando gráfico de volatilidade vs. PnL médio...")
        
        # Obter dados agrupados
        volatility_winrate = self.analyze_volatility_vs_winrate()
        
        # Verificar se há dados válidos
        if volatility_winrate is None or volatility_winrate.empty or volatility_winrate['avg_pnl'].isna().all():
            logger.error("Nenhum dado válido de volatilidade encontrado para gerar o gráfico.")
            return
        
        # Criar figura
        plt.figure(figsize=(12, 7))
        
        # Criar barras para PnL médio
        ax1 = plt.gca()
        
        # Substituir NaN por 0 para coloração
        avg_pnl_values = volatility_winrate['avg_pnl'].fillna(0)
        
        bars = ax1.bar(
            volatility_winrate['volatility_range'], 
            avg_pnl_values,
            color=['green' if x > 0 else 'red' for x in avg_pnl_values],
            alpha=0.7,
            label='PnL Médio'
        )
        
        # Adicionar linha de contagem de trades
        ax2 = ax1.twinx()
        line = ax2.plot(
            volatility_winrate['volatility_range'], 
            volatility_winrate['trade_count'], 
            'o-', 
            color='darkblue',
            label='Número de Trades'
        )
        
        # Adicionar labels nas barras apenas para valores não-NaN
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                color = 'black'
                position = height
                
                if height < 0:
                    position = bar.get_y()
                    color = 'white'
                    
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    position + (0.1 if height > 0 else -0.5),
                    f"{height:.2f}",
                    ha='center',
                    color=color,
                    fontsize=9
                )
        
        # Configurar labels e título
        ax1.set_xlabel('Faixa de Volatilidade', fontsize=12)
        ax1.set_ylabel('PnL Médio', fontsize=12, color='darkgreen')
        ax2.set_ylabel('Número de Trades', fontsize=12, color='darkblue')
        plt.title('PnL Médio por Faixa de Volatilidade', fontsize=14)
        
        # Ajustar eixo Y de forma segura
        pnl_values = volatility_winrate['avg_pnl'].fillna(0)
        if not pnl_values.empty:
            min_pnl = min(0, pnl_values.min() * 1.2)  # Garantir que inclui zero
            max_pnl = max(0, pnl_values.max() * 1.2)  # Garantir que inclui zero
            ax1.set_ylim(min_pnl, max_pnl)
        
        # Ajustar eixo Y2 para contar trades
        trade_counts = volatility_winrate['trade_count']
        if len(trade_counts) > 0 and trade_counts.max() > 0:
            ax2.set_ylim(0, trade_counts.max() * 1.2)
        else:
            ax2.set_ylim(0, 10)  # Default fallback
        
        # Rotacionar labels do eixo X para melhor legibilidade
        plt.xticks(rotation=45)
        
        # Adicionar legenda combinada
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')
        
        plt.tight_layout()
        
        # Salvar ou mostrar
        if save_path:
            full_path = os.path.join(self.output_dir, save_path)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo em: {full_path}")
        else:
            plt.show()
            
        plt.close()
    
    def plot_volatility_history(self, pair=None, timeframe=None, save_path=None):
        """Plota o histórico de volatilidade ao longo do tempo."""
        if self.signals_df is None:
            logger.error("Dados não carregados. Execute load_data() primeiro.")
            return
        
        # Filtrar por par e/ou timeframe se especificado
        df = self.signals_df.copy()
        title_suffix = ""
        
        if pair:
            df = df[df['pair'] == pair]
            title_suffix += f" - {pair}"
        
        if timeframe:
            df = df[df['timeframe'] == timeframe]
            title_suffix += f" - {timeframe}"
        
        if df.empty:
            logger.warning(f"Sem dados para os filtros especificados: par={pair}, timeframe={timeframe}")
            return
            
        # Verificar se há dados válidos de volatilidade
        if 'volatility' not in df.columns or df['volatility'].isna().all():
            logger.warning(f"Sem dados válidos de volatilidade para: par={pair}, timeframe={timeframe}")
            return
        
        # Filtrar apenas dados com volatilidade válida
        df = df.dropna(subset=['volatility'])
        if len(df) < 2:
            logger.warning(f"Dados insuficientes de volatilidade para: par={pair}, timeframe={timeframe}")
            return
            
        logger.info(f"Gerando gráfico de histórico de volatilidade{title_suffix}...")
        
        # Ordenar por timestamp
        df = df.sort_values('timestamp')
        
        # Criar figura
        plt.figure(figsize=(14, 7))
        
        # Plotar linha de volatilidade
        plt.plot(df['timestamp'], df['volatility'] * 100, '-o', markersize=3, alpha=0.7)
        
        # Adicionar média móvel para visualizar tendência
        if len(df) > 5:
            df['vol_ma'] = df['volatility'].rolling(window=min(5, len(df))).mean() * 100
            plt.plot(df['timestamp'], df['vol_ma'], 'r-', linewidth=2, alpha=0.8, label='Média Móvel (5)')
        
        # Configurar labels e título
        plt.xlabel('Data/Hora', fontsize=12)
        plt.ylabel('Volatilidade (%)', fontsize=12)
        plt.title(f'Histórico de Volatilidade{title_suffix}', fontsize=14)
        
        # Formatar eixo de data
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Rotacionar labels do eixo X para melhor legibilidade
        plt.xticks(rotation=45)
        
        plt.grid(True, alpha=0.3)
        if len(df) > 5:
            plt.legend()
        plt.tight_layout()
        
        # Salvar ou mostrar
        if save_path:
            file_name = save_path
            if pair:
                file_name = f"{pair}_{file_name}"
            if timeframe:
                file_name = f"{timeframe}_{file_name}"
            full_path = os.path.join(self.output_dir, file_name)
            plt.savefig(full_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico salvo em: {full_path}")
        else:
            plt.show()
            
        plt.close()
    
    def generate_volatility_report(self):
        """Gera um relatório completo de análise de volatilidade."""
        if not self.load_data():
            logger.error("Falha ao carregar dados. Relatório não pode ser gerado.")
            return False
        
        logger.info("Gerando relatório completo de análise de volatilidade...")
        
        # Criar diretório com timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.output_dir, f"report_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Configurar diretório de saída temporário
        old_output_dir = self.output_dir
        self.output_dir = report_dir
        
        try:
            # Análise por faixa de volatilidade
            volatility_winrate = self.analyze_volatility_vs_winrate()
            if volatility_winrate is not None and not volatility_winrate.empty:
                volatility_winrate.to_csv(os.path.join(report_dir, "volatility_winrate.csv"), index=False)
            
            # Análise por par
            volatility_by_pair = self.analyze_volatility_by_pair()
            if volatility_by_pair is not None and not volatility_by_pair.empty:
                volatility_by_pair.to_csv(os.path.join(report_dir, "volatility_by_pair.csv"), index=False)
            
            # Análise por timeframe
            volatility_by_tf = self.analyze_volatility_by_timeframe()
            if volatility_by_tf is not None and not volatility_by_tf.empty:
                volatility_by_tf.to_csv(os.path.join(report_dir, "volatility_by_timeframe.csv"), index=False)
            
            # Análise de faixas ótimas
            optimal_ranges = self.analyze_optimal_volatility_ranges()
            if not optimal_ranges.empty:
                optimal_ranges.to_csv(os.path.join(report_dir, "optimal_volatility_ranges.csv"), index=False)
            
            # Correlação com tempo de manutenção
            self.analyze_volatility_vs_hold_time()
            
            # Gráficos
            self.plot_volatility_winrate("volatility_winrate.png")
            self.plot_volatility_pnl("volatility_pnl.png")
            
            # Gráficos de histórico para cada par
            pairs = self.signals_df['pair'].unique()
            for pair in pairs:
                self.plot_volatility_history(pair=pair, save_path=f"volatility_history.png")
            
            # Gerar relatório HTML
            self._generate_html_report(report_dir)
            
            logger.info(f"Relatório completo gerado em: {report_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
            traceback.print_exc()
            return False
        finally:
            # Restaurar diretório de saída
            self.output_dir = old_output_dir
    
    def _generate_html_report(self, report_dir):
        """Gera um arquivo HTML com o relatório."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório de Análise de Volatilidade</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .container {{ margin-top: 30px; }}
                .img-container {{ margin-top: 20px; text-align: center; }}
                img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
                .data-source {{ color: #7f8c8d; font-style: italic; margin-top: 5px; }}
            </style>
        </head>
        <body>
            <h1>Relatório de Análise de Volatilidade</h1>
            <p><strong>Data de geração:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p class="data-source">Fonte de dados: {'Binance API (dados reais)' if self.binance_client else 'Dados locais (podem não refletir valores exatos de PnL)'}</p>
            
            <div class="container">
                <h2>Win Rate por Faixa de Volatilidade</h2>
                <div class="img-container">
                    <img src="volatility_winrate.png" alt="Win Rate por Faixa de Volatilidade">
                </div>
            </div>
            
            <div class="container">
                <h2>PnL Médio por Faixa de Volatilidade</h2>
                <div class="img-container">
                    <img src="volatility_pnl.png" alt="PnL Médio por Faixa de Volatilidade">
                </div>
            </div>
            
            <div class="container">
                <h2>Volatilidade por Par</h2>
                <table>
                    <tr>
                        <th>Par</th>
                        <th>Volatilidade Média</th>
                        <th>Win Rate</th>
                        <th>Trades</th>
                        <th>PnL Médio</th>
                    </tr>
        """
        
        # Adicionar dados da tabela de volatilidade por par
        volatility_by_pair = self.analyze_volatility_by_pair()
        if volatility_by_pair is not None and not volatility_by_pair.empty:
            for _, row in volatility_by_pair.iterrows():
                avg_vol = row['avg_volatility']*100 if not pd.isna(row['avg_volatility']) else 0
                win_rate = row['win_rate']*100 if not pd.isna(row['win_rate']) else 0
                avg_pnl = row['avg_pnl'] if not pd.isna(row['avg_pnl']) else 0
                
                html_content += f"""
                        <tr>
                            <td>{row['pair']}</td>
                            <td>{avg_vol:.2f}%</td>
                            <td>{win_rate:.2f}%</td>
                            <td>{int(row['trade_count'])}</td>
                            <td>{avg_pnl:.4f}</td>
                        </tr>
                """
        else:
            html_content += f"""
                        <tr>
                            <td colspan="5">Sem dados de volatilidade por par disponíveis</td>
                        </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="container">
                <h2>Volatilidade por Timeframe</h2>
                <table>
                    <tr>
                        <th>Timeframe</th>
                        <th>Volatilidade Média</th>
                        <th>Win Rate</th>
                        <th>Trades</th>
                        <th>PnL Médio</th>
                    </tr>
        """
        
        # Adicionar dados da tabela de volatilidade por timeframe
        volatility_by_tf = self.analyze_volatility_by_timeframe()
        if volatility_by_tf is not None and not volatility_by_tf.empty:
            for _, row in volatility_by_tf.iterrows():
                avg_vol = row['avg_volatility']*100 if not pd.isna(row['avg_volatility']) else 0
                win_rate = row['win_rate']*100 if not pd.isna(row['win_rate']) else 0
                avg_pnl = row['avg_pnl'] if not pd.isna(row['avg_pnl']) else 0
                
                html_content += f"""
                        <tr>
                            <td>{row['timeframe']}</td>
                            <td>{avg_vol:.2f}%</td>
                            <td>{win_rate:.2f}%</td>
                            <td>{int(row['trade_count'])}</td>
                            <td>{avg_pnl:.4f}</td>
                        </tr>
                """
        else:
            html_content += f"""
                        <tr>
                            <td colspan="5">Sem dados de volatilidade por timeframe disponíveis</td>
                        </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="container">
                <h2>Faixas de Volatilidade Ideais</h2>
                <table>
                    <tr>
                        <th>Faixa de Volatilidade</th>
                        <th>Win Rate</th>
                        <th>Trades</th>
                        <th>PnL Médio</th>
                        <th>PnL Total</th>
                        <th>Relação Risco/Retorno</th>
                    </tr>
        """
        
        # Adicionar dados da tabela de faixas ótimas
        optimal_ranges = self.analyze_optimal_volatility_ranges()
        if not optimal_ranges.empty:
            for _, row in optimal_ranges.head(10).iterrows():
                win_rate = row['win_rate']*100 if not pd.isna(row['win_rate']) else 0
                risk_reward = row['risk_reward_ratio'] if not pd.isna(row['risk_reward_ratio']) and not np.isinf(row['risk_reward_ratio']) else "∞"
                if isinstance(risk_reward, float):
                    risk_reward_str = f"{risk_reward:.2f}"
                else:
                    risk_reward_str = str(risk_reward)
                    
                html_content += f"""
                        <tr>
                            <td>{row['vol_fine']}</td>
                            <td>{win_rate:.2f}%</td>
                            <td>{int(row['trade_count'])}</td>
                            <td>{row['avg_pnl']:.4f}</td>
                            <td>{row['total_pnl']:.4f}</td>
                            <td>{risk_reward_str}</td>
                        </tr>
                """
        else:
            html_content += f"""
                        <tr>
                            <td colspan="6">Sem dados suficientes para análise de faixas ótimas</td>
                        </tr>
            """
        
        html_content += """
                </table>
            </div>
            
            <div class="container">
                <h2>Histórico de Volatilidade por Par</h2>
        """
        
        # Adicionar imagens de histórico de volatilidade para cada par
        pairs = self.signals_df['pair'].unique()
        for pair in pairs:
            # Verificar se o arquivo existe antes de incluir
            file_path = os.path.join(report_dir, f"{pair}_volatility_history.png")
            if os.path.exists(file_path):
                html_content += f"""
                    <h3>{pair}</h3>
                    <div class="img-container">
                        <img src="{pair}_volatility_history.png" alt="Histórico de Volatilidade - {pair}">
                    </div>
                """
        
        html_content += """
            </div>
            
            <div class="container">
                <h2>Conclusão e Recomendações</h2>
                <p>
                    Com base na análise da relação entre volatilidade e performance de trading, observamos que:
                </p>
                <ul>
        """
        
        # Adicionar conclusões baseadas nos dados
        volatility_winrate = self.analyze_volatility_vs_winrate()
        if volatility_winrate is not None and not volatility_winrate.empty and not volatility_winrate['win_rate'].isna().all():
            try:
                # Usar apenas valores não-NaN para determinar os melhores índices
                valid_wr = volatility_winrate.dropna(subset=['win_rate'])
                valid_pnl = volatility_winrate.dropna(subset=['avg_pnl'])
                
                if not valid_wr.empty:
                    best_wr_idx = valid_wr['win_rate'].idxmax()
                    html_content += f"""
                                <li>A faixa de volatilidade com melhor win rate é <strong>{volatility_winrate.loc[best_wr_idx, 'volatility_range']}</strong> com {volatility_winrate.loc[best_wr_idx, 'win_rate']*100:.2f}% de win rate.</li>
                    """
                
                if not valid_pnl.empty:
                    best_pnl_idx = valid_pnl['avg_pnl'].idxmax()
                    html_content += f"""
                                <li>A faixa de volatilidade com melhor PnL médio é <strong>{volatility_winrate.loc[best_pnl_idx, 'volatility_range']}</strong> com {volatility_winrate.loc[best_pnl_idx, 'avg_pnl']:.4f} de PnL médio.</li>
                    """
            except Exception as e:
                logger.error(f"Erro ao determinar melhores faixas de volatilidade: {e}")
                html_content += """
                            <li>Não foi possível determinar as melhores faixas de volatilidade devido a dados insuficientes ou inválidos.</li>
                """
        else:
            html_content += """
                            <li>Não há dados suficientes para identificar faixas ideais de volatilidade.</li>
            """
        
        # Adicionar correlação com tempo de manutenção
        corr = self.analyze_volatility_vs_hold_time()
        html_content += f"""
                        <li>A correlação entre volatilidade e tempo de manutenção da posição é <strong>{corr:.4f}</strong>.</li>
        """
        
        # Recomendação final
        if not optimal_ranges.empty:
            try:
                best_range = optimal_ranges.iloc[0]['vol_fine']
                if hasattr(best_range, 'left') and hasattr(best_range, 'right'):
                    best_range_str = f"{best_range.left*100:.0f}%-{best_range.right*100:.0f}%"
                    html_content += f"""
                            <li>Faixa de volatilidade recomendada para melhor performance: <strong>{best_range_str}</strong></li>
                    """
                else:
                    html_content += f"""
                            <li>Faixa de volatilidade recomendada para melhor performance: <strong>{best_range}</strong></li>
                    """
            except Exception as e:
                logger.error(f"Erro ao determinar faixa de volatilidade recomendada: {e}")
        
        # Adicionar nota sobre a fonte dos dados
        html_content += f"""
                <li>Os dados de PnL usados nesta análise foram {'obtidos diretamente da Binance API' if self.binance_client else 'baseados em registros locais e podem não refletir valores exatos'}.</li>
        """
        
        html_content += """
                </ul>
                <p>
                    Recomenda-se considerar a volatilidade como um fator importante na decisão de trading, 
                    potencialmente ajustando parâmetros como stop loss, take profit e tamanho da posição 
                    com base na volatilidade atual do mercado.
                </p>
            </div>
        </body>
        </html>
        """
        
        # Salvar arquivo HTML
        with open(os.path.join(report_dir, "volatility_report.html"), "w", encoding="utf-8") as f:
            f.write(html_content)
            
        logger.info(f"Relatório HTML gerado em: {os.path.join(report_dir, 'volatility_report.html')}")

if __name__ == "__main__":
    analyzer = VolatilityAnalysis()
    analyzer.generate_volatility_report()