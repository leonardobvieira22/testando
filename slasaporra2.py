import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import os
import time
import traceback
import warnings
warnings.filterwarnings('ignore')

# Importando módulos para integração com Binance
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from utils import logger, config_logger
from config import *
from config import REAL_API_KEY, REAL_API_SECRET

# Configurar estilo de visualização
plt.style.use('ggplot')
sns.set(style="darkgrid")

class BinanceUtils:
    """Classe para manipulação da API da Binance"""
    
    def __init__(self, api_key=None, api_secret=None):
        """Inicializa o cliente Binance com as credenciais fornecidas ou do arquivo de configuração."""
        try:
            self.api_key = api_key or REAL_API_KEY
            self.api_secret = api_secret or REAL_API_SECRET
            self.client = Client(self.api_key, self.api_secret)
            self.test_connection()
            logger.info("Conexão com Binance API estabelecida com sucesso")
        except Exception as e:
            logger.error(f"Erro ao conectar com Binance API: {e}")
            raise
    
    def test_connection(self):
        """Testa a conexão com a API da Binance."""
        try:
            self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Falha na conexão com Binance API: {e}")
            return False
    
    def get_order(self, symbol, orderId):
        """Obtém detalhes de uma ordem pelo ID."""
        try:
            return self.client.get_order(symbol=symbol, orderId=orderId)
        except BinanceAPIException as e:
            logger.error(f"Erro na API Binance ao obter ordem {orderId}: {e}")
            return None
        except BinanceRequestException as e:
            logger.error(f"Erro de requisição Binance para ordem {orderId}: {e}")
            return None
        except Exception as e:
            logger.error(f"Erro desconhecido ao obter ordem {orderId}: {e}")
            return None
    
    def get_trades(self, symbol, orderId):
        """Obtém todas as execuções de negociação para uma ordem específica."""
        try:
            # Primeiro obtém informações da ordem
            order = self.get_order(symbol, orderId)
            if not order:
                return []
            
            # Usa a janela de tempo da ordem para limitar a consulta
            start_time = int(order.get('time', int(time.time() * 1000) - 86400000))
            end_time = start_time + 86400000  # +24h para garantir que pegamos todas as execuções
            
            # Obtém as execuções da ordem
            trades = self.client.get_my_trades(symbol=symbol, startTime=start_time, endTime=end_time)
            
            # Filtra apenas as execuções relacionadas a este orderId
            order_trades = [trade for trade in trades if str(trade.get('orderId')) == str(orderId)]
            return order_trades
        except BinanceAPIException as e:
            logger.error(f"Erro na API Binance ao obter trades da ordem {orderId}: {e}")
            return []
        except BinanceRequestException as e:
            logger.error(f"Erro de requisição Binance para trades da ordem {orderId}: {e}")
            return []
        except Exception as e:
            logger.error(f"Erro desconhecido ao obter trades da ordem {orderId}: {e}")
            return []
    
    def get_position_pnl(self, entry_order_id, exit_order_id, symbol):
        """
        Calcula o PnL realizado para uma posição completa,
        considerando ordens de entrada e saída e incluindo taxas.
        """
        try:
            # Verificar argumentos
            if not entry_order_id or not exit_order_id or not symbol:
                logger.warning(f"Parâmetros incompletos: entry_order_id={entry_order_id}, exit_order_id={exit_order_id}, symbol={symbol}")
                return None
            
            # Obter detalhes das ordens e trades
            entry_trades = self.get_trades(symbol, entry_order_id)
            exit_trades = self.get_trades(symbol, exit_order_id)
            
            if not entry_trades or not exit_trades:
                logger.warning(f"Não foi possível obter trades para ordens: {entry_order_id} ou {exit_order_id}")
                return None
            
            # Calcular valores agregados de entrada
            entry_qty = sum(float(trade['qty']) for trade in entry_trades)
            entry_cost = sum(float(trade['qty']) * float(trade['price']) for trade in entry_trades)
            entry_fee = sum(float(trade['commission']) for trade in entry_trades)
            
            # Calcular valores agregados de saída
            exit_qty = sum(float(trade['qty']) for trade in exit_trades)
            exit_value = sum(float(trade['qty']) * float(trade['price']) for trade in exit_trades)
            exit_fee = sum(float(trade['commission']) for trade in exit_trades)
            
            # Calcular PnL
            # Para compras: PnL = (valor_saída - custo_entrada) - (taxa_entrada + taxa_saída)
            # Para vendas: PnL = (custo_entrada - valor_saída) - (taxa_entrada + taxa_saída)
            side = entry_trades[0].get('isBuyer', True)  # True para compra, False para venda
            
            if side:  # Compra
                pnl = (exit_value - entry_cost) - (entry_fee + exit_fee)
            else:  # Venda
                pnl = (entry_cost - exit_value) - (entry_fee + exit_fee)
            
            return pnl
        except Exception as e:
            logger.error(f"Erro ao calcular PnL para ordens {entry_order_id}/{exit_order_id}: {e}")
            return None
    
    def calculate_order_pnl(self, row):
        """
        Calcula o PnL para uma linha de dados que representa uma posição completa.
        Usa os dados da ordem e preços de entrada/saída e desconta taxas.
        """
        try:
            # Verificar dados necessários
            if pd.isna(row['entry_price']) or pd.isna(row['close_price']) or pd.isna(row['quantity']):
                return 0.0
            
            # Calcular diferença básica de preço
            price_diff = float(row['close_price']) - float(row['entry_price'])
            
            # Ajustar para direção da operação
            if row['direction'] == 'SELL':
                price_diff = -price_diff
                
            # Calcular PnL bruto
            gross_pnl = price_diff * float(row['quantity'])
            
            # Descontar taxas (se disponíveis)
            entry_fee = float(row['entry_fee']) if not pd.isna(row['entry_fee']) else 0.0
            exit_fee = float(row['exit_fee']) if not pd.isna(row['exit_fee']) else 0.0
            
            # PnL líquido
            net_pnl = gross_pnl - entry_fee - exit_fee
            
            return net_pnl
        except Exception as e:
            logger.error(f"Erro ao calcular PnL para linha: {e}\nDados: {row}")
            return 0.0

class VolatilityAnalysis:
    """Classe para análise e visualização da relação entre volatilidade e resultados de trading."""
    
    def __init__(self, signals_csv=SIGNALS_CSV, orders_pnl_csv=ORDERS_PNL_CSV, use_binance_api=True):
        """Inicializa o analisador de volatilidade."""
        self.signals_csv = signals_csv
        self.orders_pnl_csv = orders_pnl_csv
        self.use_binance_api = use_binance_api
        
        self.signals_df = None
        self.orders_df = None
        self.merged_df = None
        self.output_dir = "volatility_analysis"
        self.api_calls_count = 0
        self.pnl_cache = {}
        
        # Inicializar cliente Binance se necessário
        if use_binance_api:
            try:
                self.binance_utils = BinanceUtils()
                logger.info("Cliente Binance inicializado com sucesso")
            except Exception as e:
                logger.error(f"Erro ao inicializar cliente Binance: {e}")
                self.binance_utils = None
                self.use_binance_api = False
                logger.warning("Análise prosseguirá sem consulta à API Binance")
        else:
            logger.info("API Binance não será utilizada para esta análise")
            self.binance_utils = None
        
        # Criar diretório para outputs se não existir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logger.info(f"Diretório criado: {self.output_dir}")
    
    def _respect_api_limits(self):
        """
        Controla as chamadas à API para respeitar os limites da Binance.
        Força uma pausa se atingir muitas chamadas em um curto período.
        """
        self.api_calls_count += 1
        
        # Limitar a 10 chamadas por segundo
        if self.api_calls_count % 8 == 0:
            time.sleep(1)  # Pausa para respeitar limites de taxa
            
        # Pausa maior a cada 100 chamadas
        if self.api_calls_count % 100 == 0:
            logger.info(f"{self.api_calls_count} chamadas de API realizadas. Pausando para evitar limites.")
            time.sleep(5)
    
    def load_data(self):
        """Carrega dados de sinais e ordens finalizadas."""
        try:
            logger.info(f"Carregando dados de {self.signals_csv}...")
            
            # Verificar se o arquivo existe
            if not os.path.exists(self.signals_csv):
                logger.error(f"Arquivo de sinais não encontrado: {self.signals_csv}")
                return False
            
            # Ler o arquivo de sinais
            self.signals_df = pd.read_csv(self.signals_csv)
            logger.info(f"Carregados {len(self.signals_df)} sinais")
            
            # Verificar se a coluna de volatilidade existe
            if 'volatility' not in self.signals_df.columns:
                logger.warning("Coluna 'volatility' não encontrada nos dados de sinais.")
                # Se tivermos dados suficientes, podemos tentar calcular a volatilidade
                if 'pair' in self.signals_df.columns and 'timestamp' in self.signals_df.columns:
                    logger.info("Tentando calcular volatilidade a partir dos dados disponíveis...")
                    self._calculate_volatility()
            
            # Converter timestamp para datetime
            if 'timestamp' in self.signals_df.columns:
                self.signals_df['timestamp'] = pd.to_datetime(self.signals_df['timestamp'])
            
            # Verificar se o arquivo de ordens existe
            if not os.path.exists(self.orders_pnl_csv):
                logger.error(f"Arquivo de ordens não encontrado: {self.orders_pnl_csv}")
                return False
                
            # Carregar dados de ordens
            logger.info(f"Carregando dados de {self.orders_pnl_csv}...")
            self.orders_df = pd.read_csv(self.orders_pnl_csv)
            logger.info(f"Carregadas {len(self.orders_df)} ordens")
            
            # Converter timestamps para datetime
            if 'timestamp' in self.orders_df.columns:
                self.orders_df['timestamp'] = pd.to_datetime(self.orders_df['timestamp'])
            if 'close_timestamp' in self.orders_df.columns:
                self.orders_df['close_timestamp'] = pd.to_datetime(self.orders_df['close_timestamp'])
            
            # Verificar se temos IDs de ordem
            if 'order_id' not in self.orders_df.columns:
                logger.warning("Coluna 'order_id' não encontrada. Análise de PnL será limitada.")
            
            # Mesclar dados de sinais e ordens
            logger.info("Mesclando dados de sinais e ordens...")
            self.merged_df = self.orders_df.copy()
            
            # Se tivermos signal_id em ambos, mesclar para obter informações adicionais
            if 'signal_id' in self.orders_df.columns and 'signal_id' in self.signals_df.columns:
                signal_cols = [col for col in self.signals_df.columns if col not in self.orders_df.columns or col == 'signal_id']
                self.merged_df = self.orders_df.merge(
                    self.signals_df[signal_cols],
                    on='signal_id',
                    how='left',
                    suffixes=('', '_signal')
                )
                logger.info(f"Mesclagem completa: {len(self.merged_df)} registros")
            else:
                logger.warning("Mesclagem por signal_id não possível. Usando apenas dados de ordens.")
            
            # Se a volatilidade estiver faltando, copiar da coluna volatility nos dados de ordens (se existir)
            if 'volatility' in self.orders_df.columns and ('volatility' not in self.merged_df.columns or self.merged_df['volatility'].isna().all()):
                self.merged_df['volatility'] = self.orders_df['volatility']
            
            # Calcular PnL e win rate precisos com base na API da Binance, se disponível
            if self.use_binance_api and self.binance_utils:
                logger.info("Atualizando PnL e win rate usando dados da Binance API...")
                self._update_pnl_from_binance()
            else:
                # Caso contrário, calcular PnL usando dados locais
                logger.info("Calculando PnL usando dados locais...")
                self._calculate_local_pnl()
            
            # Calcular se a ordem foi vitoriosa (PnL > 0)
            self.merged_df['is_win'] = self.merged_df['pnl'] > 0
            
            # Garantir que volatilidade é numérica
            if 'volatility' in self.merged_df.columns:
                self.merged_df['volatility'] = pd.to_numeric(self.merged_df['volatility'], errors='coerce')
                
                # Verificar dados de volatilidade
                nan_count = self.merged_df['volatility'].isna().sum()
                pct_missing = nan_count / len(self.merged_df) * 100 if len(self.merged_df) > 0 else 0
                logger.info(f"Volatilidade: {len(self.merged_df)} entradas, {nan_count} valores NaN ({pct_missing:.2f}%)")
                
                # Se todos forem NaN, tentar imputar valores
                if pct_missing > 75:
                    logger.warning(f"Alta proporção de dados de volatilidade ausentes ({pct_missing:.2f}%). Tentando reparar...")
                    self._impute_volatility()
            else:
                logger.warning("Sem dados de volatilidade disponíveis. Tentando gerar automaticamente...")
                self._impute_volatility()
            
            logger.info(f"Carregamento de dados concluído: {len(self.merged_df)} registros prontos para análise")
            return True
        
        except Exception as e:
            logger.error(f"Erro ao carregar dados: {str(e)}")
            traceback.print_exc()
            return False
    
    def _update_pnl_from_binance(self):
        """Atualiza os dados de PnL e win rate usando a API da Binance."""
        if not self.binance_utils or not self.use_binance_api:
            logger.warning("Cliente Binance não disponível para atualizar PnL")
            return
        
        updated_count = 0
        error_count = 0
        
        # Criar colunas para armazenar valores originais
        if 'original_pnl' not in self.merged_df.columns:
            self.merged_df['original_pnl'] = self.merged_df['pnl']
        
        for idx, row in self.merged_df.iterrows():
            # Verificar se temos os dados necessários
            if pd.isna(row['order_id']) or pd.isna(row['pair']):
                continue
            
            # Verificar se já calculamos este PnL antes (usando cache)
            cache_key = f"{row['pair']}_{row['order_id']}"
            if cache_key in self.pnl_cache:
                self.merged_df.at[idx, 'pnl'] = self.pnl_cache[cache_key]
                updated_count += 1
                continue
            
            try:
                # Respeitar limites de API
                self._respect_api_limits()
                
                # Se tivermos o ID da ordem de fechamento, usamos ele para cálculo mais preciso
                if not pd.isna(row['close_order_id']):
                    pnl = self.binance_utils.get_position_pnl(
                        entry_order_id=int(row['order_id']),
                        exit_order_id=int(row['close_order_id']),
                        symbol=row['pair']
                    )
                else:
                    # Se não tivermos ordem de fechamento, calcular usando preços
                    pnl = self.binance_utils.calculate_order_pnl(row)
                
                # Atualizar PnL se for um valor válido
                if pnl is not None:
                    self.merged_df.at[idx, 'pnl'] = pnl
                    self.pnl_cache[cache_key] = pnl
                    updated_count += 1
            
            except Exception as e:
                error_count += 1
                logger.error(f"Erro ao calcular PnL para {row['pair']} ordem {row['order_id']}: {e}")
                
                # Se tivermos muitos erros consecutivos, pausar ou interromper
                if error_count > 10:
                    logger.error("Muitos erros consecutivos. Interrompendo atualização de PnL da Binance.")
                    break
        
        logger.info(f"Atualização de PnL concluída: {updated_count} ordens atualizadas, {error_count} erros")
        
        # Calcular campo is_win novamente com os novos valores de PnL
        self.merged_df['is_win'] = self.merged_df['pnl'] > 0
    
    def _calculate_local_pnl(self):
        """Calcula PnL usando apenas dados locais quando a API não está disponível."""
        for idx, row in self.merged_df.iterrows():
            try:
                # Verificar se temos os dados necessários
                if pd.isna(row['entry_price']) or pd.isna(row['close_price']) or pd.isna(row['quantity']):
                    continue
                
                # Calcular diferença básica de preço
                price_diff = float(row['close_price']) - float(row['entry_price'])
                
                # Ajustar para direção da operação
                if 'direction' in row and row['direction'] == 'SELL':
                    price_diff = -price_diff
                    
                # Calcular PnL bruto
                gross_pnl = price_diff * float(row['quantity'])
                
                # Descontar taxas (se disponíveis)
                entry_fee = float(row['entry_fee']) if 'entry_fee' in row and not pd.isna(row['entry_fee']) else 0.0
                exit_fee = float(row['exit_fee']) if 'exit_fee' in row and not pd.isna(row['exit_fee']) else 0.0
                
                # PnL líquido
                net_pnl = gross_pnl - entry_fee - exit_fee
                
                # Atualizar o valor de PnL
                self.merged_df.at[idx, 'pnl'] = net_pnl
                
            except Exception as e:
                logger.error(f"Erro ao calcular PnL local para linha {idx}: {e}")
    
    def _calculate_volatility(self):
        """
        Calcula a volatilidade para pares que não têm esse valor.
        Usa dados históricos de OHLCV se disponíveis.
        """
        # Implementação simplificada - idealmente, buscaria dados OHLCV para cálculo preciso
        logger.warning("Cálculo de volatilidade não implementado. Usando valores simulados para testes.")
        
        # Usar valores pré-definidos típicos por par
        volatility_map = {
            'BTCUSDT': 0.03,  # 3%
            'ETHUSDT': 0.04,  # 4%
            'DOGEUSDT': 0.08,  # 8%
            'TRUMPUSDT': 0.12,  # 12%
            'SHIBUSDT': 0.10,  # 10%
        }
        
        # Aplicar valores base e adicionar variação
        if self.signals_df is not None and 'pair' in self.signals_df.columns:
            np.random.seed(42)  # Para resultados reproduzíveis
            
            self.signals_df['volatility'] = self.signals_df['pair'].apply(
                lambda x: volatility_map.get(x, 0.05) * (1 + np.random.uniform(-0.5, 1.5))
            )
    
    def _impute_volatility(self):
        """
        Imputa valores de volatilidade quando estão faltando.
        Usa valores típicos por par ou timeframe quando possível.
        """
        # Se não tivermos coluna de volatilidade, criar uma
        if 'volatility' not in self.merged_df.columns:
            self.merged_df['volatility'] = np.nan
        
        # Valores típicos por par
        pair_volatility = {
            'BTCUSDT': 0.03,
            'ETHUSDT': 0.04,
            'DOGEUSDT': 0.08,
            'TRUMPUSDT': 0.12,
        }
        
        # Valores típicos por timeframe
        timeframe_volatility = {
            '1m': 0.02,
            '5m': 0.03,
            '15m': 0.04,
            '30m': 0.05,
            '1h': 0.06,
            '4h': 0.08,
            '1d': 0.10,
        }
        
        # Iterar sobre linhas e imputar valores faltantes
        for idx, row in self.merged_df.iterrows():
            if pd.isna(row['volatility']):
                # Tentar usar o valor do par
                if 'pair' in row and row['pair'] in pair_volatility:
                    base_vol = pair_volatility[row['pair']]
                # Senão, tentar usar o valor do timeframe
                elif 'timeframe' in row and row['timeframe'] in timeframe_volatility:
                    base_vol = timeframe_volatility[row['timeframe']]
                # Senão, usar valor padrão
                else:
                    base_vol = 0.05
                
                # Adicionar variação aleatória para maior realismo
                np.random.seed(int(idx) + 42)  # Seed variável mas reproduzível
                volatility = base_vol * (1 + np.random.uniform(-0.3, 0.7))
                
                # Atualizar o valor
                self.merged_df.at[idx, 'volatility'] = volatility
        
        logger.info(f"Imputação de volatilidade concluída. {self.merged_df['volatility'].isna().sum()} valores NaN restantes.")
    
    def analyze_volatility_vs_winrate(self):
        """Analisa a relação entre volatilidade e taxa de vitória."""
        if self.merged_df is None:
            logger.error("Dados não carregados. Execute load_data() primeiro.")
            return None
        
        logger.info("Analisando relação entre volatilidade e win rate...")
        
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
        
        # Registrar resultados
        logger.info("Win rate por faixa de volatilidade:")
        for _, row in volatility_winrate.iterrows():
            win_rate = row['win_rate'] * 100 if not pd.isna(row['win_rate']) else 0
            avg_pnl = row['avg_pnl'] if not pd.isna(row['avg_pnl']) else 0
            logger.info(f"Volatilidade {row['volatility_range']}: {win_rate:.2f}% win rate, {row['trade_count']} trades, PnL médio: {avg_pnl:.4f}")
        
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
            avg_pnl=('pnl', 'mean'),
            total_pnl=('pnl', 'sum')
        ).reset_index()
        
        # Registrar resultados
        logger.info("Volatilidade por par:")
        for _, row in volatility_by_pair.iterrows():
            win_rate = row['win_rate'] * 100 if not pd.isna(row['win_rate']) else 0
            avg_volatility = row['avg_volatility'] * 100 if not pd.isna(row['avg_volatility']) else 0
            logger.info(f"Par {row['pair']}: Volatilidade média {avg_volatility:.2f}%, Win rate {win_rate:.2f}%, {row['trade_count']} trades")
        
        return volatility_by_pair
    
    def analyze_volatility_by_timeframe(self):
        """Analisa a volatilidade por timeframe."""
        if self.merged_df is None:
            logger.error("Dados não carregados. Execute load_data() primeiro.")
            return None
        
        # Verificar se temos a coluna timeframe
        if 'timeframe' not in self.merged_df.columns:
            logger.warning("Coluna 'timeframe' não encontrada nos dados.")
            return None
        
        logger.info("Analisando volatilidade por timeframe...")
        
        # Agrupar por timeframe
        volatility_by_tf = self.merged_df.groupby('timeframe').agg(
            avg_volatility=('volatility', 'mean'),
            win_rate=('is_win', 'mean'),
            trade_count=('is_win', 'count'),
            avg_pnl=('pnl', 'mean'),
            total_pnl=('pnl', 'sum')
        ).reset_index()
        
        # Registrar resultados
        logger.info("Volatilidade por timeframe:")
        for _, row in volatility_by_tf.iterrows():
            win_rate = row['win_rate'] * 100 if not pd.isna(row['win_rate']) else 0
            avg_volatility = row['avg_volatility'] * 100 if not pd.isna(row['avg_volatility']) else 0
            logger.info(f"Timeframe {row['timeframe']}: Volatilidade média {avg_volatility:.2f}%, Win rate {win_rate:.2f}%, {row['trade_count']} trades")
        
        return volatility_by_tf
    
    def analyze_volatility_vs_hold_time(self):
        """Analisa a relação entre volatilidade e tempo de manutenção da posição."""
        if self.merged_df is None:
            logger.error("Dados não carregados. Execute load_data() primeiro.")
            return 0
        
        # Verificar se temos as colunas necessárias
        if 'timestamp' not in self.merged_df.columns or 'close_timestamp' not in self.merged_df.columns:
            logger.warning("Colunas de timestamp necessárias não encontradas.")
            return 0
        
        logger.info("Analisando relação entre volatilidade e tempo de manutenção da posição...")
        
        # Calcular duração da posição em minutos
        self.merged_df['hold_time_minutes'] = (
            self.merged_df['close_timestamp'].dt.tz_localize(None) - 
            self.merged_df['timestamp'].dt.tz_localize(None)
        ).dt.total_seconds() / 60
        
        # Remover outliers (posições mantidas por mais de 24 horas ou valores negativos)
        valid_df = self.merged_df[(self.merged_df['hold_time_minutes'] < 24*60) & 
                                 (self.merged_df['hold_time_minutes'] > 0)]
        
        # Verificar se há dados suficientes
        if len(valid_df) < 5:
            logger.warning("Dados insuficientes para análise de tempo de manutenção após filtragem.")
            return 0
        
        # Calcular correlação
        correlation = valid_df['volatility'].corr(valid_df['hold_time_minutes'])
        
        logger.info(f"Correlação entre volatilidade e tempo de manutenção: {correlation:.4f}")
        
        # Retornar resultados adicionais como dicionário
        results = {
            'correlation': correlation,
            'avg_hold_time': valid_df['hold_time_minutes'].mean(),
            'median_hold_time': valid_df['hold_time_minutes'].median(),
            'min_hold_time': valid_df['hold_time_minutes'].min(),
            'max_hold_time': valid_df['hold_time_minutes'].max(),
        }
        
        # Registrar estatísticas de tempo
        logger.info(f"Tempo médio de manutenção: {results['avg_hold_time']:.2f} minutos")
        logger.info(f"Tempo mediano de manutenção: {results['median_hold_time']:.2f} minutos")
        
        return correlation
    
    def analyze_optimal_volatility_ranges(self, min_trades=3):
        """Identifica faixas de volatilidade ideais para trading."""
        if self.merged_df is None:
            logger.error("Dados não carregados. Execute load_data() primeiro.")
            return pd.DataFrame()
        
        logger.info(f"Analisando faixas de volatilidade ideais (mínimo {min_trades} trades)...")
        
        # Verificar se há dados válidos
        valid_df = self.merged_df.dropna(subset=['volatility', 'pnl'])
        if len(valid_df) < min_trades:
            logger.warning("Dados insuficientes para análise de faixas ótimas.")
            return pd.DataFrame()
        
        # Criar bins mais granulares para análise detalhada
        fine_bins = np.linspace(0, 0.5, 21)  # 2.5% incrementos até 50% de volatilidade
        fine_labels = [f"{bin_start*100:.1f}%-{bin_end*100:.1f}%" for bin_start, bin_end 
                      in zip(fine_bins[:-1], fine_bins[1:])]
        
        valid_df['vol_fine'] = pd.cut(
            valid_df['volatility'], 
            bins=fine_bins,
            labels=fine_labels,
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
        
        # Calcular sharpe ratio simplificado (retorno/volatilidade)
        vol_performance['sharpe_ratio'] = vol_performance.apply(
            lambda row: row['avg_pnl'] / (row['avg_pnl'].std() if hasattr(row['avg_pnl'], 'std') else 1) 
            if row['trade_count'] > 1 else 0, 
            axis=1
        )
        
        # Filtrar apenas faixas com número mínimo de trades
        optimal_ranges = vol_performance[vol_performance['trade_count'] >= min_trades].sort_values('avg_pnl', ascending=False)
        
        # Registrar as melhores faixas
        logger.info(f"Top 5 faixas de volatilidade (mín. {min_trades} trades):")
        for i, (_, row) in enumerate(optimal_ranges.head(5).iterrows(), 1):
            win_rate = row['win_rate'] * 100 if not pd.isna(row['win_rate']) else 0
            logger.info(f"{i}. {row['vol_fine']}: Win rate {win_rate:.2f}%, PnL médio {row['avg_pnl']:.4f}, {row['trade_count']} trades")
        
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
        if volatility_winrate is None or volatility_winrate.empty:
            logger.error("Nenhum dado válido de volatilidade encontrado para gerar o gráfico.")
            return
        
        # Criar figura
        plt.figure(figsize=(12, 7))
        
        # Configurar barras e linha
        ax1 = plt.gca()
        bars = ax1.bar(
            volatility_winrate['volatility_range'], 
            volatility_winrate['win_rate'] * 100,
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
        
        # Ajustar eixos
        max_win_rate = volatility_winrate['win_rate'].max() * 100
        if not np.isnan(max_win_rate) and max_win_rate > 0:
            ax1.set_ylim(0, max_win_rate * 1.2)
        else:
            ax1.set_ylim(0, 100)
        
        # Ajustar eixo Y2 para contar trades
        max_trades = volatility_winrate['trade_count'].max()
        if not np.isnan(max_trades) and max_trades > 0:
            ax2.set_ylim(0, max_trades * 1.2)
        
        # Rotacionar labels do eixo X para melhor legibilidade
        plt.xticks(rotation=45)
        
        # Adicionar legenda combinada
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')
        
        # Adicionar grid
        ax1.grid(True, alpha=0.3)
        
        # Adicionar nota sobre a fonte de dados
        plt.figtext(
            0.5, 0.01, 
            f"Fonte: {'Binance API (valores reais)' if self.use_binance_api else 'Dados locais'}",
            ha='center', fontsize=8, style='italic'
        )
        
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
        if volatility_winrate is None or volatility_winrate.empty:
            logger.error("Nenhum dado válido de volatilidade encontrado para gerar o gráfico.")
            return
        
        # Criar figura
        plt.figure(figsize=(12, 7))
        
        # Criar barras para PnL médio
        ax1 = plt.gca()
        
        # Cores baseadas no valor de PnL
        bar_colors = ['green' if x > 0 else 'red' for x in volatility_winrate['avg_pnl']]
        
        bars = ax1.bar(
            volatility_winrate['volatility_range'], 
            volatility_winrate['avg_pnl'],
            color=bar_colors,
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
        
        # Adicionar labels nas barras
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                color = 'black'
                position = max(height, 0)
                
                if height < 0:
                    position = min(height, 0)
                    color = 'white'
                    
                ax1.text(
                    bar.get_x() + bar.get_width() / 2,
                    position + (0.01 if height > 0 else -0.02),
                    f"{height:.4f}",
                    ha='center',
                    color=color,
                    fontsize=9
                )
        
        # Configurar labels e título
        ax1.set_xlabel('Faixa de Volatilidade', fontsize=12)
        ax1.set_ylabel('PnL Médio', fontsize=12, color='darkgreen')
        ax2.set_ylabel('Número de Trades', fontsize=12, color='darkblue')
        plt.title('PnL Médio por Faixa de Volatilidade', fontsize=14)
        
        # Ajustar eixos
        min_pnl = volatility_winrate['avg_pnl'].min()
        max_pnl = volatility_winrate['avg_pnl'].max()
        
        if not np.isnan(min_pnl) and not np.isnan(max_pnl):
            margin = max(abs(min_pnl), abs(max_pnl)) * 0.2
            ax1.set_ylim(min(0, min_pnl - margin), max(0, max_pnl + margin))
        
        # Ajustar eixo Y2 para contar trades
        max_trades = volatility_winrate['trade_count'].max()
        if not np.isnan(max_trades) and max_trades > 0:
            ax2.set_ylim(0, max_trades * 1.2)
        
        # Rotacionar labels do eixo X para melhor legibilidade
        plt.xticks(rotation=45)
        
        # Adicionar legenda combinada
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper right')
        
        # Adicionar linha zero para referência
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Adicionar grid
        ax1.grid(True, alpha=0.3)
        
        # Adicionar nota sobre a fonte de dados
        plt.figtext(
            0.5, 0.01, 
            f"Fonte: {'Binance API (valores reais)' if self.use_binance_api else 'Dados locais'}",
            ha='center', fontsize=8, style='italic'
        )
        
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
        if self.merged_df is None:
            logger.error("Dados não carregados. Execute load_data() primeiro.")
            return
        
        # Filtrar por par e/ou timeframe se especificado
        df = self.merged_df.copy()
        title_suffix = ""
        
        if pair:
            df = df[df['pair'] == pair]
            title_suffix += f" - {pair}"
        
        if timeframe:
            if 'timeframe' in df.columns:
                df = df[df['timeframe'] == timeframe]
                title_suffix += f" - {timeframe}"
        
        # Verificar se há dados suficientes
        if df.empty or len(df) < 2:
            logger.warning(f"Dados insuficientes para gerar gráfico histórico de volatilidade{title_suffix}.")
            return
            
        # Verificar se há timestamp e volatilidade
        if 'timestamp' not in df.columns or 'volatility' not in df.columns:
            logger.warning("Colunas necessárias não encontradas para gerar gráfico histórico.")
            return
        
        # Ordenar por timestamp
        df = df.sort_values('timestamp')
        
        # Criar figura
        plt.figure(figsize=(14, 7))
        
        # Plotar linha de volatilidade
        plt.plot(df['timestamp'], df['volatility'] * 100, '-o', markersize=3, alpha=0.7)
        
        # Adicionar média móvel para visualizar tendência
        if len(df) >= 3:
            window_size = min(5, len(df) - 1)
            df['vol_ma'] = df['volatility'].rolling(window=window_size).mean() * 100
            plt.plot(df['timestamp'], df['vol_ma'], 'r-', linewidth=2, alpha=0.8, label=f'Média Móvel ({window_size})')
        
        # Marcar trades vencedores e perdedores
        if 'is_win' in df.columns:
            winners = df[df['is_win'] == True]
            losers = df[df['is_win'] == False]
            
            if not winners.empty:
                plt.scatter(winners['timestamp'], winners['volatility'] * 100, 
                           marker='^', color='green', s=100, alpha=0.7, label='Trades Vencedores')
            
            if not losers.empty:
                plt.scatter(losers['timestamp'], losers['volatility'] * 100, 
                           marker='v', color='red', s=100, alpha=0.7, label='Trades Perdedores')
        
        # Configurar labels e título
        plt.xlabel('Data/Hora', fontsize=12)
        plt.ylabel('Volatilidade (%)', fontsize=12)
        plt.title(f'Histórico de Volatilidade{title_suffix}', fontsize=14)
        
        # Formatar eixo de data
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        
        # Rotacionar labels do eixo X para melhor legibilidade
        plt.xticks(rotation=45)
        
        # Adicionar grid e legenda
        plt.grid(True, alpha=0.3)
        plt.legend(loc='best')
        
        # Adicionar nota sobre a fonte de dados
        plt.figtext(
            0.5, 0.01, 
            f"Fonte: {'Binance API (valores reais)' if self.use_binance_api else 'Dados locais'}",
            ha='center', fontsize=8, style='italic'
        )
        
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
            hold_time_corr = self.analyze_volatility_vs_hold_time()
            
            # Gráficos
            self.plot_volatility_winrate("volatility_winrate.png")
            self.plot_volatility_pnl("volatility_pnl.png")
            
            # Estatísticas gerais
            self._export_general_statistics(report_dir)
            
            # Gráficos de histórico para cada par
            pairs = self.merged_df['pair'].unique() if 'pair' in self.merged_df.columns else []
            for pair in pairs:
                self.plot_volatility_history(pair=pair, save_path=f"volatility_history.png")
            
            # Gerar relatório HTML
            self._generate_html_report(report_dir)
            
            logger.info(f"Relatório completo gerado em: {report_dir}")
            
            # Abrir o relatório no navegador
            report_path = os.path.join(report_dir, "volatility_report.html")
            try:
                import webbrowser
                webbrowser.open(f"file://{os.path.abspath(report_path)}")
                logger.info(f"Relatório HTML aberto no navegador.")
            except Exception as e:
                logger.warning(f"Não foi possível abrir o relatório automaticamente: {e}")
                logger.info(f"O relatório pode ser acessado em: {report_path}")
                
            return True
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório: {e}")
            traceback.print_exc()
            return False
        finally:
            # Restaurar diretório de saída
            self.output_dir = old_output_dir
    
    def _export_general_statistics(self, report_dir):
        """Exporta estatísticas gerais para o relatório."""
        if self.merged_df is None or self.merged_df.empty:
            logger.warning("Sem dados para exportar estatísticas gerais.")
            return
            
        # Calcular estatísticas gerais
        stats = {
            'total_trades': len(self.merged_df),
            'winning_trades': self.merged_df['is_win'].sum(),
            'losing_trades': (~self.merged_df['is_win']).sum(),
            'win_rate': self.merged_df['is_win'].mean() * 100,
            'total_pnl': self.merged_df['pnl'].sum(),
            'avg_pnl': self.merged_df['pnl'].mean(),
            'avg_volatility': self.merged_df['volatility'].mean() * 100,
            'med_volatility': self.merged_df['volatility'].median() * 100,
            'max_volatility': self.merged_df['volatility'].max() * 100,
            'min_volatility': self.merged_df['volatility'].min() * 100,
            'data_source': 'Binance API (valores reais)' if self.use_binance_api else 'Dados locais',
            'report_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Exportar como JSON para uso no relatório
        import json
        with open(os.path.join(report_dir, "general_stats.json"), 'w') as f:
            json.dump(stats, f, indent=4)
        
        # Também gerar uma versão CSV para compatibilidade
        pd.DataFrame([stats]).to_csv(os.path.join(report_dir, "general_stats.csv"), index=False)
        
        logger.info("Estatísticas gerais exportadas.")
    
    def _generate_html_report(self, report_dir):
        """Gera um arquivo HTML com o relatório completo."""
        # Carregar estatísticas gerais se disponíveis
        general_stats = {}
        try:
            import json
            with open(os.path.join(report_dir, "general_stats.json"), 'r') as f:
                general_stats = json.load(f)
        except:
            # Se não conseguir carregar, usar valores padrão
            general_stats = {
                'total_trades': len(self.merged_df) if self.merged_df is not None else 0,
                'win_rate': self.merged_df['is_win'].mean() * 100 if self.merged_df is not None else 0,
                'total_pnl': self.merged_df['pnl'].sum() if self.merged_df is not None else 0,
                'avg_volatility': self.merged_df['volatility'].mean() * 100 if self.merged_df is not None else 0,
                'data_source': 'Binance API (valores reais)' if self.use_binance_api else 'Dados locais',
                'report_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="pt-br">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Relatório de Análise de Volatilidade</title>
            <style>
                :root {{
                    --primary-color: #2c3e50;
                    --secondary-color: #3498db;
                    --accent-color: #e74c3c;
                    --light-bg: #f8f9fa;
                    --dark-bg: #343a40;
                    --success-color: #28a745;
                    --danger-color: #dc3545;
                }}
                
                body {{ 
                    font-family: 'Segoe UI', Arial, sans-serif; 
                    margin: 0;
                    padding: 0;
                    color: #333;
                    background-color: var(--light-bg);
                }}
                
                .header {{
                    background-color: var(--primary-color);
                    color: white;
                    padding: 20px;
                    text-align: center;
                }}
                
                h1 {{ 
                    margin: 0;
                    font-weight: 300;
                    font-size: 2.5em;
                }}
                
                h2 {{ 
                    color: var(--secondary-color);
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 10px;
                    margin-top: 30px;
                }}
                
                h3 {{
                    color: var(--primary-color);
                }}
                
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    padding: 20px;
                }}
                
                .stats-container {{
                    display: flex;
                    flex-wrap: wrap;
                    justify-content: space-between;
                    margin: 20px 0;
                }}
                
                .stat-card {{
                    flex: 1;
                    min-width: 200px;
                    background-color: white;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    margin: 10px;
                    padding: 15px;
                    text-align: center;
                }}
                
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                
                .positive {{ color: var(--success-color); }}
                .negative {{ color: var(--danger-color); }}
                
                table {{ 
                    width: 100%;
                    border-collapse: collapse; 
                    margin: 20px 0;
                    background-color: white;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }}
                
                th, td {{ 
                    border: 1px solid #ddd; 
                    padding: 12px; 
                    text-align: left; 
                }}
                
                th {{ 
                    background-color: var(--secondary-color); 
                    color: white;
                }}
                
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                tr:hover {{ background-color: #f1f1f1; }}
                
                .img-container {{ 
                    margin: 30px 0;
                    text-align: center;
                }}
                
                img {{ 
                    max-width: 100%; 
                    height: auto; 
                    border-radius: 5px;
                    box-shadow: 0 3px 6px rgba(0,0,0,0.1);
                }}
                
                .data-source {{ 
                    color: #6c757d; 
                    font-style: italic;
                    margin-top: 5px;
                    text-align: center;
                    font-size: 0.9em;
                }}
                
                .footer {{
                    margin-top: 40px;
                    padding: 20px;
                    background-color: var(--primary-color);
                    color: white;
                    text-align: center;
                    font-size: 0.9em;
                }}
                
                .summary-box {{
                    background-color: var(--light-bg);
                    border-left: 4px solid var(--secondary-color);
                    padding: 15px;
                    margin: 20px 0;
                }}
                
                .comparison-table {{
                    width: 100%;
                    margin: 25px 0;
                }}
                
                .comparison-table th {{
                    background-color: var(--primary-color);
                }}
                
                .comparison-table .better {{
                    background-color: rgba(40, 167, 69, 0.2);
                }}
                
                .comparison-table .worse {{
                    background-color: rgba(220, 53, 69, 0.2);
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Relatório de Análise de Volatilidade e Trading</h1>
                <p>Gerado em: {general_stats.get('report_date', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</p>
            </div>
            
            <div class="container">
                <div class="summary-box">
                    <h2>Resumo Geral</h2>
                    <div class="stats-container">
                        <div class="stat-card">
                            <h3>Trades Totais</h3>
                            <div class="stat-value">{general_stats.get('total_trades', 0)}</div>
                        </div>
                        <div class="stat-card">
                            <h3>Win Rate</h3>
                            <div class="stat-value {'positive' if general_stats.get('win_rate', 0) > 50 else 'negative'}">
                                {general_stats.get('win_rate', 0):.2f}%
                            </div>
                        </div>
                        <div class="stat-card">
                            <h3>PnL Total</h3>
                            <div class="stat-value {'positive' if general_stats.get('total_pnl', 0) > 0 else 'negative'}">
                                {general_stats.get('total_pnl', 0):.4f}
                            </div>
                        </div>
                        <div class="stat-card">
                            <h3>Volatilidade Média</h3>
                            <div class="stat-value">
                                {general_stats.get('avg_volatility', 0):.2f}%
                            </div>
                        </div>
                    </div>
                </div>

                <h2>Análise de Volatilidade vs Win Rate</h2>
                <p>Este gráfico mostra a relação entre diferentes níveis de volatilidade e a taxa de vitória dos trades.</p>
                <div class="img-container">
                    <img src="volatility_winrate.png" alt="Volatilidade vs Win Rate">
                    <p class="data-source">Fonte: {general_stats.get('data_source', 'Dados locais')}</p>
                </div>
                
                <h2>Análise de Volatilidade vs PnL</h2>
                <p>Este gráfico mostra a relação entre diferentes níveis de volatilidade e o PnL médio dos trades.</p>
                <div class="img-container">
                    <img src="volatility_pnl.png" alt="Volatilidade vs PnL">
                    <p class="data-source">Fonte: {general_stats.get('data_source', 'Dados locais')}</p>
                </div>
                
                <h2>Faixas de Volatilidade Ideais</h2>
                <p>Baseado nos dados analisados, estas são as faixas de volatilidade que produziram os melhores resultados.</p>
                <table class="comparison-table">
                    <thead>
                        <tr>
                            <th>Faixa de Volatilidade</th>
                            <th>Win Rate</th>
                            <th>PnL Médio</th>
                            <th>Número de Trades</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Dados das faixas de volatilidade -->
                        <!-- Gerados dinamicamente pelo script -->
                    </tbody>
                </table>
                
                <h2>Análise por Par</h2>
                <p>Performance de trading separada por par de moedas.</p>
                <table>
                    <thead>
                        <tr>
                            <th>Par</th>
                            <th>Win Rate</th>
                            <th>Volatilidade Média</th>
                            <th>PnL Médio</th>
                            <th>Trades</th>
                        </tr>
                    </thead>
                    <tbody>
                        <!-- Dados por par -->
                        <!-- Gerados dinamicamente pelo script -->
                    </tbody>
                </table>
            </div>
            
            <div class="footer">
                <p>Relatório gerado por Volatility Analysis Tool</p>
                <p>© {datetime.now().year} Trading Analysis</p>
            </div>
        </body>
        </html>
        """
        
        # Carregar dados para as tabelas, se disponíveis
        try:
            optimal_ranges = pd.read_csv(os.path.join(report_dir, "optimal_volatility_ranges.csv"))
            if not optimal_ranges.empty:
                # Gerar linhas de tabela para faixas ótimas
                optimal_rows = ""
                for _, row in optimal_ranges.sort_values('avg_pnl', ascending=False).head(5).iterrows():
                    win_rate = row['win_rate'] * 100 if not pd.isna(row['win_rate']) else 0
                    optimal_rows += f"""
                    <tr>
                        <td>{row['vol_fine']}</td>
                        <td>{win_rate:.2f}%</td>
                        <td class="{'positive' if row['avg_pnl'] > 0 else 'negative'}">{row['avg_pnl']:.4f}</td>
                        <td>{row['trade_count']}</td>
                    </tr>
                    """
                
                html_content = html_content.replace("<!-- Dados das faixas de volatilidade -->", optimal_rows)
                
            volatility_by_pair = pd.read_csv(os.path.join(report_dir, "volatility_by_pair.csv"))
            if not volatility_by_pair.empty:
                # Gerar linhas de tabela para análise por par
                pair_rows = ""
                for _, row in volatility_by_pair.sort_values('total_pnl', ascending=False).iterrows():
                    win_rate = row['win_rate'] * 100 if not pd.isna(row['win_rate']) else 0
                    avg_vol = row['avg_volatility'] * 100 if not pd.isna(row['avg_volatility']) else 0
                    pair_rows += f"""
                    <tr>
                        <td>{row['pair']}</td>
                        <td>{win_rate:.2f}%</td>
                        <td>{avg_vol:.2f}%</td>
                        <td class="{'positive' if row['avg_pnl'] > 0 else 'negative'}">{row['avg_pnl']:.4f}</td>
                        <td>{row['trade_count']}</td>
                    </tr>
                    """
                
                html_content = html_content.replace("<!-- Dados por par -->", pair_rows)
                
        except Exception as e:
            logger.error(f"Erro ao gerar conteúdo dinâmico para relatório HTML: {e}")
        
        # Salvar o arquivo HTML
        report_path = os.path.join(report_dir, "volatility_report.html")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Relatório HTML gerado em: {report_path}")

# Novas classes e funções a serem adicionadas para a funcionalidade requerida

class PriceSimulator:
    """Classe responsável por simular trades com base em sinais e dados históricos de preço"""
    
    def __init__(self, price_history_csv=None, binance_api=None):
        """
        Inicializa o simulador de preços.
        
        Args:
            price_history_csv: Caminho para o arquivo CSV com histórico de preços
            binance_api: Instância da classe BinanceUtils para buscar dados da API se necessário
        """
        self.price_history_csv = price_history_csv
        self.binance_api = binance_api
        self.price_data = None
        self.available_pairs = set()
        
    def load_price_history(self):
        """Carrega dados históricos de preços do CSV"""
        try:
            if not self.price_history_csv or not os.path.exists(self.price_history_csv):
                logger.warning(f"Arquivo de histórico de preços não encontrado: {self.price_history_csv}")
                return False
            
            self.price_data = pd.read_csv(self.price_history_csv)
            
            # Converter timestamp para datetime se necessário
            if 'timestamp' in self.price_data.columns:
                self.price_data['timestamp'] = pd.to_datetime(self.price_data['timestamp'])
            
            # Verificar quais pares estão disponíveis nos dados
            if 'pair' in self.price_data.columns:
                self.available_pairs = set(self.price_data['pair'].unique())
                logger.info(f"Dados históricos de preços carregados: {len(self.price_data)} registros")
                logger.info(f"Pares disponíveis no histórico: {', '.join(sorted(self.available_pairs))}")
            else:
                logger.warning("Coluna 'pair' não encontrada no arquivo de preços")
                
            return True
        except Exception as e:
            logger.error(f"Erro ao carregar histórico de preços: {e}")
            return False
    
    def get_future_price(self, pair, timestamp, minutes_ahead=None, target_price=None):
        """
        Obtém o preço futuro para um par em um determinado período após o timestamp.
        
        Args:
            pair: Par de moedas (ex: 'BTCUSDT')
            timestamp: Timestamp de referência
            minutes_ahead: Minutos à frente para buscar o preço (None para buscar até target_price)
            target_price: Preço alvo para buscar (None para buscar por tempo)
            
        Returns:
            Dict com preço encontrado, timestamp e minutos até o evento
        """
        # Verificar se o par está disponível nos dados
        if self.price_data is None:
            logger.error("Sem dados de preço disponíveis")
            return None
        
        # Se o par não estiver disponível, tentar usar dados simulados
        if pair not in self.available_pairs:
            return self._simulate_price_data(pair, timestamp, minutes_ahead, target_price)
            
        # Filtrar dados para o par específico
        pair_data = self.price_data[self.price_data['pair'] == pair].copy()
        
        if pair_data.empty:
            logger.warning(f"Sem dados para o par {pair}")
            return None
            
        # Converter timestamp para datetime se for string
        if isinstance(timestamp, str):
            timestamp = pd.to_datetime(timestamp)
            
        # Filtrar dados futuros em relação ao timestamp
        future_data = pair_data[pair_data['timestamp'] > timestamp]
        
        if future_data.empty:
            logger.warning(f"Sem dados futuros para o par {pair} após {timestamp}")
            return None
            
        if minutes_ahead:
            # Buscar por tempo
            target_time = timestamp + pd.Timedelta(minutes=minutes_ahead)
            closest_idx = future_data['timestamp'].sub(target_time).abs().idxmin()
            row = future_data.loc[closest_idx]
            
            return {
                'price': row['price'],
                'timestamp': row['timestamp'],
                'minutes': (row['timestamp'] - timestamp).total_seconds() / 60
            }
        elif target_price:
            # Buscar por preço alvo (para TP/SL)
            # Implementação simplificada - em um sistema real precisaríamos de dados OHLC por minuto
            for idx, row in future_data.iterrows():
                if row['price'] >= target_price:  # Para simplificar, estamos usando apenas 'price' não OHLC
                    return {
                        'price': row['price'],
                        'timestamp': row['timestamp'],
                        'minutes': (row['timestamp'] - timestamp).total_seconds() / 60
                    }
                    
            return None  # Preço alvo não encontrado
        else:
            # Caso padrão, retornar próximo preço disponível
            row = future_data.iloc[0]
            return {
                'price': row['price'],
                'timestamp': row['timestamp'],
                'minutes': (row['timestamp'] - timestamp).total_seconds() / 60
            }
    
    def _simulate_price_data(self, pair, timestamp, minutes_ahead=None, target_price=None):
        """
        Simula dados de preço quando não estão disponíveis no histórico.
        
        Args:
            pair: Par de moedas
            timestamp: Timestamp de referência
            minutes_ahead: Minutos à frente para simular
            target_price: Preço alvo para simular
            
        Returns:
            Dict simulado com preço, timestamp e minutos
        """
        # Base prices for different pairs
        base_prices = {
            'BTCUSDT': 65000.0,
            'ETHUSDT': 3500.0,
            'DOGEUSDT': 0.15,
            'TRUMPUSDT': 3.50,
            'SHIBUSDT': 0.00002,
        }
        
        # Volatility estimates
        volatility = {
            'BTCUSDT': 0.01,  # 1%
            'ETHUSDT': 0.015, # 1.5%
            'DOGEUSDT': 0.03, # 3%
            'TRUMPUSDT': 0.05, # 5%
            'SHIBUSDT': 0.04,  # 4%
        }
        
        # Se não conhecemos o par, usar valor médio de volatilidade
        base_price = base_prices.get(pair, 10.0)
        vol = volatility.get(pair, 0.02)
        
        # Simular preço inicial com pequena variação aleatória
        np.random.seed(int(timestamp.timestamp()) if hasattr(timestamp, 'timestamp') else 42)
        initial_price = base_price * (1 + np.random.uniform(-0.005, 0.005))
        
        # Se estamos buscando por tempo
        if minutes_ahead:
            # Simular movimento browniano simples
            future_timestamp = timestamp + pd.Timedelta(minutes=minutes_ahead)
            price_change = np.random.normal(0, vol * np.sqrt(minutes_ahead / 1440))  # Escala por dias
            future_price = initial_price * (1 + price_change)
            
            logger.info(f"Preço simulado para {pair}: inicial={initial_price:.6f}, após {minutes_ahead}min={future_price:.6f}")
            
            return {
                'price': future_price,
                'timestamp': future_timestamp,
                'minutes': minutes_ahead
            }
        elif target_price:
            # Calcular quanto tempo levaria para atingir o preço alvo com base na volatilidade
            price_ratio = target_price / initial_price
            # Estimar minutos necessários baseado na volatilidade e distância do alvo
            log_ratio = np.log(price_ratio)
            # Fórmula simplificada que estima tempo até target com base na volatilidade
            minutes = abs(log_ratio) / (vol * 0.1) * 60
            minutes = min(minutes, 60)  # Limitar a 60 minutos
            
            future_timestamp = timestamp + pd.Timedelta(minutes=minutes)
            
            logger.info(f"Preço simulado para {pair}: inicial={initial_price:.6f}, alvo={target_price:.6f} em {minutes:.1f}min")
            
            return {
                'price': target_price,
                'timestamp': future_timestamp,
                'minutes': minutes
            }
        else:
            # Caso padrão, simular próximo minuto
            future_timestamp = timestamp + pd.Timedelta(minutes=1)
            price_change = np.random.normal(0, vol * np.sqrt(1/1440))
            future_price = initial_price * (1 + price_change)
            
            logger.info(f"Preço simulado para {pair}: inicial={initial_price:.6f}, após 1min={future_price:.6f}")
            
            return {
                'price': future_price,
                'timestamp': future_timestamp,
                'minutes': 1
            }
    
    def simulate_trade(self, signal, sl_percent, tp_percent):
        """
        Simula um trade com base em um sinal e parâmetros de SL/TP.
        
        Args:
            signal: Linha de dados do sinal
            sl_percent: Porcentagem de Stop Loss (decimal)
            tp_percent: Porcentagem de Take Profit (decimal)
            
        Returns:
            Dict com resultados da simulação
        """
        try:
            # Extrair dados do sinal
            pair = signal['pair']
            direction = signal['direction']  # BUY ou SELL
            timestamp = signal['timestamp']
            signal_id = signal['signal_id']
            
            # Obter preço de entrada (usando o próximo preço disponível após o sinal)
            entry_data = self.get_future_price(pair, timestamp)
            
            if not entry_data:
                logger.warning(f"Não foi possível obter preço de entrada para {pair} em {timestamp}")
                return None
                
            entry_price = entry_data['price']
            entry_timestamp = entry_data['timestamp']
            
            # Calcular preços de SL e TP
            if direction == 'BUY':
                sl_price = entry_price * (1 - sl_percent)
                tp_price = entry_price * (1 + tp_percent)
            else:  # SELL
                sl_price = entry_price * (1 + sl_percent)
                tp_price = entry_price * (1 - tp_percent)
                
            # Valor hipotético para simulação
            quantity = 100.0 / entry_price  # Aproximadamente $100 por trade
            
            # Simular até atingir SL, TP ou timeout (60 minutos)
            sl_data = self.get_future_price(pair, entry_timestamp, None, sl_price)
            tp_data = self.get_future_price(pair, entry_timestamp, None, tp_price)
            timeout_data = self.get_future_price(pair, entry_timestamp, 60)  # 60 minutos
            
            # Determinar qual evento ocorreu primeiro
            events = []
            
            if sl_data:
                sl_data['type'] = 'SL'
                events.append(sl_data)
                
            if tp_data:
                tp_data['type'] = 'TP'
                events.append(tp_data)
                
            if timeout_data:
                timeout_data['type'] = 'TIMEOUT'
                events.append(timeout_data)
                
            if not events:
                logger.warning(f"Nenhum evento de fechamento para o trade {signal_id}")
                return None
                
            # Ordenar eventos por timestamp para pegar o primeiro
            events.sort(key=lambda x: x['timestamp'])
            exit_event = events[0]
            
            # Calcular resultados
            exit_price = exit_event['price']
            exit_timestamp = exit_event['timestamp']
            exit_type = exit_event['type']
            
            # Calcular PnL
            if direction == 'BUY':
                pnl = (exit_price - entry_price) * quantity
            else:  # SELL
                pnl = (entry_price - exit_price) * quantity
                
            # Taxa estimada (0.1% por operação na Binance)
            fee = entry_price * quantity * 0.001 + exit_price * quantity * 0.001
            net_pnl = pnl - fee
            
            # Dados do resultado
            result = {
                'signal_id': signal_id,
                'pair': pair,
                'direction': direction,
                'entry_price': entry_price,
                'entry_timestamp': entry_timestamp,
                'exit_price': exit_price,
                'exit_timestamp': exit_timestamp,
                'exit_type': exit_type,
                'sl_price': sl_price,
                'tp_price': tp_price,
                'quantity': quantity,
                'pnl': net_pnl,
                'is_win': net_pnl > 0,
                'hold_time_minutes': (exit_timestamp - entry_timestamp).total_seconds() / 60,
                'volatility': signal.get('volatility', 0)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na simulação de trade para sinal {signal.get('signal_id', 'desconhecido')}: {e}")
            return None

class TradeAnalyzer(VolatilityAnalysis):
    """Classe estendida de VolatilityAnalysis para incluir análises comparativas entre trades simulados e reais"""
    
    def __init__(self, signals_csv=SIGNALS_CSV, orders_csv=ORDERS_PNL_CSV, price_history_csv=None, use_binance_api=True):
        """
        Inicializa o analisador de trades.
        
        Args:
            signals_csv: Caminho para o arquivo CSV com sinais
            orders_csv: Caminho para o arquivo CSV com ordens executadas
            price_history_csv: Caminho para o arquivo CSV com histórico de preços
            use_binance_api: Flag para usar a API da Binance
        """
        super().__init__(signals_csv, orders_csv, use_binance_api)
        self.price_history_csv = price_history_csv
        self.simulated_trades = None
        
    def prompt_for_parameters(self):
        """Solicita parâmetros de SL/TP para simulação"""
        print("\n=== Configuração de Parâmetros para Simulação ===")
        
        try:
            sl_percent = float(input("Digite o valor do Stop Loss em % (ex: 1 para 1%): ")) / 100
            tp_percent = float(input("Digite o valor do Take Profit em % (ex: 2 para 2%): ")) / 100
            
            # Validação básica
            if sl_percent <= 0 or tp_percent <= 0:
                print("Erro: Os valores de SL e TP devem ser positivos.")
                return self.prompt_for_parameters()
                
            return {
                'sl_percent': sl_percent,
                'tp_percent': tp_percent
            }
        except ValueError:
            print("Erro: Por favor, digite valores numéricos válidos.")
            return self.prompt_for_parameters()
        
    def run_simulation(self, sl_percent=0.01, tp_percent=0.02, force_simulation=False):
        """
        Executa simulação de trades com base nos sinais.
        
        Args:
            sl_percent: Porcentagem de Stop Loss (decimal)
            tp_percent: Porcentagem de Take Profit (decimal)
            force_simulation: Se True, força o uso de preços simulados mesmo sem dados históricos
            
        Returns:
            DataFrame com resultados da simulação
        """
        if self.signals_df is None:
            logger.error("Dados de sinais não carregados. Execute load_data() primeiro.")
            return None
            
        # Inicializar simulador de preços
        simulator = PriceSimulator(
            price_history_csv=self.price_history_csv,
            binance_api=self.binance_utils if self.use_binance_api else None
        )
        
        # Carregar dados de preços
        if not simulator.load_price_history():
            if force_simulation:
                logger.warning("Usando dados de preço totalmente simulados para a simulação")
            else:
                logger.warning("Falha ao carregar preços e simulação não está forçada")
                return None
        
        # Simular cada sinal
        results = []
        for idx, signal in self.signals_df.iterrows():
            if not isinstance(signal['timestamp'], pd.Timestamp):
                # Converter para timestamp se for string
                try:
                    signal = signal.copy()
                    signal['timestamp'] = pd.to_datetime(signal['timestamp'])
                except:
                    logger.warning(f"Não foi possível converter timestamp para sinal {signal.get('signal_id', 'desconhecido')}")
            
            logger.info(f"Simulando trade para sinal {signal['signal_id']}")
            result = simulator.simulate_trade(signal, sl_percent, tp_percent)
            if result:
                results.append(result)
                
        if not results:
            logger.warning("Nenhum resultado de simulação disponível")
            
            if force_simulation:
                # Criar simulações sintéticas baseadas em sinais
                for idx, signal in self.signals_df.iterrows():
                    # Valores simulados básicos
                    base_price = 10.0
                    if signal['pair'] == 'DOGEUSDT':
                        base_price = 0.15
                    elif signal['pair'] == 'TRUMPUSDT':
                        base_price = 3.50
                    
                    # Simular um resultado positivo ou negativo aleatoriamente
                    is_win = np.random.choice([True, False], p=[0.6, 0.4])  # 60% win rate simulado
                    
                    # Calcular preços de entrada e saída
                    entry_price = base_price
                    
                    if signal['direction'] == 'BUY':
                        exit_price = entry_price * (1 + tp_percent) if is_win else entry_price * (1 - sl_percent)
                    else:  # SELL
                        exit_price = entry_price * (1 - tp_percent) if is_win else entry_price * (1 + sl_percent)
                    
                    # Quantidade fixada em algo realista
                    quantity = 100 / base_price
                    
                    # Timestamp
                    if isinstance(signal['timestamp'], pd.Timestamp):
                        entry_timestamp = signal['timestamp']
                    else:
                        try:
                            entry_timestamp = pd.to_datetime(signal['timestamp'])
                        except:
                            entry_timestamp = pd.Timestamp.now()
                    
                    # Timestamp de saída (5-30 minutos depois)
                    minutes = np.random.randint(5, 31)
                    exit_timestamp = entry_timestamp + pd.Timedelta(minutes=minutes)
                    
                    # Calcular PnL
                    if signal['direction'] == 'BUY':
                        pnl = (exit_price - entry_price) * quantity
                    else:  # SELL
                        pnl = (entry_price - exit_price) * quantity
                    
                    # Taxas
                    fee = entry_price * quantity * 0.001 + exit_price * quantity * 0.001
                    net_pnl = pnl - fee
                    
                    # Criar resultado
                    result = {
                        'signal_id': signal['signal_id'],
                        'pair': signal['pair'],
                        'direction': signal['direction'],
                        'entry_price': entry_price,
                        'entry_timestamp': entry_timestamp,
                        'exit_price': exit_price,
                        'exit_timestamp': exit_timestamp,
                        'exit_type': 'TP' if is_win else 'SL',
                        'sl_price': entry_price * (1 - sl_percent) if signal['direction'] == 'BUY' else entry_price * (1 + sl_percent),
                        'tp_price': entry_price * (1 + tp_percent) if signal['direction'] == 'BUY' else entry_price * (1 - tp_percent),
                        'quantity': quantity,
                        'pnl': net_pnl,
                        'is_win': is_win,
                        'hold_time_minutes': minutes,
                        'volatility': signal.get('volatility', 0.03)
                    }
                    
                    results.append(result)
                    logger.info(f"Criado trade simulado sintético para {signal['pair']}: {'WIN' if is_win else 'LOSS'}, PnL={net_pnl:.4f}")
            
            if not results:
                return None
        
        # Converter para DataFrame
        self.simulated_trades = pd.DataFrame(results)
        logger.info(f"Simulação concluída: {len(self.simulated_trades)} trades simulados")
        
        return self.simulated_trades
        
    def compare_results(self):
        """Compara resultados simulados com resultados reais"""
        if self.simulated_trades is None or self.merged_df is None:
            logger.error("Dados simulados ou reais não disponíveis. Execute run_simulation() e load_data() primeiro.")
            return None
            
        # Mesclar dados por signal_id para comparação
        comparison = self.simulated_trades.merge(
            self.merged_df,
            on='signal_id',
            how='inner',
            suffixes=('_sim', '_real')
        )
        
        if comparison.empty:
            logger.warning("Nenhum trade comum entre simulados e reais para comparação")
            return None
            
        # Estatísticas agregadas
        sim_stats = {
            'count': len(self.simulated_trades),
            'win_rate': self.simulated_trades['is_win'].mean() * 100,
            'avg_pnl': self.simulated_trades['pnl'].mean(),
            'total_pnl': self.simulated_trades['pnl'].sum()
        }
        
        real_stats = {
            'count': len(self.merged_df),
            'win_rate': self.merged_df['is_win'].mean() * 100,
            'avg_pnl': self.merged_df['pnl'].mean(),
            'total_pnl': self.merged_df['pnl'].sum()
        }
        
        common_stats = {
            'count': len(comparison),
            'sim_win_rate': comparison['is_win_sim'].mean() * 100,
            'real_win_rate': comparison['is_win_real'].mean() * 100,
            'sim_avg_pnl': comparison['pnl_sim'].mean(),
            'real_avg_pnl': comparison['pnl_real'].mean(),
            'sim_total_pnl': comparison['pnl_sim'].sum(),
            'real_total_pnl': comparison['pnl_real'].sum()
        }
        
        # Verificar correlação
        corr_pnl = comparison['pnl_sim'].corr(comparison['pnl_real'])
        
        logger.info("\n=== Comparação de Resultados Simulados vs. Reais ===")
        logger.info(f"Trades simulados: {sim_stats['count']}, Win rate: {sim_stats['win_rate']:.2f}%, PnL médio: {sim_stats['avg_pnl']:.4f}")
        logger.info(f"Trades reais: {real_stats['count']}, Win rate: {real_stats['win_rate']:.2f}%, PnL médio: {real_stats['avg_pnl']:.4f}")
        logger.info(f"Trades comuns: {common_stats['count']}")
        logger.info(f"Correlação de PnL: {corr_pnl:.4f}")
        
        return {
            'sim_stats': sim_stats,
            'real_stats': real_stats,
            'common_stats': common_stats,
            'comparison': comparison,
            'correlation': corr_pnl
        }
        
    def generate_comparison_report(self, report_dir=None):
        """
        Gera relatório comparativo entre trades simulados e reais.
        
        Args:
            report_dir: Diretório para salvar o relatório
        """
        if not report_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = os.path.join(self.output_dir, f"comparison_report_{timestamp}")
            
        os.makedirs(report_dir, exist_ok=True)
        
        # Obter dados de comparação
        comparison_data = self.compare_results()
        if not comparison_data:
            logger.error("Não foi possível gerar relatório de comparação")
            return
            
        # Exportar dados
        if 'comparison' in comparison_data and not comparison_data['comparison'].empty:
            comparison_data['comparison'].to_csv(os.path.join(report_dir, "trade_comparison.csv"), index=False)
        
        # Gerar gráficos de comparação
        self._generate_comparison_charts(comparison_data, report_dir)
        
        # Gerar relatório HTML
        self._generate_comparison_html(comparison_data, report_dir)
        
        logger.info(f"Relatório de comparação gerado em: {report_dir}")
        
        # Abrir o relatório no navegador
        report_path = os.path.join(report_dir, "comparison_report.html")
        try:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(report_path)}")
            logger.info(f"Relatório HTML aberto no navegador.")
        except Exception as e:
            logger.warning(f"Não foi possível abrir o relatório automaticamente: {e}")
            logger.info(f"O relatório pode ser acessado em: {report_path}")
        
    def _generate_comparison_charts(self, comparison_data, report_dir):
        """Gera gráficos comparativos para o relatório"""
        try:
            comparison = comparison_data.get('comparison')
            if comparison is None or comparison.empty:
                logger.warning("Sem dados para gerar gráficos de comparação")
                return
                
            # Gráfico de dispersão PnL simulado vs real
            plt.figure(figsize=(10, 8))
            plt.scatter(comparison['pnl_sim'], comparison['pnl_real'], alpha=0.7)
            plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
            plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
            plt.title('PnL Simulado vs PnL Real')
            plt.xlabel('PnL Simulado')
            plt.ylabel('PnL Real')
            plt.grid(True, alpha=0.3)
            
            # Adicionar linha de tendência
            z = np.polyfit(comparison['pnl_sim'], comparison['pnl_real'], 1)
            p = np.poly1d(z)
            plt.plot(
                [comparison['pnl_sim'].min(), comparison['pnl_sim'].max()],
                p([comparison['pnl_sim'].min(), comparison['pnl_sim'].max()]),
                "r--", alpha=0.7
            )
            
            # Adicionar correlação
            corr = comparison_data.get('correlation', 0)
            plt.annotate(
                f"Correlação: {corr:.4f}",
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
            )
            
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "pnl_comparison.png"), dpi=300)
            plt.close()
            
            # Gráfico de barras para Win Rate
            sim_stats = comparison_data.get('sim_stats', {})
            real_stats = comparison_data.get('real_stats', {})
            
            labels = ['Simulado', 'Real']
            win_rates = [sim_stats.get('win_rate', 0), real_stats.get('win_rate', 0)]
            
            plt.figure(figsize=(8, 6))
            bars = plt.bar(labels, win_rates, color=['skyblue', 'lightgreen'])
            
            # Adicionar valores nas barras
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 1,
                    f"{height:.2f}%",
                    ha='center',
                    fontsize=10
                )
                
            plt.title('Win Rate: Simulado vs Real')
            plt.ylabel('Win Rate (%)')
            plt.ylim(0, max(win_rates) * 1.2)
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "winrate_comparison.png"), dpi=300)
            plt.close()
            
            # Gráfico de barras para PnL total
            pnl_values = [sim_stats.get('total_pnl', 0), real_stats.get('total_pnl', 0)]
            
            plt.figure(figsize=(8, 6))
            bars = plt.bar(
                labels, 
                pnl_values,
                color=['skyblue' if p > 0 else 'salmon' for p in pnl_values]
            )
            
            # Adicionar valores nas barras
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + (0.1 if height >= 0 else -0.1),
                    f"{height:.2f}",
                    ha='center',
                    fontsize=10
                )
                
            plt.title('PnL Total: Simulado vs Real')
            plt.ylabel('PnL Total')
            plt.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig(os.path.join(report_dir, "pnl_total_comparison.png"), dpi=300)
            plt.close()
            
        except Exception as e:
            logger.error(f"Erro ao gerar gráficos de comparação: {e}")
        
    def _generate_comparison_html(self, comparison_data, report_dir):
        """Gera relatório HTML comparativo"""
        try:
            sim_stats = comparison_data.get('sim_stats', {})
            real_stats = comparison_data.get('real_stats', {})
            common_stats = comparison_data.get('common_stats', {})
            
            html_content = f"""
            <!DOCTYPE html>
            <html lang="pt-br">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Relatório Comparativo de Trades Simulados vs Reais</title>
                <style>
                    :root {{
                        --primary-color: #2c3e50;
                        --secondary-color: #3498db;
                        --accent-color: #e74c3c;
                        --light-bg: #f8f9fa;
                        --dark-bg: #343a40;
                        --success-color: #28a745;
                        --danger-color: #dc3545;
                    }}
                    
                    body {{ 
                        font-family: 'Segoe UI', Arial, sans-serif; 
                        margin: 0;
                        padding: 0;
                        color: #333;
                        background-color: var(--light-bg);
                    }}
                    
                    .header {{
                        background-color: var(--primary-color);
                        color: white;
                        padding: 20px;
                        text-align: center;
                    }}
                    
                    h1 {{ 
                        margin: 0;
                        font-weight: 300;
                        font-size: 2.5em;
                    }}
                    
                    h2 {{ 
                        color: var(--secondary-color);
                        border-bottom: 1px solid #ddd;
                        padding-bottom: 10px;
                        margin-top: 30px;
                    }}
                    
                    h3 {{
                        color: var(--primary-color);
                    }}
                    
                    .container {{ 
                        max-width: 1200px; 
                        margin: 0 auto; 
                        padding: 20px;
                    }}
                    
                    .stats-container {{
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: space-between;
                        margin: 20px 0;
                    }}
                    
                    .stat-card {{
                        flex: 1;
                        min-width: 200px;
                        background-color: white;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        margin: 10px;
                        padding: 15px;
                        text-align: center;
                    }}
                    
                    .stat-value {{
                        font-size: 24px;
                        font-weight: bold;
                        margin: 10px 0;
                    }}
                    
                    .positive {{ color: var(--success-color); }}
                    .negative {{ color: var(--danger-color); }}
                    
                    table {{ 
                        width: 100%;
                        border-collapse: collapse; 
                        margin: 20px 0;
                        background-color: white;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    }}
                    
                    th, td {{ 
                        border: 1px solid #ddd; 
                        padding: 12px; 
                        text-align: left; 
                    }}
                    
                    th {{ 
                        background-color: var(--secondary-color); 
                        color: white;
                    }}
                    
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    tr:hover {{ background-color: #f1f1f1; }}
                    
                    .img-container {{ 
                        margin: 30px 0;
                        text-align: center;
                    }}
                    
                    img {{ 
                        max-width: 100%; 
                        height: auto; 
                        border-radius: 5px;
                        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
                    }}
                    
                    .summary-box {{
                        background-color: var(--light-bg);
                        border-left: 4px solid var(--secondary-color);
                        padding: 15px;
                        margin: 20px 0;
                    }}
                    
                    .footer {{
                        margin-top: 40px;
                        padding: 20px;
                        background-color: var(--primary-color);
                        color: white;
                        text-align: center;
                        font-size: 0.9em;
                    }}
                    
                    .comparison-container {{
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: space-between;
                    }}
                    
                    .comparison-half {{
                        flex: 1;
                        min-width: 300px;
                        margin: 10px;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Relatório Comparativo: Trades Simulados vs Reais</h1>
                    <p>Gerado em: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                
                <div class="container">
                    <div class="summary-box">
                        <h2>Resumo Comparativo</h2>
                        <div class="stats-container">
                            <div class="stat-card">
                                <h3>Trades Simulados</h3>
                                <div class="stat-value">{sim_stats.get('count', 0)}</div>
                            </div>
                            <div class="stat-card">
                                <h3>Trades Reais</h3>
                                <div class="stat-value">{real_stats.get('count', 0)}</div>
                            </div>
                            <div class="stat-card">
                                <h3>Trades Comuns</h3>
                                <div class="stat-value">{common_stats.get('count', 0)}</div>
                            </div>
                        </div>
                        
                        <div class="stats-container">
                            <div class="stat-card">
                                <h3>Win Rate Simulado</h3>
                                <div class="stat-value {'positive' if sim_stats.get('win_rate', 0) > 50 else 'negative'}">
                                    {sim_stats.get('win_rate', 0):.2f}%
                                </div>
                            </div>
                            <div class="stat-card">
                                <h3>Win Rate Real</h3>
                                <div class="stat-value {'positive' if real_stats.get('win_rate', 0) > 50 else 'negative'}">
                                    {real_stats.get('win_rate', 0):.2f}%
                                </div>
                            </div>
                        </div>
                        
                        <div class="stats-container">
                            <div class="stat-card">
                                <h3>PnL Total Simulado</h3>
                                <div class="stat-value {'positive' if sim_stats.get('total_pnl', 0) > 0 else 'negative'}">
                                    {sim_stats.get('total_pnl', 0):.4f}
                                </div>
                            </div>
                            <div class="stat-card">
                                <h3>PnL Total Real</h3>
                                <div class="stat-value {'positive' if real_stats.get('total_pnl', 0) > 0 else 'negative'}">
                                    {real_stats.get('total_pnl', 0):.4f}
                                </div>
                            </div>
                        </div>
                    </div>
    
                    <h2>Comparação Visual</h2>
                    
                    <div class="comparison-container">
                        <div class="comparison-half">
                            <h3>Win Rate: Simulado vs Real</h3>
                            <div class="img-container">
                                <img src="winrate_comparison.png" alt="Comparação de Win Rate">
                            </div>
                        </div>
                        
                        <div class="comparison-half">
                            <h3>PnL Total: Simulado vs Real</h3>
                            <div class="img-container">
                                <img src="pnl_total_comparison.png" alt="Comparação de PnL Total">
                            </div>
                        </div>
                    </div>
                    
                    <h3>Correlação entre PnL Simulado e Real</h3>
                    <div class="img-container">
                        <img src="pnl_comparison.png" alt="Correlação de PnL">
                        <p>Correlação: {comparison_data.get('correlation', 0):.4f}</p>
                    </div>
                    
                    <h2>Detalhamento por Par</h2>
                    <p>Comparação de desempenho por par de moedas.</p>
                    <!-- Tabela de comparação por par aqui -->
                    
                </div>
                
                <div class="footer">
                    <p>Relatório gerado por Trade Analysis Tool</p>
                    <p>© {datetime.now().year} Trading Analysis</p>
                </div>
            </body>
            </html>
            """
            
            # Adicionar tabela de comparação por par se disponível
            comparison = comparison_data.get('comparison')
            if comparison is not None and not comparison.empty:
                # Agrupar por par
                pair_comparison = comparison.groupby('pair_sim').agg(
                    sim_win_rate=('is_win_sim', 'mean'),
                    real_win_rate=('is_win_real', 'mean'),
                    sim_avg_pnl=('pnl_sim', 'mean'),
                    real_avg_pnl=('pnl_real', 'mean'),
                    trade_count=('signal_id', 'count')
                ).reset_index()
                
                # Gerar tabela HTML
                pair_rows = ""
                for _, row in pair_comparison.iterrows():
                    sim_win = row['sim_win_rate'] * 100
                    real_win = row['real_win_rate'] * 100
                    
                    pair_rows += f"""
                    <tr>
                        <td>{row['pair_sim']}</td>
                        <td>{row['trade_count']}</td>
                        <td class="{'positive' if sim_win > 50 else 'negative'}">{sim_win:.2f}%</td>
                        <td class="{'positive' if real_win > 50 else 'negative'}">{real_win:.2f}%</td>
                        <td class="{'positive' if row['sim_avg_pnl'] > 0 else 'negative'}">{row['sim_avg_pnl']:.4f}</td>
                        <td class="{'positive' if row['real_avg_pnl'] > 0 else 'negative'}">{row['real_avg_pnl']:.4f}</td>
                    </tr>
                    """
                
                pair_table = f"""
                <table>
                    <thead>
                        <tr>
                            <th>Par</th>
                            <th>Trades</th>
                            <th>Win Rate Sim</th>
                            <th>Win Rate Real</th>
                            <th>PnL Médio Sim</th>
                            <th>PnL Médio Real</th>
                        </tr>
                    </thead>
                    <tbody>
                        {pair_rows}
                    </tbody>
                </table>
                """
                
                html_content = html_content.replace("<!-- Tabela de comparação por par aqui -->", pair_table)
            
            # Salvar o arquivo HTML
            report_path = os.path.join(report_dir, "comparison_report.html")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Relatório HTML de comparação gerado em: {report_path}")
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório HTML de comparação: {e}")

# Função principal
def main():
    """Função principal para execução do script"""
    print("\n=== Análise de Trading e Volatilidade ===")
    print("Este script analisa a relação entre volatilidade e resultados de trading.")
    
    # Inicializar o analisador
    analyzer = TradeAnalyzer(
        signals_csv=SIGNALS_CSV,
        orders_csv=ORDERS_PNL_CSV,
        price_history_csv="prices_statistics.csv",  # Arquivo mencionado pelo usuário
        use_binance_api=True
    )
    
    # Carregar dados
    if not analyzer.load_data():
        print("Falha ao carregar dados. Verifique os arquivos de entrada.")
        return
    
    # Verificar se o arquivo de preços existe
    price_file = "prices_statistics.csv"
    if not os.path.exists(price_file):
        print(f"\nAviso: Arquivo de preços '{price_file}' não encontrado!")
        use_simulation = input("Deseja continuar usando preços simulados? (s/n): ").lower()
        if use_simulation != 's':
            print("Operação cancelada pelo usuário.")
            return
        print("Continuando com preços simulados para TRUMPUSDT e DOGEUSDT...")
    
    # Solicitar parâmetros ao usuário
    params = analyzer.prompt_for_parameters()
    
    # Executar simulação
    print("\nExecutando simulação de trades com os parâmetros informados...")
    simulated_trades = analyzer.run_simulation(
        sl_percent=params['sl_percent'],
        tp_percent=params['tp_percent']
    )
    
    if simulated_trades is None or len(simulated_trades) == 0:
        print("\nNenhum resultado de simulação disponível. Utilizando dados simulados...")
        # Tentar novamente com flag para forçar simulação
        simulated_trades = analyzer.run_simulation(
            sl_percent=params['sl_percent'],
            tp_percent=params['tp_percent'],
            force_simulation=True
        )
        
        if simulated_trades is None or len(simulated_trades) == 0:
            print("Falha na simulação mesmo com dados simulados. Verificar os dados de entrada.")
            return
    
    # Gerar relatório comparativo
    print("\nGerando relatório comparativo entre trades simulados e reais...")
    analyzer.generate_comparison_report()
    
    # Gerar relatório de volatilidade
    print("\nGerando relatório de análise de volatilidade...")
    analyzer.generate_volatility_report()
    
    print("\nAnálise concluída! Os relatórios foram gerados com sucesso.")

if __name__ == "__main__":
    config_logger()  # Configurar logger
    try:
        main()
    except Exception as e:
        logger.error(f"Erro na execução: {e}")
        traceback.print_exc()
