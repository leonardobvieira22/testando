import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import os
import time
from datetime import datetime, timedelta
import traceback
from typing import Dict, List, Tuple
import warnings
from scipy import stats
import random
warnings.filterwarnings('ignore')

def evaluate_rsi_params(data_params):
    """Função para avaliação paralela de parâmetros RSI"""
    data, params, tp_pct, sl_pct = data_params
    
    # Gerar sinais com estes parâmetros
    signals = generate_signals(data, params)
    
    # Calcular quantos sinais atingem TP vs SL
    results = evaluate_signals(signals, tp_pct, sl_pct)
    
    return {
        'params': params,
        'win_rate': results['win_rate'],
        'win_count': results['win_count'],
        'loss_count': results['loss_count'],
        'total_trades': results['total_trades'],
        'expected_value': results['expected_value']
    }

def generate_signals(data, params):
    """Gera sinais baseados nos parâmetros RSI"""
    df = data.copy()
    df['signal'] = None
    df['signal_price'] = None
    
    # Extrair parâmetros
    rsi_buy = params['rsi_buy']
    rsi_sell = params['rsi_sell']
    rsi_delta = params['rsi_delta']
    rsi_min_long = params['rsi_min_long']
    rsi_max_short = params['rsi_max_short']
    timeout_min = params.get('timeout', 5)  # Default 5 minutos
    
    results = []
    
    # Processar cada par separadamente
    for pair, group in df.groupby('pair'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        
        # Variáveis para controle do modo atento
        in_attentive_mode = False
        attentive_mode_type = None
        attentive_rsi_extreme = None
        attentive_start_time = None
        
        for i in range(1, len(group)):
            curr_time = group.at[i, 'timestamp']
            curr_price = group.at[i, 'price']
            curr_rsi = group.at[i, 'rsi']
            
            # Verificar timeout do modo atento
            if in_attentive_mode and attentive_start_time:
                time_diff = (curr_time - attentive_start_time).total_seconds() / 60
                if time_diff > timeout_min:
                    in_attentive_mode = False
                    attentive_mode_type = None
            
            # Lógica para entrar em modo atento
            if not in_attentive_mode:
                # Condição para modo atento Long
                if curr_rsi <= rsi_buy:
                    in_attentive_mode = True
                    attentive_mode_type = 'long'
                    attentive_rsi_extreme = curr_rsi
                    attentive_start_time = curr_time
                
                # Condição para modo atento Short
                elif curr_rsi >= rsi_sell:
                    in_attentive_mode = True
                    attentive_mode_type = 'short'
                    attentive_rsi_extreme = curr_rsi
                    attentive_start_time = curr_time
            
            # Atualizar extremos de RSI no modo atento
            elif in_attentive_mode:
                if attentive_mode_type == 'long' and curr_rsi < attentive_rsi_extreme:
                    attentive_rsi_extreme = curr_rsi
                elif attentive_mode_type == 'short' and curr_rsi > attentive_rsi_extreme:
                    attentive_rsi_extreme = curr_rsi
                
                # Verificar condições para gerar sinais
                if attentive_mode_type == 'long':
                    # Verificar se RSI subiu o suficiente do extremo
                    if curr_rsi >= attentive_rsi_extreme + rsi_delta:
                        # Verificar se RSI está acima do mínimo para long
                        if curr_rsi >= rsi_min_long:
                            group.at[i, 'signal'] = 'BUY'
                            group.at[i, 'signal_price'] = curr_price
                        
                        # Desligar modo atento após verificação
                        in_attentive_mode = False
                        attentive_mode_type = None
                
                elif attentive_mode_type == 'short':
                    # Verificar se RSI caiu o suficiente do extremo
                    if curr_rsi <= attentive_rsi_extreme - rsi_delta:
                        # Verificar se RSI está abaixo do máximo para short
                        if curr_rsi <= rsi_max_short:
                            group.at[i, 'signal'] = 'SELL'
                            group.at[i, 'signal_price'] = curr_price
                        
                        # Desligar modo atento após verificação
                        in_attentive_mode = False
                        attentive_mode_type = None
        
        results.append(group)
    
    return pd.concat(results).reset_index(drop=True)

def evaluate_signals(signals_df, tp_pct, sl_pct):
    """
    Avalia sinais para determinar quantos atingem TP vs SL
    Foco simples em win rate e expectativa matemática
    """
    # Filtrar apenas registros com sinais
    signals = signals_df.dropna(subset=['signal']).copy()
    
    # Se não houver sinais, retornar resultados vazios
    if signals.empty:
        return {
            'win_rate': 0,
            'win_count': 0,
            'loss_count': 0,
            'total_trades': 0,
            'expected_value': 0
        }
    
    wins = 0
    losses = 0
    
    for pair, pair_signals in signals.groupby('pair'):
        # Ordenar cronologicamente
        pair_signals = pair_signals.sort_values('timestamp')
        
        for i, row in pair_signals.iterrows():
            # Obter preço e tipo de sinal
            entry_time = row['timestamp']
            entry_price = row['signal_price'] or row['price']
            signal_type = row['signal']
            
            # Definir preços target para TP e SL
            if signal_type == 'BUY':
                tp_price = entry_price * (1 + tp_pct)
                sl_price = entry_price * (1 - sl_pct)
            else:  # SELL
                tp_price = entry_price * (1 - tp_pct)
                sl_price = entry_price * (1 + sl_pct)
            
            # Filtrar preços futuros para verificar se atinge TP ou SL primeiro
            future_prices = signals_df[(signals_df['pair'] == pair) & 
                                     (signals_df['timestamp'] > entry_time)]
            
            hit_tp = False
            hit_sl = False
            
            for _, price_row in future_prices.iterrows():
                current_price = price_row['price']
                
                # Verificar se atingiu TP
                if signal_type == 'BUY' and current_price >= tp_price:
                    wins += 1
                    hit_tp = True
                    break
                elif signal_type == 'SELL' and current_price <= tp_price:
                    wins += 1
                    hit_tp = True
                    break
                
                # Verificar se atingiu SL
                if signal_type == 'BUY' and current_price <= sl_price:
                    losses += 1
                    hit_sl = True
                    break
                elif signal_type == 'SELL' and current_price >= sl_price:
                    losses += 1
                    hit_sl = True
                    break
            
            # Se não atingiu nem TP nem SL, considerar como posição aberta
            if not hit_tp and not hit_sl:
                # Verificar último preço para determinar resultado
                last_price = future_prices['price'].iloc[-1] if not future_prices.empty else entry_price
                
                if signal_type == 'BUY':
                    if last_price > entry_price:
                        wins += 1
                    else:
                        losses += 1
                else:  # SELL
                    if last_price < entry_price:
                        wins += 1
                    else:
                        losses += 1
    
    # Calcular estatísticas
    total_trades = wins + losses
    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    
    # Expectativa matemática: (Win% × Average Win) − (Loss% × Average Loss)
    # Simplificando para usar apenas TP e SL fixos:
    win_pct = wins / total_trades if total_trades > 0 else 0
    loss_pct = losses / total_trades if total_trades > 0 else 0
    expected_value = (win_pct * tp_pct) - (loss_pct * sl_pct)
    
    return {
        'win_rate': win_rate,
        'win_count': wins,
        'loss_count': losses,
        'total_trades': total_trades,
        'expected_value': expected_value
    }

class SimpleRSIOptimizer:
    """Otimizador inteligente para parâmetros RSI focando em win rate"""
    
    def __init__(self):
        self.data = None
        self.results_dir = "resultados_rsi"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def load_data(self, filename='prices_statistics.csv'):
        """Carregar dados de preço e RSI"""
        try:
            print(f"\nCarregando dados de {filename}...")
            
            data = pd.read_csv(filename)
            
            # Verificar colunas necessárias
            required_cols = ['pair', 'timestamp', 'price', 'rsi']
            missing = [col for col in required_cols if col not in data.columns]
            
            if missing:
                print(f"❌ Erro: Colunas obrigatórias ausentes: {', '.join(missing)}")
                return False
            
            # Converter timestamp para datetime
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            
            # Ordenar dados
            data = data.sort_values(['pair', 'timestamp'])
            
            print("\n✅ Dados carregados com sucesso:")
            print(f"- {len(data)} registros de preço e RSI")
            print(f"- Pares disponíveis: {', '.join(data['pair'].unique())}")
            print(f"- Período: {data['timestamp'].min()} a {data['timestamp'].max()}")
            
            self.data = data
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {str(e)}")
            traceback.print_exc()
            return False
    
    def suggest_rsi_ranges(self, data):
        """Sugere ranges de parâmetros RSI baseados na análise dos dados históricos"""
        rsi_values = data['rsi'].dropna()
        
        # Calcular percentis do RSI para sugerir thresholds
        rsi_percentiles = {
            'p5': rsi_values.quantile(0.05),
            'p10': rsi_values.quantile(0.10),
            'p20': rsi_values.quantile(0.20),
            'p80': rsi_values.quantile(0.80),
            'p90': rsi_values.quantile(0.90),
            'p95': rsi_values.quantile(0.95)
        }
        
        # Sugerir ranges com base nos percentis encontrados
        suggested_ranges = {
            'rsi_buy': [round(rsi_percentiles['p5']), round(rsi_percentiles['p10']), round(rsi_percentiles['p20'])],
            'rsi_sell': [round(rsi_percentiles['p80']), round(rsi_percentiles['p90']), round(rsi_percentiles['p95'])],
            'rsi_delta': [3, 5, 7],  # Valores padrão para delta
            'rsi_min_long': [round(rsi_percentiles['p20']) + 5],
            'rsi_max_short': [round(rsi_percentiles['p80']) - 5]
        }
        
        return suggested_ranges
    
    def detect_market_regime(self, data, window=20):
        """Detecta se o mercado está em tendência ou lateralização"""
        regimes = {}
        for pair, group in data.groupby('pair'):
            # Calcular a tendência usando regressão linear
            prices = group['price'].values
            x = np.arange(len(prices))
            slope, _, r_value, _, _ = stats.linregress(x, prices)
            
            # Calcular volatilidade
            volatility = group['price'].pct_change().std()
            
            # Classificar o regime
            if abs(r_value) > 0.7:  # Forte correlação linear = tendência
                if slope > 0:
                    regimes[pair] = "uptrend"
                else:
                    regimes[pair] = "downtrend"
            else:
                if volatility > 0.02:  # Threshold arbitrário
                    regimes[pair] = "volatile_range"
                else:
                    regimes[pair] = "stable_range"
        
        return regimes
    
    def optimize_with_genetic_algorithm(self, data, tp_pct, sl_pct, generations=8, population_size=30):
        """Otimiza parâmetros RSI usando algoritmo genético"""
        # Definir limites dos parâmetros
        param_bounds = {
            'rsi_buy': (10, 40),
            'rsi_sell': (60, 90),
            'rsi_delta': (2, 10),
            'rsi_min_long': (25, 45),
            'rsi_max_short': (55, 75),
            'timeout': (3, 15)
        }
        
        # Função para criar indivíduo aleatório
        def create_individual():
            return {
                'rsi_buy': random.uniform(*param_bounds['rsi_buy']),
                'rsi_sell': random.uniform(*param_bounds['rsi_sell']),
                'rsi_delta': random.uniform(*param_bounds['rsi_delta']),
                'rsi_min_long': random.uniform(*param_bounds['rsi_min_long']),
                'rsi_max_short': random.uniform(*param_bounds['rsi_max_short']),
                'timeout': random.randint(*param_bounds['timeout'])
            }
        
        # Função para avaliar fitness de um indivíduo
        def evaluate_fitness(individual):
            signals = generate_signals(data, individual)
            results = evaluate_signals(signals, tp_pct, sl_pct)
            
            # Fitness considera win rate e quantidade de trades
            fitness = results['win_rate'] * 0.8
            # Penalizar se tiver poucos trades
            if results['total_trades'] < 5:
                fitness *= 0.5
            
            return {
                'fitness': fitness,
                'win_rate': results['win_rate'],
                'total_trades': results['total_trades'],
                'expected_value': results['expected_value']
            }
        
        # Inicializar população aleatória
        population = [create_individual() for _ in range(population_size)]
        
        best_individual = None
        best_fitness = -float('inf')
        
        print(f"\nExecutando otimização genética por {generations} gerações...")
        
        for gen in range(generations):
            print(f"Geração {gen+1}/{generations}...")
            
            # Avaliar fitness de cada indivíduo
            fitness_results = []
            for ind in population:
                fitness_data = evaluate_fitness(ind)
                fitness_results.append({
                    'individual': ind,
                    'fitness': fitness_data['fitness'],
                    'win_rate': fitness_data['win_rate'],
                    'total_trades': fitness_data['total_trades'],
                    'expected_value': fitness_data['expected_value']
                })
            
            # Ordenar por fitness
            fitness_results.sort(key=lambda x: x['fitness'], reverse=True)
            
            # Atualizar melhor indivíduo encontrado
            if fitness_results[0]['fitness'] > best_fitness:
                best_fitness = fitness_results[0]['fitness']
                best_individual = fitness_results[0]
            
            # Selecionar os melhores para reprodução (elitismo)
            elite_count = int(population_size * 0.2)  # 20% da população
            elite = [res['individual'] for res in fitness_results[:elite_count]]
            
            # Criar nova população
            new_population = elite.copy()
            
            # Adicionar cruzamentos até completar a população
            while len(new_population) < population_size:
                # Seleção dos pais (torneio)
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                
                # Crossover
                child = {}
                for key in parent1:
                    if random.random() < 0.5:
                        child[key] = parent1[key]
                    else:
                        child[key] = parent2[key]
                
                # Mutação
                if random.random() < 0.2:  # 20% chance de mutação
                    param_to_mutate = random.choice(list(param_bounds.keys()))
                    if param_to_mutate == 'timeout':
                        child[param_to_mutate] = random.randint(*param_bounds[param_to_mutate])
                    else:
                        child[param_to_mutate] = random.uniform(*param_bounds[param_to_mutate])
                
                new_population.append(child)
            
            # Substituir população
            population = new_population
            
            # Mostrar status atual
            print(f"  Melhor fitness: {fitness_results[0]['fitness']:.2f}")
            print(f"  Melhor win rate: {fitness_results[0]['win_rate']:.2f}%")
            print(f"  Trades: {fitness_results[0]['total_trades']}")
        
        # Retornar o melhor indivíduo
        return best_individual
    
    def staged_optimization(self, data, tp_pct, sl_pct):
        """Otimização em várias etapas para refinar os melhores parâmetros"""
        # Etapa 1: Busca ampla com poucos valores para mapear o espaço
        print("Etapa 1/3: Busca inicial em espaço amplo...")
        coarse_params = {
            'rsi_buy': [20, 30],
            'rsi_sell': [70, 80],
            'rsi_delta': [3, 7],
            'rsi_min_long': [30],
            'rsi_max_short': [70],
            'timeout': [5]
        }
        
        # Executar primeira otimização
        best_coarse = self.find_best_parameters(tp_pct, sl_pct, coarse_params, validation_split=0.3, verbose=False)
        
        if not best_coarse:
            print("❌ Etapa 1 falhou: não foi possível encontrar parâmetros iniciais válidos.")
            return None
        
        # Etapa 2: Refinar em torno dos melhores resultados
        print("Etapa 2/3: Refinando busca em área promissora...")
        best_p = best_coarse['best_params']
        refined_params = {
            'rsi_buy': [best_p['rsi_buy']-5, best_p['rsi_buy'], best_p['rsi_buy']+5],
            'rsi_sell': [best_p['rsi_sell']-5, best_p['rsi_sell'], best_p['rsi_sell']+5],
            'rsi_delta': [best_p['rsi_delta']-1, best_p['rsi_delta'], best_p['rsi_delta']+1],
            'rsi_min_long': [best_p['rsi_min_long']-2, best_p['rsi_min_long'], best_p['rsi_min_long']+2],
            'rsi_max_short': [best_p['rsi_max_short']-2, best_p['rsi_max_short'], best_p['rsi_max_short']+2],
            'timeout': [best_p['timeout']]
        }
        
        # Executar otimização refinada
        best_refined = self.find_best_parameters(tp_pct, sl_pct, refined_params, validation_split=0.3, verbose=False)
        
        if not best_refined:
            print("❌ Etapa 2 falhou: não foi possível refinar os parâmetros.")
            return best_coarse  # Retorna resultado da etapa 1
        
        # Etapa 3: Otimização final com valores precisos
        print("Etapa 3/3: Otimização final com valores precisos...")
        best_p = best_refined['best_params']
        
        # Valores mais precisos para busca final
        final_params = {
            'rsi_buy': [best_p['rsi_buy']-2, best_p['rsi_buy']-1, best_p['rsi_buy'], 
                       best_p['rsi_buy']+1, best_p['rsi_buy']+2],
            'rsi_sell': [best_p['rsi_sell']-2, best_p['rsi_sell']-1, best_p['rsi_sell'], 
                        best_p['rsi_sell']+1, best_p['rsi_sell']+2],
            'rsi_delta': [max(1, best_p['rsi_delta']-0.5), best_p['rsi_delta'], 
                         best_p['rsi_delta']+0.5],
            'rsi_min_long': [best_p['rsi_min_long']-1, best_p['rsi_min_long'], 
                            best_p['rsi_min_long']+1],
            'rsi_max_short': [best_p['rsi_max_short']-1, best_p['rsi_max_short'], 
                             best_p['rsi_max_short']+1],
            'timeout': [max(1, best_p['timeout']-1), best_p['timeout'], 
                       best_p['timeout']+1]
        }
        
        # Executar otimização final
        best_final = self.find_best_parameters(tp_pct, sl_pct, final_params, validation_split=0.3, verbose=True)
        
        if not best_final:
            print("❌ Etapa 3 falhou: usando resultados da etapa 2.")
            return best_refined
        
        return best_final
    
    def find_best_parameters(self, tp_pct, sl_pct, rsi_params_ranges, validation_split=0.3, verbose=True):
        """
        Encontra os melhores parâmetros RSI para maximizar win rate
        com base nos TP/SL especificados
        """
        if self.data is None:
            print("❌ Erro: Dados não carregados.")
            return None
        
        data = self.data.copy()
        
        # Dividir dados em treino e validação
        timestamp_threshold = data['timestamp'].min() + (data['timestamp'].max() - data['timestamp'].min()) * (1 - validation_split)
        train_data = data[data['timestamp'] <= timestamp_threshold].copy()
        validation_data = data[data['timestamp'] > timestamp_threshold].copy()
        
        if verbose:
            print(f"\nDados divididos em conjuntos de treino e validação:")
            print(f"- Treino: {len(train_data)} registros ({train_data['timestamp'].min()} a {train_data['timestamp'].max()})")
            print(f"- Validação: {len(validation_data)} registros ({validation_data['timestamp'].min()} a {validation_data['timestamp'].max()})")
        
        # Extrair ranges de parâmetros
        rsi_buy_range = rsi_params_ranges.get('rsi_buy', [20, 25, 30])
        rsi_sell_range = rsi_params_ranges.get('rsi_sell', [70, 75, 80])
        rsi_delta_range = rsi_params_ranges.get('rsi_delta', [3, 4, 5])
        rsi_min_long_range = rsi_params_ranges.get('rsi_min_long', [30, 35])
        rsi_max_short_range = rsi_params_ranges.get('rsi_max_short', [60, 65])
        timeout_range = rsi_params_ranges.get('timeout', [5])
        
        # Gerar todas as combinações
        param_combinations = []
        for rsi_buy in rsi_buy_range:
            for rsi_sell in rsi_sell_range:
                for rsi_delta in rsi_delta_range:
                    for rsi_min_long in rsi_min_long_range:
                        for rsi_max_short in rsi_max_short_range:
                            for timeout in timeout_range:
                                params = {
                                    'rsi_buy': rsi_buy,
                                    'rsi_sell': rsi_sell,
                                    'rsi_delta': rsi_delta,
                                    'rsi_min_long': rsi_min_long,
                                    'rsi_max_short': rsi_max_short,
                                    'timeout': timeout
                                }
                                param_combinations.append((train_data, params, tp_pct, sl_pct))
        
        total_combinations = len(param_combinations)
        if verbose:
            print(f"\nTestando {total_combinations} combinações de parâmetros RSI...")
        start_time = time.time()
        
        # Executar avaliações em paralelo
        results = []
        with ProcessPoolExecutor() as executor:
            for idx, result in enumerate(executor.map(evaluate_rsi_params, param_combinations)):
                results.append(result)
                if verbose and ((idx + 1) % 10 == 0 or (idx + 1) == total_combinations):
                    print(f"Progresso: {idx + 1}/{total_combinations} combinações testadas")
        
        # Converter resultados para DataFrame
        results_df = pd.DataFrame(results)
        
        # Ordenar por win rate decrescente
        results_df = results_df.sort_values(['win_rate', 'expected_value', 'total_trades'], 
                                          ascending=[False, False, False])
        
        # Extrair os melhores parâmetros
        if not results_df.empty:
            best_params = results_df.iloc[0]['params']
            best_win_rate = results_df.iloc[0]['win_rate']
            best_trades = results_df.iloc[0]['total_trades']
            best_ev = results_df.iloc[0]['expected_value']
            
            elapsed = time.time() - start_time
            if verbose:
                print(f"\nOtimização concluída em {elapsed:.1f} segundos")
            
            # Validar os melhores parâmetros no conjunto de validação
            if verbose:
                print("\nValidando os melhores parâmetros no conjunto de validação...")
            
            validation_signals = generate_signals(validation_data, best_params)
            validation_results = evaluate_signals(validation_signals, tp_pct, sl_pct)
            
            if verbose:
                print("\n===== RESULTADOS DA OTIMIZAÇÃO =====")
                print("\nMelhores parâmetros RSI encontrados:")
                print(f"RSI_BUY_THRESHOLD = {best_params['rsi_buy']}")
                print(f"RSI_SELL_THRESHOLD = {best_params['rsi_sell']}")
                print(f"RSI_DELTA_MIN = {best_params['rsi_delta']}")
                print(f"RSI_MIN_LONG = {best_params['rsi_min_long']}")
                print(f"RSI_MAX_SHORT = {best_params['rsi_max_short']}")
                print(f"ATTENTIVE_MODE_TIMEOUT_MINUTES = {best_params['timeout']}")
                
                print("\nPerformance no conjunto de treino:")
                print(f"- Win Rate: {best_win_rate:.2f}%")
                print(f"- Número de Trades: {best_trades}")
                print(f"- Expectativa matemática: {best_ev:.4f}")
                
                print("\nPerformance no conjunto de validação:")
                print(f"- Win Rate: {validation_results['win_rate']:.2f}%")
                print(f"- Número de Trades: {validation_results['total_trades']}")
                print(f"- Expectativa matemática: {validation_results['expected_value']:.4f}")
                
                # Verificar diferença entre treino e validação
                win_rate_diff = abs(best_win_rate - validation_results['win_rate'])
                
                if win_rate_diff > 10:
                    print("\n⚠️ AVISO: Grande diferença entre win rates de treino e validação!")
                    print("Isso pode indicar overfitting. Considere parâmetros mais conservadores.")
                
                # Gerar visualizações
                self.visualize_results(results_df, best_params)
                
                # Salvar configuração
                with open(f"{self.results_dir}/rsi_config_recomendado.py", "w") as f:
                    f.write("# Configurações RSI otimizadas\n\n")
                    f.write(f"RSI_PERIOD = 10  # Período fixo conforme solicitado\n")
                    f.write(f"RSI_BUY_THRESHOLD = {best_params['rsi_buy']}\n")
                    f.write(f"RSI_SELL_THRESHOLD = {best_params['rsi_sell']}\n")
                    f.write(f"RSI_DELTA_MIN = {best_params['rsi_delta']}\n")
                    f.write(f"RSI_MIN_LONG = {best_params['rsi_min_long']}\n")
                    f.write(f"RSI_MAX_SHORT = {best_params['rsi_max_short']}\n")
                    f.write(f"RSI_EXIT_LONG_THRESHOLD = 70  # Mantido valor padrão\n")
                    f.write(f"RSI_EXIT_SHORT_THRESHOLD = 35  # Mantido valor padrão\n")
                    f.write(f"ATTENTIVE_MODE_TIMEOUT_MINUTES = {best_params['timeout']}\n")
                
                print(f"\nConfigurações RSI recomendadas salvas em: {self.results_dir}/rsi_config_recomendado.py")
            
            return {
                'best_params': best_params,
                'train_results': {
                    'win_rate': best_win_rate,
                    'trades': best_trades,
                    'expected_value': best_ev
                },
                'validation_results': validation_results
            }
        else:
            print("❌ Erro: Nenhum resultado válido encontrado.")
            return None
    
    def visualize_results(self, results_df, best_params):
        """Gera visualizações dos resultados da otimização"""
        try:
            # Extrair parâmetros para análise
            params_data = pd.DataFrame([
                {
                    'rsi_buy': p['rsi_buy'],
                    'rsi_sell': p['rsi_sell'], 
                    'rsi_delta': p['rsi_delta'],
                    'rsi_min_long': p['rsi_min_long'],
                    'rsi_max_short': p['rsi_max_short'],
                    'win_rate': wr,
                    'trades': t
                }
                for p, wr, t in zip(results_df['params'], 
                                   results_df['win_rate'], 
                                   results_df['total_trades'])
            ])
            
            # Mapas de calor para visualizar impacto dos parâmetros
            plt.figure(figsize=(16, 12))
            
            # 1. RSI Buy vs RSI Sell
            plt.subplot(2, 2, 1)
            pivot_buy_sell = params_data.pivot_table(
                index='rsi_buy', 
                columns='rsi_sell', 
                values='win_rate',
                aggfunc='mean'
            )
            sns.heatmap(pivot_buy_sell, annot=True, fmt=".1f", cmap="RdYlGn")
            plt.title('Win Rate: RSI Buy vs RSI Sell')
            
            # 2. RSI Buy vs RSI Delta
            plt.subplot(2, 2, 2)
            pivot_buy_delta = params_data.pivot_table(
                index='rsi_buy', 
                columns='rsi_delta', 
                values='win_rate',
                aggfunc='mean'
            )
            sns.heatmap(pivot_buy_delta, annot=True, fmt=".1f", cmap="RdYlGn")
            plt.title('Win Rate: RSI Buy vs RSI Delta')
            
            # 3. RSI Min Long vs RSI Max Short
            plt.subplot(2, 2, 3)
            pivot_min_max = params_data.pivot_table(
                index='rsi_min_long', 
                columns='rsi_max_short', 
                values='win_rate',
                aggfunc='mean'
            )
            sns.heatmap(pivot_min_max, annot=True, fmt=".1f", cmap="RdYlGn")
            plt.title('Win Rate: RSI Min Long vs RSI Max Short')
            
            # 4. Gráfico de barras para impacto de cada parâmetro
            plt.subplot(2, 2, 4)
            # Criar gráficos de linhas para cada parâmetro
            x_values = np.arange(5)
            width = 0.15
            
            # Agrupar por cada parâmetro e calcular win rate médio
            buy_impact = params_data.groupby('rsi_buy')['win_rate'].mean()
            sell_impact = params_data.groupby('rsi_sell')['win_rate'].mean()
            delta_impact = params_data.groupby('rsi_delta')['win_rate'].mean()
            
            plt.bar(x_values - 2*width, 
                   [0, buy_impact.max() - buy_impact.min(), 0, 0, 0], 
                   width=width, label='RSI Buy')
            
            plt.bar(x_values - width, 
                   [0, 0, sell_impact.max() - sell_impact.min(), 0, 0], 
                   width=width, label='RSI Sell')
            
            plt.bar(x_values, 
                   [0, 0, 0, delta_impact.max() - delta_impact.min(), 0], 
                   width=width, label='RSI Delta')
            
            plt.bar(x_values + width, 
                   [0, 0, 0, 0, params_data.groupby('rsi_min_long')['win_rate'].mean().max() - 
                    params_data.groupby('rsi_min_long')['win_rate'].mean().min()], 
                   width=width, label='Min Long')
            
            plt.bar(x_values + 2*width, 
                   [params_data.groupby('rsi_max_short')['win_rate'].mean().max() - 
                    params_data.groupby('rsi_max_short')['win_rate'].mean().min(), 0, 0, 0, 0], 
                   width=width, label='Max Short')
            
            plt.legend()
            plt.title('Impacto de cada parâmetro no Win Rate')
            plt.xticks(x_values, ['Max Short', 'Buy', 'Sell', 'Delta', 'Min Long'])
            plt.ylabel('Variação no Win Rate (%)')
            
            plt.tight_layout()
            plt.savefig(f"{self.results_dir}/rsi_param_impact.png")
            plt.close()
            
            print(f"Visualizações salvas em {self.results_dir}/rsi_param_impact.png")
        
        except Exception as e:
            print(f"Erro ao gerar visualizações: {str(e)}")
            traceback.print_exc()


def main():
    print("\n" + "="*60)
    print("    OTIMIZADOR INTELIGENTE DE RSI    ")
    print("="*60)
    
    try:
        optimizer = SimpleRSIOptimizer()
        
        if not optimizer.load_data():
            return
        
        # Perguntar sobre objetivos de trading
        print("\n📊 QUAL SEU PERFIL DE TRADING?")
        print("1. Conservador (prefere menos trades, maior win rate)")
        print("2. Moderado (equilíbrio entre frequência e qualidade)")
        print("3. Agressivo (mais trades, aceitando win rate menor)")
        profile = input("Escolha (1-3): ")
        
        # Configurar TP/SL com base no perfil
        if profile == "1":  # Conservador
            suggested_tp = 1.5
            suggested_sl = 0.5
        elif profile == "3":  # Agressivo
            suggested_tp = 0.5
            suggested_sl = 0.8
        else:  # Moderado
            suggested_tp = 0.8
            suggested_sl = 0.7
        
        print(f"\n📊 CONFIGURAÇÕES DE TP/SL (sugeridas para seu perfil):")
        tp_input = input(f"Take Profit em % (sugerido: {suggested_tp}): ")
        tp = float(tp_input if tp_input.strip() else suggested_tp) / 100
        
        sl_input = input(f"Stop Loss em % (sugerido: {suggested_sl}): ")
        sl = float(sl_input if sl_input.strip() else suggested_sl) / 100
        
        print("\n🧠 ANÁLISE DE MERCADO EM ANDAMENTO...")
        # Detectar regimes e sugerir parâmetros
        regimes = optimizer.detect_market_regime(optimizer.data)
        suggested_ranges = optimizer.suggest_rsi_ranges(optimizer.data)
        
        # Mostrar estatísticas dos dados e regimes detectados
        pairs = optimizer.data['pair'].unique()
        print(f"\nRegimes de mercado detectados para {len(pairs)} pares:")
        for pair in pairs:
            if pair in regimes:
                print(f"- {pair}: {regimes[pair]}")
        
        print("\n📈 PARÂMETROS SUGERIDOS BASEADOS NOS DADOS:")
        print(f"RSI_BUY_THRESHOLD: {suggested_ranges['rsi_buy']}")
        print(f"RSI_SELL_THRESHOLD: {suggested_ranges['rsi_sell']}")
        print(f"RSI_DELTA_MIN: {suggested_ranges['rsi_delta']}")
        print(f"RSI_MIN_LONG: {suggested_ranges['rsi_min_long']}")
        print(f"RSI_MAX_SHORT: {suggested_ranges['rsi_max_short']}")
        
        print("\n⚙️ OPÇÕES DE OTIMIZAÇÃO:")
        print("1. Usar parâmetros sugeridos")
        print("2. Definir parâmetros personalizados")
        print("3. Usar algoritmo genético (mais rápido)")
        print("4. Otimização por etapas (mais preciso)")
        option = input("Escolha (1-4): ")
        
        # Implementar lógica baseada na opção escolhida
        if option == "1":
            # Usar parâmetros sugeridos
            param_ranges = suggested_ranges
            param_ranges['timeout'] = [5]  # Valor padrão para timeout
            
            # Executar otimização com os parâmetros sugeridos
            results = optimizer.find_best_parameters(tp, sl, param_ranges, validation_split=0.3)
            
        elif option == "2":
            # Definir parâmetros personalizados
            print("\n📊 DEFINA INTERVALOS PARA OS PARÂMETROS:")
            print("(Os valores podem ser inseridos separados por vírgula, ex: 20,25,30)")
            
            # RSI Buy Threshold
            rsi_buy_input = input(f"RSI_BUY_THRESHOLD a testar (sugerido: {','.join(map(str, suggested_ranges['rsi_buy']))}): ")
            rsi_buy_values = [float(x.strip()) for x in rsi_buy_input.split(',')] if rsi_buy_input.strip() else suggested_ranges['rsi_buy']
            
            # RSI Sell Threshold
            rsi_sell_input = input(f"RSI_SELL_THRESHOLD a testar (sugerido: {','.join(map(str, suggested_ranges['rsi_sell']))}): ")
            rsi_sell_values = [float(x.strip()) for x in rsi_sell_input.split(',')] if rsi_sell_input.strip() else suggested_ranges['rsi_sell']
            
            # RSI Delta Min
            rsi_delta_input = input(f"RSI_DELTA_MIN a testar (sugerido: {','.join(map(str, suggested_ranges['rsi_delta']))}): ")
            rsi_delta_values = [float(x.strip()) for x in rsi_delta_input.split(',')] if rsi_delta_input.strip() else suggested_ranges['rsi_delta']
            
            # RSI Min Long
            rsi_min_long_input = input(f"RSI_MIN_LONG a testar (sugerido: {','.join(map(str, suggested_ranges['rsi_min_long']))}): ")
            rsi_min_long_values = [float(x.strip()) for x in rsi_min_long_input.split(',')] if rsi_min_long_input.strip() else suggested_ranges['rsi_min_long']
            
            # RSI Max Short
            rsi_max_short_input = input(f"RSI_MAX_SHORT a testar (sugerido: {','.join(map(str, suggested_ranges['rsi_max_short']))}): ")
            rsi_max_short_values = [float(x.strip()) for x in rsi_max_short_input.split(',')] if rsi_max_short_input.strip() else suggested_ranges['rsi_max_short']
            
            # Timeout
            timeout_input = input("ATTENTIVE_MODE_TIMEOUT_MINUTES a testar (padrão: 5): ")
            timeout_values = [int(x.strip()) for x in timeout_input.split(',')] if timeout_input.strip() else [5]
            
            # Porcentagem de validação
            validation_pct = float(input("\nPorcentagem de dados para validação (ex: 30): ")) / 100
            
            # Criar dicionário de parâmetros
            param_ranges = {
                'rsi_buy': rsi_buy_values,
                'rsi_sell': rsi_sell_values,
                'rsi_delta': rsi_delta_values,
                'rsi_min_long': rsi_min_long_values,
                'rsi_max_short': rsi_max_short_values,
                'timeout': timeout_values
            }
            
            # Executar otimização
            results = optimizer.find_best_parameters(tp, sl, param_ranges, validation_split=validation_pct)
            
        elif option == "3":
            # Usar algoritmo genético
            print("\nExecutando otimização com algoritmo genético...")
            
            # Perguntar número de gerações e tamanho da população
            gen_input = input("Número de gerações (padrão: 8): ")
            generations = int(gen_input) if gen_input.strip() else 8
            
            pop_input = input("Tamanho da população (padrão: 30): ")
            population = int(pop_input) if pop_input.strip() else 30
            
            # Executar otimização genética
            results = optimizer.optimize_with_genetic_algorithm(
                optimizer.data, tp, sl, generations=generations, population_size=population)
            
            if results:
                best_params = results['individual']
                
                print("\n===== RESULTADOS DA OTIMIZAÇÃO GENÉTICA =====")
                print("\nMelhores parâmetros RSI encontrados:")
                print(f"RSI_BUY_THRESHOLD = {best_params['rsi_buy']}")
                print(f"RSI_SELL_THRESHOLD = {best_params['rsi_sell']}")
                print(f"RSI_DELTA_MIN = {best_params['rsi_delta']}")
                print(f"RSI_MIN_LONG = {best_params['rsi_min_long']}")
                print(f"RSI_MAX_SHORT = {best_params['rsi_max_short']}")
                print(f"ATTENTIVE_MODE_TIMEOUT_MINUTES = {best_params['timeout']}")
                
                print("\nPerformance:")
                print(f"- Win Rate: {results['win_rate']:.2f}%")
                print(f"- Número de Trades: {results['total_trades']}")
                print(f"- Expectativa matemática: {results['expected_value']:.4f}")
                
                # Salvar configuração
                with open(f"{optimizer.results_dir}/rsi_config_genetico.py", "w") as f:
                    f.write("# Configurações RSI otimizadas com algoritmo genético\n\n")
                    f.write(f"RSI_PERIOD = 10  # Período fixo conforme solicitado\n")
                    f.write(f"RSI_BUY_THRESHOLD = {best_params['rsi_buy']}\n")
                    f.write(f"RSI_SELL_THRESHOLD = {best_params['rsi_sell']}\n")
                    f.write(f"RSI_DELTA_MIN = {best_params['rsi_delta']}\n")
                    f.write(f"RSI_MIN_LONG = {best_params['rsi_min_long']}\n")
                    f.write(f"RSI_MAX_SHORT = {best_params['rsi_max_short']}\n")
                    f.write(f"RSI_EXIT_LONG_THRESHOLD = 70  # Mantido valor padrão\n")
                    f.write(f"RSI_EXIT_SHORT_THRESHOLD = 35  # Mantido valor padrão\n")
                    f.write(f"ATTENTIVE_MODE_TIMEOUT_MINUTES = {best_params['timeout']}\n")
                
                print(f"\nConfigurações RSI recomendadas salvas em: {optimizer.results_dir}/rsi_config_genetico.py")
            
        elif option == "4":
            # Otimização por etapas
            print("\nExecutando otimização por etapas (mais detalhada)...")
            
            # Executar otimização por etapas
            results = optimizer.staged_optimization(optimizer.data, tp, sl)
            
            if results:
                print("\n===== RESULTADO FINAL DA OTIMIZAÇÃO POR ETAPAS =====")
                best_params = results['best_params']
                validation_results = results['validation_results']
                
                print("\nMelhores parâmetros RSI encontrados:")
                print(f"RSI_BUY_THRESHOLD = {best_params['rsi_buy']}")
                print(f"RSI_SELL_THRESHOLD = {best_params['rsi_sell']}")
                print(f"RSI_DELTA_MIN = {best_params['rsi_delta']}")
                print(f"RSI_MIN_LONG = {best_params['rsi_min_long']}")
                print(f"RSI_MAX_SHORT = {best_params['rsi_max_short']}")
                print(f"ATTENTIVE_MODE_TIMEOUT_MINUTES = {best_params['timeout']}")
                
                print("\nPerformance no conjunto de validação:")
                print(f"- Win Rate: {validation_results['win_rate']:.2f}%")
                print(f"- Número de Trades: {validation_results['total_trades']}")
                print(f"- Expectativa matemática: {validation_results['expected_value']:.4f}")
                
                # Salvar configuração
                with open(f"{optimizer.results_dir}/rsi_config_etapas.py", "w") as f:
                    f.write("# Configurações RSI otimizadas por etapas\n\n")
                    f.write(f"RSI_PERIOD = 10  # Período fixo conforme solicitado\n")
                    f.write(f"RSI_BUY_THRESHOLD = {best_params['rsi_buy']}\n")
                    f.write(f"RSI_SELL_THRESHOLD = {best_params['rsi_sell']}\n")
                    f.write(f"RSI_DELTA_MIN = {best_params['rsi_delta']}\n")
                    f.write(f"RSI_MIN_LONG = {best_params['rsi_min_long']}\n")
                    f.write(f"RSI_MAX_SHORT = {best_params['rsi_max_short']}\n")
                    f.write(f"RSI_EXIT_LONG_THRESHOLD = 70  # Mantido valor padrão\n")
                    f.write(f"RSI_EXIT_SHORT_THRESHOLD = 35  # Mantido valor padrão\n")
                    f.write(f"ATTENTIVE_MODE_TIMEOUT_MINUTES = {best_params['timeout']}\n")
                
                print(f"\nConfigurações RSI recomendadas salvas em: {optimizer.results_dir}/rsi_config_etapas.py")
            else:
                print("❌ Otimização por etapas falhou ao encontrar parâmetros ideais.")
    
    except KeyboardInterrupt:
        print("\n\nOtimização interrompida pelo usuário.")
    except ValueError as e:
        print(f"\nErro de valor: {str(e)}")
    except Exception as e:
        print(f"\nErro durante a execução: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()