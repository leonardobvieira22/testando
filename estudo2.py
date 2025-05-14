import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import multiprocessing as mp
from datetime import datetime, timedelta
import os
import time
import joblib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
warnings.filterwarnings('ignore')

@dataclass
class TradeParameters:
    """Structured trade parameters for cleaner code"""
    tp: float
    sl: float
    capital: float
    order_size: float
    leverage: float
    max_consecutive_losses: int = 3
    position_reducer: float = 0.8
    min_position_size: float = 0.25

# Função externa para processamento paralelo - deve estar FORA da classe
def _process_params_for_grid(args):
    """Função auxiliar para processamento de parâmetros em paralelo"""
    simulator, tp, sl, capital, order_size, leverage, max_sl, pause_min, metric = args
    print(f"Testando TP={tp*100:.2f}%, SL={sl*100:.2f}%")
    df_results, df_capital, stats = simulator.simulate(
        tp, sl, capital, order_size, leverage, max_sl, pause_min, verbose=False)
    
    # Determinar valor da métrica
    metric_value = stats.get('total_pnl' if metric == 'total_pnl' else 
                           'return_pct' if metric == 'return_pct' else
                           'profit_factor' if metric == 'profit_factor' else
                           'win_rate', 0)
    
    return {
        'tp': tp,
        'sl': sl,
        'results': df_results,
        'capital_history': df_capital,
        'stats': stats,
        'metric': metric,
        'metric_value': metric_value
    }
    
class EstudoTraderV2:
    """
    Next-generation trading simulator with high performance and realistic behavior
    """
    
    def __init__(self, signals_file='signals.csv', prices_file='prices_statistics.csv'):
        self.signals_file = signals_file
        self.prices_file = prices_file
        self.signals = None
        self.prices = None
        self.price_index = None
        self.market_params = {
            'commission': 0.0004,  # 0.04% commission
            'spread': 0.0001,      # 0.01% spread
            'slippage': 0.0001     # 0.01% slippage
        }
        self.results_dir = 'resultados'
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Novo - Parâmetros de risco
        self.risk_params = {
            'max_leverage': 10,                # Alavancagem máxima recomendada
            'min_capital': 100,                # Capital mínimo recomendado em USDT
            'liquidation_threshold_factor': 2, # Multiplicador para threshold de liquidação
            'high_leverage_warning': 5,        # Nível a partir do qual mostra aviso de alavancagem
            'extreme_leverage_warning': 10     # Nível a partir do qual mostra aviso extremo
        }
        
    def set_market_params(self, commission=0.0004, spread=0.0001, slippage=0.0001):
        """
        Define parâmetros realistas de mercado
        
        Args:
            commission (float): Taxa de comissão (ex: 0.0004 = 0.04%)
            spread (float): Spread médio (ex: 0.0001 = 0.01%)
            slippage (float): Slippage médio (ex: 0.0001 = 0.01%)
        """
        self.market_params = {
            'commission': commission,
            'spread': spread,
            'slippage': slippage
        }
        print(f"Parâmetros de mercado definidos: comissão={commission*100:.4f}%, "
              f"spread={spread*100:.4f}%, slippage={slippage*100:.4f}%")
        
    def validate_parameters(self, capital: float, order_size: float, leverage: float) -> Tuple[bool, str]:
        """
        Valida parâmetros de trading para garantir que são realistas e seguros
        
        Args:
            capital (float): Capital inicial
            order_size (float): Tamanho da ordem
            leverage (float): Alavancagem

        Returns:
            Tuple[bool, str]: (válido, mensagem)
        """
        # Verificar capital mínimo
        if capital < self.risk_params['min_capital']:
            return False, f"Capital muito baixo. Recomendado mínimo de {self.risk_params['min_capital']} USDT para trading realista."
            
        # Verificar alavancagem máxima
        if leverage > self.risk_params['max_leverage']:
            return False, f"Alavancagem {leverage}x excede o limite recomendado de {self.risk_params['max_leverage']}x. Risco extremamente alto."
            
        # Verificar tamanho da ordem
        if order_size > capital * 0.5:
            return False, f"Tamanho da ordem ({order_size} USDT) excede 50% do capital ({capital} USDT). Risco de ruína."
            
        # Verificar tamanho mínimo da ordem
        if order_size < 5:
            return False, f"Tamanho da ordem muito pequeno ({order_size} USDT). Recomendado mínimo de 5 USDT para considerar comissões e custos."
            
        return True, "Parâmetros válidos"
        
    def load_data(self) -> bool:
        """Load and preprocess data with optimized performance"""
        try:
            print(f"\nCarregando dados de {self.signals_file} e {self.prices_file}...")
            
            # Tentar carregar sinais
            signals = pd.read_csv(self.signals_file)
            
            # Verificar e converter timestamps
            timestamp_cols = ['timestamp', 'bb_lower', 'time', 'timeframe']
            timestamp_col = next((col for col in timestamp_cols if col in signals.columns and not signals[col].isna().all()), None)
            
            if not timestamp_col:
                print("❌ Erro: Não foi possível encontrar coluna de timestamp válida nos sinais.")
                return False
                
            if timestamp_col != 'timestamp':
                print(f"Aviso: Usando coluna '{timestamp_col}' como timestamp para sinais.")
                signals['timestamp'] = pd.to_datetime(signals[timestamp_col], errors='coerce')
            else:
                signals['timestamp'] = pd.to_datetime(signals['timestamp'], errors='coerce')
                
            # Verificar se temos valores válidos de timestamp
            if signals['timestamp'].isna().all():
                print("❌ Erro: Todos os timestamps dos sinais são inválidos.")
                return False
                
            # Verificar coluna de direção
            direction_col = None
            if 'direction' in signals.columns:
                direction_col = 'direction'
            elif 'side' in signals.columns:
                direction_col = 'side'
                signals.rename(columns={'side': 'direction'}, inplace=True)
                
            if not direction_col:
                print("❌ Erro: Não foi encontrada coluna 'direction' ou 'side' nos sinais.")
                return False
                
            # Carregar preços
            prices = pd.read_csv(self.prices_file)
            prices['timestamp'] = pd.to_datetime(prices['timestamp'], errors='coerce')
            
            # Verificar colunas essenciais nos preços
            if 'price' not in prices.columns:
                print("❌ Erro: Coluna 'price' não encontrada nos dados de preço.")
                return False
                
            # Verificar pares em comum
            signal_pairs = set(signals['pair'].unique())
            price_pairs = set(prices['pair'].unique())
            common_pairs = signal_pairs.intersection(price_pairs)
            
            if not common_pairs:
                print("❌ Erro: Não há pares de trading em comum entre sinais e preços.")
                print(f"Pares nos sinais: {signal_pairs}")
                print(f"Pares nos preços: {price_pairs}")
                return False
                
            # Filtrar sinais por pares em comum
            signals = signals[signals['pair'].isin(common_pairs)]
            
            # Normalizar direções
            signals['direction'] = signals['direction'].str.upper()
            signals['direction'] = signals['direction'].replace({'COMPRA': 'BUY', 'VENDA': 'SELL'})
            
            # Criar índice de preços para acesso rápido
            prices = prices.sort_values(['pair', 'timestamp'])
            price_index = {pair: grp for pair, grp in prices.groupby('pair')}
            
            # Estatísticas dos dados
            min_signal_time = signals['timestamp'].min()
            max_signal_time = signals['timestamp'].max()
            min_price_time = prices['timestamp'].min()
            max_price_time = prices['timestamp'].max()
            
            print("\n✅ Dados carregados com sucesso:")
            print(f"- {len(signals)} sinais de trading")
            print(f"- {len(prices)} registros de preço")
            print(f"- Pares de trading em comum: {', '.join(common_pairs)}")
            print(f"- Período dos sinais: {min_signal_time} a {max_signal_time}")
            print(f"- Período dos preços: {min_price_time} a {max_price_time}")
            
            if min_signal_time < min_price_time:
                print(f"⚠️ Aviso: Alguns sinais são anteriores ao primeiro preço disponível.")
            if max_signal_time > max_price_time:
                print(f"⚠️ Aviso: Alguns sinais são posteriores ao último preço disponível.")
                
            # Salvar dados processados
            self.signals = signals
            self.prices = prices
            self.price_index = price_index
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao carregar dados: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def simulate_trade(self, signal: pd.Series, params: TradeParameters) -> dict:
        """Process a single trade with realistic conditions"""
        # Get price data for this signal
        pair = signal['pair']
        direction = signal['direction']
        timestamp = signal['timestamp']
        
        if pair not in self.price_index:
            return {'status': 'no_price_data', 'pnl': 0}
            
        # Get prices after signal timestamp
        pair_prices = self.price_index[pair]
        prices_after = pair_prices[pair_prices['timestamp'] >= timestamp]
        
        if len(prices_after) < 2:
            return {'status': 'insufficient_price_data', 'pnl': 0}
            
        # Calculate entry with spread and slippage
        entry_price = prices_after.iloc[0]['price']
        
        # MELHORADO: Slippage dinâmico baseado em múltiplos fatores de volatilidade
        # 1. Volatilidade recente (1 hora)
        recent_prices = pair_prices[
            (pair_prices['timestamp'] < timestamp) & 
            (pair_prices['timestamp'] >= timestamp - pd.Timedelta(hours=1))
        ]
        
        volatility_factor = 1.0
        if len(recent_prices) > 5:
            # Volatilidade baseada no desvio padrão dos retornos
            volatility = recent_prices['price'].pct_change().std() * 100
            
            # Volatilidade baseada na amplitude (high-low)
            if 'high' in recent_prices.columns and 'low' in recent_prices.columns:
                high_low_range = (recent_prices['high'].max() - recent_prices['low'].min()) / recent_prices['price'].mean()
                volatility = max(volatility, high_low_range * 100)
                
            # Aplicar fator de volatilidade com limite mais realista
            volatility_factor = 1.0 + min(volatility * 3, 10.0)  # Até 10x em volatilidade extrema
            
        # 2. Horário de negociação (maior durante horários de baixa liquidez)
        hour = timestamp.hour
        time_factor = 1.0
        if hour >= 22 or hour <= 4:  # Horários de menor liquidez (noite/madrugada)
            time_factor = 1.5
            
        # 3. Volume (se disponível)
        volume_factor = 1.0
        if 'volume' in recent_prices.columns:
            avg_volume = recent_prices['volume'].mean()
            if avg_volume > 0:
                recent_volume = prices_after.iloc[0].get('volume', avg_volume)
                volume_ratio = avg_volume / max(recent_volume, 1)  # Evitar divisão por zero
                volume_factor = min(1.5, max(1.0, volume_ratio))  # Limite entre 1.0 e 1.5
                
        # Slippage dinâmico final (combinação de todos os fatores)
        dynamic_slippage = self.market_params['slippage'] * volatility_factor * time_factor * volume_factor
        
        # MELHORADO: Liquidação mais realista baseada na alavancagem
        # Cálculo do threshold de liquidação (% de movimento adverso que causa liquidação)
        # Fator 1.5 representa margem de segurança da exchange antes da liquidação total
        base_liquidation_threshold = 0.9 / params.leverage  # 90% da margem  
        
        if params.leverage >= 20:
            # Em alavancagens extremas (20x+), as exchanges são mais agressivas
            # e liquidam mais rapidamente para proteger seu capital
            safety_factor = 2.0
        elif params.leverage >= 10:
            safety_factor = 1.7
        else:
            safety_factor = 1.5
            
        liquidation_threshold = base_liquidation_threshold / safety_factor
        
        # Calculate effective entry price with spread and slippage
        if direction == 'BUY':
            effective_entry = entry_price * (1 + self.market_params['spread'] + dynamic_slippage)
            tp_price = effective_entry * (1 + params.tp)
            sl_price = effective_entry * (1 - params.sl)
            
            # Liquidation price (mais realista agora)
            liquidation_price = effective_entry * (1 - liquidation_threshold)
        else:  # SELL
            effective_entry = entry_price * (1 - self.market_params['spread'] - dynamic_slippage)
            tp_price = effective_entry * (1 - params.tp)
            sl_price = effective_entry * (1 + params.sl)
            
            # Liquidation price (mais realista agora)
            liquidation_price = effective_entry * (1 + liquidation_threshold)
            
        # Calculate entry commission
        entry_commission = params.order_size * self.market_params['commission']
        
        # Track price movement
        result = None
        exit_price = None
        exit_time = None
        max_adverse_move = 0
        
        for _, price_row in prices_after.iloc[1:].iterrows():
            current_price = price_row['price']
            current_time = price_row['timestamp']
            
            # Calculate adverse movement
            if direction == 'BUY':
                adverse_move = (effective_entry - current_price) / effective_entry
                
                # MELHORADO: Check for liquidation first (realistic behavior)
                # A liquidação acontece antes do SL, especialmente com alta alavancagem
                if current_price <= liquidation_price:
                    result = 'LIQUIDATION'
                    exit_price = current_price
                    exit_time = current_time
                    break
                    
                # Check TP
                if current_price >= tp_price:
                    result = 'TP'
                    # Slippage dinâmico também na saída, mas menor em TP (mercado a favor)
                    exit_price = current_price * (1 - self.market_params['spread'] - (dynamic_slippage * 0.5))
                    exit_time = current_time
                    break
                    
                # Check SL
                if current_price <= sl_price:
                    result = 'SL'
                    # Slippage pior em SL (mercado contra + possível "gap" no livro de ordens)
                    exit_price = current_price * (1 - self.market_params['spread'] - (dynamic_slippage * 1.5))
                    exit_time = current_time
                    break
            else:  # SELL
                adverse_move = (current_price - effective_entry) / effective_entry
                
                # Similar checks for SELL direction
                if current_price >= liquidation_price:
                    result = 'LIQUIDATION'
                    exit_price = current_price
                    exit_time = current_time
                    break
                    
                if current_price <= tp_price:
                    result = 'TP'
                    exit_price = current_price * (1 + self.market_params['spread'] + (dynamic_slippage * 0.5))
                    exit_time = current_time
                    break
                    
                if current_price >= sl_price:
                    result = 'SL'
                    exit_price = current_price * (1 + self.market_params['spread'] + (dynamic_slippage * 1.5))
                    exit_time = current_time
                    break
                    
            # Track maximum adverse movement
            max_adverse_move = max(max_adverse_move, adverse_move)
        
        # If no result by end of available data
        if result is None:
            last_price = prices_after.iloc[-1]['price']
            result = 'OPEN'
            exit_price = last_price
            exit_time = prices_after.iloc[-1]['timestamp']
        
        # Calculate position size and PnL
        position_value = params.order_size * params.leverage
        
        # Calculate PnL
        if result == 'LIQUIDATION':
            # Liquidação perde quase todo o capital da ordem
            # Exchanges tipicamente ficam com 97-100% da margem em liquidações
            pnl = -params.order_size * 0.97  # 97% de perda na liquidação
        else:
            # Calculate percentage price change
            if direction == 'BUY':
                price_change = exit_price / effective_entry - 1
            else:
                price_change = 1 - exit_price / effective_entry
                
            # Calculate PnL
            gross_pnl = position_value * price_change
            exit_commission = position_value * self.market_params['commission']
            pnl = gross_pnl - entry_commission - exit_commission
        
        # Duration in minutes
        duration = (exit_time - timestamp).total_seconds() / 60
        
        return {
            'signal_id': signal.get('signal_id', ''),
            'pair': pair,
            'direction': direction,
            'entry_time': timestamp,
            'exit_time': exit_time,
            'entry_price': effective_entry,
            'exit_price': exit_price,
            'result': result,
            'pnl': pnl,
            'duration_minutes': duration,
            'max_adverse_move': max_adverse_move * 100,  # as percentage
            'order_size': params.order_size,
            'leverage': params.leverage,
            'commissions': entry_commission + (position_value * self.market_params['commission']),
            'dynamic_slippage': dynamic_slippage * 100,  # como percentual
            'liquidation_threshold': liquidation_threshold * 100  # como percentual
        }

    def simulate(self, tp: float, sl: float, initial_capital: float, 
                order_size: float, leverage: float, max_consecutive_sl: int = 3,
                pause_minutes: int = 120, verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Simula backtest completo com regras realistas de mercado
        
        Args:
            tp (float): Take Profit em formato decimal (ex: 0.01 = 1%)
            sl (float): Stop Loss em formato decimal (ex: 0.005 = 0.5%)
            initial_capital (float): Capital inicial em USDT
            order_size (float): Tamanho da ordem em USDT
            leverage (float): Multiplicador de alavancagem
            max_consecutive_sl (int): Número máximo de SL consecutivos antes de pausar par
            pause_minutes (int): Duração da pausa em minutos
            verbose (bool): Se True, exibe progresso da simulação
            
        Returns:
            Tuple: (resultados, histórico_capital, estatísticas)
        """
        # MELHORADO: Validação de parâmetros de trading
        is_valid, validation_message = self.validate_parameters(initial_capital, order_size, leverage)
        if not is_valid:
            print(f"\n⚠️ AVISO DE PARÂMETROS INVÁLIDOS: {validation_message}")
            print("Continuando simulação apenas para fins educacionais...")
        
        if leverage > self.risk_params['extreme_leverage_warning']:
            print(f"\n⚠️⚠️⚠️ ALERTA DE RISCO EXTREMO: Alavancagem {leverage}x")
            print(f"Com esta alavancagem, movimentos de apenas {(1/leverage*100):.2f}% podem causar liquidação!")
            print(f"Em alavancagem elevada, o modelo usa threshold de liquidação de ~{(0.9/leverage/1.7*100):.2f}%")
        elif leverage > self.risk_params['high_leverage_warning']:
            print(f"\n⚠️ ALERTA DE RISCO ALTO: Alavancagem {leverage}x")
            print(f"Esta alavancagem pode causar liquidações com pequenos movimentos de mercado.")
        
        # MELHORADO: Capital mínimo verificado
        if initial_capital < self.risk_params['min_capital']:
            print(f"\n⚠️ AVISO: Capital inicial de {initial_capital} USDT está abaixo do recomendado de {self.risk_params['min_capital']} USDT.")
            print("Operações com capital baixo não cobrem adequadamente custos operacionais e slippage.")
        
        if verbose:
            print(f"\nIniciando simulação com capital inicial de {initial_capital} USDT...")
            print(f"Tamanho da ordem: {order_size} USDT × {leverage}x alavancagem = {order_size * leverage} USDT por posição")
            print(f"Take Profit: {tp*100:.2f}%, Stop Loss: {sl*100:.2f}%")
            
            # MELHORADO: Mostrar threshold de liquidação para maior transparência
            liquidation_threshold = 0.9 / (leverage * (1.5 if leverage < 10 else 1.7 if leverage < 20 else 2.0))
            print(f"Threshold de liquidação estimado: {liquidation_threshold*100:.2f}% (movimento adverso que causa liquidação)")
        
        # Important risk management parameters
        max_consecutive_losses = min(max_consecutive_sl, int(15 / leverage))  # Higher leverage = fewer allowed consecutive losses
        minimum_capital_ratio = 0.5  # Stop if capital drops to 50% of initial
        
        # Set up initial parameters
        starting_params = TradeParameters(
            tp=tp, sl=sl, capital=initial_capital,
            order_size=order_size, leverage=leverage,
            max_consecutive_losses=max_consecutive_losses
        )
        
        # Simple returns dict with base parameters
        results = []
        capital_history = {'timestamps': [], 'balances': [], 'events': []}
        
        # Track capital changes
        current_capital = initial_capital
        peak_capital = initial_capital
        max_drawdown_abs = 0
        max_drawdown_pct = 0
        consecutive_losses = 0
        
        # Add starting point to capital history
        capital_history['timestamps'].append(self.signals['timestamp'].min())
        capital_history['balances'].append(current_capital)
        capital_history['events'].append('Start')
        
        # Process signals in chronological order
        signals_sorted = self.signals.sort_values('timestamp')
        
        # Track active positions by pair
        active_positions = {}
        
        # Dictionary to track consecutive SLs by pair
        pair_stats = {}
        
        # Count of ignored signals with reasons
        ignored_signals = {
            'par_em_pausa': 0,
            'capital_insuficiente': 0, 
            'outros': 0
        }
        
        for _, signal in signals_sorted.iterrows():
            pair = signal['pair']
            
            # Skip if capital is insufficient
            if current_capital < order_size:
                ignored_signals['capital_insuficiente'] += 1
                
                if len(ignored_signals) < 5:  # Limitar logs
                    capital_history['timestamps'].append(signal['timestamp'])
                    capital_history['balances'].append(current_capital)
                    capital_history['events'].append('Insufficient Capital')
                
                continue
                
            # Check if capital fell below minimum threshold
            if current_capital < initial_capital * minimum_capital_ratio and current_capital < initial_capital:
                if len(ignored_signals) < 5:  # Limitar logs
                    capital_history['timestamps'].append(signal['timestamp'])
                    capital_history['balances'].append(current_capital)
                    capital_history['events'].append('Capital Below Safety Threshold')
                break
                
            # Initialize pair stats if needed
            if pair not in pair_stats:
                pair_stats[pair] = {
                    'consecutive_losses': 0,
                    'pause_until': None
                }
            
            # Check if pair is in cooldown after consecutive losses
            if pair_stats[pair]['pause_until'] and signal['timestamp'] <= pair_stats[pair]['pause_until']:
                ignored_signals['par_em_pausa'] += 1
                continue
            
            # Simulate this trade with current parameters
            trade_params = TradeParameters(
                tp=tp, sl=sl, capital=current_capital,
                order_size=order_size, leverage=leverage
            )
            
            trade_result = self.simulate_trade(signal, trade_params)
            
            # Skip if no valid result
            if 'status' in trade_result:
                ignored_signals['outros'] += 1
                continue
                
            # Update capital
            prev_capital = current_capital
            current_capital += trade_result['pnl']
            
            # Ensure capital never goes negative (account protection)
            current_capital = max(0, current_capital)
            
            # Update capital history
            capital_history['timestamps'].append(trade_result['exit_time'])
            capital_history['balances'].append(current_capital)
            capital_history['events'].append(f"{trade_result['result']} {pair} {trade_result['pnl']:.2f}")
            
            # Track wins/losses
            if trade_result['pnl'] <= 0:
                consecutive_losses += 1
                pair_stats[pair]['consecutive_losses'] += 1
                
                # If pair hit max consecutive losses, put it on pause
                if pair_stats[pair]['consecutive_losses'] >= max_consecutive_sl:
                    pause_duration = timedelta(minutes=pause_minutes)
                    pair_stats[pair]['pause_until'] = trade_result['exit_time'] + pause_duration
                    pair_stats[pair]['consecutive_losses'] = 0
            else:
                consecutive_losses = 0
                pair_stats[pair]['consecutive_losses'] = 0
            
            # Update peak capital & drawdown
            if current_capital > peak_capital:
                peak_capital = current_capital
            else:
                drawdown_abs = peak_capital - current_capital
                drawdown_pct = drawdown_abs / peak_capital * 100
                if drawdown_pct > max_drawdown_pct:
                    max_drawdown_pct = drawdown_pct
                    max_drawdown_abs = drawdown_abs
            
            # Update trade with capital info
            trade_result['prev_capital'] = prev_capital
            trade_result['new_capital'] = current_capital
            
            # Add trade to results
            results.append(trade_result)
            
            # Break if account is wiped out
            if current_capital <= 0:
                capital_history['timestamps'].append(trade_result['exit_time'])
                capital_history['balances'].append(0)
                capital_history['events'].append('Account Liquidation')
                break
                
        # Create DataFrame from results
        df_results = pd.DataFrame(results) if results else pd.DataFrame()
        
        # Create capital history DataFrame
        df_capital = pd.DataFrame({
            'timestamp': capital_history['timestamps'],
            'balance': capital_history['balances'],
            'event': capital_history['events']
        })
        
        # Calculate statistics
        stats = {
            'initial_capital': initial_capital,
            'final_capital': current_capital,
            'total_pnl': current_capital - initial_capital,
            'return_pct': ((current_capital / initial_capital) - 1) * 100 if initial_capital > 0 else 0,
            'max_drawdown_abs': max_drawdown_abs,
            'max_drawdown_pct': max_drawdown_pct,
            'win_count': len(df_results[df_results['pnl'] > 0]) if not df_results.empty else 0,
            'loss_count': len(df_results[df_results['pnl'] <= 0]) if not df_results.empty else 0,
            'max_consecutive_losses': consecutive_losses,
            'tp': tp * 100,
            'sl': sl * 100,
            'leverage': leverage,
            'liquidations': len(df_results[df_results['result'] == 'LIQUIDATION']) if not df_results.empty else 0,
            'ignored_signals': ignored_signals
        }
        
        if not df_results.empty:
            stats['win_rate'] = stats['win_count'] / len(df_results) * 100
            stats['avg_win'] = df_results[df_results['pnl'] > 0]['pnl'].mean() if stats['win_count'] > 0 else 0
            stats['avg_loss'] = df_results[df_results['pnl'] <= 0]['pnl'].mean() if stats['loss_count'] > 0 else 0
            stats['profit_factor'] = abs(df_results[df_results['pnl'] > 0]['pnl'].sum() / df_results[df_results['pnl'] < 0]['pnl'].sum()) if df_results[df_results['pnl'] < 0]['pnl'].sum() != 0 else float('inf')
            
            # Média de slippage dinâmico
            if 'dynamic_slippage' in df_results.columns:
                stats['avg_dynamic_slippage'] = df_results['dynamic_slippage'].mean()
            
            # Resultados por par
            stats['pair_results'] = {}
            for pair in df_results['pair'].unique():
                pair_data = df_results[df_results['pair'] == pair]
                wins = len(pair_data[pair_data['pnl'] > 0])
                losses = len(pair_data[pair_data['pnl'] <= 0])
                total = len(pair_data)
                
                stats['pair_results'][pair] = {
                    'trades': total,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': (wins / total * 100) if total > 0 else 0,
                    'pnl': pair_data['pnl'].sum()
                }
                
            # Resultados por direção
            stats['direction_results'] = {}
            for direction in df_results['direction'].unique():
                dir_data = df_results[df_results['direction'] == direction]
                wins = len(dir_data[dir_data['pnl'] > 0])
                losses = len(dir_data[dir_data['pnl'] <= 0])
                total = len(dir_data)
                
                stats['direction_results'][direction] = {
                    'trades': total,
                    'wins': wins,
                    'losses': losses,
                    'win_rate': (wins / total * 100) if total > 0 else 0,
                    'pnl': dir_data['pnl'].sum()
                }
                
            # Resultados por hora do dia
            df_results['hour'] = pd.to_datetime(df_results['entry_time']).dt.hour
            stats['hour_results'] = df_results.groupby('hour')['pnl'].sum().to_dict()
            
        if verbose:
            self._print_simulation_summary(df_results, stats)
            
        return df_results, df_capital, stats
        
    def _print_simulation_summary(self, df_results, stats):
        """Print simulation results summary"""
        trades_total = len(df_results) if not df_results.empty else 0
        signals_total = len(self.signals)
        signals_used = trades_total / signals_total * 100 if signals_total > 0 else 0
        
        ignored = stats.get('ignored_signals', {})
        ignored_details = []
        for reason, count in ignored.items():
            if count > 0:
                ignored_details.append(f"- {reason}: {count}")
        
        print(f"\nSimulação concluída: {trades_total} trades executados.")
        print(f"Sinais utilizados: {trades_total} de {signals_total} ({signals_used:.1f}%)")
        if ignored_details:
            print(f"Sinais ignorados: {sum(ignored.values())} de {signals_total} ({sum(ignored.values())/signals_total*100:.1f}%)")
            for detail in ignored_details:
                print(detail)
        
        print(f"\nCapital inicial: {stats['initial_capital']:.2f} USDT → Capital final: {stats['final_capital']:.2f} USDT")
        print(f"Lucro total: {stats['total_pnl']:.2f} USDT ({stats['return_pct']:.2f}%)")
        print(f"Drawdown máximo: {stats['max_drawdown_pct']:.2f}%")
        
        if trades_total > 0:
            print(f"\nTrades vencedores: {stats['win_count']} ({stats['win_rate']:.2f}%)")
            print(f"Trades perdedores: {stats['loss_count']} ({100-stats['win_rate']:.2f}%)")
            
            if stats.get('liquidations', 0) > 0:
                print(f"⚠️ Liquidações: {stats['liquidations']}")
                
            # Mostrar melhores pares
            if 'pair_results' in stats:
                print("\nDesempenho por par:")
                for pair, data in sorted(stats['pair_results'].items(), key=lambda x: x[1]['pnl'], reverse=True):
                    print(f"  {pair}: {data['trades']} trades, {data['pnl']:.2f} USDT, {data['win_rate']:.1f}% win rate")
                
    def grid_search(self, capital: float, order_size: float, leverage: float,
                    tp_range: List[float], sl_range: List[float], metric: str = 'total_pnl',
                    max_consecutive_sl: int = 3, pause_minutes: int = 120, 
                    n_jobs: int = -1) -> pd.DataFrame:
        """
        Efficient parallel grid search for optimal parameters
        
        Args:
            capital (float): Capital inicial
            order_size (float): Tamanho da ordem
            leverage (float): Alavancagem
            tp_range (List[float]): Lista de valores de TP a testar
            sl_range (List[float]): Lista de valores de SL a testar
            metric (str): Métrica para otimização ('total_pnl', 'return_pct', 'profit_factor')
            max_consecutive_sl (int): Número máximo de SLs consecutivos
            pause_minutes (int): Duração da pausa em minutos
            n_jobs (int): Número de processos (-1 = todos disponíveis)
            
        Returns:
            DataFrame: Resultados do grid search
        """
        start_time = time.time()
        
        # Verificar número de combinações
        num_combinations = len(tp_range) * len(sl_range)
        if num_combinations > 1000:
            print(f"\n⚠️ AVISO: Você está tentando testar {num_combinations:,} combinações!")
            print("Isso pode levar muito tempo. Recomendamos no máximo 1000 combinações.")
            print("Considere aumentar o tamanho do passo para reduzir o número de combinações.")
            if num_combinations > 5000:
                confirm = input("Digite 'CONTINUAR' para prosseguir ou qualquer outra coisa para cancelar: ")
                if confirm != "CONTINUAR":
                    print("Operação cancelada pelo usuário.")
                    return pd.DataFrame(), None
        
        # Generate all parameter combinations with self reference
        param_grid = [(self, tp, sl, capital, order_size, leverage, max_consecutive_sl, pause_minutes, metric) 
                      for tp in tp_range for sl in sl_range]
        
        print(f"\nIniciando busca de parâmetros com {len(param_grid)} combinações usando " +
              f"{min(mp.cpu_count(), n_jobs) if n_jobs != -1 else mp.cpu_count()} processos")
        
        # Set up multiprocessing pool
        n_jobs = mp.cpu_count() if n_jobs == -1 else min(n_jobs, 16)  # Limitar a 16 processos
        
        # Run in parallel with the external function
        all_results = []
        try:
            with mp.Pool(processes=n_jobs) as pool:
                all_results = pool.map(_process_params_for_grid, param_grid)
        except Exception as e:
            print(f"Erro na execução paralela: {str(e)}")
            print("Tentando processar de forma serial...")
            
            # Fallback para execução serial
            all_results = []
            for params in param_grid:
                try:
                    result = _process_params_for_grid(params)
                    all_results.append(result)
                except Exception as inner_e:
                    print(f"Erro ao processar parâmetros TP={params[1]}, SL={params[2]}: {str(inner_e)}")
        
        # Extract results into DataFrame
        grid_results = []
        for result in all_results:
            grid_results.append({
                'tp': result['tp'],
                'sl': result['sl'],
                'metric_value': result['metric_value'],
                'total_pnl': result['stats']['total_pnl'],
                'return_pct': result['stats']['return_pct'],
                'win_rate': result['stats'].get('win_rate', 0),
                'max_drawdown_pct': result['stats']['max_drawdown_pct'],
                'max_consecutive_losses': result['stats']['max_consecutive_losses'],
                'profit_factor': result['stats'].get('profit_factor', 0),
                'final_capital': result['stats']['final_capital'],
                'liquidations': result['stats']['liquidations'],
                'risk_reward_ratio': result['tp']/result['sl'] if result['sl'] > 0 else float('inf')
            })
        
        # Convert to DataFrame
        df_grid = pd.DataFrame(grid_results)
        
        # Find best result
        if not df_grid.empty:
            best_idx = df_grid['metric_value'].idxmax()
            best_params = df_grid.loc[best_idx]
            
            # Save best parameters
            best_result = next((r for r in all_results if r['tp'] == best_params['tp'] and r['sl'] == best_params['sl']), None)
            
            elapsed = time.time() - start_time
            print(f"\nBusca de parâmetros concluída em {elapsed:.1f} segundos")
            print(f"Melhores parâmetros: TP={best_params['tp']*100:.2f}%, SL={best_params['sl']*100:.2f}%")
            print(f"{metric}: {best_params['metric_value']:.2f}")
            print(f"Retorno: {best_params['return_pct']:.2f}%, Win Rate: {best_params['win_rate']:.2f}%")
            
            # Generate heatmap
            self._generate_heatmap(df_grid, tp_range, sl_range, metric)
            
            return df_grid, best_result
        else:
            print("Nenhum resultado válido encontrado na busca de parâmetros.")
            return pd.DataFrame(), None
        
    def _generate_heatmap(self, df_grid, tp_range, sl_range, metric):
        """Generate heatmap of grid search results"""
        try:
            # Create a pivot table for the heatmap
            heatmap_data = df_grid.pivot_table(
                index='tp', 
                columns='sl', 
                values='metric_value',
                aggfunc='first'
            )
            
            # Plot heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(
                heatmap_data, 
                annot=True, 
                fmt=".2f", 
                cmap='RdYlGn', 
                linewidths=0.5
            )
            
            plt.title(f'Otimização de Parâmetros TP/SL por {metric}')
            plt.xlabel('Stop Loss (%)')
            plt.ylabel('Take Profit (%)')
            
            # Format tick labels as percentages
            plt.xticks([i+0.5 for i in range(len(sl_range))], [f"{x*100:.2f}%" for x in sl_range])
            plt.yticks([i+0.5 for i in range(len(tp_range))], [f"{x*100:.2f}%" for x in tp_range])
            
            # Save figure
            plt.savefig(f'{self.results_dir}/heatmap_{metric}.png', bbox_inches='tight')
            plt.close()
            print(f"\nMapa de calor salvo em {self.results_dir}/heatmap_{metric}.png")
        except Exception as e:
            print(f"Erro ao gerar mapa de calor: {str(e)}")
            
    def generate_report(self, df_results, df_capital, stats):
        """Generate comprehensive report with visualizations"""
        if df_results.empty:
            print("Sem dados suficientes para gerar relatório")
            return
            
        print("\n" + "="*60)
        print("📊 RELATÓRIO DE DESEMPENHO DO BACKTEST")
        print("="*60)
        
        print(f"\n💰 Resumo Financeiro:")
        print(f"▶ Lucro Total: {stats['total_pnl']:.2f} USDT ({stats['return_pct']:.2f}%)")
        print(f"▶ Maior Drawdown: {stats['max_drawdown_pct']:.2f}%")
        
        print(f"\n📈 Estatísticas de Trades:")
        print(f"▶ Total de Trades: {len(df_results)}")
        print(f"▶ Trades Vencedores: {stats['win_count']} ({stats['win_rate']:.2f}%)")
        print(f"▶ Trades Perdedores: {stats['loss_count']} ({100-stats['win_rate']:.2f}%)")
        print(f"▶ Ganho Médio: {stats['avg_win']:.2f} USDT")
        print(f"▶ Perda Média: {stats['avg_loss']:.2f} USDT")
        
        if 'profit_factor' in stats:
            print(f"▶ Fator de Lucro: {stats['profit_factor']:.2f}")
            
        if 'avg_dynamic_slippage' in stats:
            print(f"▶ Slippage Dinâmico Médio: {stats['avg_dynamic_slippage']:.4f}%")
        
        # Distribuição por tipo de saída
        result_counts = df_results['result'].value_counts()
        print(f"\n🎯 Distribuição por Tipo de Saída:")
        for result, count in result_counts.items():
            pct = count / len(df_results) * 100
            print(f"▶ {result}: {count} ({pct:.2f}%)")
        
        # Análise por par de trading
        print("\n📊 DESEMPENHO POR PAR DE TRADING:")
        for pair, data in sorted(stats['pair_results'].items(), key=lambda x: x[1]['pnl'], reverse=True):
            print(f"▶ {pair}: {data['trades']} trades, {data['pnl']:.2f} USDT, {data['win_rate']:.1f}% win rate")
        
        # Análise por direção
        print("\n📊 DESEMPENHO POR DIREÇÃO:")
        for direction, data in sorted(stats['direction_results'].items(), key=lambda x: x[1]['pnl'], reverse=True):
            print(f"▶ {direction}: {data['trades']} trades, {data['pnl']:.2f} USDT, {data['win_rate']:.1f}% win rate")
        
        # Análise por hora do dia
        print("\n📊 DESEMPENHO POR HORA DO DIA (UTC):")
        hour_results = pd.Series(stats['hour_results']).sort_values(ascending=False)
        print("▶ Melhores horários:")
        for hour, pnl in hour_results.head(3).items():
            print(f"  ✓ {hour:02d}:00 UTC: {pnl:.2f} USDT")
        
        print("▶ Horários a evitar:")
        for hour, pnl in hour_results.tail(3).items():
            print(f"  ✗ {hour:02d}:00 UTC: {pnl:.2f} USDT")
        
        # Plotar gráficos
        self._plot_report_charts(df_results, df_capital, stats)
        
    def _plot_report_charts(self, df_results, df_capital, stats):
        """Create and save visualization charts"""
        try:
            # Set up plots
            plt.figure(figsize=(16, 14))
            
            # 1. Equity Curve
            plt.subplot(3, 2, 1)
            plt.plot(df_capital['timestamp'], df_capital['balance'], color='blue', linewidth=2)
            plt.title('Evolução do Capital')
            plt.xlabel('Data')
            plt.ylabel('USDT')
            plt.grid(True, alpha=0.3)
            
            # Format x-axis dates
            plt.gcf().autofmt_xdate()
            
            # 2. Trade PnL Distribution
            plt.subplot(3, 2, 2)
            sns.histplot(df_results['pnl'], kde=True, color='green')
            plt.axvline(x=0, color='red', linestyle='--')
            plt.title('Distribuição de PnL por Trade')
            plt.xlabel('PnL (USDT)')
            plt.ylabel('Frequência')
            
            # 3. Performance by Pair
            plt.subplot(3, 2, 3)
            pair_pnl = df_results.groupby('pair')['pnl'].sum().sort_values(ascending=False)
            colors = ['green' if x > 0 else 'red' for x in pair_pnl.values]
            pair_pnl.plot(kind='bar', color=colors)
            plt.title('PnL por Par')
            plt.xlabel('Par')
            plt.ylabel('PnL Total (USDT)')
            plt.xticks(rotation=45)
            
            # 4. Win Rate by Pair
            plt.subplot(3, 2, 4)
            pair_winrate = {}
            for pair in df_results['pair'].unique():
                pair_data = df_results[df_results['pair'] == pair]
                wins = (pair_data['pnl'] > 0).sum()
                total = len(pair_data)
                pair_winrate[pair] = wins / total * 100 if total > 0 else 0
                
            pair_wr_series = pd.Series(pair_winrate).sort_values(ascending=False)
            pair_wr_series.plot(kind='bar', color='skyblue')
            plt.axhline(y=50, color='red', linestyle='--')
            plt.title('Win Rate por Par')
            plt.xlabel('Par')
            plt.ylabel('Win Rate (%)')
            plt.xticks(rotation=45)
            
            # 5. Performance by Hour
            plt.subplot(3, 2, 5)
            hour_pnl = df_results.groupby(pd.to_datetime(df_results['entry_time']).dt.hour)['pnl'].sum()
            colors = ['green' if x > 0 else 'red' for x in hour_pnl.values]
            hour_pnl.plot(kind='bar', color=colors)
            plt.title('PnL por Hora do Dia (UTC)')
            plt.xlabel('Hora')
            plt.ylabel('PnL (USDT)')
            
            # 6. Drawdown Chart
            plt.subplot(3, 2, 6)
            # Calculate drawdown
            df_capital['peak'] = df_capital['balance'].cummax()
            df_capital['drawdown'] = 100 * (1 - df_capital['balance'] / df_capital['peak'])
            plt.fill_between(range(len(df_capital)), df_capital['drawdown'], color='coral', alpha=0.5)
            plt.title('Drawdown (%)')
            plt.xlabel('Operações')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)
            
            # Layout
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/relatorio_backtest.png', bbox_inches='tight')
            plt.close()
            
            print(f"\nGráficos salvos em {self.results_dir}/relatorio_backtest.png")
            
        except Exception as e:
            print(f"Erro ao gerar gráficos: {str(e)}")


# Exemplo de uso
def main():
    """Função principal para demonstração do simulador"""
    print("\n" + "="*60)
    print("     SIMULADOR AVANÇADO DE TRADING V2.5     ")
    print("="*60)
    print("\nSimule estratégias de trading com condições realistas de mercado.")
    
    try:
        # Inicializar simulador
        simulador = EstudoTraderV2()
        
        # Carregar dados
        if not simulador.load_data():
            print("Erro ao carregar dados. Verifique os arquivos.")
            return
        
        # Solicitar parâmetros de mercado
        print("\nParâmetros de mercado (pressione ENTER para usar valores padrão):")
        
        comissao_input = input("Taxa de comissão % (padrão: 0.04%): ")
        comissao = float(comissao_input) / 100 if comissao_input else 0.0004  # 0.04% (taker)
        
        spread_input = input("Spread % (padrão: 0.01%): ")
        spread = float(spread_input) / 100 if spread_input else 0.0001  # 0.01%
        
        slippage_input = input("Slippage % (padrão: 0.01%): ")
        slippage = float(slippage_input) / 100 if slippage_input else 0.0001  # 0.01%
        
        # Definir parâmetros de mercado
        simulador.set_market_params(
            commission=comissao,
            spread=spread,
            slippage=slippage
        )
        
        print("\n🏦 CONFIGURAÇÕES DA CONTA E ORDENS:")
        
        # Solicitar parâmetros da conta
        capital_inicial = float(input("Capital inicial em USDT (recomendado: 100+): "))
        tamanho_ordem = float(input("Valor da ordem em USDT (recomendado: 5-20): "))
        alavancagem = float(input("Alavancagem (recomendado: 1-10x): "))
        
        # Verificação de segurança para alavancagem alta
        if alavancagem > 10:
            print("\n⚠️⚠️⚠️ ALERTA DE RISCO ALTO ⚠️⚠️⚠️")
            print(f"Alavancagem de {alavancagem}x é EXTREMAMENTE ARRISCADA!")
            print(f"Com esta alavancagem, movimentos de apenas {(0.9/alavancagem/1.7*100):.2f}% podem causar liquidação!")
            confirmar = input("Digite 'CONFIRMO' em maiúsculas para prosseguir, ou qualquer outra coisa para ajustar: ")
            if confirmar != "CONFIRMO":
                alavancagem = float(input("Nova alavancagem (recomendado <= 10x): "))
        
        print("\n📊 CONFIGURAÇÕES DE TRADING:")
        
        # Solicitar parâmetros de negociação
        tp = float(input("Take-Profit em % (ex: 0.5 = 0.5%): ")) / 100
        sl = float(input("Stop-Loss em % (ex: 0.3 = 0.3%): ")) / 100
        
        # Verificar se SL é compatível com alavancagem
        liquidation_threshold = 0.9 / (alavancagem * (1.5 if alavancagem < 10 else 1.7 if alavancagem < 20 else 2.0))
        if sl > liquidation_threshold:
            print(f"\n⚠️ AVISO: Seu Stop-Loss ({sl*100:.2f}%) é maior que o threshold de liquidação ({liquidation_threshold*100:.2f}%)")
            print("Isto significa que a liquidação ocorrerá antes do Stop-Loss em condições normais.")
            print("O Stop-Loss só seria atingido em casos de gaps grandes de preço.")
            confirmar = input("Digite 'CONTINUAR' para prosseguir ou qualquer outra coisa para ajustar: ")
            if confirmar != "CONTINUAR":
                sl = float(input(f"Novo Stop-Loss em % (recomendado < {liquidation_threshold*100:.2f}%): ")) / 100
        
        # Solicitar parâmetros de pausa SL
        print("\n📊 CONFIGURAÇÕES DO GERENCIADOR DE RISCO:")
        sl_consecutivos = int(input("Pausar par após quantos SLs consecutivos (ex: 3): "))
        duracao_pausa = int(input("Duração da pausa em minutos (ex: 60): "))
        
        # Executar simulação
        resultados, historico_capital, estatisticas = simulador.simulate(
            tp, sl, capital_inicial, tamanho_ordem, alavancagem,
            max_consecutive_sl=sl_consecutivos, pause_minutes=duracao_pausa
        )
        
        # Gerar relatório detalhado
        simulador.generate_report(resultados, historico_capital, estatisticas)
        
        # Perguntar sobre busca de melhores parâmetros
        print("\nDeseja buscar melhores parâmetros TP/SL?")
        print("1. Sim, fazer otimização de parâmetros")
        print("2. Não, finalizar programa")
        opcao = input("Selecione (1-2): ")
        
        if opcao == "1":
            print("\nDefina o intervalo de valores para a busca:")
            
            # TP range
            tp_min = float(input("TP mínimo em % (ex: 0.3): ")) / 100
            tp_max = float(input("TP máximo em % (ex: 1.5): ")) / 100
            tp_step = float(input("Passo do TP em % (ex: 0.1): ")) / 100
            
            # SL range
            sl_min = float(input("SL mínimo em % (ex: 0.2): ")) / 100
            sl_max = float(input("SL máximo em % (ex: 0.8): ")) / 100
            sl_step = float(input("Passo do SL em % (ex: 0.1): ")) / 100
            
            # Criar arrays de valores
            tp_range = np.arange(tp_min, tp_max + tp_step/2, tp_step).tolist()
            sl_range = np.arange(sl_min, sl_max + sl_step/2, sl_step).tolist()
            
            # Métrica para otimização
            print("\nSelecione a métrica para otimização:")
            print("1. Lucro total (USDT)")
            print("2. Retorno percentual (%)")
            print("3. Fator de lucro (Ganhos/Perdas)")
            print("4. Taxa de acertos (Win Rate)")
            
            metrica_opcao = input("Selecione (1-4): ")
            
            if metrica_opcao == "2":
                metrica = "return_pct"
            elif metrica_opcao == "3":
                metrica = "profit_factor"
            elif metrica_opcao == "4":
                metrica = "win_rate"
            else:
                metrica = "total_pnl"
            
            # Executar grid search
            df_grid, melhor_resultado = simulador.grid_search(
                capital=capital_inicial, 
                order_size=tamanho_ordem, 
                leverage=alavancagem,
                tp_range=tp_range, 
                sl_range=sl_range, 
                metric=metrica,
                max_consecutive_sl=sl_consecutivos,
                pause_minutes=duracao_pausa
            )
            
            # Executar simulação com melhores parâmetros
            if melhor_resultado and not df_grid.empty:
                print("\nExecutando simulação com os melhores parâmetros encontrados...")
                melhores_resultados, melhores_capital, melhores_stats = simulador.simulate(
                    melhor_resultado['tp'], 
                    melhor_resultado['sl'], 
                    capital_inicial, 
                    tamanho_ordem, 
                    alavancagem,
                    max_consecutive_sl=sl_consecutivos,
                    pause_minutes=duracao_pausa
                )
                
                # Gerar relatório detalhado com melhores parâmetros
                simulador.generate_report(melhores_resultados, melhores_capital, melhores_stats)
    
    except KeyboardInterrupt:
        print("\n\nSimulação interrompida pelo usuário.")
    except ValueError as e:
        print(f"\nErro de valor: {str(e)}")
    except Exception as e:
        print(f"\nErro durante a simulação: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()