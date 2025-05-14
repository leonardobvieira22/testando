#!/usr/bin/env python
import os
import sys
import logging
import time
import traceback
from datetime import datetime
import pandas as pd

# Importar os módulos do sistema
try:
    from binance_utils import (
        get_klines, calculate_rsi, get_symbol_info, 
        get_server_time, test_api_connection
    )
    import config
except ImportError as e:
    print(f"Erro ao importar módulos do sistema: {e}")
    print("Verifique se você está executando este script no diretório correto.")
    sys.exit(1)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

def check_system_files():
    """Verifica a existência e integridade dos arquivos principais do sistema."""
    required_files = [
        'sinal_engine.py',
        'binance_utils.py',
        'config.py',
        'requirements.txt'
    ]
    
    results = {}
    for file in required_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            last_modified = datetime.fromtimestamp(os.path.getmtime(file)).strftime("%Y-%m-%d %H:%M:%S")
            results[file] = {
                'exists': True,
                'size': size,
                'last_modified': last_modified
            }
        else:
            results[file] = {'exists': False}
    
    return results

def test_api():
    """Testa a conexão com a API da Binance."""
    try:
        result = test_api_connection()
        return result
    except Exception as e:
        return {
            'success': False,
            'message': f"Erro ao testar API: {str(e)}",
            'traceback': traceback.format_exc()
        }

def test_market_data(symbol='BTCUSDT', timeframe='1h'):
    """Testa a obtenção de dados de mercado."""
    try:
        klines = get_klines(symbol, timeframe, 100)
        if klines and len(klines) > 0:
            return {
                'success': True,
                'message': f"Obtidos {len(klines)} candles para {symbol} {timeframe}",
                'last_candle': {
                    'open_time': datetime.fromtimestamp(klines[-1][0] / 1000).strftime("%Y-%m-%d %H:%M:%S"),
                    'open': klines[-1][1],
                    'high': klines[-1][2],
                    'low': klines[-1][3],
                    'close': klines[-1][4],
                    'volume': klines[-1][5]
                }
            }
        else:
            return {
                'success': False,
                'message': f"Nenhum dado retornado para {symbol} {timeframe}"
            }
    except Exception as e:
        return {
            'success': False,
            'message': f"Erro ao obter dados de mercado: {str(e)}",
            'traceback': traceback.format_exc()
        }

def test_rsi_calculation(symbol='BTCUSDT', timeframe='1h'):
    """Testa o cálculo de RSI."""
    try:
        klines = get_klines(symbol, timeframe, config.RSI_PERIOD * 3)
        if not klines or len(klines) < config.RSI_PERIOD + 10:
            return {
                'success': False,
                'message': f"Dados insuficientes para calcular RSI. Obtidos {len(klines) if klines else 0} candles"
            }
        
        close_prices = [float(k[4]) for k in klines]
        rsi_values = calculate_rsi(close_prices, config.RSI_PERIOD)
        
        return {
            'success': True,
            'message': f"RSI calculado com sucesso para {symbol} {timeframe}",
            'current_rsi': rsi_values[-1],
            'previous_rsi': rsi_values[-2],
            'rsi_min_long': config.RSI_MIN_LONG,
            'rsi_max_short': config.RSI_MAX_SHORT,
            'would_trigger_attentive': (
                rsi_values[-1] <= config.RSI_MIN_LONG or 
                rsi_values[-1] >= config.RSI_MAX_SHORT
            )
        }
    except Exception as e:
        return {
            'success': False,
            'message': f"Erro ao calcular RSI: {str(e)}",
            'traceback': traceback.format_exc()
        }

def check_log_file():
    """Verifica o arquivo de log."""
    log_file = 'bot.log'
    try:
        if os.path.exists(log_file):
            size = os.path.getsize(log_file)
            last_modified = datetime.fromtimestamp(os.path.getmtime(log_file)).strftime("%Y-%m-%d %H:%M:%S")
            
            # Ler as últimas 10 linhas do log
            with open(log_file, 'r') as f:
                lines = f.readlines()
                last_lines = lines[-10:] if len(lines) >= 10 else lines
            
            return {
                'success': True,
                'exists': True,
                'size': size,
                'last_modified': last_modified,
                'last_lines': last_lines
            }
        else:
            return {
                'success': True,
                'exists': False,
                'message': "Arquivo de log não existe"
            }
    except Exception as e:
        return {
            'success': False,
            'message': f"Erro ao verificar arquivo de log: {str(e)}",
            'traceback': traceback.format_exc()
        }

def check_config_integrity():
    """Verifica a integridade das configurações."""
    results = {
        'rsi_settings': {},
        'filter_settings': {},
        'trading_settings': {},
        'issues': []
    }
    
    # Verificar configurações de RSI
    try:
        results['rsi_settings'] = {
            'RSI_PERIOD': config.RSI_PERIOD,
            'RSI_MIN_LONG': config.RSI_MIN_LONG,
            'RSI_MAX_SHORT': config.RSI_MAX_SHORT,
            'MAX_RSI_FOR_BUY': config.MAX_RSI_FOR_BUY,
            'MIN_RSI_FOR_SELL': config.MIN_RSI_FOR_SELL,
            'RSI_DELTA_REQUIRED': config.RSI_DELTA_REQUIRED
        }
        
        if config.RSI_MIN_LONG >= config.RSI_MAX_SHORT:
            results['issues'].append("ERRO: RSI_MIN_LONG deve ser menor que RSI_MAX_SHORT")
            
        if config.MAX_RSI_FOR_BUY >= 100 or config.MAX_RSI_FOR_BUY <= 0:
            results['issues'].append(f"AVISO: MAX_RSI_FOR_BUY ({config.MAX_RSI_FOR_BUY}) está fora do intervalo normal (0-100)")
            
        if config.MIN_RSI_FOR_SELL >= 100 or config.MIN_RSI_FOR_SELL <= 0:
            results['issues'].append(f"AVISO: MIN_RSI_FOR_SELL ({config.MIN_RSI_FOR_SELL}) está fora do intervalo normal (0-100)")
    except Exception as e:
        results['issues'].append(f"ERRO ao verificar configurações RSI: {str(e)}")
    
    # Verificar configurações de filtros
    try:
        results['filter_settings'] = {
            'MTF_FILTER_ENABLED': config.MTF_FILTER_ENABLED,
            'MTF_RSI_FILTER_ENABLED': config.MTF_RSI_FILTER_ENABLED,
            'MTF_TREND_EMA_FILTER_ENABLED': config.MTF_TREND_EMA_FILTER_ENABLED,
            'SR_FILTER_ENABLED': config.SR_FILTER_ENABLED,
            'SR_SWING_POINTS_ENABLED': config.SR_SWING_POINTS_ENABLED,
            'SR_PIVOT_POINTS_ENABLED': config.SR_PIVOT_POINTS_ENABLED,
            'VOLUME_FILTER_ENABLED': config.VOLUME_FILTER_ENABLED,
            'TREND_EMA_FILTER_ENABLED': config.TREND_EMA_FILTER_ENABLED
        }
        
        all_filters_enabled = (
            config.MTF_FILTER_ENABLED and
            config.SR_FILTER_ENABLED and
            config.VOLUME_FILTER_ENABLED and
            config.TREND_EMA_FILTER_ENABLED
        )
        if all_filters_enabled:
            results['issues'].append("AVISO: Todos os filtros estão ativados simultaneamente, o que pode bloquear muitos sinais")
    except Exception as e:
        results['issues'].append(f"ERRO ao verificar configurações de filtros: {str(e)}")
    
    # Verificar configurações de negociação
    try:
        results['trading_settings'] = {
            'ORDER_QUANTITY': config.ORDER_QUANTITY,
            'TAKE_PROFIT_PERCENT': config.TAKE_PROFIT_PERCENT,
            'STOP_LOSS_PERCENT': config.STOP_LOSS_PERCENT,
            'SYMBOLS_TO_TRADE': config.SYMBOLS_TO_TRADE,
            'TIMEFRAMES': config.TIMEFRAMES
        }
        
        if not config.SYMBOLS_TO_TRADE:
            results['issues'].append("ERRO: Nenhum par configurado para negociação")
            
        if not config.TIMEFRAMES:
            results['issues'].append("ERRO: Nenhum timeframe configurado para negociação")
    except Exception as e:
        results['issues'].append(f"ERRO ao verificar configurações de negociação: {str(e)}")
    
    return results

def run_diagnostics():
    """Executa todos os testes de diagnóstico e retorna os resultados."""
    logging.info("Iniciando diagnóstico do sistema de Trading v2.0")
    
    results = {}
    
    # Verificar arquivos do sistema
    logging.info("Verificando arquivos do sistema...")
    results['system_files'] = check_system_files()
    
    # Verificar configurações
    logging.info("Verificando integridade das configurações...")
    results['config'] = check_config_integrity()
    
    # Testar conexão com a API
    logging.info("Testando conexão com a API da Binance...")
    results['api_connection'] = test_api()
    
    # Testar obtenção de dados de mercado
    logging.info("Testando obtenção de dados de mercado...")
    results['market_data'] = {}
    for symbol in config.SYMBOLS_TO_TRADE[:2]:  # Testar apenas os 2 primeiros símbolos
        for timeframe in config.TIMEFRAMES:
            key = f"{symbol}_{timeframe}"
            logging.info(f"Testando dados para {key}...")
            results['market_data'][key] = test_market_data(symbol, timeframe)
    
    # Testar cálculo de RSI
    logging.info("Testando cálculo de RSI...")
    results['rsi_calculation'] = {}
    for symbol in config.SYMBOLS_TO_TRADE[:2]:  # Testar apenas os 2 primeiros símbolos
        for timeframe in config.TIMEFRAMES:
            key = f"{symbol}_{timeframe}"
            logging.info(f"Testando RSI para {key}...")
            results['rsi_calculation'][key] = test_rsi_calculation(symbol, timeframe)
    
    # Verificar arquivo de log
    logging.info("Verificando arquivo de log...")
    results['log_file'] = check_log_file()
    
    return results

def print_diagnostic_report(results):
    """Imprime um relatório de diagnóstico formatado."""
    print("\n\n===== RELATÓRIO DE DIAGNÓSTICO DO SISTEMA DE TRADING V2.0 =====")
    print(f"Data e Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*60 + "\n")
    
    # 1. Arquivos do Sistema
    print("1. ARQUIVOS DO SISTEMA")
    print("-"*60)
    for file, info in results['system_files'].items():
        if info['exists']:
            print(f"✓ {file} - Tamanho: {info['size']} bytes, Última modificação: {info['last_modified']}")
        else:
            print(f"✗ {file} - NÃO ENCONTRADO")
    print("\n")
    
    # 2. Configurações
    print("2. CONFIGURAÇÕES")
    print("-"*60)
    
    print("RSI:")
    for key, value in results['config']['rsi_settings'].items():
        print(f"  {key}: {value}")
    
    print("\nFiltros:")
    for key, value in results['config']['filter_settings'].items():
        print(f"  {key}: {value}")
    
    print("\nNegociação:")
    for key, value in results['config']['trading_settings'].items():
        print(f"  {key}: {value}")
    
    if results['config']['issues']:
        print("\nProblemas de Configuração Detectados:")
        for issue in results['config']['issues']:
            print(f"  ! {issue}")
    else:
        print("\n  ✓ Nenhum problema de configuração detectado")
    print("\n")
    
    # 3. Conexão com API
    print("3. CONEXÃO COM API")
    print("-"*60)
    api_result = results['api_connection']
    if api_result.get('success'):
        print(f"✓ Conexão bem-sucedida: {api_result.get('message', '')}")
    else:
        print(f"✗ Falha na conexão: {api_result.get('message', '')}")
        if 'traceback' in api_result:
            print("\nStacktrace:")
            print(api_result['traceback'])
    print("\n")
    
    # 4. Dados de Mercado
    print("4. DADOS DE MERCADO")
    print("-"*60)
    for key, result in results['market_data'].items():
        if result.get('success'):
            print(f"✓ {key}: {result.get('message', '')}")
            if 'last_candle' in result:
                candle = result['last_candle']
                print(f"  Último candle: {candle['open_time']}, Fechamento: {candle['close']}")
        else:
            print(f"✗ {key}: {result.get('message', '')}")
    print("\n")
    
    # 5. Cálculo de RSI
    print("5. CÁLCULO DE RSI")
    print("-"*60)
    for key, result in results['rsi_calculation'].items():
        if result.get('success'):
            print(f"✓ {key}: RSI atual = {result.get('current_rsi', 'N/A'):.2f}, "
                  f"anterior = {result.get('previous_rsi', 'N/A'):.2f}")
            if result.get('would_trigger_attentive'):
                print(f"  ! Este RSI acionaria o modo atento (RSI fora dos limites configurados)")
        else:
            print(f"✗ {key}: {result.get('message', '')}")
    print("\n")
    
    # 6. Arquivo de Log
    print("6. ARQUIVO DE LOG")
    print("-"*60)
    log_result = results['log_file']
    if log_result.get('success'):
        if log_result.get('exists'):
            print(f"✓ Log encontrado - Tamanho: {log_result['size']} bytes, "
                  f"Última modificação: {log_result['last_modified']}")
            print("\nÚltimas entradas do log:")
            for line in log_result['last_lines']:
                print(f"  {line.strip()}")
        else:
            print("✗ Arquivo de log não existe - o bot pode não ter sido executado ainda ou está tendo problemas para gravar logs")
    else:
        print(f"✗ Erro ao verificar log: {log_result.get('message', '')}")
    print("\n")
    
    # Conclusão e Recomendações
    print("===== CONCLUSÃO E RECOMENDAÇÕES =====")
    
    issues_found = []
    if not all(info['exists'] for info in results['system_files'].values()):
        issues_found.append("Arquivos de sistema ausentes")
    
    if results['config']['issues']:
        issues_found.append("Problemas de configuração")
    
    if not results['api_connection'].get('success'):
        issues_found.append("Falha na conexão com API")
    
    market_data_failed = any(not result.get('success') for result in results['market_data'].values())
    if market_data_failed:
        issues_found.append("Falhas na obtenção de dados de mercado")
    
    rsi_failed = any(not result.get('success') for result in results['rsi_calculation'].values())
    if rsi_failed:
        issues_found.append("Falhas no cálculo de RSI")
    
    if not log_result.get('exists', False) and log_result.get('success', False):
        issues_found.append("Arquivo de log ausente")
    
    if issues_found:
        print("Foram detectados os seguintes problemas:")
        for issue in issues_found:
            print(f"  ! {issue}")
        print("\nRecomendações:")
        print("  1. Verifique as credenciais da API da Binance no config.py")
        print("  2. Certifique-se de que todos os filtros não estão muito restritivos")
        print("  3. Execute o sistema com modo de debug ativado para mais informações")
        print("  4. Verifique se há permissões de escrita para criar o arquivo de log")
    else:
        print("✓ Não foram detectados problemas críticos no sistema")
        print("\nRecomendações:")
        print("  1. Execute o sistema e monitore o arquivo bot.log")
        print("  2. Ajuste gradualmente os parâmetros de filtros para encontrar um equilíbrio")
        print("  3. Considere testar com pares adicionais")

if __name__ == "__main__":
    try:
        results = run_diagnostics()
        print_diagnostic_report(results)
    except Exception as e:
        logging.error(f"Erro ao executar diagnóstico: {e}")
        logging.error(traceback.format_exc())
        print(f"\nERRO CRÍTICO: {e}")
