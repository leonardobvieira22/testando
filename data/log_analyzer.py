"""
Analisador de logs do sistema de trading v2.0
Identifica padrões de falhas, erros recorrentes e problemas de performance
"""
import os
import re
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

class TradingSystemLogAnalyzer:
    def __init__(self, log_file_path='trading_system.log', rejected_signals_file='data/rejected_signals.csv'):
        self.log_file_path = log_file_path
        self.rejected_signals_file = rejected_signals_file
        self.errors = []
        self.warnings = []
        self.info_messages = []
        self.api_calls = []
        self.websocket_events = []
        self.signal_generations = []
        self.filter_rejections = []
        
    def analyze_logs(self):
        """Analisa os arquivos de log para identificar problemas"""
        print(f"Analisando arquivo de log: {self.log_file_path}")
        
        if not os.path.exists(self.log_file_path):
            print(f"ERRO: Arquivo de log não encontrado em {self.log_file_path}")
            return False
            
        # Padrões de regex para diferentes tipos de mensagens
        error_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[ERROR\] (.+)')
        warning_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[WARNING\] (.+)')
        websocket_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[DEBUG\] (.+?(websocket|socket|WebSocket).+)')
        api_call_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[DEBUG\] https://.*"GET (/.*) HTTP')
        signal_pattern = re.compile(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \[INFO\] (.+?) (gerou sinal|sinal .+? rejeitado)')
        
        with open(self.log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Verificar erros
                error_match = error_pattern.search(line)
                if error_match:
                    timestamp, message = error_match.groups()
                    self.errors.append((timestamp, message))
                    continue
                    
                # Verificar avisos
                warning_match = warning_pattern.search(line)
                if warning_match:
                    timestamp, message = warning_match.groups()
                    self.warnings.append((timestamp, message))
                    continue
                
                # Verificar eventos de WebSocket
                websocket_match = websocket_pattern.search(line)
                if websocket_match:
                    timestamp, message = websocket_match.groups()
                    self.websocket_events.append((timestamp, message))
                    continue
                
                # Verificar chamadas de API
                api_match = api_call_pattern.search(line)
                if api_match:
                    timestamp, endpoint = api_match.groups()
                    self.api_calls.append((timestamp, endpoint))
                    continue
                
                # Verificar geração de sinais e rejeições
                signal_match = signal_pattern.search(line)
                if signal_match:
                    timestamp, symbol, action = signal_match.groups()
                    if 'rejeitado' in action:
                        self.filter_rejections.append((timestamp, f"{symbol} {action}"))
                    else:
                        self.signal_generations.append((timestamp, f"{symbol} {action}"))
        
        print(f"Análise concluída. Encontrados:")
        print(f"- {len(self.errors)} mensagens de erro")
        print(f"- {len(self.warnings)} avisos")
        print(f"- {len(self.websocket_events)} eventos de WebSocket")
        print(f"- {len(self.api_calls)} chamadas de API")
        print(f"- {len(self.signal_generations)} sinais gerados")
        print(f"- {len(self.filter_rejections)} sinais rejeitados por filtros")
        
        return True
        
    def analyze_rejected_signals(self):
        """Analisa o arquivo CSV de sinais rejeitados"""
        print(f"\nAnalisando sinais rejeitados em: {self.rejected_signals_file}")
        
        if not os.path.exists(self.rejected_signals_file):
            print(f"AVISO: Arquivo de sinais rejeitados não encontrado em {self.rejected_signals_file}")
            return False
            
        try:
            df = pd.read_csv(self.rejected_signals_file)
            print(f"Total de {len(df)} sinais rejeitados encontrados no arquivo")
            
            # Análise por tipo de filtro
            filter_counts = df['filter_type'].value_counts()
            print("\nRejeições por tipo de filtro:")
            for filter_type, count in filter_counts.items():
                print(f"- {filter_type}: {count} ({count/len(df)*100:.1f}%)")
                
            # Análise por direção do sinal
            direction_counts = df['signal_direction'].value_counts()
            print("\nRejeições por direção do sinal:")
            for direction, count in direction_counts.items():
                print(f"- {direction}: {count} ({count/len(df)*100:.1f}%)")
                
            # Análise por símbolo (top 5)
            symbol_counts = df['symbol'].value_counts().head(5)
            print("\nTop 5 símbolos com mais rejeições:")
            for symbol, count in symbol_counts.items():
                print(f"- {symbol}: {count} ({count/len(df)*100:.1f}%)")
                
            # Análise temporal de rejeições
            df['date_time'] = pd.to_datetime(df['date_time'])
            df['hour'] = df['date_time'].dt.hour
            hourly_counts = df['hour'].value_counts().sort_index()
            
            print("\nDistribuição de rejeições por hora do dia:")
            for hour, count in hourly_counts.items():
                print(f"- {hour:02d}:00: {count} rejeições")
                
            return True
        except Exception as e:
            print(f"ERRO ao analisar arquivo de sinais rejeitados: {e}")
            return False
    
    def analyze_critical_errors(self):
        """Analisa padrões em erros críticos"""
        print("\nAnálise de Erros Críticos:")
        
        if not self.errors:
            print("Nenhum erro encontrado nos logs.")
            return
            
        # Agrupar erros por tipo
        error_types = defaultdict(list)
        
        for timestamp, message in self.errors:
            # Extrair o tipo básico de erro
            error_type = message.split(':')[0] if ':' in message else message[:40]
            error_types[error_type].append((timestamp, message))
        
        # Mostrar erros mais frequentes
        print(f"Encontrados {len(error_types)} tipos de erros diferentes:")
        
        for error_type, occurrences in sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\n- {error_type} ({len(occurrences)} ocorrências)")
            # Mostrar exemplos (primeiro e último)
            print(f"  Primeiro: {occurrences[0][0]} - {occurrences[0][1][:100]}")
            if len(occurrences) > 1:
                print(f"  Último: {occurrences[-1][0]} - {occurrences[-1][1][:100]}")
    
    def generate_report(self):
        """Gera um relatório completo da análise"""
        print("\n" + "="*50)
        print("RELATÓRIO DE DIAGNÓSTICO DO SISTEMA DE TRADING")
        print("="*50)
        
        # Analisar erros críticos
        self.analyze_critical_errors()
        
        # Analisar problemas de comunicação WebSocket
        print("\nAnálise de Comunicação WebSocket:")
        websocket_issues = [msg for _, msg in self.websocket_events if 'erro' in msg.lower() or 'fail' in msg.lower()]
        if websocket_issues:
            print(f"Encontrados {len(websocket_issues)} problemas de WebSocket:")
            for issue in websocket_issues[:5]:  # Mostrar apenas os 5 primeiros
                print(f"- {issue}")
            if len(websocket_issues) > 5:
                print(f"... e mais {len(websocket_issues) - 5} problemas")
        else:
            print("Nenhum problema de WebSocket encontrado nos logs analisados.")
        
        # Analisar padrões em chamadas de API
        print("\nAnálise de Chamadas de API:")
        endpoints = Counter([endpoint for _, endpoint in self.api_calls])
        print(f"Total de {len(self.api_calls)} chamadas de API para {len(endpoints)} endpoints diferentes")
        print("Top 5 endpoints mais chamados:")
        for endpoint, count in endpoints.most_common(5):
            print(f"- {endpoint}: {count} chamadas")
        
        # Recomendações automatizadas
        print("\n" + "="*50)
        print("RECOMENDAÇÕES BASEADAS NA ANÁLISE")
        print("="*50)
        
        # Baseado em erros
        if len(self.errors) > 100:
            print("⚠️ Alta taxa de erros detectada. Revisar gerenciamento de exceções e validações.")
        
        # Baseado em problemas de WebSocket
        websocket_errors = len([msg for _, msg in self.websocket_events if 'erro' in msg.lower() or 'fail' in msg.lower()])
        if websocket_errors > 20:
            print("⚠️ Frequentes problemas na comunicação WebSocket. Revisar implementação de reconexão.")
        
        # Baseado em rejeições de filtros
        if len(self.filter_rejections) > len(self.signal_generations) * 5:
            print("⚠️ Taxa muito alta de rejeição de sinais. Revisar parâmetros dos filtros que podem estar muito restritivos.")
        
        print("\n" + "="*50)
        
if __name__ == "__main__":
    analyzer = TradingSystemLogAnalyzer()
    analyzer.analyze_logs()
    analyzer.analyze_rejected_signals()
    analyzer.generate_report()