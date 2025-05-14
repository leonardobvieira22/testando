"""
Dashboard para análise de sinais rejeitados por filtros
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from rejected_signals_tracker import RejectedSignalsTracker

def generate_rejection_report(days=7):
    """
    Gera relatório e visualização de sinais rejeitados pelos filtros avançados
    
    Args:
        days: Número de dias para análise
    """
    # Inicializar o rastreador
    tracker = RejectedSignalsTracker()
    
    # Obter estatísticas
    stats = tracker.get_rejection_stats(days_back=days)
    
    print(f"\n{'='*50}")
    print(f"RELATÓRIO DE REJEIÇÕES DE SINAIS (Últimos {days} dias)")
    print(f"{'='*50}")
    
    if stats["total_rejections"] == 0:
        print("\nNenhuma rejeição registrada no período selecionado.\n")
        return
        
    print(f"\nTotal de sinais rejeitados: {stats['total_rejections']}")
    
    # Rejeições por filtro
    print(f"\n{'-'*20} REJEIÇÕES POR FILTRO {'-'*20}")
    for filter_type, count in sorted(stats['rejections_by_filter'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_rejections']) * 100
        print(f"{filter_type:<20}: {count:4d} ({percentage:5.1f}%)")
    
    # Rejeições por par
    print(f"\n{'-'*20} REJEIÇÕES POR PAR {'-'*20}")
    for symbol, count in sorted(stats['rejections_by_symbol'].items(), key=lambda x: x[1], reverse=True)[:10]:  # Top 10
        percentage = (count / stats['total_rejections']) * 100
        print(f"{symbol:<15}: {count:4d} ({percentage:5.1f}%)")
    
    # Rejeições por direção
    print(f"\n{'-'*20} REJEIÇÕES POR DIREÇÃO {'-'*20}")
    for direction, count in stats['rejections_by_direction'].items():
        percentage = (count / stats['total_rejections']) * 100 if stats['total_rejections'] > 0 else 0
        print(f"{direction:<10}: {count:4d} ({percentage:5.1f}%)")
    
    # Gerar visualizações
    try:
        plt.figure(figsize=(15, 10))
        
        # Gráfico de pizza para rejeições por filtro
        plt.subplot(221)
        filter_labels = list(stats['rejections_by_filter'].keys())
        filter_counts = list(stats['rejections_by_filter'].values())
        plt.pie(filter_counts, labels=filter_labels, autopct='%1.1f%%', startangle=90)
        plt.title('Rejeições por Filtro')
        plt.axis('equal')
        
        # Gráfico de barras para rejeições por par (top 10)
        plt.subplot(222)
        top_symbols = sorted(stats['rejections_by_symbol'].items(), key=lambda x: x[1], reverse=True)[:10]
        symbol_labels = [item[0] for item in top_symbols]
        symbol_counts = [item[1] for item in top_symbols]
        
        plt.barh(range(len(symbol_labels)), symbol_counts, align='center')
        plt.yticks(range(len(symbol_labels)), symbol_labels)
        plt.xlabel('Número de Rejeições')
        plt.title('Top 10 Pares com Sinais Rejeitados')
        
        # Gráfico de pizza para direção (BUY/SELL)
        plt.subplot(223)
        direction_labels = list(stats['rejections_by_direction'].keys())
        direction_counts = list(stats['rejections_by_direction'].values())
        plt.pie(direction_counts, labels=direction_labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999', '#66b3ff'])
        plt.title('Rejeições por Direção')
        plt.axis('equal')
        
        # Salvar o gráfico
        output_file = "rejection_analysis.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=100)
        print(f"\nGráfico salvo em: {os.path.abspath(output_file)}")
        
    except Exception as e:
        print(f"\nErro ao gerar visualizações: {e}")
    
    print(f"\n{'='*50}")
    print("Use este relatório para ajustar os parâmetros dos filtros conforme necessário.")
    print(f"{'='*50}\n")

def load_rejection_data():
    """
    Carrega dados brutos de rejeições para análise detalhada
    
    Returns:
        pandas.DataFrame: DataFrame com os dados de rejeições
    """
    tracker = RejectedSignalsTracker()
    try:
        data = pd.read_csv(tracker.rejected_signals_file)
        # Converter timestamp para datetime
        data['date'] = pd.to_datetime(data['timestamp'], unit='s')
        return data
    except Exception as e:
        print(f"Erro ao carregar dados de rejeições: {e}")
        return pd.DataFrame()

def analyze_rejections_by_time():
    """Analisa a distribuição de rejeições por hora do dia"""
    data = load_rejection_data()
    if data.empty:
        print("Sem dados de rejeições para analisar.")
        return
    
    # Extrair hora do dia
    data['hour'] = data['date'].dt.hour
    
    # Contar rejeições por hora
    hourly_counts = data['hour'].value_counts().sort_index()
    
    plt.figure(figsize=(12, 6))
    plt.bar(hourly_counts.index, hourly_counts.values)
    plt.title('Distribuição de Rejeições por Hora do Dia')
    plt.xlabel('Hora do Dia')
    plt.ylabel('Número de Rejeições')
    plt.xticks(range(0, 24))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Salvar o gráfico
    output_file = "rejections_by_hour.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=100)
    print(f"Gráfico salvo em: {os.path.abspath(output_file)}")

def interactive_menu():
    """Menu interativo para ferramentas de análise de rejeições"""
    while True:
        print("\n=== MENU DE ANÁLISE DE REJEIÇÕES ===")
        print("1. Gerar relatório de rejeições")
        print("2. Analisar rejeições por hora do dia")
        print("3. Limpar dados antigos de rejeições")
        print("0. Sair")
        
        choice = input("\nEscolha uma opção: ")
        
        if choice == "1":
            days = input("Número de dias para análise [7]: ") or "7"
            generate_rejection_report(int(days))
        elif choice == "2":
            analyze_rejections_by_time()
        elif choice == "3":
            days_to_keep = input("Manter dados de quantos dias? [30]: ") or "30"
            # Esta função precisa ser implementada
            print("Função não implementada ainda.")
        elif choice == "0":
            print("Saindo...")
            break
        else:
            print("Opção inválida.")

if __name__ == "__main__":
    interactive_menu()