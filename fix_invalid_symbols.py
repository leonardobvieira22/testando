"""
Script para corrigir símbolos inválidos no sistema de trading
Este script deve ser executado uma vez para identificar onde os problemas estão ocorrendo
"""
import os
import re
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Lista de símbolos inválidos identificados nos logs
INVALID_SYMBOLS = ['price_log_time', 'websocket_manager', 'socket_connections']

def scan_file_for_invalid_symbols(file_path):
    """
    Verifica se o arquivo contém referências aos símbolos inválidos
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for symbol in INVALID_SYMBOLS:
            matches = re.findall(r'[\'"]' + symbol + r'[\'"]', content)
            if matches:
                logger.info(f"Encontrada(s) {len(matches)} referência(s) a '{symbol}' em {file_path}")
                
                # Buscar contexto das linhas
                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if symbol in line:
                        start = max(0, i-2)
                        end = min(len(lines), i+3)
                        logger.info(f"Contexto em {file_path} (linha {i+1}):")
                        for j in range(start, end):
                            prefix = ">>> " if j == i else "    "
                            logger.info(f"{prefix}{lines[j]}")
    
    except Exception as e:
        logger.error(f"Erro ao ler arquivo {file_path}: {e}")

def scan_directory_for_invalid_symbols(directory):
    """
    Escaneia todos os arquivos .py no diretório em busca dos símbolos inválidos
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                scan_file_for_invalid_symbols(file_path)

def fix_main_issue(main_file_path):
    """
    Corrige o problema específico no arquivo main.py que está tentando
    utilizar variáveis de controle interno como símbolos de trading
    """
    try:
        with open(main_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar por padrões comuns de erro
        # 1. Chamadas para get_historical_data com variáveis de controle
        pattern = r'get_historical_data\([\'"]?(price_log_time|websocket_manager|socket_connections)[\'"]?,'
        
        # Se encontrar o padrão, alertar
        if re.search(pattern, content):
            logger.warning(f"Encontrado padrão de código problemático em {main_file_path}")
            logger.info("Este código precisa ser reescrito para não tratar variáveis internas como símbolos")
        
        # Opção para criar um arquivo com correções
        # ... (código para fazer as substituições específicas)
        
    except Exception as e:
        logger.error(f"Erro ao processar arquivo {main_file_path}: {e}")

def create_fix_for_websocket_manager(output_file):
    """
    Cria um patch para o websocket_manager.py para evitar
    que variáveis de controle sejam tratadas como símbolos
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("""
# Patch para corrigir problemas com websocket_manager.py
# Substitua a função problemática existente por esta versão corrigida

def process_metrics(self):
    \"\"\"
    Processa métricas de desempenho e conexão do WebSocket sem chamar a API da Binance
    \"\"\"
    try:
        # Obter timestamp atual
        current_time = int(time.time())
        
        # Registrar métricas sem tentar obter dados históricos
        metrics = {
            'timestamp': current_time,
            'active_connections': len(self.socket_connections),
            'messages_received': self.messages_received,
            'messages_per_second': self.calculate_messages_per_second(),
            'last_price_log_time': getattr(self, 'price_log_time', 0)
        }
        
        # Log das métricas
        if current_time % 60 == 0:  # Log a cada minuto
            logger.info(f"WebSocket Metrics: {metrics}")
            
        return metrics
    except Exception as e:
        logger.error(f"Erro ao processar métricas: {e}")
        return {}
""")
    
    logger.info(f"Arquivo de correção criado em {output_file}")
    logger.info("Substitua a função problemática no seu código pelo conteúdo deste arquivo.")

if __name__ == "__main__":
    # Diretório atual como ponto de partida
    current_dir = os.getcwd()
    
    logger.info(f"Iniciando varredura do diretório: {current_dir}")
    scan_directory_for_invalid_symbols(current_dir)
    
    # Tentar corrigir o problema no main.py
    main_file = os.path.join(current_dir, "main.py")
    if os.path.exists(main_file):
        fix_main_issue(main_file)
    
    # Criar arquivo de correção para websocket_manager.py
    create_fix_for_websocket_manager(os.path.join(current_dir, "websocket_manager_fix.py"))
    
    logger.info("\nPara corrigir estes erros:")
    logger.info("1. Verifique os arquivos listados acima que contêm referências aos símbolos inválidos")
    logger.info("2. Certifique-se de que variáveis como 'price_log_time', 'websocket_manager', e 'socket_connections'")
    logger.info("   não sejam passadas como símbolos para funções que interagem com a API da Binance")
    logger.info("3. Aplique as correções sugeridas no arquivo websocket_manager_fix.py")