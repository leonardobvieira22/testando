"""
Patch para corrigir chamadas à API da Binance com símbolos inválidos
"""
import functools
import logging

logger = logging.getLogger(__name__)

# Lista de símbolos inválidos conhecidos (variáveis de controle)
INVALID_SYMBOLS = ['price_log_time', 'websocket_manager', 'socket_connections']

def safe_binance_call(func):
    """
    Decorator para proteger chamadas de API Binance contra símbolos inválidos
    
    Uso:
        binance_utils.get_historical_data = safe_binance_call(binance_utils.get_historical_data)
    """
    @functools.wraps(func)
    def wrapper(symbol, *args, **kwargs):
        # Verificar se o símbolo é uma variável de controle conhecida
        if symbol in INVALID_SYMBOLS:
            logger.warning(f"Tentativa bloqueada de usar variável de controle '{symbol}' como símbolo de trading")
            # Retornar um valor seguro dependendo da função
            if func.__name__ == 'get_historical_data':
                import pandas as pd
                return pd.DataFrame()  # DataFrame vazio para get_historical_data
            elif func.__name__ == 'get_current_price':
                return None  # None para get_current_price
            else:
                return None  # Valor seguro padrão
                
        # Se o símbolo parece válido, chamar a função original
        return func(symbol, *args, **kwargs)
    
    return wrapper

def apply_safe_wrappers(binance_utils):
    """
    Aplica os wrappers de segurança às funções relevantes do objeto binance_utils
    
    Args:
        binance_utils: Instância da classe BinanceUtils
        
    Returns:
        O mesmo objeto binance_utils com funções protegidas
    """
    # Proteger get_historical_data
    binance_utils.get_historical_data = safe_binance_call(binance_utils.get_historical_data)
    
    # Proteger get_current_price
    binance_utils.get_current_price = safe_binance_call(binance_utils.get_current_price)
    
    # Proteger outras funções que aceitam símbolo como primeiro parâmetro
    for func_name in dir(binance_utils):
        if func_name.startswith('__'):
            continue
            
        func = getattr(binance_utils, func_name)
        if callable(func) and func.__name__ not in ['get_historical_data', 'get_current_price']:
            try:
                # Verificar se a assinatura da função tem 'symbol' como primeiro parâmetro
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                
                # Se o primeiro parâmetro após 'self' for potencialmente um símbolo
                if len(params) >= 2 and params[1] in ['symbol', 'pair', 'ticker']:
                    setattr(binance_utils, func_name, safe_binance_call(func))
            except Exception:
                pass  # Ignorar funções que não podem ser inspecionadas
    
    logger.info("Proteção contra símbolos inválidos aplicada à instância BinanceUtils")
    return binance_utils