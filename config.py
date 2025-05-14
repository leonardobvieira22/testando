# ===========================================================
# CONFIGURAÇÕES DO SISTEMA DE TRADING - CONFIGURAÇÃO COMPLETA (V2.0 Integrada)
# ===========================================================
#
# Este arquivo contém todas as configurações necessárias para o funcionamento do bot.
# Baseado no config original, com funcionalidades da V2.0 adicionadas (desativadas por padrão).
#
# COMO USAR: Altere os valores conforme sua necessidade, mas NÃO altere os nomes das variáveis.
# IMPORTANTE: Após alterar qualquer configuração, reinicie o bot para que as mudanças tenham efeito.

import os

print("[Config V2] Carregando configurações...")

# =========================
# CHAVES API PARA BINANCE
# =========================
# Estas são suas chaves de API. Nunca compartilhe-as com ninguém!

REAL_API_KEY = "VGQ0dhdCcHjDhEjj0Xuue3ZtyIZHiG9NK8chA4ew0HMQMywydjrVrLTWeN8nnZ9e"
REAL_API_SECRET = "jHrPFutd2fQH2AECeABbG6mDvbJqhEYBt1kuYmiWfcBjJV22Fwtykqx8mDFle3dO"


# =========================
# PARES E TIMEFRAMES
# =========================
# Escolha quais pares e timeframes o bot vai operar.
# True = Ativo | False = Inativo

# Pares de negociação para trading real
PAIRS = {
    "XRPUSDT": False,    # Ripple
    "DOGEUSDT": True,    # Dogecoin
    "TRXUSDT": False,    # Tron
    "BTCUSDT": False,    # Bitcoin
    "INITUSDT": False,   # Init
    "TRUMPUSDT": True,   # Trump
    "AVAAIUSDT": False,  # Avalanche AI
    "ZEREBROUSDT": False # Zerebro
}

# Timeframes disponíveis para trading real
TIMEFRAMES = {
    "1m": True,   # 1 minuto
    "5m": True,  # 5 minutos
    "15m": True, # 15 minutos
    "1h": True,  # 1 hora
    "4h": False,  # 4 horas
    "1d": False   # 1 dia
}


# =========================
# MOTORES DE SINAIS
# =========================
# Escolha quais motores de sinais estarão ativos.
# True = Ativo | False = Inativo

SIGNAL_ENGINES = {
    "RSI": True,       # Usa indicador RSI para sinais (Principal motor da V2.0)
    "Bollinger": False, # Usa Bandas de Bollinger para sinais (Placeholder do original)
    "Candle": False     # Usa padrões de candles para sinais (Placeholder do original)
}

# Configurações do indicador RSI (Valores do original mantidos onde possível)
RSI_PERIOD = 10                 # Período para cálculo do RSI (Original: 10, V2.0 usa 14 - ajuste se necessário para V2.0)
RSI_BUY_THRESHOLD = 30.0        # Abaixo deste valor, considera sinal de compra (V2.0 usa para modo atento)
RSI_SELL_THRESHOLD = 120.0      # Acima deste valor, considera sinal de venda (V2.0 usa 70 para modo atento)
RSI_DELTA_MIN = 5.0             # Mínima variação do RSI para considerar (Usado na V2.0)
MAX_RSI_FOR_BUY = 30            # Máximo RSI para entrar em compra (V2.0 usa para limitar entrada)
MIN_RSI_FOR_SELL = 90           # Mínimo RSI para entrar em venda (V2.0 usa para limitar entrada)
RSI_EXIT_LONG_THRESHOLD = 70    # Saída de posição long se RSI acima (Lógica de saída RSI não na V2.0 base)
RSI_EXIT_SHORT_THRESHOLD = 35   # Saída de posição short se RSI abaixo (Lógica de saída RSI não na V2.0 base)
# Novos parâmetros RSI da V2.0 (ajuste conforme estratégia V2.0)
RSI_MIN_LONG = 5                # RSI mínimo para efetivamente abrir uma COMPRA (após sinal V na V2.0) (Original: 5)
RSI_MAX_SHORT = 110             # RSI máximo para efetivamente abrir uma VENDA (após sinal A na V2.0) (Original: 110)
# ATTENTIVE_MODE_TIMEOUT_MINUTES = 5  # Tempo em modo atento após sinal (Original: 5, V2.0 usa 120)
ATTENTIVE_MODE_TIMEOUT_MINUTES = 120 # V2.0: Tempo em minutos para resetar o modo atento se nenhum sinal for gerado

# Configurações para Bollinger Bands (Placeholder do original)
BOLLINGER_PERIOD = 20           # Número de períodos para cálculo das bandas
BOLLINGER_DEVIATION = 2.0       # Desvio padrão para as bandas de Bollinger

# Timeframes e períodos para motor de candle (Placeholders do original)
CANDLE_TIMEFRAMES = {"1m": True, "5m": True, "15m": False, "1h": False, "4h": False, "1d": False}
CANDLE_PERIODS = {"1m": 1, "5m": 1, "15m": 3, "1h": 3, "4h": 3, "1d": 3}


# =========================
# CONFIGURAÇÕES DE ORDENS
# =========================
# Define valores para ordens, stop loss e take profit (Valores do original mantidos)
ORDER_VALUE_USDT = 0.25         # Valor fixo da ordem em USDT
LEVERAGE = 30                    # Alavancagem (obs: valores altos = maior risco)
STOP_LOSS_PERCENT = 0.008      # Stop Loss em percentual (0.008 = 0.8%)
TAKE_PROFIT_PERCENT = 0.008   # Take Profit em percentual (0.008 = 0.8%)


# =========================
# CONFIGURAÇÕES DE VOLATILIDADE (Original e Integração V2.0)
# =========================
# Configurações básicas de cálculo de volatilidade (do original, podem ser usadas por outras partes do sistema se houver)
VOLATILITY_UPDATE_INTERVAL = 60
VOLATILITY_WINDOW_SIZE = 100
VOLATILITY_TIME_WINDOW = 3600
VOLATILITY_CACHE_TTL = 300
VOLATILITY_LOW = 0.05
VOLATILITY_MEDIUM = 0.15
VOLATILITY_HIGH = 0.30
VOLATILITY_MIN_VALID = 0.001
VOLATILITY_MAX_VALID = 1.0

# === SISTEMA AVANÇADO DE VOLATILIDADE (V2.0 - AJUSTE DINÂMICO DE SL/TP PARA RSI) ===
# Permite ajuste dinâmico de stops e take profits baseado na volatilidade atual para o motor RSI.
VOLATILITY_DYNAMIC_SL_TP_RSI_ENABLED = False  # V2.0 FEATURE: Ativa/desativa SL/TP dinâmico para RSI (DEFAULT: False)

# Parâmetros para cálculo da volatilidade usada no SL/TP dinâmico do RSI (V2.0)
VOLATILITY_LOOKBACK_PERIODS_RSI = 20        # Períodos para cálculo da volatilidade para RSI
VOLATILITY_SL_MULTIPLIER_RSI = 1.5          # Multiplicador da volatilidade para o Stop Loss
VOLATILITY_TP_MULTIPLIER_RSI = 2.0          # Multiplicador da volatilidade para o Take Profit

# Limites percentuais para o SL/TP dinâmico (V2.0), para evitar valores extremos
VOLATILITY_MIN_SL_PERCENT_RSI = 0.005       # SL mínimo percentual (0.5%) se dinâmico
VOLATILITY_MAX_SL_PERCENT_RSI = 0.03        # SL máximo percentual (3%) se dinâmico
VOLATILITY_MIN_TP_PERCENT_RSI = 0.0075      # TP mínimo percentual (0.75%) se dinâmico
VOLATILITY_MAX_TP_PERCENT_RSI = 0.05        # TP máximo percentual (5%) se dinâmico

# Configurações de Volatilidade Avançada do Original (mantidas para referência)
# VOLATILITY_ADVANCED_FEATURES_ENABLED = False # No original, V2.0 usa VOLATILITY_DYNAMIC_SL_TP_RSI_ENABLED especificamente
# VOLATILITY_DYNAMIC_SL_TP_BOLLINGER_ENABLED = False # Placeholder do original
# VOLATILITY_DYNAMIC_SL_TP_SIM_ENABLED = True      # Placeholder do original
# VOLATILITY_DYNAMIC_ADJUSTMENT = True
# VOLATILITY_SL_MIN_MULTIPLIER = 0.8
# VOLATILITY_SL_MAX_MULTIPLIER = 1.5
# VOLATILITY_TP_MIN_MULTIPLIER = 0.9
# VOLATILITY_TP_MAX_MULTIPLIER = 1.5
# VOLATILITY_SL_ADJUSTMENT_STRENGTH = 0.3
# VOLATILITY_TP_ADJUSTMENT_STRENGTH = 0.2
# VOLATILITY_DYNAMIC_PAUSE_ENABLED = False
# VOLATILITY_PAUSE_REFERENCE = 0.01
# VOLATILITY_PAUSE_MIN_MULTIPLIER = 0.5
# VOLATILITY_PAUSE_MAX_MULTIPLIER = 3.0
# VOLATILITY_MAX_PAUSE_MINUTES = 30
# VOLATILITY_ENHANCED_LOGGING = True
# VOLATILITY_LOG_INCLUDE_CATEGORY = True


# =========================
# CONFIGURAÇÕES DE PAUSA E SL STREAK (Original e Integração V2.0)
# =========================
# Controla como o bot se comporta após stop losses consecutivos
SL_STREAK_PAUSE_ENABLED = False  # V2.0 FEATURE: Chave principal para ativar a pausa por SL Streak (DEFAULT: False)
SL_STREAK_NOTIFICATIONS_ENABLED = False  # Notificações quando SL streak aumenta (do original)
SL_STREAK_CONFIG = { # (do original)
    "XRPUSDT": {"threshold": 1, "pause_minutes": 2},
    "DOGEUSDT": {"threshold": 1, "pause_minutes": 2},
    "TRXUSDT": {"threshold": 1, "pause_minutes": 2},
    "BTCUSDT": {"threshold": 1, "pause_minutes": 2},
    "INITUSDT": {"threshold": 1, "pause_minutes": 2},
    "TRUMPUSDT": {"threshold": 1, "pause_minutes": 2},
    "AVAAIUSDT": {"threshold": 1, "pause_minutes": 2},
    "ZEREBROUSDT": {"threshold": 1, "pause_minutes": 2},
    "default": {"threshold": 3, "pause_minutes": 30} # V2.0: Adicionado default para pares não listados
}


# =========================
# CONFIGURAÇÕES DE MODO INVERTIDO (Original e Integração V2.0)
# =========================
# O modo invertido inverte os sinais após uma série de stop losses
INVERTED_MODE_ENABLED = False  # Habilita o modo invertido (do original)
INVERTED_MODE_NOTIFICATIONS_ENABLED = True  # Notifica quando ativar/desativar (do original)
INVERTED_MODE_CONFIG = { # (do original)
    "XRPUSDT": {"sl_threshold": 2, "invert_count": 2},
    "DOGEUSDT": {"sl_threshold": 2, "invert_count": 2},
    "TRXUSDT": {"sl_threshold": 2, "invert_count": 2},
    "BTCUSDT": {"sl_threshold": 2, "invert_count": 2},
    "default": {"sl_threshold": 5, "invert_count": 3} # V2.0: Adicionado default para pares não listados
}


# ========================================
# FILTROS AVANÇADOS PARA SINAIS RSI (V2.0 - NOVAS FEATURES)
# ========================================
# Todos os filtros abaixo podem ser ativados/desativados globalmente.
# Se um filtro principal está desativado, seus sub-filtros são ignorados.
# TODAS AS NOVAS FEATURES DE FILTRO VÊM DESATIVADAS POR PADRÃO.

# 1. Filtro Multi-Timeframe (MTF)  Especialista em Contexto Maior (Filtro Multi-Timeframe - MTF)
#O que ele faz? Este especialista é como um "supervisor" do detetive RSI. 
# Ele olha para um timeframe (tempo gráfico) maior para ver se o sinal do RSI faz sentido no contexto mais amplo. 
# Por exemplo, se você opera no gráfico de 5 minutos (seu timeframe operacional), o MTF pode olhar o gráfico de 15 minutos ou 1 hora.
MTF_FILTER_ENABLED = True  # V2.0 FEATURE (DEFAULT: False)
MTF_REFERENCE_TIMEFRAMES = { # Timeframe operacional -> Timeframe de referência
    "1m": "5m",
    "5m": "15m",
    "15m": "1h",
    "1h": "4h"
}
# Sub-filtro RSI no MTF de referência
MTF_RSI_FILTER_ENABLED = True  # V2.0 FEATURE (DEFAULT: False)
MTF_RSI_PERIOD = 10
MTF_RSI_MAX_FOR_BUY = 45     # Se RSI no TF de referência > X, não compra
MTF_RSI_MIN_FOR_SELL = 35    # Se RSI no TF de referência < Y, não vende
# Sub-filtro Tendência (EMA) no MTF de referência
MTF_TREND_FILTER_ENABLED = False  # V2.0 FEATURE (DEFAULT: False)
MTF_EMA_PERIOD = 50          # Período da EMA longa no TF de referência

# 2. Filtro de Níveis de Suporte e Resistência (S/R) Especialista em Zonas de Perigo/Oportunidade (Filtro de Níveis de Suporte e Resistência - S/R)
#O que ele faz? Este especialista é como um cartógrafo que conhece as "zonas seguras" e "zonas perigosas" do mapa de preços.
#  Suportes são como "pisos" onde o preço tende a parar de cair e pode subir. Resistências são como "tetos" onde o preço tende a parar de subir e pode cair.
SR_FILTER_ENABLED = False  # V2.0 FEATURE (DEFAULT: False) - MODIFICADO PARA DEBUG
# Sub-filtro Swing Points
SR_SWING_POINTS_ENABLED = False  # V2.0 FEATURE (DEFAULT: False) - MODIFICADO PARA DEBUG
SR_SWING_WINDOW = 5
SR_SWING_LOOKBACK_PERIOD = 60
SR_SWING_PRICE_PROXIMITY_PERCENT = 0.003 # 0.3% de proximidade
# Sub-filtro Pivot Points
SR_PIVOT_POINTS_ENABLED = False  # V2.0 FEATURE (DEFAULT: False) - MODIFICADO PARA DEBUG
SR_PIVOT_TIMEFRAME = "1d"
SR_PIVOT_PRICE_PROXIMITY_PERCENT = 0.003 # 0.3% de proximidade
SR_PIVOT_LEVELS_TO_USE = ["PP", "S1", "R1", "S2", "R2"]

# 3. Filtro de Confirmação por Volume Especialista em Força do Movimento (Filtro de Confirmação por Volume)
#O que ele faz? Este especialista mede a "energia" ou "convicção" por trás de um movimento de preço. 
# Um aumento significativo no volume de negociação durante um sinal pode indicar que há mais força e interesse naquele movimento.
VOLUME_FILTER_ENABLED = True  # V2.0 FEATURE (DEFAULT: False)
VOLUME_LOOKBACK_PERIOD = 20
VOLUME_INCREASE_PERCENT = 20 # Volume X% acima da média

# 4. Filtro de Tendência (EMA Longa no TF Operacional) Especialista em Tendência Principal (Filtro de EMA Longa no Timeframe Operacional)
# O que ele faz? Este especialista é como um "detetive de tendência".
#O que ele faz? Este especialista olha para o "mapa geral" da tendência no timeframe que você está operando (ex: 5 minutos). 
# Ele usa uma Média Móvel Exponencial (EMA) de período longo (ex: EMA de 200 períodos) para saber se o preço está, de modo geral, subindo, descendo ou andando de lado.
TREND_FILTER_ENABLED = False  # V2.0 FEATURE (DEFAULT: False)
TREND_EMA_PERIOD = 200      # Período da EMA longa no TF operacional


# =========================
# CONFIGURAÇÕES MODO SIMULADO (do original)
# =========================
SIMULATED_MODE = False
SIMULATED_TELEGRAM_ENABLED = True
SIMULATED_TIMEFRAMES = {"1m": True, "5m": True, "15m": True, "1h": True, "4h": True, "1d": True}
SIMULATED_PAIRS = {"XRPUSDT": True, "DOGEUSDT": True, "TRXUSDT": True, "BTCUSDT": True}
SIMULATED_ORDER_VALUE_USDT = 30.0
SIMULATED_LEVERAGE = 20
SIMULATED_SL_PERCENT = 0.0025
SIMULATED_TP_PERCENT = 0.0025
SIMULATED_FEE_PERCENT = 0.0004
SIMULATED_SL_STREAK_ENABLED = True
SIMULATED_SL_STREAK_NOTIFICATIONS_ENABLED = True
SIMULATED_SL_STREAK_CONFIG = {
    "XRPUSDT": {"threshold": 3, "pause_minutes": 7},
    "DOGEUSDT": {"threshold": 3, "pause_minutes": 7},
    "TRXUSDT": {"threshold": 3, "pause_minutes": 7},
    "BTCUSDT": {"threshold": 3, "pause_minutes": 5}
}
SIMULATED_INVERTED_MODE_ENABLED = True
SIMULATED_INVERTED_MODE_NOTIFICATIONS_ENABLED = True
SIMULATED_INVERTED_MODE_CONFIG = {
    "XRPUSDT": {"sl_threshold": 3, "invert_count": 2},
    "DOGEUSDT": {"sl_threshold": 3, "invert_count": 2},
    "TRXUSDT": {"sl_threshold": 3, "invert_count": 2},
    "BTCUSDT": {"sl_threshold": 3, "invert_count": 2}
}
# V2.0: Adicionado para consistência com arquivos de dados da V2.0
SIMULATION_INITIAL_BALANCE = 1000 # Saldo inicial para simulação (V2.0)


# =========================
# CONFIGURAÇÕES DE ARQUIVOS E LOGGING (Original, com adições V2.0)
# =========================
LOG_LEVEL = "INFO"  # V2.0 FEATURE: DEBUG, INFO, WARNING, ERROR, CRITICAL (DEFAULT: INFO)
LOG_FILE = "bot.log" # (do original)

# Arquivos de dados (nomes do original mantidos)
SIGNALS_CSV = "signals.csv"
ORDERS_CSV = "orders.csv"
ORDERS_PNL_CSV = "orders_pnl.csv"
PNL_CSV = "pnl.csv"
PRICES_STATISTICS_CSV = "prices_statistics.csv"
PAUSE_LOG_CSV = "pause_log.csv"
SL_STREAK_LOG_CSV = "sl_streak_log.csv"
INVERTED_MODE_LOG_CSV = "inverted_mode_log.csv"

# Arquivos para modo simulado (nomes do original mantidos)
SIMULATED_ORDERS_CSV = "simulated_orders.csv"
SIMULATED_PAUSE_LOG_CSV = "simulated_pause_log.csv"
# INVERTED_STATE_CSV era do original, V2.0 usa PAUSES_CSV e INVERTED_STATE_CSV para persistência de estado real
# Para V2.0, os estados de pausa e modo invertido são salvos em arquivos separados (definidos abaixo)
# Se o sistema V2.0 for usar os nomes do original para estado, precisará adaptar a lógica de save/load.
# V2.0 usa estes para estado real:
PAUSES_CSV = "pauses_v2.csv" # V2.0: Persistência de pausas e SL streak real
INVERTED_STATE_CSV = "inverted_state_v2.csv" # V2.0: Persistência do modo invertido real
# E estes para estado simulado:
SIMULATED_INVERTED_STATE_CSV = "simulated_inverted_state_v2.csv" # V2.0: Persistência do modo invertido simulado
# Os SIM_ORDERS_CSV, SIM_ORDERS_PNL_CSV, SIM_PAUSE_LOG_CSV do original são mantidos.

# Intervalos de atualização (do original)
PNL_UPDATE_INTERVAL = 3.0
PRICE_STATISTICS_UPDATE_INTERVAL = 0.4

# =========================
# CONFIGURAÇÕES DO TELEGRAM (Original, com adições V2.0)
# =========================
TELEGRAM_BOT_TOKEN = "7804058317:AAHyGa90DwA17jIS-j4vDmYAWFOyBZLinoY" # (do original)
TELEGRAM_CHAT_ID = "6097421181"               # (do original)
SIMULATED_TELEGRAM_CHAT_ID = "6097421182"     # (do original)
# telegram_api_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage" # (do original, pode ser reconstruído no código se necessário)

# V2.0 FEATURE: Flags granulares de notificação Telegram (DEFAULT: False, exceto geral)
TELEGRAM_NOTIFICATIONS_ENABLED = False # Chave geral para notificações (DEFAULT: False para não sobrepor original)
TELEGRAM_NOTIFY_ON_ORDER = False     # Notificar a cada nova ordem (DEFAULT: False)
TELEGRAM_NOTIFY_ON_CLOSE = False     # Notificar a cada fechamento de ordem (DEFAULT: False)
TELEGRAM_NOTIFY_ON_ERROR = False     # Notificar em erros críticos (DEFAULT: False)
# Notificações de SL Streak e Modo Invertido são controladas por suas flags específicas (SL_STREAK_NOTIFICATIONS_ENABLED, INVERTED_MODE_NOTIFICATIONS_ENABLED)


# =========================
# CONFIGURAÇÕES AUXILIARES (do original)
# =========================
SIM_LEVERAGE = SIMULATED_LEVERAGE
SIM_ORDER_VALUE_USDT = SIMULATED_ORDER_VALUE_USDT
SIM_STOP_LOSS_PERCENT = SIMULATED_SL_PERCENT
SIM_TAKE_PROFIT_PERCENT = SIMULATED_TP_PERCENT
SIM_INVERTED_MODE_ENABLED = SIMULATED_INVERTED_MODE_ENABLED
SIM_INVERTED_MODE_CONFIG = SIMULATED_INVERTED_MODE_CONFIG
SIM_SL_STREAK_CONFIG = SIMULATED_SL_STREAK_CONFIG

# =========================
# CONFIGURAÇÕES DE INTERVALOS E DESEMPENHO (V2.0 - NOVAS FEATURES)
# =========================
MAIN_LOOP_INTERVAL = 2       # V2.0 FEATURE: Intervalo em segundos para o loop principal de geração de sinais (DEFAULT: 10s, ajuste conforme necessidade)
WEBSOCKET_PRICE_TIMEOUT = 5 # V2.0 FEATURE: Timeout em segundos para esperar preço do WebSocket antes de fallback HTTP (DEFAULT: 15s)

# --- Checagens e Criação de Diretórios (V2.0 Style, adaptado) ---
# O sistema V2.0 pode precisar de um diretório de dados. Se os arquivos CSV acima não estiverem em subdiretórios,
# esta seção pode não ser necessária ou pode ser adaptada.
# Exemplo: Se for usar um DATA_DIR="data_v2_merged"
# DATA_DIR_V2 = "data_v2_merged"
# if not os.path.exists(DATA_DIR_V2):
#     try:
#         os.makedirs(DATA_DIR_V2)
#         print(f"[Config V2 Merged] Diretório '{DATA_DIR_V2}' criado.")
#     except Exception as e:
#         print(f"[Config V2 Merged] Erro ao criar diretório '{DATA_DIR_V2}': {e}")

# Validações para garantir configurações consistentes
if 'RSI_MIN_LONG' in locals() and 'RSI_MAX_SHORT' in locals():
    if RSI_MIN_LONG >= RSI_MAX_SHORT:
        print(f"[Config V2] AVISO: RSI_MIN_LONG ({RSI_MIN_LONG}) deve ser menor que RSI_MAX_SHORT ({RSI_MAX_SHORT})")

# Se todas as configurações de filtro estiverem ativas, emitir aviso
filter_settings = [
    var for var in ['MTF_FILTER_ENABLED', 'SR_FILTER_ENABLED', 
                   'VOLUME_FILTER_ENABLED', 'TREND_EMA_FILTER_ENABLED'] 
    if var in locals()
]
if all(locals().get(setting, False) for setting in filter_settings):
    print("[Config V2] AVISO: Todos os filtros estão ativos, o que pode ser muito restritivo para geração de sinais.")

# Recuperação de posições abertas após reinicialização
RECOVER_POSITIONS_ON_STARTUP = True

print("[Config V2 Merged] Configurações carregadas. Novas features da V2.0 estão DESATIVADAS por padrão.")

