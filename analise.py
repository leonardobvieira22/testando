import pandas as pd
from binance_utils import BinanceUtils
from config import REAL_API_KEY, REAL_API_SECRET, PAIRS, TAKE_PROFIT_PERCENT, STOP_LOSS_PERCENT
from utils import logger
from binance.enums import SIDE_BUY, SIDE_SELL

def analisar_pares():
    binance = BinanceUtils(REAL_API_KEY, REAL_API_SECRET)
    resultados = []

    for pair in PAIRS.keys():
        dados = binance.get_historical_data(pair, '1m', limit=500)
        if dados is None or dados.empty:
            logger.warning(f"[Analise] Dados insuficientes para {pair}")
            continue

        dados['open'] = dados['open'].astype(float)
        dados['close'] = dados['close'].astype(float)
        dados['high'] = dados['high'].astype(float)
        dados['low'] = dados['low'].astype(float)

        tp_count = 0
        sl_count = 0
        pnl_acumulado = 0.0

        for i in range(len(dados) - 1):
            candle_anterior = dados.iloc[i]
            candle_atual = dados.iloc[i + 1]

            if candle_anterior['close'] > candle_anterior['open']:
                direcao = 'BUY'
            elif candle_anterior['close'] < candle_anterior['open']:
                direcao = 'SELL'
            else:
                continue

            entry_price = candle_anterior['close']
            tp_price = entry_price * (1 + TAKE_PROFIT_PERCENT/100) if direcao == 'BUY' else entry_price * (1 - TAKE_PROFIT_PERCENT)
            sl_price = entry_price * (1 - STOP_LOSS_PERCENT) if direcao == 'BUY' else entry_price * (1 + STOP_LOSS_PERCENT)

            high = float(candle_atual['high'])
            low = float(candle_atual['low'])

            if direcao == 'BUY':
                if high >= tp_price:
                    tp_count += 1
                    pnl_acumulado += TAKE_PROFIT_PERCENT
                elif low <= sl_price:
                    sl_count += 1
                    pnl_acumulado -= STOP_LOSS_PERCENT
            elif direcao == 'SELL':
                if low <= tp_price:
                    tp_count += 1
                    pnl_acumulado += TAKE_PROFIT_PERCENT
                elif high >= sl_price:
                    sl_count += 1
                    pnl_acumulado -= STOP_LOSS_PERCENT

        resultados.append({
            'pair': pair,
            'tp_count': tp_count,
            'sl_count': sl_count,
            'pnl_acumulado': pnl_acumulado
        })

    resultados.sort(key=lambda x: x['pnl_acumulado'], reverse=True)

    for res in resultados:
        logger.info(f"[Analise] {res['pair']}: TPs={res['tp_count']}, SLs={res['sl_count']}, PnL={res['pnl_acumulado']:.2%}")

    melhor_par = resultados[0]['pair']
    melhor_pnl = resultados[0]['pnl_acumulado']
    logger.info(f"[Analise] Melhor par para operar: {melhor_par} com lucro acumulado de {melhor_pnl:.2%}")

if __name__ == "__main__":
    analisar_pares()