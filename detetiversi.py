import config
import os

def analisar_e_explicar_configuracao():
    explicacao = []
    insights = []

    explicacao.append("Olá! Sou o Detetive RSI, seu assistente para entender como seu bot de trading está configurado no momento.")
    explicacao.append("Vamos dar uma olhada no seu arquivo `config.py` e ver o que ele nos diz:\n")

    # 1. Modo de Operação
    if hasattr(config, 'SIMULATED_MODE') and config.SIMULATED_MODE:
        explicacao.append("**1. Modo de Operação:**")
        explicacao.append("   - Você está rodando em **MODO SIMULADO**. Isso é ótimo para testes, pois nenhuma ordem real será enviada para a Binance. As operações serão registradas em arquivos CSV de simulação.")
        insights.append("- Insight Modo Simulado: Lembre-se que o modo simulado usa taxas (`SIMULATED_FEE_PERCENT`) e um saldo inicial (`SIMULATION_INITIAL_BALANCE`) definidos. Os resultados podem variar do modo real devido a slippage e latência real da API.")
    else:
        explicacao.append("**1. Modo de Operação:**")
        explicacao.append("   - Você está rodando em **MODO REAL**. Atenção! As ordens serão enviadas para a Binance e operarão com seu capital real.")
        insights.append("- Insight Modo Real: Verifique sempre o saldo da sua conta e os parâmetros de risco (valor da ordem, alavancagem, stop loss) antes de deixar o bot rodando por longos períodos.")

    explicacao.append("\n**2. Mercados e Tempos Gráficos:**")
    # Pares Ativos
    pares_ativos = [p for p, ativo in config.PAIRS.items() if ativo]
    if pares_ativos:
        explicacao.append(f"   - Pares de moedas ativos: **{', '.join(pares_ativos)}**.")
    else:
        explicacao.append("   - Nenhum par de moeda está ativo! O bot não fará trades.")
        insights.append("- Alerta Pares: Nenhum par ativo! Para operar, você precisa definir pelo menos um par como `True` em `PAIRS` no `config.py`.")

    # Timeframes Ativos
    timeframes_ativos = [tf for tf, ativo in config.TIMEFRAMES.items() if ativo]
    if timeframes_ativos:
        explicacao.append(f"   - Timeframes ativos para análise: **{', '.join(timeframes_ativos)}**.")
    else:
        explicacao.append("   - Nenhum timeframe está ativo! O bot não analisará nenhum tempo gráfico.")
        insights.append("- Alerta Timeframes: Nenhum timeframe ativo! Para operar, defina pelo menos um timeframe como `True` em `TIMEFRAMES`.")

    explicacao.append("\n**3. Motor de Sinais Principal (RSI):**")
    if hasattr(config, 'SIGNAL_ENGINES') and config.SIGNAL_ENGINES.get('RSI', False):
        explicacao.append("   - O motor de sinais **RSI (Indicador de Força Relativa)** está ATIVO. Este é o cérebro principal para encontrar oportunidades.")
        explicacao.append(f"     - Período do RSI: **{config.RSI_PERIOD}**")
        explicacao.append(f"     - Limiar para modo atento de COMPRA (RSI abaixo de): **{config.RSI_BUY_THRESHOLD}**")
        explicacao.append(f"     - Limiar para modo atento de VENDA (RSI acima de): **{config.RSI_SELL_THRESHOLD}** (Parece alto, o comum é ~70. Verifique se é intencional!)")
        if config.RSI_SELL_THRESHOLD > 80:
            insights.append(f"- Alerta RSI Sell Threshold: Seu `RSI_SELL_THRESHOLD` está em {config.RSI_SELL_THRESHOLD}, que é um valor muito alto e pode nunca ser atingido, impedindo sinais de venda. O comum é em torno de 70-80. Revise este valor.")
        explicacao.append(f"     - Delta mínimo para confirmação do sinal RSI: **{config.RSI_DELTA_MIN}**")
        explicacao.append(f"     - RSI máximo para confirmar COMPRA (após modo atento): **{config.MAX_RSI_FOR_BUY}**")
        explicacao.append(f"     - RSI mínimo para confirmar VENDA (após modo atento): **{config.MIN_RSI_FOR_SELL}** (Parece alto, o comum é ~70. Verifique!)")
        if config.MIN_RSI_FOR_SELL > 70:
            insights.append(f"- Alerta MIN_RSI_FOR_SELL: Seu `MIN_RSI_FOR_SELL` está em {config.MIN_RSI_FOR_SELL}. Se o `RSI_SELL_THRESHOLD` for ~70-80, este valor pode impedir a confirmação de vendas. Revise.")
        explicacao.append(f"     - Timeout do modo atento: **{config.ATTENTIVE_MODE_TIMEOUT_MINUTES} minutos** (Se o RSI entrar em modo atento e não confirmar o sinal dentro deste tempo, ele reseta).")
    else:
        explicacao.append("   - O motor de sinais RSI está DESATIVADO. O bot não gerará sinais baseados em RSI.")
        insights.append("- Alerta Motor RSI: O motor RSI principal está desativado! Se esta é sua estratégia principal, ative-o em `SIGNAL_ENGINES`.")

    explicacao.append("\n**4. Configurações de Ordem e Risco (Valores Fixos):**")
    explicacao.append(f"   - Valor de cada ordem: **{config.ORDER_VALUE_USDT} USDT**")
    explicacao.append(f"   - Alavancagem: **{config.LEVERAGE}x**")
    explicacao.append(f"   - Stop Loss Fixo: **{config.STOP_LOSS_PERCENT*100:.2f}%**")
    explicacao.append(f"   - Take Profit Fixo: **{config.TAKE_PROFIT_PERCENT*100:.2f}%**")
    if config.LEVERAGE > 20:
        insights.append(f"- Atenção Alavancagem: Você está usando alavancagem de {config.LEVERAGE}x. Valores altos aumentam significativamente o risco de liquidação. Use com cautela.")
    if config.STOP_LOSS_PERCENT == config.TAKE_PROFIT_PERCENT:
        insights.append("- Insight Risco/Retorno: Seu Stop Loss e Take Profit fixos estão iguais ({config.STOP_LOSS_PERCENT*100:.2f}%). Considere se uma relação risco/retorno diferente (ex: TP maior que SL) seria mais adequada à sua estratégia.")

    explicacao.append("\n**5. Gerenciamento de Risco Avançado:**")
    # SL/TP Dinâmico por Volatilidade
    if hasattr(config, 'VOLATILITY_DYNAMIC_SL_TP_RSI_ENABLED') and config.VOLATILITY_DYNAMIC_SL_TP_RSI_ENABLED:
        explicacao.append("   - **SL/TP Dinâmico por Volatilidade (para RSI): ATIVO!**")
        explicacao.append(f"     - Os valores de Stop Loss e Take Profit serão ajustados com base na volatilidade do mercado, usando multiplicadores de **{config.VOLATILITY_SL_MULTIPLIER_RSI}x para SL** e **{config.VOLATILITY_TP_MULTIPLIER_RSI}x para TP**.")
        explicacao.append(f"     - Limites: SL entre {config.VOLATILITY_MIN_SL_PERCENT_RSI*100:.2f}%-{config.VOLATILITY_MAX_SL_PERCENT_RSI*100:.2f}%, TP entre {config.VOLATILITY_MIN_TP_PERCENT_RSI*100:.2f}%-{config.VOLATILITY_MAX_TP_PERCENT_RSI*100:.2f}%.")
        insights.append("- Insight SL/TP Dinâmico: Ótimo! Usar SL/TP dinâmico pode adaptar suas ordens às condições atuais do mercado. Monitore se os multiplicadores e limites estão adequados.")
    else:
        explicacao.append("   - SL/TP Dinâmico por Volatilidade (para RSI): DESATIVADO. Usará os valores fixos de SL/TP definidos acima.")

    # Pausa por SL Streak
    if hasattr(config, 'SL_STREAK_PAUSE_ENABLED') and config.SL_STREAK_PAUSE_ENABLED:
        explicacao.append("   - **Pausa por Stop Losses em Sequência: ATIVA!**")
        explicacao.append("     - Se um par atingir um certo número de stop losses seguidos (definido em `SL_STREAK_CONFIG`), ele será pausado temporariamente.")
        insights.append("- Insight Pausa SL Streak: Boa medida de proteção! Isso pode evitar grandes perdas em dias ruins ou quando a estratégia não está alinhada com o mercado.")
    else:
        explicacao.append("   - Pausa por Stop Losses em Sequência: DESATIVADA.")

    # Modo Invertido
    if hasattr(config, 'INVERTED_MODE_ENABLED') and config.INVERTED_MODE_ENABLED:
        explicacao.append("   - **Modo Invertido: ATIVO!**")
        explicacao.append("     - Após uma sequência de stop losses (definida em `INVERTED_MODE_CONFIG`), o bot poderá inverter os próximos sinais (compra vira venda, venda vira compra) por um número limitado de vezes.")
        insights.append("- Insight Modo Invertido: Estratégia avançada! O modo invertido pode ser útil se você identificar que seus sinais estão consistentemente errados em certos períodos. Use com análise cuidadosa.")
    else:
        explicacao.append("   - Modo Invertido: DESATIVADO.")

    explicacao.append("\n**6. Filtros Avançados para Sinais RSI (Os Especialistas do Detetive!):**")
    filtros_ativos_count = 0

    # Filtro de Tendência Operacional
    if hasattr(config, 'TREND_FILTER_ENABLED') and config.TREND_FILTER_ENABLED:
        filtros_ativos_count += 1
        explicacao.append("   - **Filtro de Tendência (EMA Longa no Timeframe Operacional): ATIVO!**")
        explicacao.append(f"     - Um sinal de COMPRA só será considerado se o preço estiver ACIMA da EMA de **{config.TREND_EMA_PERIOD} períodos** no gráfico que você está operando.")
        explicacao.append(f"     - Um sinal de VENDA só será considerado se o preço estiver ABAIXO desta EMA.")
        explicacao.append("     - Objetivo: Operar a favor da tendência principal do timeframe atual.")
        insights.append("- Insight Filtro Tendência Op.: Ótimo para evitar sinais contra-tendência no seu gráfico principal. Verifique se o período da EMA ({config.TREND_EMA_PERIOD}) está adequado.")
    else:
        explicacao.append("   - Filtro de Tendência (EMA Longa no Timeframe Operacional): DESATIVADO.")

    # Filtro Multi-Timeframe (MTF)
    if hasattr(config, 'MTF_FILTER_ENABLED') and config.MTF_FILTER_ENABLED:
        filtros_ativos_count += 1
        explicacao.append("   - **Filtro Multi-Timeframe (MTF): ATIVO!**")
        explicacao.append("     - Os sinais do seu timeframe operacional serão comparados com um timeframe de referência maior (ex: 5min -> 15min).")
        if hasattr(config, 'MTF_RSI_FILTER_ENABLED') and config.MTF_RSI_FILTER_ENABLED:
            explicacao.append(f"       - Sub-filtro RSI no MTF: ATIVO. Um sinal de COMPRA pode ser bloqueado se o RSI no TF maior (período {config.MTF_RSI_PERIOD}) estiver acima de {config.MTF_RSI_MAX_FOR_BUY}. Um sinal de VENDA pode ser bloqueado se o RSI no TF maior estiver abaixo de {config.MTF_RSI_MIN_FOR_SELL}.")
        else:
            explicacao.append("       - Sub-filtro RSI no MTF: DESATIVADO.")
        if hasattr(config, 'MTF_TREND_FILTER_ENABLED') and config.MTF_TREND_FILTER_ENABLED:
            explicacao.append(f"       - Sub-filtro Tendência (EMA) no MTF: ATIVO. Um sinal de COMPRA pode ser bloqueado se o preço no TF maior estiver abaixo da EMA de {config.MTF_EMA_PERIOD} períodos. Um sinal de VENDA pode ser bloqueado se estiver acima.")
        else:
            explicacao.append("       - Sub-filtro Tendência (EMA) no MTF: DESATIVADO.")
        insights.append("- Insight Filtro MTF: Excelente para confirmar sinais com o contexto de um tempo gráfico maior. Certifique-se que os `MTF_REFERENCE_TIMEFRAMES` estão configurados como deseja.")
    else:
        explicacao.append("   - Filtro Multi-Timeframe (MTF): DESATIVADO.")

    # Filtro de Suporte e Resistência (S/R)
    if hasattr(config, 'SR_FILTER_ENABLED') and config.SR_FILTER_ENABLED:
        filtros_ativos_count += 1
        explicacao.append("   - **Filtro de Níveis de Suporte e Resistência (S/R): ATIVO!**")
        explicacao.append(f"     - Os sinais RSI serão validados se ocorrerem próximos a níveis de S/R significativos (proximidade de {config.SR_SWING_PRICE_PROXIMITY_PERCENT*100:.2f}% ou {config.SR_PIVOT_PRICE_PROXIMITY_PERCENT*100:.2f}%). ")
        if hasattr(config, 'SR_SWING_POINTS_ENABLED') and config.SR_SWING_POINTS_ENABLED:
            explicacao.append(f"       - Sub-filtro Swing Points (Topos/Fundos Recentes): ATIVO (janela {config.SR_SWING_WINDOW}, lookback {config.SR_SWING_LOOKBACK_PERIOD}).")
        else:
            explicacao.append("       - Sub-filtro Swing Points: DESATIVADO.")
        if hasattr(config, 'SR_PIVOT_POINTS_ENABLED') and config.SR_PIVOT_POINTS_ENABLED:
            explicacao.append(f"       - Sub-filtro Pivot Points (Diários, etc.): ATIVO (timeframe {config.SR_PIVOT_TIMEFRAME}, níveis {', '.join(config.SR_PIVOT_LEVELS_TO_USE)}).")
        else:
            explicacao.append("       - Sub-filtro Pivot Points: DESATIVADO.")
        if not (config.SR_SWING_POINTS_ENABLED or config.SR_PIVOT_POINTS_ENABLED):
            insights.append("- Alerta Filtro S/R: O filtro S/R principal está ativo, mas nenhum sub-filtro (Swing ou Pivot) está. Ele não terá efeito. Ative `SR_SWING_POINTS_ENABLED` ou `SR_PIVOT_POINTS_ENABLED`.")
        insights.append("- Insight Filtro S/R: Operar perto de S/R pode aumentar a probabilidade de sucesso. Ajuste os parâmetros de proximidade e lookback conforme o comportamento do ativo.")
    else:
        explicacao.append("   - Filtro de Níveis de Suporte e Resistência (S/R): DESATIVADO.")

    # Filtro de Volume
    if hasattr(config, 'VOLUME_FILTER_ENABLED') and config.VOLUME_FILTER_ENABLED:
        filtros_ativos_count += 1
        explicacao.append("   - **Filtro de Confirmação por Volume: ATIVO!**")
        explicacao.append(f"     - Um sinal só será considerado se o volume no momento do sinal for pelo menos **{config.VOLUME_INCREASE_PERCENT}% maior** que a média do volume nos últimos **{config.VOLUME_LOOKBACK_PERIOD} candles**.")
        explicacao.append("     - Objetivo: Confirmar a força e convicção por trás do sinal.")
        explicacao.append(f"     - Os sinais RSI serão validados se ocorrerem próximos a níveis de S/R significativos (proximidade de {config.SR_SWING_PRICE_PROXIMITY_PERCENT*100:.2f}% ou {config.SR_PIVOT_PRICE_PROXIMITY_PERCENT*100:.2f}%).")
    else:
        explicacao.append("   - Filtro de Confirmação por Volume: DESATIVADO.")

    if filtros_ativos_count == 0:
        explicacao.append("\n   -> No momento, **nenhum filtro avançado está ativo**. O sistema operará baseado apenas nos sinais primários do RSI, similar ao seu sistema v1.0, mas com a lógica de modo atento da v2.0.")
        insights.append("- Insight Filtros: Nenhum filtro avançado ativo. O sistema usará apenas o RSI. Considere ativar alguns filtros gradualmente para refinar os sinais e potencialmente melhorar o win rate.")
    elif filtros_ativos_count > 0:
        explicacao.append(f"\n   -> Você tem **{filtros_ativos_count} filtro(s) avançado(s) ativos**. O Detetive RSI consultará esses especialistas antes de tomar uma decisão!")

    explicacao.append("\n**7. Notificações:**")
    if hasattr(config, 'TELEGRAM_NOTIFICATIONS_ENABLED') and config.TELEGRAM_NOTIFICATIONS_ENABLED:
        explicacao.append("   - Notificações gerais do Telegram: ATIVAS.")
        if hasattr(config, 'TELEGRAM_NOTIFY_ON_ORDER') and config.TELEGRAM_NOTIFY_ON_ORDER:
            explicacao.append("     - Notificará no Telegram a cada nova ordem aberta.")
        if hasattr(config, 'TELEGRAM_NOTIFY_ON_CLOSE') and config.TELEGRAM_NOTIFY_ON_CLOSE:
            explicacao.append("     - Notificará no Telegram a cada ordem fechada.")
        # Adicionar outras flags de notificação se necessário
    else:
        explicacao.append("   - Notificações do Telegram: DESATIVADAS.")

    explicacao.append("\nLembre-se: esta é uma análise da sua configuração atual. Você pode ajustar o `config.py` para mudar o comportamento do bot.")
    explicacao.append("Teste diferentes combinações de filtros no modo simulado para encontrar o que funciona melhor para você!")

    # Montar o output
    output_final = "\n".join(explicacao)
    output_final += "\n\n--- INSIGHTS E SUGESTÕES DO DETETIVE RSI ---\n"
    if insights:
        for i, insight in enumerate(insights):
            output_final += f"{i+1}. {insight}\n"
    else:
        output_final += "Nenhum insight específico no momento. Parece que suas configurações básicas estão ok ou os alertas principais já foram dados na explicação acima.\n"
    
    output_final += "\nBoa sorte e bons trades!"
    return output_final

if __name__ == "__main__":
    print("Executando o Detetive RSI para analisar sua configuração...\n")
    # Garante que o config.py seja encontrado se detetiversi.py estiver no mesmo diretório
    # Para uma solução mais robusta, config.py poderia ser um módulo instalável ou o path gerenciado.
    try:
        # Tenta recarregar o módulo config caso ele tenha sido alterado desde o último import
        import importlib
        importlib.reload(config)
        
        explicacao_completa = analisar_e_explicar_configuracao()
        print(explicacao_completa)

        # Salvar a explicação em um arquivo .txt para fácil leitura
        nome_arquivo_saida = "analise_config_detetive_rsi.txt"
        with open(nome_arquivo_saida, "w", encoding="utf-8") as f:
            f.write(explicacao_completa)
        print(f"\n\n[+] Análise salva em: {os.path.abspath(nome_arquivo_saida)}")
        
    except ImportError:
        print("ERRO: Não foi possível encontrar o arquivo 'config.py'. Certifique-se de que ele está no mesmo diretório que 'detetiversi.py' ou acessível no PYTHONPATH.")
    except AttributeError as e:
        print(f"ERRO: Parece que uma variável esperada não foi encontrada no seu 'config.py'. Detalhes: {e}")
        print("Por favor, verifique se seu 'config.py' está completo e com todas as variáveis da v2.0.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao analisar a configuração: {e}")

