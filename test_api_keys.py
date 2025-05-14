from binance.client import Client
from config import REAL_API_KEY, REAL_API_SECRET
from utils import logger

def test_api_keys():
    try:
        client = Client(REAL_API_KEY, REAL_API_SECRET)
        account_info = client.get_account()
        logger.info("Conexão com Binance API bem-sucedida!")
        logger.info(f"Informações da conta: {account_info['canTrade']}")
    except Exception as e:
        logger.error(f"Erro ao testar chaves API: {e}")

if __name__ == "__main__":
    test_api_keys()