import threading
import time
from binance import ThreadedWebsocketManager
import logging

class BinanceWebSocket:
    def __init__(self, api_key, api_secret, pairs):
        self.api_key = api_key
        self.api_secret = api_secret
        self.twm = None
        self.prices = {}
        self.orders = {}
        self.lock = threading.Lock()
        self.pairs = pairs
        self.logger = logging.getLogger("BinanceWebSocket")
        self._running = False
        self._ws_thread = None

    def _run_ws(self):
        while self._running:
            try:
                self.twm = ThreadedWebsocketManager(api_key=self.api_key, api_secret=self.api_secret)
                self.twm.start()
                for pair in self.pairs:
                    self.twm.start_symbol_ticker_socket(callback=self.price_callback, symbol=pair)
                self.logger.info("WebSocket iniciado para pares: %s", self.pairs)
                while self._running and self.twm._is_running:
                    time.sleep(1)
            except Exception as e:
                self.logger.error(f"WebSocket falhou: {e}. Tentando reconectar em 5s...")
                time.sleep(5)
            finally:
                if self.twm:
                    try:
                        self.twm.stop()
                    except Exception:
                        pass

    def start(self):
        if self._running:
            self.logger.warning("WebSocket já está rodando.")
            return
        self._running = True
        self._ws_thread = threading.Thread(target=self._run_ws, daemon=True)
        self._ws_thread.start()

    def is_active(self):
        return self._running and self._ws_thread and self._ws_thread.is_alive()

    def price_callback(self, msg):
        if msg.get('e') == 'error':
            self.logger.error(f"WebSocket error: {msg}")
            return
        symbol = msg['s']
        price = float(msg['c'])
        with self.lock:
            self.prices[symbol] = price
        self.logger.debug(f"Preço atualizado via WebSocket: {symbol} = {price}")

    def get_price(self, symbol):
        with self.lock:
            return self.prices.get(symbol)

    def stop(self):
        self._running = False
        if self.twm:
            self.twm.stop()
        self.logger.info("WebSocket parado.")
