from sinal_engine import SignalEngine
from sinal_engine_bollinger import BollingerSignalEngine
from sinal_enginecandle import CandleSignalEngine
from config import REAL_API_KEY, REAL_API_SECRET, SIGNAL_ENGINES
from utils import logger, log_configurations, ensure_orders_pnl_csv_exists
import time
import config
import threading
import traceback
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from binance import ThreadedWebsocketManager
import sys
import asyncio
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from utils import acquire_csv_lock, update_order_csv  # Importar funções utilitárias
from binance_utils import get_current_price, get_entry_price  # Importar funções relacionadas à Binance

# Define a safe shared state wrapper to prevent internal variables being used as trading symbols
class SafeSharedState:
    """
    Wrapper for shared state that prevents internal control variables from being 
    treated as trading symbols in API calls.
    """
    # Variables that should never be treated as trading symbols
    INTERNAL_VARIABLES = {
        'price_log_time', 'websocket_manager', 'socket_connections',
        'open_positions', 'active_monitors', 'sl_streak', 'pause_until',
        'lock', 'prices', 'last_price_update'
    }

    def __init__(self, original_state):
        """
        Initialize with the original shared state dictionary
        """
        self._state = original_state
        self._valid_symbols = set()
        self._initialize_valid_symbols()

    def _initialize_valid_symbols(self):
        """Initialize set of valid trading symbols from configuration"""
        for pair, active in config.PAIRS.items():
            if active:
                self._valid_symbols.add(pair)
        logger.info(f"Initialized SafeSharedState with {len(self._valid_symbols)} valid trading pairs")

    def __getitem__(self, key):
        """Safe access to shared state items"""
        return self._state[key]

    def __setitem__(self, key, value):
        """Safe setting of shared state items"""
        self._state[key] = value

    def __contains__(self, key):
        """Safe containment check"""
        return key in self._state

    def get(self, key, default=None):
        """Safe get method with default value"""
        return self._state.get(key, default)

    def items(self):
        """Safe items iterator"""
        return self._state.items()

    def keys(self):
        """Safe keys iterator"""
        return self._state.keys()

    def values(self):
        """Safe values iterator"""
        return self._state.values()

    def is_valid_symbol(self, symbol):
        """
        Check if the given symbol is a valid trading symbol
        """
        if symbol in self.INTERNAL_VARIABLES:
            logger.warning(f"Attempted to use internal variable '{symbol}' as a trading symbol")
            return False
        if symbol not in self._valid_symbols:
            logger.warning(f"Potentially invalid trading symbol: '{symbol}'")
            # We don't block unknown symbols completely as new pairs might be added
        return True

    def get_trading_symbols(self):
        """
        Return only valid trading symbols (used for safe iteration)
        """
        return [symbol for symbol in self._valid_symbols]

    def get_raw_state(self):
        """
        Return the raw underlying state dictionary (use with caution)
        """
        return self._state


def validate_configuration():
    """Validate required configuration settings."""
    if not REAL_API_KEY or not REAL_API_SECRET:
        logger.error("API key and secret must be configured")
        return False
    
    if not any(SIGNAL_ENGINES.values()):
        logger.error("At least one signal engine must be enabled in SIGNAL_ENGINES")
        return False
    
    if not any(config.PAIRS.values()):
        logger.error("At least one trading pair must be enabled in PAIRS")
        return False
    
    return True


def price_websocket_callback(shared_state):
    """Returns a callback function for the websocket price data."""
    def callback(msg):
        if msg.get('e') == 'error':
            logger.error(f"WebSocket error: {msg.get('m', 'No error message')}")
            return
        
        if msg.get('s') and msg.get('c'):
            symbol = msg['s']
            price = float(msg['c'])
            with shared_state['lock']:
                shared_state['prices'][symbol] = price
                shared_state['last_price_update'][symbol] = time.time()
            
            # Log price update periodically (not every update to avoid log spam)
            if symbol not in shared_state['price_log_time'] or \
               time.time() - shared_state['price_log_time'].get(symbol, 0) > 60:  # Log once per minute
                logger.debug(f"Updated {symbol} price: {price}")
                shared_state['price_log_time'][symbol] = time.time()
    
    return callback


def initialize_websocket(api_key, api_secret, active_pairs, shared_state):
    """Initialize WebSocket for price data with reconnection handling."""
    max_retries = 5
    retry_delay = 5
    
    for _ in range(max_retries):
        try:
            logger.info("Initializing price WebSocket connection...")
            twm = ThreadedWebsocketManager(api_key=api_key, api_secret=api_secret)
            twm.start()
            
            callback = price_websocket_callback(shared_state)
            
            socket_connections = {}
            for pair in active_pairs:
                socket_id = twm.start_symbol_ticker_socket(callback=callback, symbol=pair)
                socket_connections[pair] = socket_id
                logger.info(f"Started price stream for {pair}")
            
            shared_state['websocket_manager'] = twm
            shared_state['socket_connections'] = socket_connections
            
            # Add a reconnection monitor
            def monitor_websocket_connections():
                while True:
                    time.sleep(30)  # Check every 30 seconds
                    try:
                        current_time = time.time()
                        with shared_state['lock']:
                            for pair in active_pairs:
                                # If we haven't received an update in 2 minutes, reconnect
                                if pair not in shared_state['last_price_update'] or \
                                   current_time - shared_state['last_price_update'].get(pair, 0) > 120:
                                    logger.warning(f"No recent price updates for {pair}. Reconnecting websocket...")
                                    if pair in shared_state['socket_connections']:
                                        old_socket = shared_state['socket_connections'][pair]
                                        twm.stop_socket(old_socket)
                                    
                                    new_socket = twm.start_symbol_ticker_socket(callback=callback, symbol=pair)
                                    shared_state['socket_connections'][pair] = new_socket
                                    logger.info(f"Reconnected price stream for {pair}")
                    except Exception as e:
                        logger.error(f"Error in WebSocket monitor: {e}")
            
            # Start the monitor thread
            monitor_thread = threading.Thread(target=monitor_websocket_connections, daemon=True)
            monitor_thread.start()
            
            return twm
            
        except Exception as e:
            logger.error(f"Failed to initialize WebSocket: {e}")
            time.sleep(retry_delay)
    
    logger.critical("Failed to initialize WebSocket after multiple attempts")
    raise RuntimeError("Could not establish WebSocket connection")


def create_signal_engine_wrapper(engine_class, api_key, api_secret, safe_shared_state):
    """
    Creates a signal engine with protection against invalid symbols
    """
    # Create the original engine
    engine = engine_class(api_key, api_secret, safe_shared_state.get_raw_state())
    
    # Patch the generate_signals method to safely iterate only over valid trading symbols
    original_generate_signals = engine.generate_signals
    
    def safe_generate_signals(*args, **kwargs):
        try:
            return original_generate_signals(*args, **kwargs)
        except AttributeError as e:
            if "'dict' object has no attribute" in str(e):
                logger.warning(f"Caught potential dict access issue in {engine_class.__name__}: {e}")
                # Handle any special recovery needed here
                return None
            else:
                raise
    
    engine.generate_signals = safe_generate_signals
    return engine


def calculate_realized_pnl(entry_price, close_price, quantity, direction, fee_percent=0.0004):
    """
    Calcula o PnL realizado considerando taxas de abertura e fechamento.
    """
    gross_pnl = (close_price - entry_price) * quantity if direction == "BUY" else (entry_price - close_price) * quantity
    total_fees = (entry_price * quantity + close_price * quantity) * fee_percent
    return gross_pnl - total_fees

def close_order(symbol, side, quantity, close_reason):
    """
    Fecha uma ordem e registra detalhes no CSV.
    """
    close_price = get_current_price(config.REAL_API_KEY, config.REAL_API_SECRET, symbol)  # Função para obter o preço atual
    entry_price = get_entry_price(config.REAL_API_KEY, config.REAL_API_SECRET, symbol)  # Função para obter o preço de entrada
    direction = "BUY" if side == "SELL" else "SELL"
    realized_pnl = calculate_realized_pnl(entry_price, close_price, quantity, direction)

    # Atualizar o CSV com detalhes da ordem fechada
    with acquire_csv_lock('data/orders.csv') as orders_csv:
        update_order_csv(orders_csv, symbol, side, quantity, close_price, realized_pnl, close_reason)

    # Log detalhado
    logger.info(f"Ordem fechada: {symbol}, Lado: {side}, Quantidade: {quantity}, "
                f"Preço de Fechamento: {close_price}, PnL Realizado: {realized_pnl:.6f}, Motivo: {close_reason}")


def main():
    logger.info("[RSI_V2][DEBUG] Entered main()")
    logger.info("Starting trading system...")
    
    # Validate configuration
    if not validate_configuration():
        logger.critical("Invalid configuration. Exiting.")
        return
    
    log_configurations(config)
    ensure_orders_pnl_csv_exists()
    
    # Shared state for signal engines with better structure
    raw_shared_state = {
        'open_positions': {},
        'active_monitors': {},
        'sl_streak': {},
        'pause_until': {},
        'lock': threading.Lock(),
        'prices': {},
        'last_price_update': {},
        'price_log_time': {},
        'websocket_manager': None,
        'socket_connections': {}
    }
    
    # Wrap the shared state with our safe implementation
    shared_state = SafeSharedState(raw_shared_state)
    
    # Initialize Binance API client
    try:
        client = Client(REAL_API_KEY, REAL_API_SECRET)
        # Test the connection
        server_time = client.get_server_time()
        logger.info(f"Successfully connected to Binance API. Server time: {server_time['serverTime']}")
    except BinanceAPIException as e:
        logger.critical(f"Failed to initialize Binance client: {e}")
        return
    except BinanceRequestException as e:
        logger.critical(f"Network issue with Binance API: {e}")
        return
    except Exception as e:
        logger.critical(f"Unexpected error initializing Binance client: {e}")
        return
    
    # Get active trading pairs
    active_pairs = [pair for pair, active in config.PAIRS.items() if active]
    if not active_pairs:
        logger.critical("No active trading pairs configured")
        return
    
    logger.info(f"Active trading pairs: {', '.join(active_pairs)}")
    
    # Initialize WebSocket for price data
    try:
        twm_prices = initialize_websocket(REAL_API_KEY, REAL_API_SECRET, active_pairs, raw_shared_state)
    except RuntimeError as e:
        logger.critical(f"Failed to start WebSocket manager: {e}")
        return
    
    engines = []
    
    # Initialize signal engines with proper protection against invalid symbols
    try:
        # Initialize signal engines based on configuration
        if SIGNAL_ENGINES.get("RSI", False):
            rsi_engine = SignalEngine(REAL_API_KEY, REAL_API_SECRET, active_pairs)
            engines.append(rsi_engine)
            logger.info("RSI signal engine initialized")
        
        if SIGNAL_ENGINES.get("Bollinger", False):
            bollinger_engine = BollingerSignalEngine(REAL_API_KEY, REAL_API_SECRET, active_pairs)
            engines.append(bollinger_engine)
            logger.info("Bollinger signal engine initialized")
        
        if SIGNAL_ENGINES.get("Candle", False):
            candle_engine = CandleSignalEngine(REAL_API_KEY, REAL_API_SECRET, active_pairs)
            engines.append(candle_engine)
            logger.info("Candle signal engine initialized")
        
        if not engines:
            logger.error("No signal engines active in configuration")
            twm_prices.stop()
            return
            
    except Exception as e:
        logger.critical(f"Error initializing signal engines: {e}")
        logger.critical(traceback.format_exc())
        twm_prices.stop()
        return
    
    # Main processing loop
    running = True
    try:
        logger.info("Trading system started successfully")
        logger.info("[RSI_V2][DEBUG] Main processing loop entered")
        while running:
            try:
                logger.info("Starting new signal check cycle")
                logger.info("[RSI_V2][DEBUG] Beginning signal check cycle")
                cycle_start = time.time()
                
                # Wait for initial price data
                if not shared_state['prices']:
                    logger.info("Waiting for initial price data...")
                    time.sleep(2)
                    continue
                
                # Generate trading signals
                for engine in engines:
                    logger.info(f"[RSI_V2][DEBUG] Calling generate_signals on {engine.__class__.__name__}")
                    engine.generate_signals()
                
                cycle_duration = time.time() - cycle_start
                logger.info(f"Signal check cycle completed in {cycle_duration:.2f} seconds")
                
                # Dynamic sleep time to maintain consistent cycle time
                sleep_time = max(1, 2 - cycle_duration)  # At least 1 second, up to 2 seconds
                time.sleep(sleep_time)
                
            except Exception as e:
                logger.error(f"Error in processing cycle: {e}")
                logger.error(traceback.format_exc())
                time.sleep(5)  # Wait before retrying after an error
    
    except KeyboardInterrupt:
        logger.info("System interrupted by user")
    except Exception as e:
        logger.critical(f"Critical system error: {e}")
        logger.critical(traceback.format_exc())
    finally:
        # Clean shutdown
        logger.info("Shutting down trading system...")
        running = False
        
        # Stop all engines
        for engine in engines:
            try:
                engine.stop()
                logger.info(f"Stopped {engine.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error stopping {engine.__class__.__name__}: {e}")
        
        # Stop WebSocket manager
        try:
            if twm_prices:
                twm_prices.stop()
                logger.info("WebSocket manager stopped")
        except Exception as e:
            logger.error(f"Error stopping WebSocket manager: {e}")
        
        logger.info("Trading system shutdown complete")


if __name__ == "__main__":
    engine = SignalEngine(
        api_key=config.REAL_API_KEY,
        api_secret=config.REAL_API_SECRET,
        pairs=[pair for pair, enabled in config.PAIRS.items() if enabled]
    )

    if config.RECOVER_POSITIONS_ON_STARTUP:
        engine.recover_positions_on_startup()

    try:
        engine.start()
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Interrupção do usuário detectada")
        engine.stop()
    except Exception as e:
        logger.critical(f"Erro crítico: {str(e)}")
        engine.stop()