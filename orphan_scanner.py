import os
import csv
import time
from datetime import datetime, timezone
from filelock import FileLock, Timeout
from binance.client import Client
from binance.exceptions import BinanceAPIException
from config import REAL_API_KEY, REAL_API_SECRET, ORDERS_CSV
from utils import logger

# intervalo de scan (segundos)
SCAN_INTERVAL = 60
# arquivo de log de ordens órfãs
ORPHAN_CSV = "orphan.csv"
# locks para evitar escrita concorrente
LOCKFILE = f"{ORPHAN_CSV}.lock"
ORDERS_LOCK = f"{ORDERS_CSV}.lock"
# não cancelar ordens criadas há menos de X segundos (evita corrida com registro)
NEW_ORDER_GRACE = 10

# inicia cliente Binance
client = Client(REAL_API_KEY, REAL_API_SECRET)

def load_tracked_orders() -> dict:
    """
    Lê ORDERS_CSV (com lock) e retorna dict {order_id: status}.
    """
    tracked = {}
    lock = FileLock(ORDERS_LOCK, timeout=5)
    try:
        with lock, open(ORDERS_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                oid = row.get("order_id")
                if oid:
                    oid_str = str(int(float(oid)))
                    tracked[oid_str] = row.get("status", "").upper()
    except Timeout:
        logger.warning("[ORPHAN] Timeout ao ler ORDERS_CSV; prosseguindo sem ordens rastreadas.")
    except FileNotFoundError:
        logger.warning(f"[ORPHAN] {ORDERS_CSV} não encontrado; sem ordens rastreadas.")
    except Exception as e:
        logger.error(f"[ORPHAN] Erro lendo {ORDERS_CSV}: {e}")
    return tracked

def get_realized_pnl(symbol: str, order_id: int) -> float:
    """
    Soma realizedPnl dos trades da ordem.
    """
    try:
        trades = client.futures_account_trades(symbol=symbol, orderId=order_id)
        return sum(float(t.get("realizedPnl", 0)) for t in trades)
    except BinanceAPIException as e:
        logger.error(f"[ORPHAN] Erro ao obter PnL para {symbol} orderId={order_id}: {e}")
    except Exception as e:
        logger.error(f"[ORPHAN] Erro inesperado em get_realized_pnl: {e}")
    return 0.0

def scan_orphans():
    """
    Busca ordens abertas na Binance; identifica órfãs ou divergentes;
    cancela e registra em ORPHAN_CSV.
    """
    scanned_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    tracked = load_tracked_orders()

    try:
        open_orders = client.futures_get_open_orders()
    except BinanceAPIException as e:
        logger.error(f"[ORPHAN] Erro ao buscar open_orders: {e}")
        return
    except Exception as e:
        logger.error(f"[ORPHAN] Erro inesperado ao buscar open_orders: {e}")
        return

    total_open = len(open_orders)
    canceled = 0
    errors = 0

    lock = FileLock(LOCKFILE, timeout=5)
    try:
        with lock, open(ORPHAN_CSV, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # escreve header se criar o arquivo agora
            if f.tell() == 0:
                writer.writerow([
                    "scanned_at",
                    "order_id",
                    "symbol",
                    "side",
                    "price",
                    "origQty",
                    "executedQty",
                    "created_time",
                    "linked",
                    "original_status",
                    "cancel_status",
                    "canceled_time",
                    "realized_pnl"
                ])
            for ord in open_orders:
                try:
                    oid = str(int(ord["orderId"]))
                    # evita corrida: ignora ordens criadas há pouco
                    created_ts = ord.get("time", 0) / 1000.0
                    if time.time() - created_ts < NEW_ORDER_GRACE:
                        continue

                    linked = oid in tracked
                    orig_status = tracked.get(oid, "N/A")
                    # órfã = não rastreada OU marcada CLOSED mas ainda aberta
                    if (not linked) or (orig_status == "CLOSED"):
                        pnl = get_realized_pnl(ord["symbol"], int(ord["orderId"]))
                        try:
                            resp = client.futures_cancel_order(
                                symbol=ord["symbol"],
                                orderId=int(ord["orderId"])
                            )
                            cancel_status = resp.get("status", "CANCELED")
                            canceled += 1
                        except Exception as e:
                            cancel_status = f"ERROR:{e}"
                            errors += 1

                        canceled_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                        created_time = datetime.fromtimestamp(created_ts, timezone.utc)\
                                            .strftime("%Y-%m-%d %H:%M:%S")

                        writer.writerow([
                            scanned_at,
                            oid,
                            ord.get("symbol"),
                            ord.get("side"),
                            ord.get("price"),
                            ord.get("origQty"),
                            ord.get("executedQty"),
                            created_time,
                            linked,
                            orig_status,
                            cancel_status,
                            canceled_time,
                            f"{pnl:.8f}"
                        ])
                        f.flush()
                        os.fsync(f.fileno())

                        logger.info(
                            f"[ORPHAN] {'NãoRast.' if not linked else 'FechadoMasAberto'} "
                            f"order canceled: {oid} {ord.get('symbol')} pnl={pnl:.2f}"
                        )
                except Exception as e:
                    errors += 1
                    logger.error(f"[ORPHAN] Erro processando ordem {ord.get('orderId')}: {e}")

        logger.info(f"[ORPHAN] Scan concluído: open_orders={total_open}, canceled={canceled}, errors={errors}")
    except Timeout:
        logger.error("[ORPHAN] Timeout ao adquirir lock de ORPHAN_CSV")
    except Exception as e:
        logger.error(f"[ORPHAN] Erro gravando em {ORPHAN_CSV}: {e}")

def main():
    logger.info("[ORPHAN] Iniciando orphan scanner...")
    while True:
        scan_orphans()
        time.sleep(SCAN_INTERVAL)

if __name__ == "__main__":
    main()