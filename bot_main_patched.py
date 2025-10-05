# bot_v1_fixed.py
import os
import sys
import time
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_UP

# Hyperliquid SDK
from hyperliquid.info import Info
from hyperliquid.exchange import Exchange
from hyperliquid.utils import constants
from eth_account import Account

# ================== Utils ==================

def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> pd.Series:
    h = pd.Series(high); l = pd.Series(low); c = pd.Series(close)
    tr1 = (h - l).abs()
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(length).mean()

def pivots(series: np.ndarray, left: int, right: int, is_high: bool) -> List[float]:
    s = pd.Series(series)
    n = len(s)
    out = [np.nan] * n
    for i in range(left, n - right):
        window = s.iloc[i - left : i + right + 1]
        center = s.iloc[i]
        if is_high:
            if center == window.max():
                out[i + right] = float(center)
        else:
            if center == window.min():
                out[i + right] = float(center)
    return out

def ceil_to_tick(px: float, tick: float) -> float:
    dpx = Decimal(str(px))
    dt  = Decimal(str(tick))
    return float((dpx / dt).to_integral_value(rounding=ROUND_UP) * dt)

def floor_to_tick(px: float, tick: float) -> float:
    dpx = Decimal(str(px))
    dt  = Decimal(str(tick))
    return float((dpx / dt).to_integral_value(rounding=ROUND_DOWN) * dt)

def floor_to_step(x: float, step: float) -> float:
    dx = Decimal(str(x))
    ds = Decimal(str(step))
    return float((dx / ds).to_integral_value(rounding=ROUND_DOWN) * ds)

# ================== Timeframes ==================

INTERVAL_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "8h": 28_800_000, "12h": 43_200_000,
    "1d": 86_400_000, "3d": 259_200_000, "1w": 604_800_000, "1M": 2_592_000_000
}

def aligned_now_ms(interval_ms: int) -> int:
    now_ms = int(time.time() * 1000)
    return (now_ms // interval_ms) * interval_ms

# ================== Dataclasses ==================

@dataclass
class StratParams:
    lookback: int
    sensitivity: int
    alert_cutoff: int
    atr_len: int
    atr_mult: float
    near_pct: float
    take_profit_rr: float
    stop_pct: float
    dca_spacing_atr: float
    max_pyramid: int
    order_pct: float
    enable_shorts: bool
    break_mode: str = "close"

@dataclass
class CoinMeta:
    tick: float = 0.0          # price tick (ex.: 0.5 para BTC)
    lot: float = 0.0           # size step (ex.: 0.001)
    sz_decimals: int = 3       # usado p/ mínimo 10^-sz_decimals

    @property
    def min_size(self) -> float:
        if self.lot and self.lot > 0:
            return self.lot
        return float(Decimal(10) ** Decimal(-self.sz_decimals))

@dataclass
class PositionState:
    side: str = "flat"  # "flat" | "long" | "short"
    fills: List[Tuple[float, float]] = field(default_factory=list)  # [(price, size)]
    initial_px: Optional[float] = None
    next_dca_idx: int = 1

    def avg_entry(self) -> Optional[float]:
        if not self.fills:
            return None
        notional = sum(px * sz for px, sz in self.fills)
        qty = sum(sz for _, sz in self.fills)
        return notional / qty if qty > 0 else None

    def qty(self) -> float:
        return sum(sz for _, sz in self.fills)

# ================== Bot ==================

class CubanRangeReversalBot:
    def __init__(self):
        load_dotenv()
        self.env = os.getenv("ENV", "testnet").lower()
        self.dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s | %(levelname)s | %(message)s"
        )
        self.tz = timezone(timedelta(hours=-3))  # BRT

        # Ativos
        coins_env = os.getenv("COINS")
        if coins_env:
            self.coins = [s.strip().upper() for s in coins_env.split(",") if s.strip()]
        else:
            self.coins = ["BTC", "ETH"]

        # Timeframe / break mode
        self.tf = os.getenv("TIMEFRAME", "3m")
        if self.tf not in INTERVAL_MS:
            raise ValueError(f"TIMEFRAME '{self.tf}' não suportado.")
        self.break_mode = os.getenv("BREAK_MODE", "wick").lower().strip()

        # Fonte de candles e timeouts
        self.info_source = os.getenv("INFO_SOURCE", "testnet").lower()
        self.info_timeout = float(os.getenv("INFO_TIMEOUT_SEC", "20"))

        # Slippage IOC
        self.ioc_slip = float(os.getenv("IOC_SLIPPAGE", "0.003"))  # 0.3%

        # Estratégia
        self.params = StratParams(
            lookback=int(os.getenv("LOOKBACK", 120)),
            sensitivity=int(os.getenv("SENSITIVITY", 1)),
            alert_cutoff=int(os.getenv("ALERT_CUTOFF", 1)),
            atr_len=int(os.getenv("ATR_LEN", 14)),
            atr_mult=float(os.getenv("ATR_MULT", 0.30)),
            near_pct=float(os.getenv("NEAR_PCT", 0.006)),
            take_profit_rr=float(os.getenv("TAKE_PROFIT_RR", 0.5)),
            stop_pct=float(os.getenv("STOP_PCT", 5)),
            dca_spacing_atr=float(os.getenv("DCA_SPACING_ATR", 1.0)),
            max_pyramid=int(os.getenv("MAX_PYRAMID", 10)),
            order_pct=float(os.getenv("ORDER_PCT", 5.0)),
            enable_shorts=os.getenv("ENABLE_SHORTS", "true").lower() == "true",
            break_mode=self.break_mode
        )

        # Notional fixo por entrada quando equity indisponível
        self.fixed_order_usd = float(os.getenv("ORDER_USD", "25"))

        # SDK base
        api_url = constants.TESTNET_API_URL if self.env == "testnet" else constants.MAINNET_API_URL

        # Info opcional
        self.info = None
        sdk_info_disabled = os.getenv("SDK_INFO_DISABLED", "true").lower() == "true"
        try:
            if not sdk_info_disabled:
                self.info = Info(api_url, skip_ws=True)
        except Exception as e:
            logging.warning(f"[INIT] Info() indisponível ({e}). Prosseguindo sem SDK Info.")
            self.info = None

        # Exchange (se não for DRY)
        self.exchange = None
        self.account_address = os.getenv("ACCOUNT_ADDRESS")
        secret_key = os.getenv("SECRET_KEY")
        if not self.dry_run:
            if not self.account_address or not secret_key:
                logging.error("Preencha ACCOUNT_ADDRESS e SECRET_KEY no .env (ou use DRY_RUN=true).")
                sys.exit(1)
            wallet = Account.from_key(secret_key)
            self.exchange = Exchange(wallet, api_url, account_address=self.account_address)

        # /info base
        if self.info_source == "mainnet":
            self.base_info_url = "https://api.hyperliquid.xyz"
        elif self.info_source == "testnet":
            self.base_info_url = "https://api.hyperliquid-testnet.xyz"
        else:
            self.base_info_url = "https://api.hyperliquid-testnet.xyz" if self.env == "testnet" else "https://api.hyperliquid.xyz"
        self.info_url = f"{self.base_info_url}/info"

        # HTTP session
        self.http = requests.Session()
        retry = Retry(
            total=5, connect=5, read=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=frozenset(["POST"]),
            raise_on_status=False
        )
        adapter = HTTPAdapter(max_retries=retry)
        self.http.mount("https://", adapter)
        self.http.mount("http://", adapter)

        # Metas e estados
        self.meta_by_coin: Dict[str, CoinMeta] = {c: CoinMeta() for c in self.coins}
        self._load_meta()  # <-- puxa tick/lot/sz_decimals
        self.state: Dict[str, PositionState] = {c: PositionState() for c in self.coins}
        self.last_signal_bar: Dict[Tuple[str, str], int] = {}
        self.bar_index: Dict[str, int] = {c: 0 for c in self.coins}
        self.last_bar_ts: Dict[str, Optional[int]] = {c: None for c in self.coins}
        self.pending_entry: Dict[str, Optional[str]] = {c: None for c in self.coins}

        logging.info(f"Ambiente: {self.env} | DRY_RUN={self.dry_run} | TF={self.tf} | BREAK_MODE={self.break_mode}")

    # ================ Meta (tick/lot) =================

    def _load_meta(self):
        """
        Tenta obter tick (price increment), lot (size step) e szDecimals via /info meta.
        Falhou? Usa fallbacks razoáveis.
        """
        try:
            r = self.http.post(self.info_url, json={"type": "meta"}, timeout=self.info_timeout)
            r.raise_for_status()
            meta = r.json()
        except Exception as e:
            logging.warning(f"[META] Falha ao obter meta: {e}. Aplicando fallbacks.")
            meta = None

        # Estruturas possíveis:
        # meta["universe"] = [ {"name":"BTC","szDecimals":3, "pxDecimals":1, "tickSize":0.5, "lotSize":0.001}, ... ]
        # ou variações de chaves.
        for coin in self.coins:
            cm = CoinMeta()
            if isinstance(meta, dict):
                universe = meta.get("universe") or meta.get("assets") or meta.get("contracts") or []
                # procura por name/symbol/coin
                found = None
                for it in universe:
                    name = (it.get("name") or it.get("symbol") or it.get("coin") or "").upper()
                    if name == coin:
                        found = it
                        break
                if found:
                    sz_dec = int(found.get("szDecimals", found.get("sizeDecimals", 3)))
                    tick = found.get("tickSize", found.get("tick", None))
                    px_dec = found.get("pxDecimals", None)
                    if tick is None and px_dec is not None:
                        try:
                            tick = float(Decimal(10) ** Decimal(-int(px_dec)))
                        except Exception:
                            tick = None
                    lot = found.get("lotSize", found.get("minSize", None))
                    if lot is None:
                        lot = float(Decimal(10) ** Decimal(-sz_dec))
                    cm.sz_decimals = sz_dec
                    cm.lot = float(lot)
                    # fallbacks específicos se tick faltou
                    if tick is None or float(tick) <= 0:
                        if coin == "BTC":
                            cm.tick = 0.5
                        elif coin == "ETH":
                            cm.tick = 0.1
                        else:
                            cm.tick = float(Decimal(10) ** Decimal(-2))  # 0.01
                    else:
                        cm.tick = float(tick)
                else:
                    # não achou no meta → fallbacks
                    cm = self._fallback_meta(coin)
            else:
                cm = self._fallback_meta(coin)

            self.meta_by_coin[coin] = cm
            logging.info(f"[META] {coin} tick={cm.tick} lot={cm.lot} szDecimals={cm.sz_decimals}")

    @staticmethod
    def _fallback_meta(coin: str) -> CoinMeta:
        if coin == "BTC":
            return CoinMeta(tick=0.5, lot=0.001, sz_decimals=3)
        if coin == "ETH":
            return CoinMeta(tick=0.1, lot=0.001, sz_decimals=3)
        return CoinMeta(tick=0.01, lot=0.001, sz_decimals=3)

    # ================ Dados =================

    def fetch_candles(self, coin: str, n_bars: int) -> pd.DataFrame:
        try:
            n = max(50, min(n_bars, 5000))
            interval_ms = INTERVAL_MS[self.tf]
            end_time = aligned_now_ms(interval_ms)
            start_time = end_time - n * interval_ms

            payload = {
                "type": "candleSnapshot",
                "req": {"coin": coin, "interval": self.tf, "startTime": start_time, "endTime": end_time}
            }
            r = self.http.post(self.info_url, json=payload, timeout=self.info_timeout)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                logging.error(f"[{coin}] candleSnapshot resposta inesperada: {data}")
                return pd.DataFrame(columns=["t","open","high","low","close","volume"])

            rows = [[int(c["t"]), float(c["o"]), float(c["h"]), float(c["l"]), float(c["c"]), float(c["v"])] for c in data]
            df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume"])
            return df.dropna().sort_values("t").reset_index(drop=True)

        except Exception as e:
            logging.error(f"[{coin}] fetch_candles erro: {e}")
            return pd.DataFrame(columns=["t","open","high","low","close","volume"])

    # ================ Strategy =================

    def compute_signal(self, coin: str, df_closed: pd.DataFrame):
        p = self.params
        min_bars = max(p.lookback, p.atr_len) + (p.sensitivity * 2 + 2)
        if len(df_closed) < min_bars:
            return False, False, {}

        highs = df_closed["high"].to_numpy()
        lows  = df_closed["low"].to_numpy()
        closes = df_closed["close"].to_numpy()

        rh = pd.Series(highs).rolling(p.lookback).max()
        rl = pd.Series(lows ).rolling(p.lookback).min()
        a  = atr(highs, lows, closes, p.atr_len)

        ph = pivots(highs, p.sensitivity, p.sensitivity, True)
        pl = pivots(lows,  p.sensitivity, p.sensitivity, False)

        if p.break_mode == "wick":
            up_src = pd.Series(highs)
            dn_src = pd.Series(lows)
            cross_up = (up_src.shift(1) <= rh.shift(1)) & (up_src > rh)
            cross_dn = (dn_src.shift(1) >= rl.shift(1)) & (dn_src < rl)
        else:
            src = pd.Series(closes)
            cross_up = (src.shift(1) <= rh.shift(1)) & (src > rh)
            cross_dn = (src.shift(1) >= rl.shift(1)) & (src < rl)

        high_thresh = rh - a * p.atr_mult
        low_thresh  = rl + a * p.atr_mult

        if p.break_mode == "wick":
            src_up = pd.Series(highs)
            src_dn = pd.Series(lows)
            cross_down_high_thresh = (src_dn.shift(1) >= high_thresh.shift(1)) & (src_dn < high_thresh)
            cross_up_low_thresh    = (src_up.shift(1) <= low_thresh.shift(1))  & (src_up > low_thresh)
        else:
            src = pd.Series(closes)
            cross_down_high_thresh = (src.shift(1) >= high_thresh.shift(1)) & (src < high_thresh)
            cross_up_low_thresh    = (src.shift(1) <= low_thresh.shift(1))  & (src > low_thresh)

        bi = self.bar_index[coin]

        near_high = (not math.isnan(ph[-1])) and (ph[-1] >= float(rh.iloc[-1]) * (1 - p.near_pct))
        near_low  = (not math.isnan(pl[-1])) and (pl[-1] <= float(rl.iloc[-1]) * (1 + p.near_pct))

        reversal_after_break_up   = bool(cross_up.shift(1).iloc[-1])  and bool(cross_down_high_thresh.iloc[-1])
        reversal_after_break_down = bool(cross_dn.shift(1).iloc[-1])  and bool(cross_up_low_thresh.iloc[-1])

        raw_short = near_high or reversal_after_break_up
        raw_long  = near_low  or reversal_after_break_down

        can_long  = raw_long  and self._cooldown_ok(coin, "long",  bi, p.alert_cutoff)
        can_short = raw_short and self._cooldown_ok(coin, "short", bi, p.alert_cutoff)

        details = {
            "range_high": float(rh.iloc[-1]),
            "range_low":  float(rl.iloc[-1]),
            "atr":        float(a.iloc[-1]),
            "close":      float(closes[-1]),
            "near_high":  bool(near_high),
            "near_low":   bool(near_low),
            "rev_up":     bool(reversal_after_break_up),
            "rev_down":   bool(reversal_after_break_down),
            "bi":         bi
        }
        return can_long, can_short, details

    def _cooldown_ok(self, coin, side, bar_index, cutoff):
        key = (coin, side)
        last = self.last_signal_bar.get(key)
        if last is None or (bar_index - last) >= cutoff:
            self.last_signal_bar[key] = bar_index
            return True
        return False

    # ================ Execução =================

    def _equity_usd(self) -> float:
        if self.info is None or self.dry_run or not self.account_address:
            return 0.0
        try:
            pf = self.info.portfolio(self.account_address)
            return float(pf.get("accountValue", 0.0))
        except Exception as e:
            logging.warning(f"[EQUITY] Falha ao obter portfolio: {e} (seguindo equity=0)")
            return 0.0

    def _order_size(self, coin: str, price: float) -> float:
        """
        Calcula size pelo ORDER_PCT do equity ou ORDER_USD fixo. Garante:
        - arredondamento para o step (lot)
        - mínimo de ordem = max(lot, 10^-szDecimals)
        """
        equity = self._equity_usd()
        if equity > 0:
            risk_usd = max(1.0, equity * (self.params.order_pct / 100.0))
        else:
            risk_usd = max(1.0, self.fixed_order_usd)

        raw_sz = risk_usd / max(1e-10, price)
        meta = self.meta_by_coin.get(coin, CoinMeta(lot=0.001, tick=0.01, sz_decimals=3))
        step = meta.lot if meta.lot > 0 else float(Decimal(10) ** Decimal(-meta.sz_decimals))
        min_sz = meta.min_size

        sz = floor_to_step(raw_sz, step)
        if sz < min_sz:
            sz = min_sz
        return float(sz)

    @staticmethod
    def _resp_ok(resp: dict) -> bool:
        try:
            if not resp or resp.get("status") != "ok":
                return False
            data = resp.get("response", {}).get("data", {})
            statuses = data.get("statuses") or []
            if not statuses:
                return False
            st = (statuses[0].get("status") or "").lower()
            if "rejected" in st:
                return False
            # Alguns retornos não trazem "status", só "error"
            if statuses[0].get("error"):
                return False
            return True
        except Exception:
            return False

    def _place_limit_ioc(self, coin: str, is_buy: bool, sz: float, px_desired: float, reduce_only: bool = False):
        """
        Garante preço múltiplo do tick:
        - BUY: ceil para tick (>= px_desired)
        - SELL: floor para tick (<= px_desired)
        """
        meta = self.meta_by_coin.get(coin, CoinMeta(lot=0.001, tick=0.01, sz_decimals=3))
        if is_buy:
            px = ceil_to_tick(px_desired, meta.tick)
        else:
            px = floor_to_tick(px_desired, meta.tick)

        order_type = {"limit": {"tif": "Ioc"}}

        if self.dry_run or self.exchange is None:
            logging.info(f"[DRY] IOC {'BUY' if is_buy else 'SELL'} {sz} {coin} @ {px:.10f}{' (reduceOnly)' if reduce_only else ''}")
            return {"dry": True, "status": "ok"}

        try:
            resp = self.exchange.order(
                coin,
                bool(is_buy),
                float(sz),
                float(px),
                order_type,
                reduce_only=bool(reduce_only),
            )
            logging.info(f"[API] IOC {'BUY' if is_buy else 'SELL'} {sz} {coin} @ {px:.10f} -> {resp}")
            return resp
        except Exception as e:
            logging.error(f"Falha ao enviar ordem: {e}")
            return {"error": str(e), "status": "error"}

    def _market_like_px(self, ref_px: float, is_buy: bool) -> float:
        slip = abs(self.ioc_slip)
        return ref_px * (1.0 + slip if is_buy else 1.0 - slip)

    def _open_or_add(self, coin: str, is_buy: bool, ref_open_px: float) -> Optional[float]:
        px_desired = self._market_like_px(ref_open_px, is_buy)
        sz = self._order_size(coin, ref_open_px)
        if sz <= 0:
            logging.warning(f"{coin} tamanho 0 — pulando ordem.")
            return None
        resp = self._place_limit_ioc(coin, is_buy, sz, px_desired, reduce_only=False)
        if self._resp_ok(resp):
            return float(ref_open_px)
        else:
            logging.error(f"{coin} ordem rejeitada ao abrir/adder: {resp}")
            return None

    def _close_position(self, coin: str, ps: PositionState, ref_close_px: float):
        qty = ps.qty()
        if qty <= 0 or ps.side == "flat":
            return False
        is_buy = (ps.side == "short")
        px_desired = self._market_like_px(ref_close_px, is_buy)
        resp = self._place_limit_ioc(coin, is_buy, qty, px_desired, reduce_only=True)
        if self._resp_ok(resp):
            self.state[coin] = PositionState()
            return True
        else:
            logging.error(f"{coin} falha ao fechar posição: {resp}")
            return False

    # ================ Barra fechada =================

    def on_bar_close(self, coin: str, df_closed: pd.DataFrame):
        p = self.params
        ps = self.state[coin]
        can_long, can_short, d = self.compute_signal(coin, df_closed)
        last_close = float(df_closed["close"].iloc[-1])

        if d:
            sig_txt = "LONG" if can_long else ("SHORT" if can_short else "-")
            logging.info(
                f"{coin} signal={sig_txt} | nearH={d.get('near_high')} revUp={d.get('rev_up')} "
                f"| nearL={d.get('near_low')} revDn={d.get('rev_down')} "
                f"| range=({d.get('range_low'):.4f},{d.get('range_high'):.4f}) | ATR={d.get('atr'):.6f} | close={last_close:.6f}"
            )
        else:
            logging.info(f"{coin} sinal: histórico insuficiente (len={len(df_closed)})")

        # Saídas
        if ps.side == "long":
            avg = ps.avg_entry() or ps.initial_px or last_close
            long_stop = avg * (1.0 - p.stop_pct / 100.0)
            long_tp   = avg + (avg - long_stop) * p.take_profit_rr
            if last_close <= long_stop or last_close >= long_tp:
                ok = self._close_position(coin, ps, last_close)
                if ok:
                    logging.info(f"{coin} LONG exit @ {last_close:.6f}")

        elif ps.side == "short":
            avg = ps.avg_entry() or ps.initial_px or last_close
            short_stop = avg * (1.0 + p.stop_pct / 100.0)
            short_tp   = avg - (short_stop - avg) * p.take_profit_rr
            if last_close >= short_stop or last_close <= short_tp:
                ok = self._close_position(coin, ps, last_close)
                if ok:
                    logging.info(f"{coin} SHORT exit @ {last_close:.6f}")

        # Agenda próxima barra
        if d:
            if ps.side == "flat":
                if can_long:
                    self.pending_entry[coin] = "long"
                    logging.info(f"{coin} [SIGNAL] LONG scheduled for next bar (bi={self.bar_index[coin]+1})")
                elif self.params.enable_shorts and can_short:
                    self.pending_entry[coin] = "short"
                    logging.info(f"{coin} [SIGNAL] SHORT scheduled for next bar (bi={self.bar_index[coin]+1})")
            else:
                if ps.side == "long" and can_long and ps.next_dca_idx <= p.max_pyramid:
                    self.pending_entry[coin] = "long"
                    logging.info(f"{coin} [SIGNAL] LONG add #{ps.next_dca_idx} scheduled for next bar (bi={self.bar_index[coin]+1})")
                elif ps.side == "short" and self.params.enable_shorts and can_short and ps.next_dca_idx <= p.max_pyramid:
                    self.pending_entry[coin] = "short"
                    logging.info(f"{coin} [SIGNAL] SHORT add #{ps.next_dca_idx} scheduled for next bar (bi={self.bar_index[coin]+1})")

    def _exec_pending_if_any(self, coin: str, newbar_open: float):
        side = self.pending_entry.get(coin)
        if not side:
            return
        ps = self.state[coin]
        p = self.params

        # Entrada inicial
        if ps.side == "flat":
            is_buy = (side == "long")
            done_px = self._open_or_add(coin, is_buy, newbar_open)
            if done_px is not None:
                base_sz = self._order_size(coin, newbar_open)
                ps.side = "long" if is_buy else "short"
                ps.initial_px = newbar_open
                ps.fills = [(newbar_open, base_sz)]
                ps.next_dca_idx = 1
                self.state[coin] = ps
                logging.info(f"{coin} [EXEC] entry {side.upper()} @ open={newbar_open:.6f} (bi={self.bar_index[coin]})")
            else:
                logging.error(f"{coin} [EXEC] entry {side.upper()} falhou — estado não alterado.")
            self.pending_entry[coin] = None
            return

        # Add
        if ps.side == side and ps.next_dca_idx <= p.max_pyramid:
            is_buy = (side == "long")
            done_px = self._open_or_add(coin, is_buy, newbar_open)
            if done_px is not None:
                add_sz = self._order_size(coin, newbar_open)
                ps.fills.append((newbar_open, add_sz))
                logging.info(f"{coin} [EXEC] add {side.upper()} #{ps.next_dca_idx} @ open={newbar_open:.6f} (bi={self.bar_index[coin]})")
                ps.next_dca_idx += 1
                self.state[coin] = ps
            else:
                logging.error(f"{coin} [EXEC] add {side.upper()} #{ps.next_dca_idx} falhou — estado inalterado.")
            self.pending_entry[coin] = None
            return

        # Sinal pendente incompatível
        logging.info(f"{coin} [EXEC] pending {side.upper()} ignorado (state={ps.side}, next_dca={ps.next_dca_idx}/{p.max_pyramid})")
        self.pending_entry[coin] = None

    def _sleep_until_next_close(self):
        interval_ms = INTERVAL_MS[self.tf]
        now_ms = int(time.time() * 1000)
        next_close_ms = ((now_ms // interval_ms) + 1) * interval_ms
        sleep_sec = max(0.5, (next_close_ms - now_ms) / 1000.0 + 0.25)
        time.sleep(sleep_sec)

    # ================ Loop =================

    def run(self):
        logging.info(f"Iniciando loop {self.tf}… Ctrl+C para parar.")
        while True:
            try:
                for coin in self.coins:
                    df = self.fetch_candles(coin, self.params.lookback + 600)
                    if df.empty:
                        logging.warning(f"{coin}: sem dados. Tentando novamente…")
                        continue

                    last_t = int(df["t"].iloc[-1])
                    last_open = float(df["open"].iloc[-1])
                    last_close_prevbar = float(df["close"].iloc[-2]) if len(df) >= 2 else float(df["close"].iloc[-1])
                    last_dt = datetime.fromtimestamp(last_t/1000, self.tz).strftime("%H:%M:%S")
                    age_min = max(0.0, (int(time.time()*1000) - last_t)/60000)

                    if self.last_bar_ts[coin] is None:
                        self.last_bar_ts[coin] = last_t
                        logging.info(f"{coin} init | close={last_close_prevbar:.6f} @ {last_dt} (age {age_min:.1f}m)")
                        continue

                    if last_t > self.last_bar_ts[coin]:
                        # 1) fechamento da barra anterior
                        df_closed = df.iloc[:-1].copy()
                        self.on_bar_close(coin, df_closed)

                        # 2) nova barra
                        self.bar_index[coin] += 1
                        logging.info(f"{coin} new {self.tf} bar | open={last_open:.6f} @ {last_dt} (age {age_min:.1f}m)")

                        # 3) executa pendências na abertura
                        self._exec_pending_if_any(coin, last_open)

                        self.last_bar_ts[coin] = last_t
                    else:
                        logging.info(f"{coin} waiting next {self.tf} bar… last_close={last_close_prevbar:.6f} @ {last_dt} (age {age_min:.1f}m)")

                self._sleep_until_next_close()

            except KeyboardInterrupt:
                logging.info("Encerrando por KeyboardInterrupt.")
                break
            except Exception as e:
                logging.error(f"Loop error: {e}")
                time.sleep(2)

    # ================ Backtest =================

    def backtest(self, coin: str, n_bars: int):
        df = self.fetch_candles(coin, n_bars)
        p = self.params
        ps = PositionState()
        pend: Optional[str] = None
        wins = losses = 0
        trades = 0

        min_i = max(p.lookback, p.atr_len) + p.sensitivity*2 + 2
        for i in range(min_i, len(df)-1):
            open_i = float(df["open"].iloc[i])
            df_closed = df.iloc[:i].copy()

            self.bar_index[coin] = i - 1
            can_long, can_short, d = self.compute_signal(coin, df_closed)
            if d:
                if ps.side == "flat":
                    if can_long:
                        pend = "long"
                    elif p.enable_shorts and can_short:
                        pend = "short"
                else:
                    if ps.side == "long" and can_long and ps.next_dca_idx <= p.max_pyramid:
                        pend = "long"
                    elif ps.side == "short" and p.enable_shorts and can_short and ps.next_dca_idx <= p.max_pyramid:
                        pend = "short"

            self.bar_index[coin] = i
            if pend:
                side = pend; pend = None
                is_buy = (side == "long")
                base_sz = 1.0
                if ps.side == "flat":
                    ps.side = "long" if is_buy else "short"
                    ps.initial_px = open_i
                    ps.fills = [(open_i, base_sz)]
                    ps.next_dca_idx = 1
                    trades += 1
                elif ps.side == side and ps.next_dca_idx <= p.max_pyramid:
                    ps.fills.append((open_i, base_sz))
                    ps.next_dca_idx += 1

            if ps.side != "flat":
                last_close = float(df_closed["close"].iloc[-1])
                avg = ps.avg_entry() or ps.initial_px or last_close
                if ps.side == "long":
                    long_stop = avg * (1.0 - p.stop_pct/100.0)
                    long_tp   = avg + (avg - long_stop) * p.take_profit_rr
                    if last_close <= long_stop or last_close >= long_tp:
                        if last_close >= long_tp:
                            wins += 1
                        else:
                            losses += 1
                        ps = PositionState()
                else:
                    short_stop = avg * (1.0 + p.stop_pct/100.0)
                    short_tp   = avg - (short_stop - avg) * p.take_profit_rr
                    if last_close >= short_stop or last_close <= short_tp:
                        if last_close <= short_tp:
                            wins += 1
                        else:
                            losses += 1
                        ps = PositionState()

        logging.info(f"[BT] {coin} bars={len(df)} trades={trades} W={wins} L={losses}")

# ================== CLI ==================

if __name__ == "__main__":
    bot = CubanRangeReversalBot()
    if len(sys.argv) >= 2 and sys.argv[1].lower() == "backtest":
        coin = sys.argv[2].upper() if len(sys.argv) >= 3 else (bot.coins[0] if bot.coins else "BTC")
        n = int(sys.argv[3]) if len(sys.argv) >= 4 else bot.params.lookback + 1500
        bot.backtest(coin, n)
    else:
        bot.run()



# === Monkey patches: robust equity + unified % sizing ===
def _patched_equity_usd(self) -> float:
    import os, logging
    try:
        pf = self.info.portfolio()
        val = None
        if isinstance(pf, dict):
            for k in ("accountValue","equity","total","balance","account_value"):
                if k in pf and pf[k] is not None:
                    try: val = float(pf[k]); break
                    except Exception: pass
            if val is None:
                for ck in ("data","result","portfolio","summary"):
                    node = pf.get(ck)
                    if isinstance(node, dict):
                        for k in ("accountValue","equity","total","balance"):
                            if k in node and node[k] is not None:
                                try: val = float(node[k]); break
                                except Exception: pass
                    if val is not None: break
        elif isinstance(pf, list):
            for item in pf:
                if not isinstance(item, dict): continue
                for k in ("accountValue","equity","total","balance"):
                    if k in item and item[k] is not None:
                        try: val = float(item[k]); break
                        except Exception: pass
                if val is not None: break
            if val is None:
                usdc_total = 0.0; any_val = False
                for item in pf:
                    if not isinstance(item, dict): continue
                    if "usdValue" in item and item["usdValue"] is not None:
                        try: usdc_total += float(item["usdValue"]); any_val = True
                        except Exception: pass
                    elif "available" in item and item["available"] is not None:
                        try: usdc_total += float(item["available"]); any_val = True
                        except Exception: pass
                if any_val and usdc_total>0: val = usdc_total
        if val is not None and val>0:
            self._last_equity = float(val)
            return float(val)
        raise ValueError("Portfolio sem campo de equity reconhecido")
    except Exception as e:
        last = float(getattr(self, "_last_equity", 0.0) or 0.0)
        try: hint = float(os.getenv("EQUITY_HINT","0") or 0.0)
        except Exception: hint = 0.0
        chosen = last if last>0 else (hint if hint>0 else 0.0)
        logging.warning(f"[EQUITY] Falha ao obter portfolio: {e} (usando fallback={chosen})")
        return chosen

def _patched_order_size(self, coin: str, price: float) -> float:
    import os
    equity = float(self._equity_usd() or 0.0)
    try: fixed = float(getattr(self, "fixed_order_usd", 0.0) or os.getenv("ORDER_USD","0") or 0.0)
    except Exception: fixed = 0.0
    try: order_pct = float(getattr(self, "order_pct", None) or os.getenv("ORDER_PCT","2.5"))
    except Exception: order_pct = 2.5
    risk_usd = fixed if fixed>0 else (order_pct/100.0)*equity
    if risk_usd<=0:
        try: floor_usd = float(os.getenv("MIN_USD","10") or 10.0)
        except Exception: floor_usd = 10.0
        risk_usd = max(1.0, floor_usd)
    meta = self.meta_by_coin.get(coin, CoinMeta())
    lot = meta.lot if meta.lot and meta.lot>0 else 0.001
    sz_dec = meta.sz_decimals if meta.sz_decimals is not None else 3
    size_raw = risk_usd / max(price, 1e-9)
    steps = max(1, int(size_raw/lot))
    size = round(steps*lot, sz_dec)
    if size < lot: size = lot
    return float(size)

# Bind patches
CubanRangeReversalBot._equity_usd = _patched_equity_usd
CubanRangeReversalBot._order_size = _patched_order_size
