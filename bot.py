# bot.py — Paper-only, ready-to-run
# - Healthcheck keeps process alive until keys are accepted (prints 200/200 when OK)
# - Trades regular + pre/after-hours (limit in extended), bracket exits
# - Breakout + VWAP + RSI, ATR risk sizing

import sys, subprocess
try:
    import requests
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
    import requests

import time, json, math, csv
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
from typing import Dict, Any, List, Optional

from config import HEADERS, TRADING_BASE, DATA_BASE

# ===== Tunables =====
TIMEFRAME = "1Min"
SCAN_INTERVAL_SEC = 30
MAX_SYMBOLS_PER_SCAN = 60
MAX_CONCURRENT_POS = 6
RISK_PCT_PER_TRADE = 0.01
TAKE_PROFIT_ATR_MULT = 2.2
STOP_ATR_MULT = 1.1
MIN_PRICE = 5.0
MAX_PRICE = 800.0
MIN_AVG_VOL = 300_000
DAILY_MAX_LOSS_PCT = 0.03

# Extended hours
EXTENDED_TRADING = True
PRE_MARKET_START  = dtime(4, 0)
REGULAR_OPEN      = dtime(9, 30)
REGULAR_CLOSE     = dtime(16, 0)
AFTER_MARKET_END  = dtime(20, 0)

UNIVERSE = [
    "AAPL","MSFT","NVDA","TSLA","AMD","META","AMZN","GOOGL","GOOG","NFLX","AVGO","SMCI","ASML",
    "SHOP","UBER","CRM","ADBE","MU","INTC","COIN","PLTR","SQ","ABNB","DELL","ON","KLAC","LRCX",
    "PANW","NOW","ANET","TTD","SNOW","MDB","BABA","PDD","NIO","LI","RIVN","CVNA","DDOG","CRWD",
    "NET","ZS","OKTA","ARM","SOFI","UAL","JPM","BAC","CAT","GE","DE","MARA","RIOT","AFRM","WMT",
    "TGT","HD","LOW","DAL","AAL","NKE","COST","PEP","KO","DIS","SBUX","BA","QCOM","TXN","MRVL",
    "INTU","PYPL","NEE","ENPH","RUN","CCL","NCLH"
]

# ===== HTTP helpers =====
def _req(method: str, url: str, **kw) -> Optional[requests.Response]:
    tries = 0
    while tries < 3:
        try:
            r = requests.request(method, url, headers=HEADERS, timeout=20, **kw)
            if r.status_code == 429:
                wait = 2 ** tries
                print(f"⏳ Rate limited (429). Backing off {wait}s…"); time.sleep(wait); tries += 1; continue
            r.raise_for_status(); return r
        except requests.HTTPError as e:
            body = getattr(e.response, "text", "")
            print(f"HTTP {getattr(e.response,'status_code','')} {method} {url} -> {e} | {body[:140]}")
            return None
        except Exception as e:
            print(f"{method} {url} error: {e}"); return None
    return None

def get_json(url: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    r = _req("GET", url, params=params or {}); return r.json() if r is not None else None

def post_json(url: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    r = _req("POST", url, data=json.dumps(payload)); return r.json() if r is not None else None

# ===== Alpaca API =====
def get_clock() -> Optional[Dict[str, Any]]: return get_json(f"{TRADING_BASE}/v2/clock")
def get_account() -> Optional[Dict[str, Any]]: return get_json(f"{TRADING_BASE}/v2/account")
def list_positions() -> List[Dict[str, Any]]:
    res = get_json(f"{TRADING_BASE}/v2/positions"); return res if isinstance(res, list) else []
def get_bars(symbol: str, timeframe: str, limit: int = 180) -> Optional[List[Dict[str, Any]]]:
    res = get_json(f"{DATA_BASE}/v2/stocks/{symbol}/bars", {"timeframe": timeframe, "limit": limit})
    return res.get("bars") if res and "bars" in res else None
def get_snapshot(symbol: str) -> Optional[Dict[str, Any]]: return get_json(f"{DATA_BASE}/v2/stocks/{symbol}/snapshot")

def is_open_regular() -> bool:
    c = get_clock(); return bool(c and c.get("is_open", False))

def is_extended_session_now() -> bool:
    now = datetime.now(ZoneInfo("America/New_York")).time()
    return (PRE_MARKET_START <= now < REGULAR_OPEN) or (REGULAR_CLOSE <= now < AFTER_MARKET_END)

def place_bracket_order(symbol: str, qty: int, stop_price: float, take_profit_price: float, tif: str = "day"):
    extended = EXTENDED_TRADING and is_extended_session_now()
    snap = get_snapshot(symbol) or {}
    last = snap.get("latestTrade", {}).get("p") or snap.get("minuteBar", {}).get("c") or 0.0
    order_type = "limit" if extended else "market"
    payload = {
        "symbol": symbol.upper(),
        "qty": qty,
        "side": "buy",
        "type": order_type,
        "time_in_force": tif,
        "order_class": "bracket",
        "take_profit": {"limit_price": round(take_profit_price, 2)},
        "stop_loss": {"stop_price": round(stop_price, 2)},
        "extended_hours": extended,
    }
    if order_type == "limit":
        payload["limit_price"] = round(last * 1.003, 2) if last else round((stop_price + take_profit_price)/2, 2)
    return post_json(f"{TRADING_BASE}/v2/orders", payload)

# ===== Indicators =====
def highest(values: List[float], n: int) -> float:
    return max(values[-n:]) if len(values) >= n else float("nan")
def atr(highs: List[float], lows: List[float], closes: List[float], n: int = 14) -> float:
    if len(closes) < n+1: return float("nan")
    trs = []
    for i in range(1, len(closes)):
        h,l,pc = highs[i], lows[i], closes[i-1]
        tr = max(h-l, abs(h-pc), abs(l-pc)); trs.append(tr)
    return sum(trs[-n:])/n if len(trs)>=n else float("nan")
def rsi(closes: List[float], n: int = 14) -> float:
    if len(closes) < n+1: return float("nan")
    gains = []; losses = []
    for i in range(1, len(closes)):
        ch = closes[i]-closes[i-1]
        gains.append(max(0, ch)); losses.append(max(0, -ch))
    avg_gain = sum(gains[-n:])/n; avg_loss = sum(losses[-n:])/n
    if avg_loss == 0: return 100.0
    rs = avg_gain/avg_loss; return 100 - (100/(1+rs))

# ===== Strategy =====
def analyze_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    bars = get_bars(symbol, TIMEFRAME, limit=180)
    if not bars or len(bars) < 60: return None
    closes = [b["c"] for b in bars]; highs = [b["h"] for b in bars]; lows = [b["l"] for b in bars]; vols = [b["v"] for b in bars]
    last = closes[-1]
    if not (MIN_PRICE <= last <= MAX_PRICE): return None
    avg_vol = sum(vols[-30:]) / 30
    if avg_vol < MIN_AVG_VOL: return None

    hi20 = highest(highs[:-1], 20)
    breakout = last > hi20 and vols[-1] > 1.5 * (sum(vols[-21:-1])/20 if len(vols)>21 else avg_vol)

    typical = [(b["h"]+b["l"]+b["c"])/3 for b in bars]
    vwap = (sum([typical[i]*vols[i] for i in range(-30,0)]) / sum(vols[-30:])) if sum(vols[-30:])>0 else last
    r = rsi(closes, 14)
    vwap_ok = last > vwap
    rsi_ok = r < 75

    a = atr(highs, lows, closes, 14)
    if a != a: return None

    if not (breakout and vwap_ok and rsi_ok):
        return None

    stop = last - STOP_ATR_MULT * a
    take = last + TAKE_PROFIT_ATR_MULT * a

    snap = get_snapshot(symbol) or {}
    if snap.get("trading_status") in {"Halted","T1"}: return None

    strength = (last - hi20) / max(0.01, a)
    score = strength + (last - vwap)/max(0.01, a)
    return {"symbol": symbol, "entry": float(last), "stop": float(stop), "take": float(take),
            "atr": float(a), "avg_vol": float(avg_vol), "strength": float(score)}

# ===== Risk / Guardrails =====
def position_size(buying_power: float, entry: float, stop: float, risk_pct: float) -> int:
    risk_dollars = max(0.0, buying_power) * risk_pct
    per_share_risk = max(0.01, entry - stop)
    shares = math.floor(risk_dollars / per_share_risk)
    if shares * entry > buying_power:
        shares = math.floor(buying_power / entry)
    return max(0, shares)

def daily_loss_exceeded(acct: Dict[str, Any]) -> bool:
    try:
        eq = float(acct.get("equity", 0)); last_eq = float(acct.get("last_equity", eq))
        dd = (eq - last_eq) / last_eq if last_eq else 0.0
        if dd < -DAILY_MAX_LOSS_PCT:
            print(f"🛑 Daily drawdown {dd:.2%} exceeds {DAILY_MAX_LOSS_PCT:.0%}. Pausing new entries today.")
            return True
    except Exception:
        pass
    return False

def log_trade(row: List[Any]):
    path = "trades.csv"
    new_file = False
    try:
        with open(path, "r"): pass
    except FileNotFoundError:
        new_file = True
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["ts","symbol","entry","stop","take","qty","session","order_response"])
        w.writerow(row)

# ===== Diagnostics =====
def keys_healthcheck(loop_sleep=30):
    """Stay alive until keys are accepted; prints clear status."""
    while True:
        try:
            a = requests.get(f"{TRADING_BASE}/v2/account", headers=HEADERS, timeout=15)
            c = requests.get(f"{TRADING_BASE}/v2/clock",   headers=HEADERS, timeout=15)
            print(f"🔑 Account={a.status_code} Clock={c.status_code}")
            if a.status_code == 200 and c.status_code == 200:
                print("✅ Keys OK, connected to Alpaca Paper Trading API."); return
            elif a.status_code in (401,403) or c.status_code in (401,403):
                print("❌ Auth rejected (401/403). Ensure TRADING API → PAPER keys. Retrying…")
            else:
                print("⚠️ Unexpected response. Retrying…")
        except Exception as e:
            print("Error during healthcheck:", e)
        time.sleep(loop_sleep)

# ===== Core loop =====
def scan_and_trade():
    acct = get_account()
    if not acct: print("❌ Cannot fetch account."); return
    if acct.get("trading_blocked"): print("🚫 Trading blocked."); return
    if daily_loss_exceeded(acct): return

    buying_power = float(acct.get("buying_power", 0.0))
    positions = list_positions()
    print(f"✅ BP ${buying_power:,.2f} | Pos {len(positions)}/{MAX_CONCURRENT_POS}")

    if len(positions) >= MAX_CONCURRENT_POS:
        print("ℹ️ Position cap reached."); return

    symbols = UNIVERSE[:MAX_SYMBOLS_PER_SCAN]
    candidates = []
    for s in symbols:
        idea = analyze_symbol(s)
        if idea: candidates.append(idea)

    if not candidates:
        print("🔎 No setups."); return
    candidates.sort(key=lambda d: d["strength"], reverse=True)

    for idea in candidates:
        qty = position_size(buying_power, idea["entry"], idea["stop"], RISK_PCT_PER_TRADE)
        if qty <= 0:
            print(f"⚠️ {idea['symbol']}: size=0."); continue
        session = "extended" if is_extended_session_now() else ("regular" if is_open_regular() else "unknown")
        print(f"🚀 {idea['symbol']} {session} | e≈{idea['entry']:.2f} s={idea['stop']:.2f} t={idea['take']:.2f} q={qty}")
        resp = place_bracket_order(idea["symbol"], qty, idea["stop"], idea["take"])
        log_trade([datetime.now().isoformat(), idea["symbol"], idea["entry"], idea["stop"], idea["take"], qty, session, json.dumps(resp) if resp else "err"])
        break

def main():
    print("🤖 Aurum X — PAPER ONLY. Regular + extended hours.")
    keys_healthcheck(loop_sleep=30)
    while True:
        try:
            if is_open_regular() or (EXTENDED_TRADING and is_extended_session_now()):
                scan_and_trade()
            else:
                now_et = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S ET")
                print(f"⏸ Waiting for regular or extended session… ({now_et})")
            time.sleep(SCAN_INTERVAL_SEC)
        except KeyboardInterrupt:
            print("👋 Stopped."); break
        except Exception as e:
            print("🔥 Loop error:", e); time.sleep(5)

if __name__ == "__main__":
    main()
