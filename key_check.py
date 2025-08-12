# key_check.py — quick one-off tester
import requests
from config import HEADERS, TRADING_BASE
a=requests.get(f"{TRADING_BASE}/v2/account", headers=HEADERS)
c=requests.get(f"{TRADING_BASE}/v2/clock", headers=HEADERS)
print("Account:", a.status_code, a.text[:120])
print("Clock:  ", c.status_code, c.text[:120])
