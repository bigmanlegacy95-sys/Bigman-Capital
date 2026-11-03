"""Fetches your Myfxbook account results and writes results.json (no broker name, no personal details)."""
import json, os, sys, urllib.parse, urllib.request
from datetime import datetime, timezone

EMAIL = os.environ["MYFXBOOK_EMAIL"]
PASSWORD = os.environ["MYFXBOOK_PASSWORD"]
ACCOUNT_ID = str(os.environ.get("MYFXBOOK_ACCOUNT_ID", "12292187"))
BASE = "https://www.myfxbook.com/api/"

def call(name, **params):
    url = BASE + name + ".json?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)

login = call("login", email=EMAIL, password=PASSWORD)
if login.get("error"):
    sys.exit("Myfxbook login failed: " + str(login.get("message")))
session = login["session"]
try:
    data = call("get-my-accounts", session=session)
    if data.get("error"):
        sys.exit("Myfxbook error: " + str(data.get("message")))
    accounts = data.get("accounts", [])
    acc = next((a for a in accounts if str(a.get("id")) == ACCOUNT_ID), None)
    if acc is None:
        sys.exit("Account %s not found. Ids seen: %s" % (ACCOUNT_ID, [a.get("id") for a in accounts]))
    keep = ["gain", "absGain", "daily", "monthly", "drawdown", "balance", "equity", "profit", "interest", "deposits", "withdrawals"]
    out = {k: acc[k] for k in keep if k in acc}
    out["lastUpdateDate"] = acc.get("lastUpdateDate")
    out["updated"] = datetime.now(timezone.utc).isoformat()
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Wrote results.json with keys:", list(out))
finally:
    try:
        call("logout", session=session)
    except Exception:
        pass
