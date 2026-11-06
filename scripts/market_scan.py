#!/usr/bin/env python3
"""Deep market scan. Reads real candles for each pair, finds trend, momentum and every key level on three timeframes per
trading style (scalp, intraday, swing), builds a bias and a trade plan, and writes scan.json for the website.
If ANTHROPIC_API_KEY is set, Claude also writes a written analysis from its own market knowledge plus the data."""
import json, os, sys, urllib.request, re
from datetime import datetime, timezone
import numpy as np
import pandas as pd

PAIRS = {"EURUSD": ["EURUSD=X"], "GBPUSD": ["GBPUSD=X"], "USDJPY": ["USDJPY=X"], "AUDUSD": ["AUDUSD=X"],
         "USDCAD": ["USDCAD=X"], "USDCHF": ["USDCHF=X"], "NZDUSD": ["NZDUSD=X"], "XAUUSD": ["XAUUSD=X", "GC=F"]}
STYLES = {
    "scalp":    {"tfs": ["5m", "15m", "1h"], "w": [.25, .40, .35], "h": "Minutes to a few hours", "stop": .6, "rr": [1.2, 2.0, 3.0]},
    "intraday": {"tfs": ["15m", "1h", "4h"], "w": [.25, .35, .40], "h": "Within the trading day", "stop": .6, "rr": [1.5, 2.5, 3.5]},
    "swing":    {"tfs": ["4h", "1d", "1w"],  "w": [.20, .35, .45], "h": "Several days to weeks", "stop": .7, "rr": [1.5, 2.5, 4.0]},
}
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
KEY = os.environ.get("ANTHROPIC_API_KEY", "")
NARR_MAX_AGE_H = 3

# ---------- data ----------
def dl(sym, interval, period):
    import yfinance as yf
    df = yf.download(sym, interval=interval, period=period, progress=False, auto_adjust=False, threads=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df

def frames(sym):
    f = {"5m": dl(sym, "5m", "5d"), "15m": dl(sym, "15m", "10d"), "1h": dl(sym, "60m", "60d"),
         "1d": dl(sym, "1d", "2y"), "1w": dl(sym, "1wk", "5y")}
    if any(v is None or len(v) < 30 for v in f.values()):
        return None
    f["4h"] = f["1h"].resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    return f

# ---------- indicators ----------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))
def atr(df, n=14):
    pc = df.Close.shift(); tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()
def macd_hist(c):
    m = ema(c, 12) - ema(c, 26); return m - ema(m, 9)
def swings(df, k=3):
    h, l = df.High.values, df.Low.values; hi, lo = [], []
    for i in range(k, len(df) - k):
        if h[i] == h[i-k:i+k+1].max(): hi.append((df.index[i], float(h[i])))
        if l[i] == l[i-k:i+k+1].min(): lo.append((df.index[i], float(l[i])))
    return hi, lo
def trend(hi, lo):
    if len(hi) < 2 or len(lo) < 2: return "range"
    if hi[-1][1] > hi[-2][1] and lo[-1][1] > lo[-2][1]: return "up"
    if hi[-1][1] < hi[-2][1] and lo[-1][1] < lo[-2][1]: return "down"
    return "range"

def tf_info(df):
    c = df.Close; e50, e200 = ema(c, 50), ema(c, 200); hi, lo = swings(df, 3)
    r = float(rsi(c).iloc[-1]); a = float(atr(df).iloc[-1]); mh = float(macd_hist(c).iloc[-1]); px = float(c.iloc[-1])
    tr = trend(hi, lo); es = ((px > e50.iloc[-1]) + (e50.iloc[-1] > e200.iloc[-1])) - 1
    ema_txt = {1: "above EMA 50 and 200", 0: "mixed around the EMAs", -1: "below EMA 50 and 200"}[int(es)]
    score = 0.45 * {"up": 1, "down": -1, "range": 0}[tr] + 0.25 * es + 0.15 * float(np.clip((r - 50) / 25, -1, 1)) + 0.15 * (1 if mh > 0 else -1)
    return {"trend": tr, "rsi": round(r, 1), "ema": ema_txt, "macd": "positive" if mh > 0 else "negative", "atr": a, "score": score,
            "e50": float(e50.iloc[-1]), "e200": float(e200.iloc[-1]), "hi": hi, "lo": lo}

# ---------- levels ----------
def build_levels(f, cfg, P, A, info):
    tfs = cfg["tfs"]; pr, hi_df = f[tfs[1]], f[tfs[2]]; L = []
    pts = []
    for df in (pr, hi_df):
        h, l = swings(df, 3); pts += [x[1] for x in h[-12:]] + [x[1] for x in l[-12:]]
    pts.sort(); cl = []
    for p in pts:
        if cl and p - cl[-1][-1] <= 0.35 * A: cl[-1].append(p)
        else: cl.append([p])
    for c in cl:
        if len(c) >= 2:
            p = sum(c) / len(c); L.append({"p": p, "label": "Support/resistance zone (%d touches)" % len(c), "kind": "resistance" if p > P else "support"})
    d = f["1d"]
    if len(d) >= 3:
        b = d.iloc[-2]; H, Lo, C = float(b.High), float(b.Low), float(b.Close); pv = (H + Lo + C) / 3
        for n, v in (("Pivot", pv), ("R1", 2*pv-Lo), ("R2", pv+H-Lo), ("R3", H+2*(pv-Lo)), ("S1", 2*pv-H), ("S2", pv-(H-Lo)), ("S3", Lo-2*(H-pv))):
            L.append({"p": v, "label": "Daily pivot " + n if n == "Pivot" else "Daily " + n, "kind": "pivot"})
        L += [{"p": H, "label": "Previous day high", "kind": "prev"}, {"p": Lo, "label": "Previous day low", "kind": "prev"}, {"p": float(d.iloc[-1].Open), "label": "Daily open", "kind": "prev"}]
    w = f["1w"]
    if len(w) >= 3: L += [{"p": float(w.iloc[-2].High), "label": "Previous week high", "kind": "prev"}, {"p": float(w.iloc[-2].Low), "label": "Previous week low", "kind": "prev"}]
    if tfs[0] != "4h":
        h1 = f["1h"]; day = h1[h1.index.date == h1.index[-1].date()]
        for nm, a0, b0 in (("Asian", 0, 7), ("London", 7, 12), ("New York", 12, 17)):
            s = day[(day.index.hour >= a0) & (day.index.hour < b0)]
            if len(s): L += [{"p": float(s.High.max()), "label": nm + " session high", "kind": "session"}, {"p": float(s.Low.min()), "label": nm + " session low", "kind": "session"}]
    # Fibonacci on the higher timeframe leg
    h, l = info[tfs[2]]["hi"], info[tfs[2]]["lo"]; rng = None
    if h and l:
        (ht, hp), (lt, lp) = h[-1], l[-1]; up = ht > lt; rng = (hp, lp, up)
        span = hp - lp
        for r in (.382, .5, .618, .705, .786):
            L.append({"p": hp - r * span if up else lp + r * span, "label": "Fib %.1f%% retracement" % (r * 100), "kind": "fib"})
        for r in (1.272, 1.618):
            L.append({"p": lp + r * span if up else hp - r * span, "label": "Fib %.1f%% extension" % (r * 100), "kind": "fib"})
    # FVG and order blocks on the primary timeframe
    x = pr.tail(150).reset_index(drop=True); a = atr(pr).tail(150).reset_index(drop=True)
    for i in range(2, len(x)):
        if x.Low[i] > x.High[i-2] and not (x.Low[i+1:] <= x.High[i-2]).any():
            L.append({"p": (x.Low[i] + x.High[i-2]) / 2, "lo": float(x.High[i-2]), "hi": float(x.Low[i]), "label": "Bullish fair value gap (%s)" % tfs[1], "kind": "fvg-bull"})
        if x.High[i] < x.Low[i-2] and not (x.High[i+1:] >= x.Low[i-2]).any():
            L.append({"p": (x.High[i] + x.Low[i-2]) / 2, "lo": float(x.High[i]), "hi": float(x.Low[i-2]), "label": "Bearish fair value gap (%s)" % tfs[1], "kind": "fvg-bear"})
        body = abs(x.Close[i] - x.Open[i])
        if body > 1.5 * a[i] and i >= 1:
            j = i - 1
            if x.Close[i] > x.Open[i] and x.Close[j] < x.Open[j] and not (x.Close[i+1:] < x.Low[j]).any():
                L.append({"p": (x.High[j] + x.Low[j]) / 2, "lo": float(x.Low[j]), "hi": float(x.High[j]), "label": "Bullish order block (%s)" % tfs[1], "kind": "ob-bull"})
            if x.Close[i] < x.Open[i] and x.Close[j] > x.Open[j] and not (x.Close[i+1:] > x.High[j]).any():
                L.append({"p": (x.High[j] + x.Low[j]) / 2, "lo": float(x.Low[j]), "hi": float(x.High[j]), "label": "Bearish order block (%s)" % tfs[1], "kind": "ob-bear"})
    # liquidity pools: equal highs and lows
    for src, kind, nm in ((info[tfs[1]]["hi"][-20:], "liquidity", "Equal highs (liquidity above)"), (info[tfs[1]]["lo"][-20:], "liquidity", "Equal lows (liquidity below)")):
        v = sorted(p for _, p in src)
        for i in range(len(v) - 1):
            if abs(v[i+1] - v[i]) <= 0.15 * A: L.append({"p": max(v[i], v[i+1]) if "highs" in nm else min(v[i], v[i+1]), "label": nm, "kind": kind})
    L += [{"p": info[tfs[1]]["e50"], "label": "EMA 50 (%s)" % tfs[1], "kind": "ema"}, {"p": info[tfs[1]]["e200"], "label": "EMA 200 (%s)" % tfs[1], "kind": "ema"}]
    L.sort(key=lambda z: abs(z["p"] - P)); L = L[:44]; L.sort(key=lambda z: -z["p"])
    return L, rng

# ---------- plan ----------
def make_plan(side, P, A, L, cfg):
    buy = side == "buy"; pref = ("fvg-bull", "ob-bull", "support") if buy else ("fvg-bear", "ob-bear", "resistance")
    pool = [z for z in L if (z["p"] < P if buy else z["p"] > P) and abs(z["p"] - P) <= 2 * A]
    pick = [z for z in pool if z["kind"] in pref] or pool
    if pick:
        z = min(pick, key=lambda q: abs(q["p"] - P)); lo = z.get("lo", z["p"] - .1 * A); hi = z.get("hi", z["p"] + .1 * A); note = z["label"]
    else:
        lo, hi, note = (P - .5 * A, P - .1 * A, "Pullback of about half an ATR") if buy else (P + .1 * A, P + .5 * A, "Pullback of about half an ATR")
    em = (lo + hi) / 2
    stop = lo - cfg["stop"] * A * .5 if buy else hi + cfg["stop"] * A * .5
    if abs(em - stop) < .5 * A: stop = em - .5 * A if buy else em + .5 * A
    risk = abs(em - stop)
    tg = sorted([z["p"] for z in L if (z["p"] > hi if buy else z["p"] < lo) and risk <= abs(z["p"] - em) <= min(6 * A, 5 * risk)], key=lambda p: abs(p - em))
    t = []
    for p in tg:
        if all(abs(p - q) >= .5 * risk for q in t): t.append(p)
        if len(t) == 3: break
    for m in cfg["rr"]:
        if len(t) == 3: break
        p = em + m * risk if buy else em - m * risk
        if all(abs(p - q) >= .5 * risk for q in t): t.append(p)
    t = sorted(t, key=lambda p: abs(p - em))[:3]
    return {"side": side, "entry": [lo, hi], "stop": stop, "targets": t, "rr": [abs(p - em) / risk for p in t], "note": note}

def analyze(f, style):
    cfg = STYLES[style]; tfs = cfg["tfs"]; P = float(f["5m"].Close.iloc[-1])
    info = {tf: tf_info(f[tf]) for tf in tfs}; A = info[tfs[1]]["atr"]
    score = sum(w * info[tf]["score"] for w, tf in zip(cfg["w"], tfs)); trs = [info[tf]["trend"] for tf in tfs]
    bias = "Bullish" if score > .2 else "Bearish" if score < -.2 else "Neutral"
    conf = int(min(95, abs(score) * 120 + (10 if len(set(trs)) == 1 and trs[0] != "range" else 0)))
    L, rng = build_levels(f, cfg, P, A, info)
    primary = "buy" if bias == "Bullish" else "sell" if bias == "Bearish" else None
    pb, ps = make_plan("buy", P, A, L, cfg), make_plan("sell", P, A, L, cfg)
    if primary is None: primary = "buy" if pb["rr"][0] >= ps["rr"][0] else "sell"
    plans = {"primary": pb if primary == "buy" else ps, "alt": ps if primary == "buy" else pb}
    out = {"bias": bias, "confidence": conf, "score": round(score, 3), "horizon": cfg["h"], "atr": A, "tfs": [], "levels": L, "plan": plans, "neutral": bias == "Neutral"}
    for tf in tfs: out["tfs"].append({"tf": tf, "trend": info[tf]["trend"], "rsi": info[tf]["rsi"], "ema": info[tf]["ema"], "macd": info[tf]["macd"]})
    if rng:
        hp, lp, up = rng; eq = (hp + lp) / 2; out["range"] = {"hi": hp, "lo": lp, "eq": eq, "zone": "premium" if P > eq else "discount", "pos": int(max(0, min(100, round((P - lp) / (hp - lp) * 100)))) if hp != lp else 50, "outside": bool(P > hp or P < lp)}
    return P, out

def rnd(o, d):
    if isinstance(o, float): return round(o, d)
    if isinstance(o, list): return [rnd(x, d) for x in o]
    if isinstance(o, dict): return {k: (v if k in ("confidence", "score", "pos") else rnd(v, d)) for k, v in o.items()}
    return o

# ---------- optional Claude narrative ----------
SYSTEM = ("You are a senior FX and gold market analyst. Using your own market knowledge (macro drivers, central bank policy context, correlations "
          "such as the dollar index, yields and risk sentiment, seasonality, typical behaviour around sessions and events) together with the technical "
          "data provided, write a concise deep analysis for the chosen pair and trading style. Do not invent live news or prices; treat the data as given, "
          "and say plainly when you are unsure about current conditions. No guarantees and no financial advice. Reply with JSON only, with keys: "
          "headline (string, max 120 chars), outlook (2 to 3 sentences), drivers (array of 3 to 5 short strings), bull_case (string), bear_case (string), "
          "watch (array of 3 to 4 short strings), risk_note (string).")
def narrative(payload):
    body = json.dumps({"model": MODEL, "max_tokens": 1000, "system": SYSTEM, "messages": [{"role": "user", "content": json.dumps(payload)}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r: txt = json.load(r)["content"][0]["text"]
    txt = re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip()
    return json.loads(txt)

def main():
    old = {}
    if os.path.exists("scan.json"):
        try: old = json.load(open("scan.json"))
        except Exception: pass
    news = {}
    if os.path.exists("news.json"):
        try: news = json.load(open("news.json"))
        except Exception: pass
    heads = [h.get("title", "") for h in (news.get("headlines") or [])[:8]]
    now = datetime.now(timezone.utc); res = {"updated": now.isoformat(), "pairs": {}}
    for pair, syms in PAIRS.items():
        f = None
        for s in syms:
            try: f = frames(s)
            except Exception as e: print(pair, s, "download failed:", e)
            if f: break
        if not f:
            if pair in old.get("pairs", {}): res["pairs"][pair] = old["pairs"][pair]
            print("skipped", pair); continue
        styles = {}
        for st in STYLES:
            P, o = analyze(f, st); d = 2 if P > 500 else 3 if P > 20 else 5; o = rnd(o, d)
            prev = ((old.get("pairs", {}).get(pair, {}).get("styles", {}) or {}).get(st, {}) or {}).get("narrative")
            fresh = prev and prev.get("_at") and (now - datetime.fromisoformat(prev["_at"])).total_seconds() < NARR_MAX_AGE_H * 3600 and prev.get("_bias") == o["bias"]
            if fresh: o["narrative"] = prev
            elif KEY:
                try:
                    n = narrative({"pair": pair, "style": st, "price": round(P, d), "bias": o["bias"], "confidence": o["confidence"], "timeframes": o["tfs"],
                                   "nearest_levels": sorted(o["levels"], key=lambda z: abs(z["p"] - P))[:10], "plan": o["plan"]["primary"], "headlines": heads,
                                   "fundamentals": (news.get("fundamentals") or "")[:600]})
                    n["_at"] = now.isoformat(); n["_bias"] = o["bias"]; o["narrative"] = n
                except Exception as e: print(pair, st, "narrative failed:", e)
            styles[st] = o
        res["pairs"][pair] = {"price": round(P, d), "decimals": d, "styles": styles}
        print("done", pair)
    if not res["pairs"]: sys.exit("No pairs could be scanned")
    json.dump(res, open("scan.json", "w"), separators=(",", ":"))
    print("Wrote scan.json for", list(res["pairs"]))

if __name__ == "__main__":
    main()
