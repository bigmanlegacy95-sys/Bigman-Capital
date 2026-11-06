#!/usr/bin/env python3
"""Deep market scan v2. For each pair and trading style it reads real candles on three timeframes and builds:
trend, structure (BOS / CHoCH), liquidity sweeps, divergence, momentum, trend strength, volatility, killzone context,
every key level, high-confluence zones, a graded trade plan (A+ to No trade) with trigger, stop, targets and management,
and a quick historical edge check of the bias rule. Writes scan.json for the website.
If ANTHROPIC_API_KEY is set, Claude stress-tests each setup and writes a view from its own market knowledge."""
import json, os, sys, urllib.request, re
from datetime import datetime, timezone
import numpy as np
import pandas as pd

PAIRS = {"EURUSD": ["EURUSD=X"], "GBPUSD": ["GBPUSD=X"], "USDJPY": ["USDJPY=X"], "AUDUSD": ["AUDUSD=X"],
         "USDCAD": ["USDCAD=X"], "USDCHF": ["USDCHF=X"], "NZDUSD": ["NZDUSD=X"], "XAUUSD": ["XAUUSD=X", "GC=F"]}
STYLES = {
    "scalp":    {"tfs": ["5m", "15m", "1h"], "h": "Minutes to a few hours", "stop": .6, "rr": [1.2, 2.0, 3.0], "valid": "about 5 hours", "kz": True},
    "intraday": {"tfs": ["15m", "1h", "4h"], "h": "Within the trading day", "stop": .6, "rr": [1.5, 2.5, 3.5], "valid": "the rest of the trading day", "kz": True},
    "swing":    {"tfs": ["4h", "1d", "1w"],  "h": "Several days to weeks", "stop": .7, "rr": [1.5, 2.5, 4.0], "valid": "about one to two weeks", "kz": False},
}
MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5")
KEY = os.environ.get("ANTHROPIC_API_KEY", "")
NARR_MAX_AGE_H = 3

# ---------------- data ----------------
def dl(sym, interval, period):
    import yfinance as yf
    df = yf.download(sym, interval=interval, period=period, progress=False, auto_adjust=False, threads=False)
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close"]].dropna()
    df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
    return df

def frames(sym):
    f = {"5m": dl(sym, "5m", "5d"), "15m": dl(sym, "15m", "10d"), "1h": dl(sym, "60m", "60d"), "1d": dl(sym, "1d", "2y"), "1w": dl(sym, "1wk", "5y")}
    if any(v is None or len(v) < 30 for v in f.values()): return None
    f["4h"] = f["1h"].resample("4h").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"}).dropna()
    return f

# ---------------- indicators ----------------
def ema(s, n): return s.ewm(span=n, adjust=False).mean()
def rsi(c, n=14):
    d = c.diff(); up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean(); dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))
def true_range(df):
    pc = df.Close.shift(); return pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
def atr(df, n=14): return true_range(df).ewm(alpha=1/n, adjust=False).mean()
def macd_hist(c):
    m = ema(c, 12) - ema(c, 26); return m - ema(m, 9)
def adx(df, n=14):
    up, dn = df.High.diff(), -df.Low.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index); minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = atr(df, n).replace(0, np.nan); pdi = 100 * plus.ewm(alpha=1/n, adjust=False).mean() / a; mdi = 100 * minus.ewm(alpha=1/n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan); return dx.ewm(alpha=1/n, adjust=False).mean()
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
    c = df.Close; e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200); hi, lo = swings(df, 3)
    r = float(rsi(c).iloc[-1]); at = atr(df); a = float(at.iloc[-1]); mh = float(macd_hist(c).iloc[-1]); px = float(c.iloc[-1]); ax = float(adx(df).iloc[-1])
    tr = trend(hi, lo); es = int(((px > e50.iloc[-1]) + (e50.iloc[-1] > e200.iloc[-1])) - 1)
    pct = float((at.tail(120) < a).mean() * 100)
    return {"trend": tr, "rsi": round(r, 1), "adx": round(ax, 1), "es": es, "ema": {1: "above EMA 50 and 200", 0: "mixed around the EMAs", -1: "below EMA 50 and 200"}[es],
            "macd": "positive" if mh > 0 else "negative", "mh": mh, "atr": a, "vol_pct": pct, "e50": float(e50.iloc[-1]), "e200": float(e200.iloc[-1]),
            "slope": float((e50.iloc[-1] - e50.iloc[-6]) / a) if a else 0.0, "hi": hi, "lo": lo}

def structure(df, i):
    hi, lo, tr = i["hi"], i["lo"], i["trend"]; c = float(df.Close.iloc[-1]); ev = None
    if hi and lo:
        if tr == "up" and c > hi[-1][1]: ev = ("bull", "Bullish break of structure: price closed above the last swing high")
        if tr == "down" and c < lo[-1][1]: ev = ("bear", "Bearish break of structure: price closed below the last swing low")
        if len(hi) >= 2 and len(lo) >= 2:
            if hi[-1][1] < hi[-2][1] and lo[-1][1] < lo[-2][1] and c > hi[-1][1]: ev = ("bull", "Bullish change of character: price closed above the last lower high")
            if hi[-1][1] > hi[-2][1] and lo[-1][1] > lo[-2][1] and c < lo[-1][1]: ev = ("bear", "Bearish change of character: price closed below the last higher low")
    return ev

def sweep(df, i):
    out = None; n = len(df)
    for k in range(max(0, n - 8), n):
        row = df.iloc[k]
        for t, p in i["hi"][-8:]:
            if t < df.index[k] and row.High > p and row.Close < p: out = ("bear", p, "Swept buy-side liquidity above a swing high, then closed back below it")
        for t, p in i["lo"][-8:]:
            if t < df.index[k] and row.Low < p and row.Close > p: out = ("bull", p, "Swept sell-side liquidity below a swing low, then closed back above it")
    return out

def diverge(df, i):
    r = rsi(df.Close); n = len(df); hi, lo = i["hi"], i["lo"]
    try:
        if len(hi) >= 2 and df.index.get_loc(hi[-1][0]) > n - 40 and hi[-1][1] > hi[-2][1] and r.loc[hi[-1][0]] < r.loc[hi[-2][0]] - 2: return ("bear", "Bearish RSI divergence: price made a higher high but momentum did not")
        if len(lo) >= 2 and df.index.get_loc(lo[-1][0]) > n - 40 and lo[-1][1] < lo[-2][1] and r.loc[lo[-1][0]] > r.loc[lo[-2][0]] + 2: return ("bull", "Bullish RSI divergence: price made a lower low but momentum did not")
    except Exception: pass
    return None

def killzone(now):
    try:
        from zoneinfo import ZoneInfo; h = now.astimezone(ZoneInfo("America/New_York")).hour
    except Exception: h = (now.hour - 5) % 24
    for n, a, b in (("Asian killzone", 20, 24), ("London killzone", 2, 5), ("New York killzone", 7, 10), ("London close", 10, 12)):
        if a <= h < b: return n
    return None

# ---------------- levels ----------------
W = {"fvg-bull": 2, "fvg-bear": 2, "ob-bull": 2, "ob-bear": 2, "liquidity": 2, "resistance": 1.5, "support": 1.5, "prev": 1.5, "pivot": 1, "fib": 1, "session": 1, "ema": 1}
def build_levels(f, cfg, P, A, info):
    tfs = cfg["tfs"]; pr, hi_df = f[tfs[1]], f[tfs[2]]; L = []
    pts = []
    for df in (pr, hi_df):
        h, l = swings(df, 3); pts += [x[1] for x in h[-12:]] + [x[1] for x in l[-12:]]
    pts.sort(); cl = []
    for p in pts:
        if cl and p - cl[-1][-1] <= .35 * A: cl[-1].append(p)
        else: cl.append([p])
    for c in cl:
        if len(c) >= 2:
            p = sum(c) / len(c); L.append({"p": p, "label": "Support/resistance zone (%d touches)" % len(c), "kind": "resistance" if p > P else "support"})
    d = f["1d"]
    if len(d) >= 3:
        b = d.iloc[-2]; H, Lo, C = float(b.High), float(b.Low), float(b.Close); pv = (H + Lo + C) / 3
        for n, v in (("pivot", pv), ("R1", 2*pv-Lo), ("R2", pv+H-Lo), ("R3", H+2*(pv-Lo)), ("S1", 2*pv-H), ("S2", pv-(H-Lo)), ("S3", Lo-2*(H-pv))):
            L.append({"p": v, "label": "Daily pivot" if n == "pivot" else "Daily " + n, "kind": "pivot"})
        L += [{"p": H, "label": "Previous day high", "kind": "prev"}, {"p": Lo, "label": "Previous day low", "kind": "prev"}, {"p": float(d.iloc[-1].Open), "label": "Daily open", "kind": "prev"}]
    w = f["1w"]
    if len(w) >= 3: L += [{"p": float(w.iloc[-2].High), "label": "Previous week high", "kind": "prev"}, {"p": float(w.iloc[-2].Low), "label": "Previous week low", "kind": "prev"}]
    if tfs[0] != "4h":
        h1 = f["1h"]; day = h1[h1.index.date == h1.index[-1].date()]
        for nm, a0, b0 in (("Asian", 0, 7), ("London", 7, 12), ("New York", 12, 17)):
            s = day[(day.index.hour >= a0) & (day.index.hour < b0)]
            if len(s): L += [{"p": float(s.High.max()), "label": nm + " session high", "kind": "session"}, {"p": float(s.Low.min()), "label": nm + " session low", "kind": "session"}]
    h, l = info[tfs[2]]["hi"], info[tfs[2]]["lo"]; rng = None
    if h and l:
        (ht, hp), (lt, lp) = h[-1], l[-1]; up = ht > lt; rng = (hp, lp, up); span = hp - lp
        for r in (.382, .5, .618, .705, .786): L.append({"p": hp - r * span if up else lp + r * span, "label": "Fib %.1f%% retracement" % (r * 100), "kind": "fib"})
        for r in (1.272, 1.618): L.append({"p": lp + r * span if up else hp - r * span, "label": "Fib %.1f%% extension" % (r * 100), "kind": "fib"})
    x = pr.tail(150).reset_index(drop=True); a = atr(pr).tail(150).reset_index(drop=True)
    for i in range(2, len(x)):
        if x.Low[i] > x.High[i-2] and not (x.Low[i+1:] <= x.High[i-2]).any():
            L.append({"p": (x.Low[i] + x.High[i-2]) / 2, "lo": float(x.High[i-2]), "hi": float(x.Low[i]), "label": "Bullish fair value gap (%s)" % tfs[1], "kind": "fvg-bull"})
        if x.High[i] < x.Low[i-2] and not (x.High[i+1:] >= x.Low[i-2]).any():
            L.append({"p": (x.High[i] + x.Low[i-2]) / 2, "lo": float(x.High[i]), "hi": float(x.Low[i-2]), "label": "Bearish fair value gap (%s)" % tfs[1], "kind": "fvg-bear"})
        if abs(x.Close[i] - x.Open[i]) > 1.5 * a[i]:
            j = i - 1
            if x.Close[i] > x.Open[i] and x.Close[j] < x.Open[j] and not (x.Close[i+1:] < x.Low[j]).any():
                L.append({"p": (x.High[j] + x.Low[j]) / 2, "lo": float(x.Low[j]), "hi": float(x.High[j]), "label": "Bullish order block (%s)" % tfs[1], "kind": "ob-bull"})
            if x.Close[i] < x.Open[i] and x.Close[j] > x.Open[j] and not (x.Close[i+1:] > x.High[j]).any():
                L.append({"p": (x.High[j] + x.Low[j]) / 2, "lo": float(x.Low[j]), "hi": float(x.High[j]), "label": "Bearish order block (%s)" % tfs[1], "kind": "ob-bear"})
    for src, nm in ((info[tfs[1]]["hi"][-20:], "Equal highs (liquidity above)"), (info[tfs[1]]["lo"][-20:], "Equal lows (liquidity below)")):
        v = sorted(p for _, p in src)
        for i in range(len(v) - 1):
            if abs(v[i+1] - v[i]) <= .15 * A: L.append({"p": max(v[i], v[i+1]) if "highs" in nm else min(v[i], v[i+1]), "label": nm, "kind": "liquidity"})
    L += [{"p": info[tfs[1]]["e50"], "label": "EMA 50 (%s)" % tfs[1], "kind": "ema"}, {"p": info[tfs[1]]["e200"], "label": "EMA 200 (%s)" % tfs[1], "kind": "ema"}]
    return L, rng

def confluence(L, A, P):
    pts = sorted(L, key=lambda z: z["p"]); zones = []
    for z in pts:
        lo, hi = z.get("lo", z["p"]), z.get("hi", z["p"])
        if zones and z["p"] - zones[-1]["_last"] <= .3 * A: zs = zones[-1]; zs["items"].append(z); zs["lo"] = min(zs["lo"], lo); zs["hi"] = max(zs["hi"], hi); zs["_last"] = z["p"]
        else: zones.append({"lo": lo, "hi": hi, "_last": z["p"], "items": [z]})
    out = []
    for zs in zones:
        st = sum(W.get(i["kind"], 1) for i in zs["items"]); kinds = {i["kind"] for i in zs["items"]}
        if len(zs["items"]) >= 3 and len(kinds) >= 2 or st >= 5:
            mid = (zs["lo"] + zs["hi"]) / 2
            out.append({"lo": zs["lo"], "hi": zs["hi"], "mid": mid, "count": len(zs["items"]), "strength": round(st, 1), "labels": [i["label"] for i in sorted(zs["items"], key=lambda q: -W.get(q["kind"], 1))][:4], "side": "above" if mid > P else "below"})
    out.sort(key=lambda z: abs(z["mid"] - P)); return out[:10]

# ---------------- plan ----------------
def make_plan(side, P, A, L, Z, cfg, info_pr, lo_tf):
    buy = side == "buy"; fv = ("fvg-bull", "ob-bull") if buy else ("fvg-bear", "ob-bear")
    zones = [{"lo": z["lo"], "hi": z["hi"], "s": z["strength"], "note": "Confluence zone: " + "; ".join(z["labels"][:3])} for z in Z if (z["mid"] < P if buy else z["mid"] > P) and abs(z["mid"] - P) <= 2.5 * A]
    zones += [{"lo": l["lo"], "hi": l["hi"], "s": 2.5, "note": l["label"]} for l in L if l["kind"] in fv and (l["p"] < P if buy else l["p"] > P) and abs(l["p"] - P) <= 2.5 * A]
    if zones:
        z = max(zones, key=lambda q: q["s"] - abs((q["lo"] + q["hi"]) / 2 - P) / A * .8); lo, hi, note, s = z["lo"], z["hi"], z["note"], z["s"]
    else:
        lo, hi, note, s = ((P - .5 * A, P - .1 * A) if buy else (P + .1 * A, P + .5 * A)) + ("Pullback of about half an ATR", 0)
    if hi - lo < .15 * A: m = (lo + hi) / 2; lo, hi = m - .075 * A, m + .075 * A
    em = (lo + hi) / 2
    if buy:
        sw = [p for _, p in info_pr["lo"] if p < lo and lo - p <= 1.2 * A]; stop = (min(lo, max(sw)) if sw else lo) - .3 * A
    else:
        sw = [p for _, p in info_pr["hi"] if p > hi and p - hi <= 1.2 * A]; stop = (max(hi, min(sw)) if sw else hi) + .3 * A
    if abs(em - stop) < .5 * A: stop = em - .5 * A if buy else em + .5 * A
    risk = abs(em - stop)
    tg = sorted([z["p"] for z in L if (z["p"] > hi if buy else z["p"] < lo) and risk <= abs(z["p"] - em) <= min(6 * A, 5 * risk)], key=lambda p: abs(p - em)); t = []
    for p in tg:
        if all(abs(p - q) >= .5 * risk for q in t): t.append(p)
        if len(t) == 3: break
    for m in cfg["rr"]:
        if len(t) == 3: break
        p = em + m * risk if buy else em - m * risk
        if all(abs(p - q) >= .5 * risk for q in t): t.append(p)
    t = sorted(t, key=lambda p: abs(p - em))[:3]
    return {"side": side, "entry": [lo, hi], "stop": stop, "targets": t, "rr": [abs(p - em) / risk for p in t], "note": note, "zone_strength": s, "lower_tf": lo_tf}

# ---------------- historical edge check ----------------
def edge(df):
    c = df.Close; e20, e50, e200 = ema(c, 20), ema(c, 50), ema(c, 200); r = rsi(c); mh = macd_hist(c); ax = adx(df); at = atr(df)
    sc = .35 * np.where(e20 > e50, 1, -1) + .25 * np.where(c > e200, 1, -1) + .2 * np.clip((r - 50) / 25, -1, 1) + .2 * np.where(mh > 0, 1, -1)
    sc = np.asarray(sc, dtype=float); ax = ax.values; H, Lw, C, A = df.High.values, df.Low.values, c.values, at.values; res = []; i = 200; n = len(df)
    while i < n - 24:
        if ax[i] > 18 and abs(sc[i]) > .5 and A[i] > 0:
            side = 1 if sc[i] > 0 else -1; e = C[i]; stop = e - side * A[i]; tgt = e + side * 1.5 * A[i]; out = None
            for j in range(i + 1, i + 25):
                hs = (Lw[j] <= stop) if side == 1 else (H[j] >= stop); ht = (H[j] >= tgt) if side == 1 else (Lw[j] <= tgt)
                if hs: out = -1.0; break
                if ht: out = 1.5; break
            if out is None: out = max(-1.0, min(1.5, side * (C[i + 24] - e) / A[i]))
            res.append(out); i += 10
        else: i += 1
    if len(res) < 8: return None
    r_ = np.array(res); return {"n": int(len(r_)), "win": int(round(100 * (r_ > 0).mean())), "avg_r": round(float(r_.mean()), 2), "horizon": "24 candles", "rule": "trend + momentum bias, stop 1 ATR, target 1.5 ATR"}

# ---------------- analysis ----------------
def sgn(x): return 1 if x > 0 else -1 if x < 0 else 0
def analyze(f, style, now):
    cfg = STYLES[style]; tfs = cfg["tfs"]; lo_tf, pr_tf, hi_tf = tfs; P = float(f["5m"].Close.iloc[-1]); d = 2 if P > 500 else 3 if P > 20 else 5
    info = {tf: tf_info(f[tf]) for tf in tfs}; ip, ih, il = info[pr_tf], info[hi_tf], info[lo_tf]; A = ip["atr"]
    L, rng = build_levels(f, cfg, P, A, info); Z = confluence(L, A, P)
    st = structure(f[pr_tf], ip); sw = sweep(f[pr_tf], ip); dv = diverge(f[pr_tf], ip)
    T = {"up": 1, "down": -1, "range": 0}; chop = .6 if ip["adx"] < 18 else 1.0; F = []
    def fac(name, s, w, det): F.append({"name": name, "score": round(float(max(-1, min(1, s))), 2), "weight": w, "detail": det})
    tn = {"up": "Uptrend, higher highs and higher lows", "down": "Downtrend, lower highs and lower lows", "range": "Range, no clear sequence"}
    fac("Higher timeframe trend (%s)" % hi_tf, T[ih["trend"]] * chop, .22, tn[ih["trend"]])
    fac("Primary trend (%s)" % pr_tf, T[ip["trend"]] * chop, .14, tn[ip["trend"]])
    fac("Lower timeframe trend (%s)" % lo_tf, T[il["trend"]], .06, tn[il["trend"]])
    fac("Moving averages", (ip["es"] * .6 + ih["es"] * .4) * (1 if abs(ip["slope"]) > .15 else .6), .10, "%s on %s, %s on %s" % (ip["ema"], pr_tf, ih["ema"], hi_tf))
    fac("Momentum", .5 * float(np.clip((ip["rsi"] - 50) / 25, -1, 1)) + .5 * (1 if ip["mh"] > 0 else -1), .08, "RSI %s, MACD %s on %s" % (ip["rsi"], ip["macd"], pr_tf))
    fac("Market structure", (1 if st and st[0] == "bull" else -1 if st and st[0] == "bear" else 0), .10, st[1] if st else "No fresh break of structure")
    fac("Liquidity sweep", (1 if sw and sw[0] == "bull" else -1 if sw and sw[0] == "bear" else 0), .10, sw[2] if sw else "No recent sweep of a swing high or low")
    pd_s, pd_t = 0, "No dealing range found"
    if rng:
        hp, lp, _ = rng; eq = (hp + lp) / 2; pos = (P - lp) / (hp - lp) if hp != lp else .5; pd_s = float(np.clip(1 - 2 * pos, -1, 1)); pd_t = "Price is in %s (%d%% of the range)" % ("discount" if pos < .5 else "premium", int(max(0, min(100, pos * 100))))
    fac("Premium or discount", pd_s * .8, .08, pd_t)
    fac("Divergence", (1 if dv and dv[0] == "bull" else -1 if dv and dv[0] == "bear" else 0), .05, dv[1] if dv else "No RSI divergence")
    score = sum(x["score"] * x["weight"] for x in F) / sum(x["weight"] for x in F)
    bias = "Bullish" if score > .16 else "Bearish" if score < -.16 else "Neutral"; bs = 1 if bias == "Bullish" else -1 if bias == "Bearish" else 0
    stats = edge(f[pr_tf]); kzn = killzone(now)
    conf = abs(score) * 120
    if bs and T[ih["trend"]] == bs and T[ip["trend"]] == bs: conf += 8
    if bs and T[ih["trend"]] == -bs: conf -= 10
    if ip["adx"] < 18: conf -= 10
    if stats: conf += 5 if stats["avg_r"] > 0 else -10
    conf = int(max(5, min(95, conf)))
    side = "buy" if bs >= 0 and bias != "Bearish" else "sell"
    pb, ps = make_plan("buy", P, A, L, Z, cfg, ip, lo_tf), make_plan("sell", P, A, L, Z, cfg, ip, lo_tf)
    if bias == "Neutral": side = "buy" if pb["rr"][0] + pb["zone_strength"] / 5 >= ps["rr"][0] + ps["zone_strength"] / 5 else "sell"
    prim, alt = (pb, ps) if side == "buy" else (ps, pb); want = 1 if side == "buy" else -1
    pts = 0; why = []
    if prim["zone_strength"] >= 4: pts += 2; why.append("Entry zone has strong confluence")
    if prim["rr"][0] >= 1.5: pts += 1
    if T[ih["trend"]] == want and T[ip["trend"]] == want: pts += 2; why.append("Higher and primary timeframes agree")
    elif T[ih["trend"]] == -want: pts -= 2; why.append("Higher timeframe trend is against this side")
    if (st and (1 if st[0] == "bull" else -1) == want) or (sw and (1 if sw[0] == "bull" else -1) == want): pts += 1; why.append("Structure or liquidity sweep supports it")
    if ip["adx"] >= 20: pts += 1; why.append("Trend strength is healthy (ADX %.0f)" % ip["adx"])
    else: why.append("Trend strength is weak (ADX %.0f), expect chop" % ip["adx"])
    if cfg["kz"] and kzn: pts += 1; why.append("Inside the " + kzn)
    if stats and stats["avg_r"] > 0: pts += 1
    elif stats: pts -= 1
    if bias == "Neutral": pts -= 2
    grade = "A+" if pts >= 8 else "A" if pts >= 6 else "B" if pts >= 4 else "C" if pts >= 2 else "No trade"
    above = [z for z in L if z["p"] > P and z["kind"] in ("resistance", "ob-bear", "fvg-bear", "prev", "liquidity", "pivot", "session") and z["p"] - P <= 4 * A]; below = [z for z in L if z["p"] < P and z["kind"] in ("support", "ob-bull", "fvg-bull", "prev", "liquidity", "pivot", "session") and P - z["p"] <= 4 * A]
    ua = min(above, key=lambda z: z["p"] - P) if above else None; db = max(below, key=lambda z: z["p"]) if below else None
    watch = {"bull": ("A strong close above %s (%s) on %s would improve the bullish case" % (round(ua["p"], d), ua["label"], pr_tf)) if ua else "", "bear": ("A strong close below %s (%s) on %s would improve the bearish case" % (round(db["p"], d), db["label"], pr_tf)) if db else ""}
    buy = side == "buy"
    prim["trigger"] = "Wait for price to trade into the zone, then enter only after a %s confirmation on the %s chart, such as a %s. Do not %s blindly into the zone." % ("bullish" if buy else "bearish", lo_tf, "break above the last lower high or a strong bullish close" if buy else "break below the last higher low or a strong bearish close", "buy" if buy else "sell")
    prim["management"] = "Take about half the position at Target 1 and move the stop to break-even. Trail the rest behind the last swing. Avoid holding through high-impact news."
    prim["valid"] = "Valid for " + cfg["valid"] + ". If it has not triggered by then, discard it."
    out = {"bias": bias, "confidence": conf, "score": round(score, 3), "grade": grade, "points": pts, "horizon": cfg["h"], "atr": A, "factors": F, "why": why[:5],
           "tfs": [{"tf": tf, "trend": info[tf]["trend"], "rsi": info[tf]["rsi"], "adx": info[tf]["adx"], "ema": info[tf]["ema"], "macd": info[tf]["macd"]} for tf in tfs],
           "context": {"killzone": kzn, "volatility": "high" if ip["vol_pct"] > 75 else "low" if ip["vol_pct"] < 25 else "normal", "structure": st[1] if st else None, "sweep": sw[2] if sw else None, "divergence": dv[1] if dv else None},
           "confluence": Z, "levels": sorted(sorted(L, key=lambda z: abs(z["p"] - P))[:46], key=lambda z: -z["p"]), "plan": {"primary": prim, "alt": alt}, "watch": watch, "stats": stats, "neutral": bias == "Neutral"}
    if rng:
        hp, lp, _ = rng; out["range"] = {"hi": hp, "lo": lp, "eq": (hp + lp) / 2, "zone": "premium" if P > (hp + lp) / 2 else "discount", "pos": int(max(0, min(100, round((P - lp) / (hp - lp) * 100)))) if hp != lp else 50, "outside": bool(P > hp or P < lp)}
    return P, d, out

def rnd(o, d):
    if isinstance(o, float): return round(o, d)
    if isinstance(o, list): return [rnd(x, d) for x in o]
    if isinstance(o, dict): return {k: (v if k in ("confidence", "score", "pos", "points", "weight", "avg_r", "rsi", "adx", "win", "n", "strength", "zone_strength", "rr") else rnd(v, d)) for k, v in o.items()}
    return o

# ---------------- optional Claude view ----------------
SYSTEM = ("You are a senior FX and gold market analyst reviewing a systematic technical scan. Use your own market knowledge (macro drivers, central bank policy context, "
          "correlations such as the dollar index, yields and risk sentiment, seasonality, behaviour around sessions and events) together with the data provided. Stress-test the "
          "proposed setup: say what supports it, what could invalidate it, and whether you agree. Do not invent live news or prices; treat the data as given and say plainly when you are "
          "unsure about current conditions. No guarantees and no financial advice. Reply with JSON only, with keys: headline (max 120 chars), verdict (exactly one of: Agree, Agree with caution, Disagree), "
          "verdict_reason (one sentence), outlook (3 to 4 sentences), drivers (3 to 5 short strings), bull_case (string), bear_case (string), watch (3 to 4 short strings), invalidation (string), risk_note (string).")
def narrative(payload):
    body = json.dumps({"model": MODEL, "max_tokens": 1400, "system": SYSTEM, "messages": [{"role": "user", "content": json.dumps(payload)}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, headers={"x-api-key": KEY, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r: txt = json.load(r)["content"][0]["text"]
    return json.loads(re.sub(r"^```(?:json)?|```$", "", txt.strip(), flags=re.M).strip())

def main():
    old, news = {}, {}
    for nm, tgt in (("scan.json", "old"), ("news.json", "news")):
        if os.path.exists(nm):
            try: (old if tgt == "old" else news).update(json.load(open(nm)))
            except Exception: pass
    heads = [h.get("title", "") for h in (news.get("headlines") or [])[:8]]
    now = datetime.now(timezone.utc); res = {"version": 2, "updated": now.isoformat(), "pairs": {}}
    for pair, syms in PAIRS.items():
        f = None
        for s in syms:
            try: f = frames(s)
            except Exception as e: print(pair, s, "download failed:", e)
            if f: break
        if not f:
            if pair in old.get("pairs", {}): res["pairs"][pair] = old["pairs"][pair]
            print("skipped", pair); continue
        styles = {}; d = 5
        for st in STYLES:
            try: P, d, o = analyze(f, st, now)
            except Exception as e: print(pair, st, "analysis failed:", e); continue
            o = rnd(o, d)
            prev = ((old.get("pairs", {}).get(pair, {}).get("styles", {}) or {}).get(st, {}) or {}).get("narrative")
            fresh = prev and prev.get("_at") and (now - datetime.fromisoformat(prev["_at"])).total_seconds() < NARR_MAX_AGE_H * 3600 and prev.get("_bias") == o["bias"] and prev.get("_grade") == o["grade"]
            if fresh: o["narrative"] = prev
            elif KEY:
                try:
                    near = sorted(o["levels"], key=lambda z: abs(z["p"] - P))[:12]
                    n = narrative({"pair": pair, "style": st, "price": round(P, d), "bias": o["bias"], "confidence": o["confidence"], "grade": o["grade"], "factors": o["factors"], "context": o["context"],
                                   "timeframes": o["tfs"], "confluence_zones": o["confluence"][:5], "nearest_levels": near, "primary_plan": o["plan"]["primary"], "edge_check": o["stats"],
                                   "headlines": heads, "fundamentals": (news.get("fundamentals") or "")[:600]})
                    n["_at"] = now.isoformat(); n["_bias"] = o["bias"]; n["_grade"] = o["grade"]; o["narrative"] = n
                except Exception as e: print(pair, st, "narrative failed:", e)
            styles[st] = o
        if styles: res["pairs"][pair] = {"price": round(float(f["5m"].Close.iloc[-1]), d), "decimals": d, "styles": styles}; print("done", pair)
    if not res["pairs"]: sys.exit("No pairs could be scanned")
    json.dump(res, open("scan.json", "w"), separators=(",", ":")); print("Wrote scan.json for", list(res["pairs"]))

if __name__ == "__main__":
    main()
