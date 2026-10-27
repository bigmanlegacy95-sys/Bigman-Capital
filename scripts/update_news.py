"""Fetch forex headlines, ask Google Gemini (free tier) for a summary + outlook, save news.json.
Auto-detects a working Gemini model, so model renames can't break it. Needs env GEMINI_API_KEY."""
import os, re, json, sys, time, datetime, urllib.request, urllib.error, urllib.parse, xml.etree.ElementTree as ET

KEY = os.environ["GEMINI_API_KEY"]
PREFERRED = os.environ.get("GEMINI_MODEL", "").strip()
BASE = "https://generativelanguage.googleapis.com/v1beta"
FEEDS = [
    ("Investing.com", "https://www.investing.com/rss/news_1.rss"),
    ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
    ("CNBC", "https://www.cnbc.com/id/10000664/device/rss/rss.html"),
]
PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD (Gold)"]


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 BigManCapitalBot"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read()


def get_headlines():
    items = []
    for name, url in FEEDS:
        try:
            root = ET.fromstring(fetch(url))
            n = 0
            for it in root.iter("item"):
                title = (it.findtext("title") or "").strip()
                if not title:
                    continue
                desc = re.sub(r"<[^>]+>", "", it.findtext("description") or "").strip()[:240]
                items.append({"source": name, "title": title, "link": (it.findtext("link") or "").strip(), "summary": desc})
                n += 1
                if n >= 12:
                    break
        except Exception as e:
            print(f"Feed failed ({name}): {e}", file=sys.stderr)
    return items


def api(path, body=None):
    headers = {"x-goog-api-key": KEY}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(BASE + path, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def candidate_models():
    names = []
    try:
        for m in api("/models?pageSize=100").get("models", []):
            if "generateContent" in m.get("supportedGenerationMethods", []):
                names.append(m["name"].split("/", 1)[1])
    except Exception as e:
        print(f"Could not list models: {e}", file=sys.stderr)
    bad = ("image", "tts", "live", "audio", "embedding", "exp", "thinking", "vision", "robotics", "computer", "gemma", "learnlm")
    good = [n for n in names if "flash" in n and not any(b in n for b in bad)]
    good.sort(reverse=True)
    good.sort(key=lambda n: "latest" not in n)
    cands = ([PREFERRED] if PREFERRED in names else []) + good
    cands = list(dict.fromkeys(cands)) or ["gemini-flash-latest", "gemini-2.5-flash"]
    print("Models to try:", cands[:8])
    return cands[:8]


CCY = ["EUR", "GBP", "AUD", "NZD", "JPY", "CAD", "CHF"]
TECH = [("EUR/USD", "EUR", False), ("GBP/USD", "GBP", False), ("USD/JPY", "JPY", True), ("AUD/USD", "AUD", False),
        ("USD/CAD", "CAD", True), ("USD/CHF", "CHF", True), ("NZD/USD", "NZD", False)]
COT_URL = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
COT_MKTS = [("EUR", "EURO FX - "), ("GBP", "BRITISH POUND"), ("JPY", "JAPANESE YEN"), ("AUD", "AUSTRALIAN DOLLAR"),
            ("CAD", "CANADIAN DOLLAR"), ("CHF", "SWISS FRANC"), ("NZD", "NZ DOLLAR"), ("USD Index", "USD INDEX"), ("Gold", "GOLD - COMMODITY")]


def safe(fn, *a):
    try:
        return fn(*a)
    except Exception as e:
        print(f"{fn.__name__} failed: {e}", file=sys.stderr)
        return None


def fx_series():
    """Daily ECB rates from Frankfurter (free, no key). Returns value of each currency in USD, oldest to newest."""
    start = (datetime.date.today() - datetime.timedelta(days=130)).isoformat()
    last = None
    for host in ("https://api.frankfurter.dev/v1/", "https://api.frankfurter.app/"):
        try:
            data = json.loads(fetch(f"{host}{start}..?base=USD&symbols={','.join(CCY)}"))
            break
        except Exception as e:
            last = e
    else:
        raise last
    days = [d for d in sorted(data["rates"]) if all(c in data["rates"][d] for c in CCY)]
    v = {c: [1 / data["rates"][d][c] for d in days] for c in CCY}
    v["USD"] = [1.0] * len(days)
    return v


def strength(v):
    out = {}
    for w in (1, 5, 20):
        ch = {c: s[-1] / s[-1 - w] - 1 for c, s in v.items() if len(s) > w}
        avg = sum(ch.values()) / len(ch)
        out[str(w)] = {c: round((x - avg) * 100, 2) for c, x in ch.items()}
    return out


def tech_row(name, cl):
    if len(cl) < 50:
        return None
    sma = lambda n: sum(cl[-n:]) / n
    diffs = [b - a for a, b in zip(cl[-15:], cl[-14:])]
    g, l = sum(max(d, 0) for d in diffs), sum(max(-d, 0) for d in diffs)
    rsi = 100 if l == 0 else 100 - 100 / (1 + g / l)
    px, s20, s50 = cl[-1], sma(20), sma(50)
    trend = "Uptrend" if px > s20 > s50 else "Downtrend" if px < s20 < s50 else "Sideways"
    d = 3 if "JPY" in name else 2 if "XAU" in name else 5
    return {"pair": name, "price": round(px, d), "trend": trend, "rsi": round(rsi, 1),
            "support": round(min(cl[-20:]), d), "resistance": round(max(cl[-20:]), d)}


def technical(v):
    rows = [tech_row(n, [1 / x for x in v[c]] if inv else v[c]) for n, c, inv in TECH]
    gold = safe(lambda: [float(r.split(",")[4]) for r in fetch("https://stooq.com/q/d/l/?s=xauusd&i=d").decode().strip().splitlines()[1:]][-120:])
    if gold:
        rows.append(tech_row("XAU/USD", gold))
    return [r for r in rows if r]


def cot():
    out = []
    for label, pre in COT_MKTS:
        q = urllib.parse.urlencode({"$where": f"market_and_exchange_names like '{pre}%'", "$order": "report_date_as_yyyy_mm_dd DESC", "$limit": "8"})
        rows = [r for r in json.loads(fetch(COT_URL + "?" + q)) if "/" not in r.get("market_and_exchange_names", "")]
        if not rows:
            continue
        name = rows[0]["market_and_exchange_names"]
        rows = [r for r in rows if r["market_and_exchange_names"] == name][:2]
        net = lambda r: int(float(r["noncomm_positions_long_all"])) - int(float(r["noncomm_positions_short_all"]))
        L, S = int(float(rows[0]["noncomm_positions_long_all"])), int(float(rows[0]["noncomm_positions_short_all"]))
        out.append({"name": label, "date": rows[0]["report_date_as_yyyy_mm_dd"][:10], "long": L, "short": S, "net": L - S,
                    "change": net(rows[0]) - net(rows[1]) if len(rows) > 1 else None,
                    "pct_long": round(100 * L / (L + S), 1) if L + S else None})
    return out


def calendar():
    ev = json.loads(fetch("https://nfs.faireconomy.media/ff_calendar_thisweek.json"))
    now = datetime.datetime.now(datetime.timezone.utc)
    out = []
    for e in ev:
        if e.get("impact") != "High":
            continue
        try:
            t = datetime.datetime.fromisoformat(e["date"])
        except Exception:
            continue
        if t >= now - datetime.timedelta(hours=3):
            out.append({"date": t.astimezone(datetime.timezone.utc).isoformat(timespec="minutes"), "country": e.get("country", ""),
                        "title": e.get("title", ""), "forecast": e.get("forecast", ""), "previous": e.get("previous", "")})
    return sorted(out, key=lambda x: x["date"])[:12]


def get_market():
    m = {}
    v = safe(fx_series)
    if v:
        m["strength"] = safe(strength, v)
        m["technical"] = safe(technical, v)
    m["cot"] = safe(cot)
    m["calendar"] = safe(calendar)
    return {k: x for k, x in m.items() if x}


def ask_ai(items, market=None):
    lines = "\n".join(f"- [{i['source']}] {i['title']}: {i['summary']}" for i in items)
    mtxt = ("\nMARKET DATA (technical, COT, calendar):\n" + json.dumps({k: market[k] for k in ("technical", "cot", "calendar") if k in market})[:4500]) if market else ""
    prompt = f"""You are a forex market analyst writing for a trading company website.
Read these latest headlines and respond with JSON only, in exactly this shape:
{{"summary": "3-4 sentence plain-English summary of what is moving markets today",
 "outlook": [{{"pair": "...", "bias": "Bullish|Bearish|Neutral", "reason": "one short sentence"}}],
 "watch": ["up to 4 key events or data releases to watch"],
 "fundamentals": "3-4 sentences on the macro and economic drivers behind the news and upcoming events",
 "technical_view": "2-3 sentences interpreting the technical data provided",
 "cot_view": "2-3 sentences interpreting the COT positioning provided"}}
Give an outlook for exactly these instruments: {", ".join(PAIRS)}.
Base everything only on the headlines and market data provided; if a data section is missing, return an empty string for it. Be calm and measured. Do not promise outcomes.

HEADLINES:
{lines}
{mtxt}"""
    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
    }
    for model in candidate_models():
        try:
            data = api(f"/models/{model}:generateContent", body)
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
            print("Success with model:", model)
            return json.loads(text)
        except urllib.error.HTTPError as e:
            print(f"Model {model} failed: HTTP {e.code} {e.read()[:300]!r}", file=sys.stderr)
        except Exception as e:
            print(f"Model {model} failed: {e}", file=sys.stderr)
        time.sleep(10)
    raise SystemExit("All AI models failed; keeping the old news.json")


def call_ai(prompt):
    body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"responseMimeType": "application/json", "temperature": 0.5}}
    for model in candidate_models():
        try:
            data = api(f"/models/{model}:generateContent", body)
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip())
        except urllib.error.HTTPError as e:
            print(f"Article model {model} failed: HTTP {e.code}", file=sys.stderr)
        except Exception as e:
            print(f"Article model {model} failed: {e}", file=sys.stderr)
        time.sleep(10)
    return None


def daily_article(items, market, old):
    today = datetime.date.today().isoformat()
    if old and old[0].get("date") == today:
        return old
    lines = "\n".join(f"- {i['title']}" for i in items[:20])
    mk = json.dumps({k: market[k] for k in ("technical", "cot", "calendar") if k in market})[:3500] if market else ""
    prompt = f"""You are a senior forex analyst writing the daily market brief article for the BigMan Capital website.
Write for retail traders in clear, calm, professional English, about 350 words in total. Respond with JSON only:
{{"title": "engaging headline", "intro": "2 sentence hook", "sections": [{{"h": "Market mood", "p": "paragraph"}}, {{"h": "Key drivers", "p": "paragraph"}}, {{"h": "Technical and positioning", "p": "paragraph"}}, {{"h": "What to watch next", "p": "paragraph"}}]}}
Use only the headlines and data below. Do not give personal financial advice, do not promise outcomes, and do not invent numbers.

HEADLINES:
{lines}

MARKET DATA:
{mk}"""
    a = call_ai(prompt)
    if not a or not a.get("title"):
        return old
    return ([{"date": today, "title": a["title"], "intro": a.get("intro", ""), "sections": a.get("sections", [])}] + old)[:30]


def main():
    items = get_headlines()
    if len(items) < 3:
        raise SystemExit("Not enough headlines; keeping the old news.json")
    market = get_market()
    ai = ask_ai(items, market)
    out = {
        "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"),
        "summary": ai.get("summary", ""),
        "outlook": ai.get("outlook", []),
        "watch": ai.get("watch", []),
        "fundamentals": ai.get("fundamentals", ""),
        "technical_view": ai.get("technical_view", ""),
        "cot_view": ai.get("cot_view", ""),
        "market": market,
        "headlines": [{"source": i["source"], "title": i["title"], "link": i["link"]} for i in items[:8]],
    }
    try:
        old = json.load(open("news.json", encoding="utf-8")).get("articles", [])
    except Exception:
        old = []
    out["articles"] = safe(daily_article, items, market, old) or old
    with open("news.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("news.json updated")


if __name__ == "__main__":
    main()
