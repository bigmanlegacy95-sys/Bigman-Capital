"""Fetch forex headlines, ask Google Gemini (free tier) for a summary + outlook, save news.json.
Auto-detects a working Gemini model, so model renames can't break it. Needs env GEMINI_API_KEY."""
import os, re, json, sys, time, datetime, urllib.request, urllib.error, xml.etree.ElementTree as ET

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
    bad = ("image", "tts", "live", "audio", "embedding", "lite", "exp", "thinking", "vision", "robotics", "computer", "gemma", "learnlm")
    good = [n for n in names if "flash" in n and not any(b in n for b in bad)]
    good.sort(reverse=True)
    good.sort(key=lambda n: "latest" not in n)
    cands = ([PREFERRED] if PREFERRED in names else []) + good
    cands = list(dict.fromkeys(cands)) or ["gemini-flash-latest", "gemini-2.5-flash"]
    print("Models to try:", cands[:4])
    return cands[:4]


def ask_ai(items):
    lines = "\n".join(f"- [{i['source']}] {i['title']}: {i['summary']}" for i in items)
    prompt = f"""You are a forex market analyst writing for a trading company website.
Read these latest headlines and respond with JSON only, in exactly this shape:
{{"summary": "3-4 sentence plain-English summary of what is moving markets today",
 "outlook": [{{"pair": "...", "bias": "Bullish|Bearish|Neutral", "reason": "one short sentence"}}],
 "watch": ["up to 4 key events or data releases to watch"]}}
Give an outlook for exactly these instruments: {", ".join(PAIRS)}.
Base everything only on the headlines. Be calm and measured. Do not promise outcomes.

HEADLINES:
{lines}"""
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
        time.sleep(3)
    raise SystemExit("All AI models failed; keeping the old news.json")


def main():
    items = get_headlines()
    if len(items) < 3:
        raise SystemExit("Not enough headlines; keeping the old news.json")
    ai = ask_ai(items)
    out = {
        "updated": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="minutes"),
        "summary": ai.get("summary", ""),
        "outlook": ai.get("outlook", []),
        "watch": ai.get("watch", []),
        "headlines": [{"source": i["source"], "title": i["title"], "link": i["link"]} for i in items[:8]],
    }
    with open("news.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("news.json updated")


if __name__ == "__main__":
    main()
