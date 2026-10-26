"""Fetch forex headlines, ask Google Gemini (free tier) for a summary + outlook, save news.json.
Runs on GitHub Actions. Needs env var GEMINI_API_KEY. No extra packages required."""
import os, re, json, sys, time, datetime, urllib.request, xml.etree.ElementTree as ET

KEY = os.environ["GEMINI_API_KEY"]
MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  # change in the workflow if Google renames it
FEEDS = [
    ("FXStreet", "https://www.fxstreet.com/rss/news"),
    ("Investing.com", "https://www.investing.com/rss/news_1.rss"),
    ("BBC Business", "https://feeds.bbci.co.uk/news/business/rss.xml"),
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
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.3},
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json", "x-goog-api-key": KEY})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
            return json.loads(data["candidates"][0]["content"]["parts"][0]["text"])
        except Exception as e:
            print(f"AI attempt {attempt + 1} failed: {e}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))
    raise SystemExit("AI request failed; keeping the old news.json")


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
