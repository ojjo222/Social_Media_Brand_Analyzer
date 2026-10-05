import os
import json
import time
import random
import emoji
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from google import genai

from config import (
    BRANDS, SENTIMENT_WEIGHTS, PLATFORM_TARGETS,
    GEMINI_SHARE, COMMENTS_PER_REQUEST, GEMINI_OUTPUT, PROGRESS_FILE
)

load_dotenv()
client = genai.Client()

MODEL = "gemini-3.8-flash"


def load_progress():
    p = Path(PROGRESS_FILE)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"gemini_done": 0, "groq_done": 0}


def save_progress(progress):
    Path(PROGRESS_FILE).write_text(json.dumps(progress, indent=2), encoding="utf-8")


def weighted_sentiment():
    r = random.random()
    cum = 0
    for s, w in SENTIMENT_WEIGHTS.items():
        cum += w
        if r <= cum:
            return s
    return "neutral"


def build_prompt(brand, platform, sentiment, n):
    platform_desc = {
        "youtube":   "casual YouTube comment (short, conversational)",
        "twitter":   "short Twitter/X post (punchy, under 200 chars)",
        "instagram": "Instagram comment (short, casual praise or criticism)",
    }[platform]

    return (
        f"Generate {n} realistic {platform_desc} comments about the brand '{brand}'. "
        f"All {n} comments should have a {sentiment} sentiment. "
        f"Make them varied in length and tone. "
        f"Do NOT use emojis. Do NOT use hashtags. Do NOT number them. "
        f"Output ONLY the comments, one per line. No headers, no explanation."
    )


def parse_response(text):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cleaned = []
    for ln in lines:
        ln = ln.lstrip("0123456789.-)* ").strip()
        ln = emoji.replace_emoji(ln, replace="")
        ln = ln.strip()
        if len(ln) > 5 and len(ln) < 300:
            cleaned.append(ln)
    return cleaned


def main():
    progress = load_progress()
    done = progress["gemini_done"]

    total_synth = sum(PLATFORM_TARGETS.values())
    gemini_target = int(total_synth * GEMINI_SHARE)

    print(f"Gemini target: {gemini_target} comments. Already done: {done}.")

    tasks = []
    for platform, total_n in PLATFORM_TARGETS.items():
        gemini_n = int(total_n * GEMINI_SHARE)
        for _ in range(0, gemini_n, COMMENTS_PER_REQUEST):
            tasks.append((platform, min(COMMENTS_PER_REQUEST, gemini_n - _)))

    tasks = tasks[done:]

    batch_rows = []
    for i, (platform, n) in enumerate(tasks):
        brand = random.choice(BRANDS)
        sentiment = weighted_sentiment()
        prompt = build_prompt(brand, platform, sentiment, n)

        try:
            resp = client.models.generate_content(model=MODEL, contents=prompt)
            comments = parse_response(resp.text)
        except Exception as e:
            print(f"[{i}] Error: {e}. Sleeping 30s...")
            time.sleep(30)
            continue

        for c in comments[:n]:
            batch_rows.append({
                "text": c,
                "sentiment": sentiment,
                "brand": brand,
                "platform": platform,
                "source": "synthetic-gemini",
            })

        df = pd.DataFrame(batch_rows)
        df.to_csv(
            GEMINI_OUTPUT,
            mode="a",
            header=not Path(GEMINI_OUTPUT).exists(),
            index=False,
            encoding="utf-8",
        )
        batch_rows = []

        done_so_far = done + i + 1
        progress["gemini_done"] = done_so_far
        save_progress(progress)

        if Path(GEMINI_OUTPUT).exists():
            with open(GEMINI_OUTPUT, encoding="utf-8") as f:
                total_written = sum(1 for _ in f) - 1
        else:
            total_written = 0

        print(f"[{i+1}/{len(tasks)}] {platform}/{brand}/{sentiment}: +{len(comments)}  |  total so far: {total_written}")

    print("\nGemini run complete.")


if __name__ == "__main__":
    main()