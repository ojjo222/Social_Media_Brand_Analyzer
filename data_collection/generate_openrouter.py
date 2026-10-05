import os
import json
import time
import random
import emoji
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from config import (
    BRANDS, SENTIMENT_WEIGHTS, PLATFORM_TARGETS,
    OPENROUTER_SHARE, COMMENTS_PER_REQUEST,
    OPENROUTER_OUTPUT, PROGRESS_FILE,
)

load_dotenv()
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

MODEL = "nvidia/nemotron-3-super-120b-a12b:free"


def load_progress():
    p = Path(PROGRESS_FILE)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"gemini_done": 0, "groq_done": 0, "openrouter_done": 0}


def save_progress(progress):
    Path(PROGRESS_FILE).write_text(
        json.dumps(progress, indent=2), encoding="utf-8"
    )


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
    done = progress.get("openrouter_done", 0)

    total_synth = sum(PLATFORM_TARGETS.values())
    or_target = int(total_synth * OPENROUTER_SHARE)

    print(f"OpenRouter target: {or_target} comments. Already done: {done}.")

    tasks = []
    for platform, total_n in PLATFORM_TARGETS.items():
        or_n = int(total_n * OPENROUTER_SHARE)
        for _ in range(0, or_n, COMMENTS_PER_REQUEST):
            tasks.append((platform, min(COMMENTS_PER_REQUEST, or_n - _)))

    tasks = tasks[done:]

    batch_rows = []
    for i, (platform, n) in enumerate(tasks):
        brand = random.choice(BRANDS)
        sentiment = weighted_sentiment()
        prompt = build_prompt(brand, platform, sentiment, n)

        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
            )
            comments = parse_response(resp.choices[0].message.content)
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
                "source": "synthetic-openrouter",
            })

        df = pd.DataFrame(batch_rows)
        df.to_csv(
            OPENROUTER_OUTPUT,
            mode="a",
            header=not Path(OPENROUTER_OUTPUT).exists(),
            index=False,
            encoding="utf-8",
        )
        batch_rows = []

        done_so_far = done + i + 1
        progress["openrouter_done"] = done_so_far
        save_progress(progress)

        if Path(OPENROUTER_OUTPUT).exists():
            with open(OPENROUTER_OUTPUT, encoding="utf-8") as f:
                total_written = sum(1 for _ in f) - 1
        else:
            total_written = 0

        print(f"[{i+1}/{len(tasks)}] {platform}/{brand}/{sentiment}: "
              f"+{len(comments)} | total so far: {total_written}")

    print("\nOpenRouter run complete.")


if __name__ == "__main__":
    main()