"""Scores every evaluation chunk with a hosted decision model, zero-shot, and saves one row per
chunk in results/decision-<model>.jsonl. A rerun skips chunks already saved.

    TYPESAFE_API_KEY=... uv run decision.py jev
    GEMINI_API_KEY=... uv run decision.py gemini

Request shapes follow compare.py in jonasneves/decision-models.
"""
import hashlib
import http.client
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd

from gate import EVASION_NOTE, QUESTION, chunks

ROOT = Path(__file__).parent
MODELS = {"jev": ("https://api.typesafe.ai/v1/systemone", "TYPESAFE_API_KEY", "jev-1.14.0"),
          "gemini": ("https://generativelanguage.googleapis.com/v1beta/decisions", "GEMINI_API_KEY", "gemini-decision-1")}
PAUSE = {"jev": 0, "gemini": 0.45}  # Gemini's free tier allows 50,000 input tokens a minute


def key(text):
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def evaluation_chunks():
    """Every chunk of every row outside the training split, plus the evasion variant of each malicious MCP description."""
    sets = pd.read_parquet(ROOT / "data" / "sets.parquet")
    texts = list(sets.loc[sets["split"] != "train", "text"])
    texts += [text + EVASION_NOTE for text in sets.loc[(sets["set"] == "mcp") & (sets["label"] == 1), "text"]]
    return list(dict.fromkeys(piece for text in texts for piece in chunks(text)))


def session(name, timeout=8, attempts=6):
    """A function that asks one hosted model about one text. The app passes a short timeout and few
    attempts, because a person is waiting; the batch run keeps the defaults."""
    url, key_env, model = MODELS[name]
    u = urlsplit(url)
    auth = {"x-goog-api-key": os.environ[key_env]} if name == "gemini" else {"authorization": f"Bearer {os.environ[key_env]}"}
    conn = http.client.HTTPSConnection(u.netloc, timeout=timeout)

    def ask(text):
        nonlocal conn
        body = {"model": model, "state" if name == "jev" else "input": text,
                "questions": {"malicious": {"type": "noul", "instructions": QUESTION}}}
        for attempt in range(attempts):
            start = time.perf_counter()
            try:
                conn.request("POST", u.path, json.dumps(body), {**auth, "content-type": "application/json"})
                res = conn.getresponse()
                data = res.read()
            except (http.client.HTTPException, OSError) as error:
                print(f"{name} {type(error).__name__}, retry {attempt + 1}", file=sys.stderr, flush=True)
                conn.close()
                conn = http.client.HTTPSConnection(u.netloc, timeout=timeout)
                continue
            ms = (time.perf_counter() - start) * 1000
            if res.status in (429, 499, 529) or res.status >= 500:
                print(f"{name} {res.status}, retry {attempt + 1}: {data[:160]!r}", file=sys.stderr, flush=True)
                time.sleep(2 ** attempt)
                continue
            response = json.loads(data)
            if res.status == 400 and response.get("error", {}).get("code") == "prohibited_content":
                # Gemini's input filter refuses some texts before the model sees them. Saved with no
                # probability; the notebook counts a refusal as a block.
                return {"p": None, "ms": round(ms, 1), "input_tokens": None}
            if res.status != 200:
                raise RuntimeError(f"{name} {res.status}: {data[:300]!r}")
            usage = response["usage"]
            return {"p": response["answers"]["malicious"]["noul"], "ms": round(ms, 1),
                    "input_tokens": usage.get("input_tokens", usage.get("total_input_tokens"))}
        raise RuntimeError(f"{name}: retries exhausted")
    return ask


if __name__ == "__main__":
    name = sys.argv[1]
    out = ROOT / "results" / f"decision-{name}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = {json.loads(line)["key"] for line in out.read_text().splitlines()} if out.exists() else set()
    todo = [piece for piece in evaluation_chunks() if key(piece) not in done]
    if len(sys.argv) > 2:
        todo = todo[:int(sys.argv[2])]
    print(f"{name}: {len(done)} chunks saved, {len(todo)} to score", flush=True)
    ask = session(name)
    with out.open("a") as f:
        for i, piece in enumerate(todo, 1):
            f.write(json.dumps({"key": key(piece), **ask(piece)}) + "\n")
            f.flush()
            time.sleep(PAUSE[name])
            if i % 500 == 0:
                print(f"{name}: {i}/{len(todo)}", flush=True)
