"""Does the second way of asking carry over to a second model? Scores the development set and the six evasion
wrappers with Gemini Decision 1 under the three purpose questions, beside its saved answers to the one question.

    GEMINI_API_KEY=... uv run variants/gemini.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import dev as D  # noqa: E402
import decision  # noqa: E402
from gate import EVASION_VARIANTS, PURPOSE_QUESTIONS, chunks, framed, purpose_score  # noqa: E402

OUT = D.HERE / "scores" / "gemini-three.jsonl"
REFUSED = {q: 1.0 for q in PURPOSE_QUESTIONS}  # the provider's input filter refused the text: counted as a block
key = lambda text: hashlib.sha1(text.encode()).hexdigest()[:16]


def answers(texts):
    saved = {r["key"]: r["answers"] for r in map(json.loads, OUT.read_text().splitlines())} if OUT.exists() else {}
    todo = [t for t in dict.fromkeys(texts) if key(t) not in saved]
    print(f"{len(saved)} saved, {len(todo)} to ask", file=sys.stderr, flush=True)
    ask = decision.session("gemini", attempts=10, questions=PURPOSE_QUESTIONS)
    with OUT.open("a") as f:
        for i, t in enumerate(todo, 1):
            saved[key(t)] = ask(t).get("answers", REFUSED)
            f.write(json.dumps({"key": key(t), "answers": saved[key(t)]}) + "\n")
            f.flush()
            time.sleep(0.6)
            if i % 200 == 0:
                print(i, file=sys.stderr, flush=True)
    return lambda t: saved[key(t)]


if __name__ == "__main__":
    dev = D.dev_set()
    one = pd.read_json(D.ROOT / "results" / "decision-gemini.jsonl", lines=True).drop_duplicates("key").set_index("key")["p"].fillna(1.0)
    bad = dev[(dev["set"] == "mcp") & (dev["label"] == 1)]
    wrapped = {i: [pre + row["text"] + suf for pre, suf in EVASION_VARIANTS] for i, row in bad.iterrows()}
    texts = [framed(row["context"], c) for _, row in dev.iterrows() for c in chunks(row["text"])]
    texts += [framed(dev.loc[i, "context"], c) for i, g in wrapped.items() for p in g for c in chunks(p)]
    get = answers(texts)
    tool = (dev["set"] == "mcp").values
    three = lambda context, text, is_tool: purpose_score([get(framed(context, c)) for c in chunks(text)], is_tool)
    first = lambda text: max(one[decision.key(c)] for c in chunks(text))
    S = pd.DataFrame({"one question": [first(t) for t in dev["text"]],
                      "three questions": [three(row["context"], row["text"], t) for (_, row), t in zip(dev.iterrows(), tool)]})
    y = dev["label"].values
    rows = {}
    for name in S:
        r = {}
        for domain, m in (("mcp", tool), ("skills", ~tool)):
            s = S[name].values
            cut = np.sort(s[m & (y == 0)])[-3]
            r |= {f"{domain} AUROC": roc_auc_score(y[m], s[m]), f"{domain} caught at 0.5": (s[m & (y == 1)] >= .5).mean(),
                  f"{domain} benign blocked at 0.5": (s[m & (y == 0)] >= .5).mean(), f"{domain} caught, 2 benign blocked": (s[m & (y == 1)] > cut).mean()}
        rows[name] = r
    pd.set_option("display.width", 250)
    print(pd.DataFrame(rows).round(2).to_string(), "\n")
    table = pd.DataFrame(rows).T
    for name, score in (("one question", lambda i, p: first(p)), ("three questions", lambda i, p: three(dev.loc[i, "context"], p, True))):
        each = np.array([[score(i, p) for p in g] for i, g in wrapped.items()])
        print(f"{name}: wrapped tools caught at 0.5, per wrapper {(each >= .5).mean(axis=0).round(2).tolist()}, attacker's pick {(each.min(axis=1) >= .5).mean():.2f}")
        table.loc[name, "mcp caught at 0.5 with the attacker's pick"] = (each.min(axis=1) >= .5).mean()
        table.loc[name, "mcp caught at 0.5 with the quoted-example wrapper"] = (each[:, 2] >= .5).mean()
    table.rename_axis("question").to_csv(D.ROOT / "results" / "question-variant-gemini.csv")
