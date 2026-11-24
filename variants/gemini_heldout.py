"""Gemini Decision 1 under the three purpose questions on the held-out connector files, beside its saved answers to
the one question. The questions, rule and development thresholds are the ones variants/heldout.py used for Jev 1.14.

    GEMINI_KEYS=key1,key2 uv run variants/gemini_heldout.py
"""
import hashlib
import json
import os
import sys
import threading
from pathlib import Path
from queue import Empty, Queue

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import dev as D  # noqa: E402
import decision  # noqa: E402
from gate import PURPOSE_QUESTIONS, chunks, framed, purpose_score  # noqa: E402

REFUSED = {q: 1.0 for q in PURPOSE_QUESTIONS}  # the provider's input filter refused the text: counted as a block
key = lambda text: hashlib.sha1(text.encode()).hexdigest()[:16]
PER_KEY = 4


def answers(texts, out):
    saved = {r["key"]: r["answers"] for r in map(json.loads, out.read_text().splitlines())} if out.exists() else {}
    todo = Queue()
    for t in dict.fromkeys(texts):
        if key(t) not in saved:
            todo.put(t)
    print(f"{len(saved)} saved, {todo.qsize()} to ask", file=sys.stderr, flush=True)
    lock, f = threading.Lock(), out.open("a")

    def work(ask):
        while True:
            try:
                t = todo.get_nowait()
            except Empty:
                return
            a = ask(t).get("answers", REFUSED)
            with lock:
                saved[key(t)] = a
                f.write(json.dumps({"key": key(t), "answers": a}) + "\n")
                f.flush()
                if len(saved) % 500 == 0:
                    print(len(saved), file=sys.stderr, flush=True)

    threads = []
    for k in os.environ["GEMINI_KEYS"].split(","):
        os.environ["GEMINI_API_KEY"] = k
        threads += [threading.Thread(target=work, args=(decision.session("gemini", attempts=10, questions=PURPOSE_QUESTIONS),)) for _ in range(PER_KEY)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return lambda t: saved[key(t)]


def scored(files, out):
    get = answers([framed(row["context"], c) for _, row in files.iterrows() for c in chunks(row["text"])], out)
    one = pd.read_json(D.ROOT / "results" / "decision-gemini.jsonl", lines=True).drop_duplicates("key").set_index("key")["p"].fillna(1.0)
    mcp = (files["set"] == "mcp").values
    return pd.DataFrame({"one question": [max(one[decision.key(c)] for c in chunks(t)) for t in files["text"]],
                         "three questions": [purpose_score([get(framed(row["context"], c)) for c in chunks(row["text"])], m)
                                             for (_, row), m in zip(files.iterrows(), mcp)]}), mcp


if __name__ == "__main__":
    dev, held = D.dev_set(), D.held_out()
    dscore, dmcp = scored(dev, D.HERE / "scores" / "gemini-three.jsonl")
    hscore, hmcp = scored(held, D.HERE / "scores" / "gemini-three-heldout.jsonl")
    rows = []
    for method in dscore:
        cuts = {True: np.sort(dscore.loc[dmcp & (dev["label"] == 0).values, method].values)[-3],
                False: np.sort(dscore.loc[~dmcp & (dev["label"] == 0).values, method].values)[-3]}
        groups = {"MCP tool descriptions": (held["set"] == "mcp").values, "skill files, first set": (held["set"] == "skills").values,
                  "skill files, other sources": (held["set"] == "skillbench").values, "all connector files": np.ones(len(held), bool)}
        for group, mask in groups.items():
            y, s = held.loc[mask, "label"].values, hscore.loc[mask, method].values
            cut = np.where(hmcp[mask], cuts[True], cuts[False])
            rows.append({"group": group, "method": method, "malicious": (y == 1).sum(), "benign": (y == 0).sum(),
                         "AUROC": roc_auc_score(y, s) if group != "all connector files" else np.nan,
                         "caught at 0.5": (s[y == 1] >= .5).mean(), "benign blocked at 0.5": (s[y == 0] >= .5).mean(),
                         "caught at development threshold": (s > cut)[y == 1].mean(), "benign blocked at it": (s > cut)[y == 0].mean()})
    out = pd.DataFrame(rows)
    out.to_csv(D.ROOT / "results" / "question-variant-gemini-heldout.csv", index=False)
    pd.set_option("display.width", 250)
    print(out.round(3).to_string(index=False))
