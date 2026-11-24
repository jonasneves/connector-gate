"""Downloads the five datasets at pinned revisions and writes them as one table, data/sets.parquet.

Columns: set, split, id, text, label (1 = malicious), kind.
The primary set keeps its publisher's train, validation and test splits. Every row of a
secondary set has split "unseen": no model is trained or tuned on it.

    uv run data.py
"""
import json
import re
import urllib.request
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "sets.parquet"

PRIMARY = ("neuralchemy/Prompt-injection-dataset", "7d70432dfcf47a821612cbf9d34e9d9e3ad20e75")
SKILLS = ("TongxiQu/agent-skill-malware", "68717b9758e78f68679c89e993a855c9d7ccddcb")
DEEPSET = ("deepset/prompt-injections", "4f61ecb038e9c3fb77e21034b22511b523772cdd")
SKILLBENCH = ("ProtectSkills/MaliciousSkillBench", "d4b42ce5766a6e0359c987cf59c1007cb3795a90")
MCPTOX = "https://raw.githubusercontent.com/zhiqiangwang4/MCPTox-Benchmark/630121b3275f74cc72d000a7a056e4d3c68648ef/"


def hub(source, filename):
    repo, revision = source
    return hf_hub_download(repo, filename, repo_type="dataset", revision=revision,
                           local_dir=RAW / repo.replace("/", "_"))


def mcptox(filename):
    path = RAW / "mcptox" / filename
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MCPTOX + filename, path)
    return json.loads(path.read_text())


def primary():
    """Prompts labeled attack or benign: 29 attack categories from HackAPrompt, HarmBench and the publisher's own."""
    for split in ("train", "validation", "test"):
        df = pd.read_parquet(hub(PRIMARY, f"core/{split}-00000-of-00001.parquet"))
        yield pd.DataFrame({"set": "primary", "split": split, "id": df["group_id"].astype(str), "text": df["text"],
                            "label": df["label"], "kind": df["category"]})


def skills():
    """SKILL.md files from ClawHub: malicious ones from the February 2026 campaigns, benign community skills."""
    df = pd.read_json(hub(SKILLS, "skills.jsonl"), lines=True)
    return pd.DataFrame({"set": "skills", "split": "unseen", "id": df["id"], "text": df["content"],
                         "label": (df["label"] == "malicious").astype(int), "kind": df["label"]})


def skillbench():
    """Skill files from MaliciousSkillBench: the test part of its source-disjoint split, which is three sources
    (SkillHarm, ATR Skill Security Benchmark, SkillFortifyBench) and none of them the source of the skills set.
    `kind` is the source and how the file came to be: wild, injected, backdoored, synthetic or test_fixture."""
    df = pd.read_parquet(hub(SKILLBENCH, "primary.parquet"))
    split = pd.read_parquet(hub(SKILLBENCH, "splits/source_disjoint.parquet"))
    df = df[df["benchmark_id"].isin(split.loc[split["split"] == "test", "benchmark_id"])]
    return pd.DataFrame({"set": "skillbench", "split": "unseen", "id": df["benchmark_id"], "text": df["skill_text"],
                         "label": df["label"].astype(int), "kind": df["source_name"] + ", " + df["provenance"]})


def mcp():
    """MCP tool descriptions: MCPTox's poisoned ones, and the clean ones of the real servers it was built on."""
    poisoned = [tool for server in mcptox("pure_tool.json") for tool in server.values()]
    rows = [{"id": tool["tool_address"], "text": tool["tool_content"], "label": 1, "kind": tool["paradigm"]}
            for tool in poisoned]
    for server in mcptox("response_all.json")["servers"].values():
        # The clean descriptions exist only inside the system prompt MCPTox gave its agents.
        listed = re.findall(r"Tool: (\S+)\nDescription: (.*?)\nArguments:", server["clean_system_promot"], flags=re.S)
        rows += [{"id": f"{server['server_name']}/{name}", "text": text, "label": 0, "kind": "clean"}
                 for name, text in listed]
    df = pd.DataFrame(rows)
    # Some poisoned descriptions carry the argument list that followed them in the prompt, with the
    # newline real or written as backslash-n; clean ones never do. Tools with no description say "None".
    df["text"] = df["text"].str.split(r"(?:\n|\\n)Arguments:", regex=True).str[0]
    return df[df["text"].str.strip() != "None"].assign(set="mcp", split="unseen")


def deepset():
    """Chat prompts labeled injection or not, in English and German. Both published splits are unseen here."""
    df = pd.concat(pd.read_parquet(hub(DEEPSET, f"data/{name}")) for name in
                   ("train-00000-of-00001-9564e8b05b4757ab.parquet", "test-00000-of-00001-701d16158af87368.parquet"))
    df = df.reset_index(drop=True)
    return pd.DataFrame({"set": "deepset", "split": "unseen", "id": df.index.astype(str), "text": df["text"],
                         "label": df["label"], "kind": df["label"].map({0: "benign", 1: "injection"})})


def clean(df):
    """Trims whitespace, then drops empty texts, repeated texts within a set, and any secondary
    text that also appears in the primary set. Prints what it dropped."""
    before = df.groupby("set", sort=False).size()
    df = df.assign(text=df["text"].str.strip())
    df = df[df["text"].str.len() > 0]
    # A text repeated under both labels has no usable label, so every copy goes.
    conflicting = df.groupby(["set", "text"])["label"].transform("nunique") > 1
    df = df[~conflicting].drop_duplicates(["set", "text"])
    seen = set(df.loc[df["set"] == "primary", "text"])
    df = df[(df["set"] == "primary") | ~df["text"].isin(seen)]
    after = df.groupby("set", sort=False).size()
    print(pd.DataFrame({"downloaded": before, "kept": after, "dropped": before - after}).to_string())
    return df.reset_index(drop=True)


if __name__ == "__main__":
    sets = clean(pd.concat([*primary(), skills(), mcp(), deepset(), skillbench()], ignore_index=True))
    sets.to_parquet(OUT, index=False)
    print(sets.groupby(["set", "split"], sort=False)["label"].agg(rows="size", malicious="sum").to_string())
    print(f"wrote {OUT.relative_to(ROOT)}")
