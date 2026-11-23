"""Publishes the app as a Hugging Face Docker Space: the app, the scoring code and the saved models.

    hf auth login
    TYPESAFE_API_KEY=... uv run space.py <user>/connector-gate

With TYPESAFE_API_KEY set, the key is stored as a Space secret and the app also offers Jev 1.14.
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

CARD = """---
title: Connector gate
emoji: 🚪
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
short_description: Screens MCP tool descriptions and skill files
---

Screens the text an AI agent loads from a connector (MCP tool descriptions, skill files) for
instructions that turn the agent against its user. Two models trained on a public prompt-injection
dataset (TF-IDF with logistic regression, and a fine-tuned DistilBERT) run in this container.
Jev 1.14, a hosted decision model, answers zero-shot through its provider's API.

CYBERSEC 520 final project, Duke University, November 2026.
"""

if __name__ == "__main__":
    repo, api, root = sys.argv[1], HfApi(), Path(__file__).parent
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    if os.environ.get("TYPESAFE_API_KEY"):
        api.add_space_secret(repo, "TYPESAFE_API_KEY", os.environ["TYPESAFE_API_KEY"])
    api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md", repo_id=repo, repo_type="space")
    api.upload_folder(folder_path=root, repo_id=repo, repo_type="space",
                      allow_patterns=["app.py", "gate.py", "decision.py", "requirements.txt", "Dockerfile",
                                      "models/baseline.joblib", "models/thresholds.json", "models/distilbert/*"])
    print(f"https://huggingface.co/spaces/{repo}")
