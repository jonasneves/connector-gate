# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Connector gate: screening what an AI agent loads
#
# CYBERSEC 520 final project · Jonas Neves · November 2026
#
# An AI agent loads text it did not write: tool descriptions from MCP servers, skill files from
# public registries, and prompts. Any of them can carry instructions that turn the agent against
# its user. This notebook trains detectors on general prompt-injection data and tests them on
# malicious skill files and poisoned MCP tool descriptions that no model saw in training.
#
# | Step | Section |
# |---|---|
# | Data: one primary set, three secondary sets | 1 |
# | Model 1: TF-IDF and logistic regression | 2 |
# | Model 2: fine-tuned DistilBERT | 3 |
# | Block thresholds, fixed on validation | 4 |
# | Reference: hosted decision models, zero-shot | 5 |
# | Evaluation on the test split and the secondary sets | 6 |
# | Evasion test | 7 |
# | Latency and cost | 8 |
#
# **Threat model.** *Actor:* anyone who can publish a skill, an MCP server or a document the agent
# reads; no access to the victim is needed. *Target:* the agent's credentials and the systems it
# can act on. *Evasion:* hiding the instruction inside a long benign file, rewording it, or
# addressing the detector itself (section 7). *Operational context:* the gate runs once, when a
# connector is installed or a tool list changes, before the text reaches the agent. It complements
# scoped credentials and does not replace them.
#
# **AI use.** Claude Code (Claude Opus 5.5) wrote this notebook and `data.py`, `gate.py`,
# `decision.py`, `app.py` and `space.py` on November 22, 2026.

# %%
import json
import math
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import transformers

from decision import key
from gate import EVASION_NOTE, MAX_TOKENS, MODELS, QUESTION, chunks, device, predictor, score

transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
pd.set_option("display.max_colwidth", 200, "display.width", 200)
RESULTS = Path("results")
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)
MODELS.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Data
#
# `data.py` downloads four public datasets at pinned revisions and writes them in one format:
# `set, split, id, text, label` (1 = malicious) and `kind`.
#
# | Set | Source | What a row is | Role |
# |---|---|---|---|
# | primary | `neuralchemy/Prompt-injection-dataset` (Apache 2.0) | a prompt, attack or benign | train, validation, test |
# | skills | `TongxiQu/agent-skill-malware` (MIT) | a `SKILL.md` file from ClawHub | secondary, never trained on |
# | mcp | MCPTox, Wang et al., AAAI 2026 (CC BY 4.0) | an MCP tool description, poisoned or clean | secondary |
# | deepset | `deepset/prompt-injections` (Apache 2.0) | a chat prompt, English or German | secondary |
#
# Cleaning (in `data.py`): trim whitespace, drop empty texts, drop texts repeated within a set,
# drop texts that appear under both labels, and drop any secondary text that is also in the
# primary set.

# %%
sets = pd.read_parquet("data/sets.parquet")
summary = sets.groupby(["set", "split"], sort=False).agg(
    rows=("label", "size"), malicious=("label", "sum"),
    median_chars=("text", lambda t: int(t.str.len().median())),
    max_chars=("text", lambda t: t.str.len().max()))
summary

# %% [markdown]
# The sets differ in three ways that matter later.
#
# - **Length.** A primary prompt has a median of 56 characters; a skill file has a median near
#   4,800. The models read at most about 1,000 characters at a time, so a long text is cut into
#   overlapping 1,000-character chunks and gets the score of its worst chunk (`gate.score`).
# - **What "malicious" means.** Primary: prompt injections, jailbreaks and harmful requests. Skills:
#   files that make the agent download and run malware. MCP: a tool description with an added
#   instruction to the agent. Deepset: chat injections.
# - **Language.** Deepset is partly German. DistilBERT here is the English uncased model.

# %%
for name, group in sets.groupby("set", sort=False):
    for label in (0, 1):
        text = group.loc[group["label"] == label, "text"].iloc[0]
        print(f"[{name}, label {label}] {text[:220]!r}\n")

# %%
train = sets[sets["split"] == "train"]
validation = sets[sets["split"] == "validation"]
evaluation = sets[sets["split"] != "train"]  # validation, test and every secondary row
EVAL_SETS = {"primary test": sets["split"] == "test", "skills": sets["set"] == "skills",
             "mcp": sets["set"] == "mcp", "deepset": sets["set"] == "deepset"}

# %% [markdown]
# ## 2. Model 1: TF-IDF and logistic regression
#
# The baseline. TF-IDF turns a text into counts of its n-grams, weighted down when an n-gram is
# common everywhere. Logistic regression learns one weight per n-gram. It is fast, small and every
# weight can be read, and it cannot use word order beyond the n-gram.
#
# Tuned on the validation split: word 1-2-grams or character 3-5-grams, and the regularization
# strength `C` (smaller is stronger).

# %%
ANALYZERS = {"word 1-2": dict(analyzer="word", ngram_range=(1, 2)),
             "char 3-5": dict(analyzer="char_wb", ngram_range=(3, 5))}


def fit_baseline(analyzer, C):
    pipeline = make_pipeline(TfidfVectorizer(sublinear_tf=True, min_df=2, **ANALYZERS[analyzer]),
                             LogisticRegression(C=C, max_iter=2000))
    return pipeline.fit(train["text"], train["label"])


def probability(pipeline):
    return lambda texts: pipeline.predict_proba(texts)[:, 1]


grid = []
for analyzer in ANALYZERS:
    for C in (0.1, 1, 10, 100):
        pipeline = fit_baseline(analyzer, C)
        auroc = roc_auc_score(validation["label"], score(probability(pipeline), list(validation["text"])))
        grid.append({"analyzer": analyzer, "C": C, "validation AUROC": auroc})
grid = pd.DataFrame(grid).sort_values("validation AUROC", ascending=False)
grid

# %%
best = grid.iloc[0]
baseline = fit_baseline(best["analyzer"], best["C"])
joblib.dump(baseline, MODELS / "baseline.joblib")
vectorizer, classifier = baseline.steps[0][1], baseline.steps[1][1]
weights = pd.Series(classifier.coef_[0], index=vectorizer.get_feature_names_out()).sort_values()
print(f"kept: {best['analyzer']}, C={best['C']}, {len(weights):,} n-grams")
print("pushes toward malicious:", list(weights.index[-20:][::-1]))
print("pushes toward benign:   ", list(weights.index[:20]))

# %% [markdown]
# ## 3. Model 2: fine-tuned DistilBERT
#
# DistilBERT is a 66-million-parameter transformer pretrained on English text. Fine-tuning adds a
# two-class layer on top and updates all weights on the primary training split. Unlike the
# baseline it reads each word in the context of the others, so it can in principle recognise an
# instruction it has never seen worded that way.
#
# Settings: 256 tokens per input, batch 16, AdamW with weight decay 0.01, linear warm-up over the
# first 6% of steps then linear decay, up to 3 epochs. Two learning rates are tried, and the
# epoch with the highest validation AUROC across both is the one saved.

# %%
BASE = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(BASE)


def fine_tune(lr, best_so_far, epochs=3, batch_size=16, seed=0):
    torch.manual_seed(seed)
    order = np.random.default_rng(seed)
    model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2).to(device())
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * math.ceil(len(train) / batch_size)
    schedule = get_linear_schedule_with_warmup(optimizer, int(0.06 * steps), steps)
    texts, labels = list(train["text"]), torch.tensor(train["label"].values)
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        shuffled, losses = order.permutation(len(texts)), []
        for i in range(0, len(texts), batch_size):
            rows = shuffled[i:i + batch_size]
            batch = tokenizer([texts[j] for j in rows], truncation=True, max_length=MAX_TOKENS,
                              padding=True, return_tensors="pt").to(model.device)
            loss = model(**batch, labels=labels[rows].to(model.device)).loss
            loss.backward()
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        auroc = roc_auc_score(validation["label"], score(predictor(model, tokenizer), list(validation["text"])))
        log.append({"lr": lr, "epoch": epoch, "train loss": np.mean(losses), "validation AUROC": auroc})
        print(log[-1], flush=True)
        if auroc > best_so_far:
            best_so_far = auroc
            model.save_pretrained(MODELS / "distilbert")
            tokenizer.save_pretrained(MODELS / "distilbert")
    return log, best_so_far


started, training_log, best_auroc = time.time(), [], 0.0
for lr in (2e-5, 5e-5):
    log, best_auroc = fine_tune(lr, best_auroc)
    training_log += log
print(f"trained in {(time.time() - started) / 60:.1f} minutes on {device()}")
pd.DataFrame(training_log)

# %% [markdown]
# ## 4. Scores and block thresholds
#
# Every model gives each text a score from 0 to 1. The gate blocks a text when its score reaches
# the model's threshold. Each threshold is set once, on the primary validation split: the lowest
# value at which at most 1% of benign validation texts would be blocked. It is then applied
# unchanged to the test split and to the secondary sets, as it would be in deployment, where
# nobody has labels for the new traffic.

# %%
def block_threshold(scores, labels, max_fpr=0.01):
    benign = scores[labels == 0]
    candidates = np.append(np.unique(scores), np.nextafter(scores.max(), 2))
    return float(min(t for t in candidates if (benign >= t).mean() <= max_fpr))


from gate import baseline as load_baseline, distilbert as load_distilbert  # the saved models, as the app loads them

predictors = {"logistic regression": load_baseline(), "DistilBERT": load_distilbert()}
scores = pd.DataFrame({name: score(predict, list(evaluation["text"])) for name, predict in predictors.items()},
                      index=evaluation.index)

# %% [markdown]
# ## 5. Reference: hosted decision models, zero-shot
#
# A decision model answers a yes/no question about a text with a probability, without training.
# `decision.py` sent every chunk to two of them, Jev 1.14 and Gemini Decision 1, with the one
# question below, written before any result was seen. They are a reference for what a detector
# with no training data achieves. They get thresholds by the same rule. When a provider's input filter refuses
# a text, the gate counts that as a block (score 1).

# %%
print(QUESTION)
decision_rows = {}
for name, file in {"Jev 1.14": "decision-jev.jsonl", "Gemini Decision 1": "decision-gemini.jsonl"}.items():
    rows = pd.read_json(RESULTS / file, lines=True).drop_duplicates("key").set_index("key")
    print(f"{name}: {len(rows):,} chunks scored, {rows['p'].isna().sum()} refused by the provider's input filter")
    rows["p"] = rows["p"].fillna(1.0)
    decision_rows[name] = rows
    predictors[name] = lambda pieces, rows=rows: rows.loc[[key(piece) for piece in pieces], "p"].values
    scores[name] = score(predictors[name], list(evaluation["text"]))

on_validation = evaluation["split"] == "validation"
thresholds = {name: {"validation": block_threshold(scores.loc[on_validation, name].values,
                                                   evaluation.loc[on_validation, "label"].values),
                     "half": 0.5} for name in scores}
(MODELS / "thresholds.json").write_text(json.dumps(thresholds, indent=1))
scores.join(evaluation[["set", "split", "id", "label", "kind"]]).to_parquet(RESULTS / "scores.parquet")
pd.DataFrame(thresholds).T

# %% [markdown]
# `validation` is the threshold from the rule above and is the one the protocol commits to.
# `half` is 0.5, the untuned reading of a probability. It was added after the first results, when
# the validation thresholds turned out to sit at the far end of each model's scale (see the table
# above), and it is reported next to the committed one for every model, not in place of it.

# %% [markdown]
# ## 6. Evaluation
#
# - **AUROC**: how well the score ranks malicious above benign, at no particular threshold. 0.5 is chance.
# - **Recall**: the share of malicious texts blocked. A miss is a poisoned connector reaching the agent.
# - **FPR** (false positive rate): the share of benign texts blocked. Each one is a legitimate tool or skill a team cannot install.
# - **Precision**: of the texts blocked, the share that were malicious.

# %%
def metrics(scored, labels, at):
    row = {"AUROC": roc_auc_score(labels, scored)}
    for name, threshold in at.items():
        blocked = scored >= threshold
        suffix = "" if name == "validation" else " at 0.5"
        row |= {f"recall{suffix}": blocked[labels == 1].mean(), f"FPR{suffix}": blocked[labels == 0].mean(),
                f"precision{suffix}": labels[blocked].mean() if blocked.any() else float("nan")}
    return row


table = pd.DataFrame([{"set": set_name, "model": model, "malicious": int(evaluation.loc[rows[evaluation.index], "label"].sum()),
                       "benign": int((1 - evaluation.loc[rows[evaluation.index], "label"]).sum()),
                       **metrics(scores.loc[rows[evaluation.index], model].values,
                                 evaluation.loc[rows[evaluation.index], "label"].values, thresholds[model])}
                      for set_name, rows in EVAL_SETS.items() for model in scores])
table.to_csv(RESULTS / "metrics.csv", index=False)
table.set_index(["set", "model"]).round(3)

# %%
COLORS = dict(zip(scores, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})

figure, axes = plt.subplots(1, 4, figsize=(15, 4.1), sharey=True)
for axis, (set_name, rows) in zip(axes, EVAL_SETS.items()):
    rows = rows[evaluation.index]
    for model in scores:
        fpr, tpr, _ = roc_curve(evaluation.loc[rows, "label"], scores.loc[rows, model])
        axis.plot(fpr, tpr, color=COLORS[model], linewidth=2, label=model)
    axis.plot([0, 1], [0, 1], color="#c3c2b7", linewidth=1, linestyle=(0, (4, 4)))
    axis.set(title=set_name, xlabel="false positive rate", xlim=(0, 1), ylim=(0, 1.005))
axes[0].set_ylabel("recall")
axes[0].legend(loc="lower right")
figure.suptitle("ROC by set. Trained models saw only the primary set; decision models saw none.", x=0.008, ha="left", fontweight="bold")
figure.tight_layout()
figure.savefig(RESULTS / "figures" / "roc.png")

# %%
def bars(suffix, title, filename):
    figure, axes = plt.subplots(1, 2, figsize=(15, 4.3), sharey=True)
    width = 0.21
    for axis, measure, name in zip(axes, ("recall", "FPR"), ("Malicious texts blocked (recall)", "Benign texts blocked (false positive rate)")):
        for i, model in enumerate(scores):
            values = table.loc[table["model"] == model].set_index("set").loc[list(EVAL_SETS), measure + suffix]
            drawn = axis.bar(np.arange(len(EVAL_SETS)) + (i - 1.5) * width, values, width * 0.9, color=COLORS[model], label=model)
            axis.bar_label(drawn, [f"{v:.0%}" for v in values], fontsize=7, color=MUTED, padding=2)
        axis.set(title=name, ylim=(0, 1.1), xticks=np.arange(len(EVAL_SETS)), xticklabels=list(EVAL_SETS))
        axis.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        axis.grid(axis="x", visible=False)
    figure.suptitle(title, x=0.008, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4)
    figure.savefig(RESULTS / "figures" / filename)


bars("", "Blocking at the threshold fixed on primary validation (at most 1% of benign validation texts blocked)", "recall-fpr.png")
bars(" at 0.5", "Blocking at a score of 0.5", "recall-fpr-half.png")

# %%
figure, axes = plt.subplots(len(scores.columns), 3, figsize=(13, 2.1 * len(scores.columns)), sharex=True)
bins = np.linspace(0, 1, 26)
for row, model in zip(axes, scores):
    for axis, set_name in zip(row, ("skills", "mcp", "deepset")):
        rows = EVAL_SETS[set_name][evaluation.index]
        for label, color, name in ((0, "#898781", "benign"), (1, "#d03b3b", "malicious")):
            axis.hist(scores.loc[rows & (evaluation["label"] == label), model], bins=bins, color=color, alpha=0.75, label=name)
        axis.axvline(thresholds[model]["validation"], color=INK, linewidth=1)
        axis.axvline(0.5, color=INK, linewidth=1, linestyle=(0, (4, 4)))
        axis.set_title(f"{model} on {set_name}", fontsize=9)
        axis.grid(axis="x", visible=False)
axes[0, 0].legend()
for axis in axes[-1]:
    axis.set_xlabel("score (solid line: validation threshold, dashed: 0.5)")
figure.tight_layout()
figure.savefig(RESULTS / "figures" / "scores.png")

# %% [markdown]
# ### What the trained models learned
#
# Section 2 printed the n-grams with the largest weights in the baseline. The rows below are the
# primary validation rows where the dataset's label and the zero-shot answer of Jev 1.14 disagree
# most: first rows labeled benign that it scores high, then rows labeled attack that it scores low.

# %%
rows = evaluation[on_validation].assign(jev=scores.loc[on_validation, "Jev 1.14"])
for label, ordered in ((0, rows[rows["label"] == 0].nlargest(6, "jev")), (1, rows[rows["label"] == 1].nsmallest(6, "jev"))):
    print("labeled", "attack" if label else "benign")
    for _, row in ordered.iterrows():
        print(f"   {row['jev']:.2f}  [{row['kind']}]  {row['text'][:130]!r}")

# %% [markdown]
# ### Is the MCP result a length effect?
#
# In the MCP set a poisoned description is longer than a clean one (the attacker's sentence is
# added to it). A detector could score well there by measuring length. The check: use the number
# of characters as the score and compare its AUROC with the models'.

# %%
for set_name in ("mcp", "skills", "deepset"):
    group = sets[sets["set"] == set_name]
    print(f"{set_name}: AUROC of text length alone = {roc_auc_score(group['label'], group['text'].str.len()):.3f}"
          f"   median chars benign {int(group.loc[group['label'] == 0, 'text'].str.len().median())},"
          f" malicious {int(group.loc[group['label'] == 1, 'text'].str.len().median())}")

# %% [markdown]
# ## 7. Evasion test
#
# The attacker knows a detector reads the text, so they add a sentence addressed to it. The note
# below is appended to every malicious MCP description, and each model scores the result at its
# unchanged thresholds. A drop in recall is the share of attacks this one sentence gets through.

# %%
print(repr(EVASION_NOTE))
poisoned = sets[(sets["set"] == "mcp") & (sets["label"] == 1)]
evasion = []
for model, predict in predictors.items():
    with_note = score(predict, list(poisoned["text"] + EVASION_NOTE))
    row = {"model": model, "median score": scores.loc[poisoned.index, model].median(), "median score with the note": np.median(with_note)}
    for name, threshold in thresholds[model].items():
        at = "" if name == "validation" else " at 0.5"
        row |= {f"recall{at}": (scores.loc[poisoned.index, model] >= threshold).mean(),
                f"recall{at} with the note": (with_note >= threshold).mean()}
    evasion.append(row)
evasion = pd.DataFrame(evasion)
evasion.to_csv(RESULTS / "evasion.csv", index=False)
evasion.set_index("model").round(3)

# %% [markdown]
# ## 8. Latency and cost
#
# Local models are timed on this machine over 200 MCP descriptions, one at a time, as the gate
# would receive them. Hosted models report the time of each request as recorded by `decision.py`,
# network included. Jev 1.14 is priced at $0.042 per million input tokens.

# %%
sample = list(sets.loc[sets["set"] == "mcp", "text"].head(200))
latency = []
for model in ("logistic regression", "DistilBERT"):
    predictors[model](sample[:8])  # the first call pays one-time setup
    times = []
    for text in sample:
        start = time.perf_counter()
        score(predictors[model], [text])
        times.append((time.perf_counter() - start) * 1000)
    latency.append({"model": model, "median ms": np.median(times), "p95 ms": np.percentile(times, 95)})
for model, rows in decision_rows.items():
    latency.append({"model": model, "median ms": rows["ms"].median(), "p95 ms": rows["ms"].quantile(0.95),
                    "median input tokens": rows["input_tokens"].median()})
latency = pd.DataFrame(latency)
latency.loc[latency["model"] == "Jev 1.14", "USD per 1,000 requests"] = latency["median input tokens"] * 0.042 / 1e6 * 1000
latency.to_csv(RESULTS / "latency.csv", index=False)
latency.set_index("model").round(4)

# %% [markdown]
# ## 9. Findings
#
# Numbers are from the tables above (run of November 22, 2026).
#
# 1. **Both trained models score well on their own test split.** AUROC 0.995 (logistic
#    regression) and 0.997 (DistilBERT), and at the validation threshold each blocks 90% of attacks
#    while blocking under 1% of benign prompts.
# 2. **Neither transfers to the unseen sets.** Their AUROC there is 0.60 to 0.82. At its
#    validation threshold DistilBERT blocks every malicious skill file and also 99.6% of the benign
#    ones. Text length alone ranks MCP descriptions at 0.74 and deepset prompts at 0.81, about the
#    same as DistilBERT (0.74 and 0.82) and above the baseline (0.63 and 0.67).
# 3. **The zero-shot decision models rank all three unseen sets at 0.95 to 0.99**, with no training
#    data. Their weakest set is the primary one (0.94 and 0.96). Section 6 lists primary rows where
#    the label and Jev 1.14's answer disagree most; whether label noise accounts for the lower
#    AUROC was not tested.
# 4. **A threshold does not transfer either.** The 1%-on-validation rule puts the decision models'
#    thresholds near 0.98, and at that threshold they block 2% to 23% of malicious texts in the
#    skills and MCP sets. At 0.5, Jev 1.14 blocks 91% of malicious skill files with 1.8% of benign
#    ones, and 58% of poisoned MCP descriptions with none of the clean ones. The 0.5 threshold was
#    added after seeing the first result, so it is an observation to confirm on new data, not a
#    tuned operating point.
# 5. **The baseline's heaviest weights are pieces of one phrase**, "I have been PWNED", which
#    attackers had to produce in the HackAPrompt competition (a source of the primary set, per its
#    dataset card). That is a property of the dataset. It does not explain the false blocks on
#    skill files, which do not contain the phrase, and nothing here shows what DistilBERT relies
#    on. Two candidates were not examined: taking the maximum over a file's chunks (a median of 6
#    per skill file), and the shift from 56-character training prompts to 1,000-character chunks.
# 6. **The appended note worked only at one threshold.** It raised the median score of every
#    model, and at 0.5 no model lost recall. At the validation thresholds, where the decision
#    models block little to begin with, their recall fell (Jev 1.14 from 1.9% to 1.4%, Gemini
#    Decision 1 from 9.1% to 0.2%).
# 7. **Latency and cost:** the local models answer one MCP description in a median of 7 ms or
#    less. Jev 1.14 answered one chunk in a median of 58 ms at about $0.018 per 1,000 chunks; a
#    median skill file is 6 chunks.
#
# **Limits.** One primary dataset with noisy labels; one run of DistilBERT per learning rate; the
# MCP positives are synthetic (written for a benchmark) and longer than the clean descriptions;
# the skills set comes from one registry and two campaigns; the hosted models can change without
# notice; the evasion test is one fixed sentence, not an adaptive attacker.
