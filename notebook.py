# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.7
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
# malicious skill files and poisoned MCP tool descriptions that no model here was trained on.
#
# | Step | Section |
# |---|---|
# | Data: one primary set, four secondary sets | 1 |
# | Model 1: TF-IDF and logistic regression | 2 |
# | Model 2: fine-tuned DistilBERT | 3 |
# | Block thresholds, fixed on validation | 4 |
# | Reference: two off-the-shelf detectors, as published | 4 |
# | Reference: hosted decision models, zero-shot | 5 |
# | Evaluation on the test split and the secondary sets, with intervals | 6 |
# | A threshold set on connector data; false blocks per server | 6 |
# | Evasion test: one note, then an attacker who picks among six | 7 |
# | Latency and cost | 8 |
# | Findings and limits | 9 |
# | Operating the gate | 10 |
#
# **Threat model.** *Actor:* anyone who can publish a skill, an MCP server or a document the agent
# reads; no access to the victim is needed. *Target:* the agent's credentials and the systems it
# can act on. *Evasion:* hiding the instruction inside a long benign file, rewording it, or
# addressing the detector itself (section 7). *Operational context:* the gate runs once, when a
# connector is installed or a tool list changes, before the text reaches the agent. It complements
# scoped credentials and does not replace them.
#
# **AI use.** Claude Code wrote this notebook and `data.py`, `gate.py`, `decision.py` and `app.py`
# (Claude Opus 5.5 on November 22, 2026, and Claude Fable 5.2 on November 23).

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
from gate import EVASION_NOTE, EVASION_VARIANTS, MAX_TOKENS, MODELS, QUESTION, chunks, device, predictor, score

transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
pd.set_option("display.max_colwidth", 200, "display.width", 200)
RESULTS = Path("results")
(RESULTS / "figures").mkdir(parents=True, exist_ok=True)
MODELS.mkdir(exist_ok=True)

# %% [markdown]
# ## 1. Data
#
# `data.py` downloads five public datasets at pinned revisions and writes them in one format:
# `set, split, id, text, label` (1 = malicious) and `kind`.
#
# | Set | Source | What a row is | Role |
# |---|---|---|---|
# | primary | `neuralchemy/Prompt-injection-dataset` (Apache 2.0) | a prompt, attack or benign | train, validation, test |
# | skills | `TongxiQu/agent-skill-malware` (MIT) | a `SKILL.md` file from ClawHub | secondary, never trained on |
# | mcp | MCPTox, Wang et al., AAAI 2026 (CC BY 4.0) | an MCP tool description, poisoned or clean | secondary |
# | deepset | `deepset/prompt-injections` (Apache 2.0) | a chat prompt, English or German | secondary |
# | skillbench | MaliciousSkillBench, Wang et al., 2026, [arXiv:2608.19901](https://arxiv.org/abs/2608.19901) (CC BY 4.0 for the benchmark; each skill file keeps its source's terms) | a skill file from one of three sources | secondary |
#
# The skillbench set is the test part of the benchmark's source-disjoint split: 728 malicious files
# from SkillHarm, 455 benign and 21 malicious from the ATR Skill Security Benchmark, and 90 of each
# from SkillFortifyBench. None of the three is the source of the skills set, and the benchmark's
# authors and one scanner vendor publish results on exactly these 1,384 files, so the models here
# can be read against those. It was added on November 23, after the results on the other sets were
# seen, and nothing was changed after its results were seen.
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
             "mcp": sets["set"] == "mcp", "deepset": sets["set"] == "deepset", "skillbench": sets["set"] == "skillbench"}

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
# ### Reference: two off-the-shelf detectors
#
# Both are published prompt-injection classifiers, used here as downloaded at a pinned revision,
# with no training. They stand in for what a team would get by installing one. Each gets a
# threshold by the same rule as every other model, and is scored through the same 1,000-character
# chunks.
#
# - **ProtectAI v2** (`protectai/deberta-v3-base-prompt-injection-v2`, a fine-tuned
#   DeBERTa-v3-base, 2024). The 2026 papers that re-evaluate detectors compare against it. Its card
#   says it was trained on 22 public datasets and names only some of them, so whether it saw any
#   text from the sets evaluated here is unknown. It is English-only, and its card now says the
#   project is archived.
# - **Horizon Guard** (`Horizon-Labs/prompt-injection-guard-small`, a fine-tuned mmBERT-small with
#   141 million parameters, October 2026). It was trained on documents and tool outputs with
#   planted injections, which is closer to connector content than chat prompts are, and it is
#   multilingual. Its card lists this notebook's primary dataset and the train split of
#   `deepset/prompt-injections` among its training data, so its results on the primary test split
#   and on deepset are not held-out results. Its card lists neither the skills set nor MCPTox.

# %%
from gate import HORIZON, PROTECTAI, published

for name, source in {"ProtectAI v2": PROTECTAI, "Horizon Guard": HORIZON}.items():
    predictors[name] = published(source)
    scores[name] = score(predictors[name], list(evaluation["text"]))

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
# - **F1**: the harmonic mean of precision and recall. It moves with a set's class balance, so it
#   is given beside the confusion matrices below and the findings rest on recall and FPR.

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

# %% [markdown]
# ### Confusion matrices
#
# The four counts behind each row above, at both thresholds. TP is a malicious text blocked, FN a
# malicious text allowed, FP a benign text blocked and TN a benign text allowed.

# %%
confusion = []
for set_name, rows in EVAL_SETS.items():
    labels = evaluation.loc[rows[evaluation.index], "label"].values
    for model in scores:
        for name, threshold in thresholds[model].items():
            blocked = scores.loc[rows[evaluation.index], model].values >= threshold
            tp, fp = int((blocked & (labels == 1)).sum()), int((blocked & (labels == 0)).sum())
            fn, tn = int((~blocked & (labels == 1)).sum()), int((~blocked & (labels == 0)).sum())
            confusion.append({"set": set_name, "model": model, "threshold": name if name == "validation" else "0.5",
                              "TP": tp, "FN": fn, "FP": fp, "TN": tn, "F1": 2 * tp / (2 * tp + fp + fn)})
confusion = pd.DataFrame(confusion)
confusion.to_csv(RESULTS / "confusion.csv", index=False)
confusion.set_index(["set", "model", "threshold"]).round(3)

# %% [markdown]
# ### How much the sample sizes allow
#
# The secondary sets are small (347 skill files, 810 MCP descriptions), so every rate above carries
# sampling error. The table gives a 95% interval for AUROC and for recall and FPR at 0.5, from
# 1,000 bootstrap resamples drawn within each label. The intervals cover sampling error only.
# They say nothing about a new registry or a new campaign.

# %%
def interval(values):
    low, high = np.percentile(values, [2.5, 97.5])
    return f"{low:.3f} to {high:.3f}"


resample = np.random.default_rng(0)
intervals = []
for set_name, rows in EVAL_SETS.items():
    labels = evaluation.loc[rows[evaluation.index], "label"].values
    benign, malicious = np.flatnonzero(labels == 0), np.flatnonzero(labels == 1)
    draws = [np.concatenate([resample.choice(benign, len(benign)), resample.choice(malicious, len(malicious))])
             for _ in range(1000)]
    for model in scores:
        scored = scores.loc[rows[evaluation.index], model].values
        blocked = scored >= thresholds[model]["half"]
        intervals.append({"set": set_name, "model": model,
                          "AUROC": interval([roc_auc_score(labels[d], scored[d]) for d in draws]),
                          "recall at 0.5": interval([blocked[d][labels[d] == 1].mean() for d in draws]),
                          "FPR at 0.5": interval([blocked[d][labels[d] == 0].mean() for d in draws])})
intervals = pd.DataFrame(intervals)
intervals.to_csv(RESULTS / "intervals.csv", index=False)
intervals.set_index(["set", "model"])

# %%
COLORS = {"logistic regression": "#2a78d6", "DistilBERT": "#eb6834", "ProtectAI v2": "#8a63d2",
          "Horizon Guard": "#d6408f", "Jev 1.14": "#1baf7a", "Gemini Decision 1": "#eda100"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
                     "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight"})

figure, axes = plt.subplots(1, len(EVAL_SETS), figsize=(18, 4.1), sharey=True)
for axis, (set_name, rows) in zip(axes, EVAL_SETS.items()):
    rows = rows[evaluation.index]
    for model in scores:
        fpr, tpr, _ = roc_curve(evaluation.loc[rows, "label"], scores.loc[rows, model])
        axis.plot(fpr, tpr, color=COLORS[model], linewidth=2, label=model)
    axis.plot([0, 1], [0, 1], color="#c3c2b7", linewidth=1, linestyle=(0, (4, 4)))
    axis.set(title=set_name, xlabel="false positive rate", xlim=(0, 1), ylim=(0, 1.005))
axes[0].set_ylabel("recall")
axes[0].legend(loc="lower right")
figure.suptitle("ROC by set. The two trained models saw only the primary set; the other four were not trained here.", x=0.008, ha="left", fontweight="bold")
figure.tight_layout()
figure.savefig(RESULTS / "figures" / "roc.png")

# %%
def bars(suffix, title, filename):
    figure, axes = plt.subplots(1, 2, figsize=(15, 4.3), sharey=True)
    count = len(scores.columns)
    width = 0.84 / count
    for axis, measure, name in zip(axes, ("recall", "FPR"), ("Malicious texts blocked (recall)", "Benign texts blocked (false positive rate)")):
        for i, model in enumerate(scores):
            values = table.loc[table["model"] == model].set_index("set").loc[list(EVAL_SETS), measure + suffix]
            drawn = axis.bar(np.arange(len(EVAL_SETS)) + (i - (count - 1) / 2) * width, values, width * 0.9, color=COLORS[model], label=model)
            axis.bar_label(drawn, [f"{v:.0%}" for v in values], fontsize=6, color=MUTED, padding=2)
        axis.set(title=name, ylim=(0, 1.1), xticks=np.arange(len(EVAL_SETS)), xticklabels=list(EVAL_SETS))
        axis.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        axis.grid(axis="x", visible=False)
    figure.suptitle(title, x=0.008, ha="left", fontweight="bold")
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=count)
    figure.savefig(RESULTS / "figures" / filename)


bars("", "Blocking at the threshold fixed on primary validation (at most 1% of benign validation texts blocked)", "recall-fpr.png")
bars(" at 0.5", "Blocking at a score of 0.5", "recall-fpr-half.png")

# %%
figure, axes = plt.subplots(len(scores.columns), 4, figsize=(17, 2.1 * len(scores.columns)), sharex=True)
bins = np.linspace(0, 1, 26)
for row, model in zip(axes, scores):
    for axis, set_name in zip(row, ("skills", "skillbench", "mcp", "deepset")):
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
# ### The second skills set, by source
#
# The skillbench set has three sources, and they are not alike. SkillHarm supplies malicious files
# only: real skills with a harmful instruction injected into the text (570) or with a backdoor in
# a file the skill ships with (158). The ATR benchmark supplies nearly all the benign files, taken
# from public registries. So a model could separate the set by telling the sources apart.
# SkillFortifyBench has 90 files of each label from one generator, which makes it the one source
# where both labels share an origin. Its files are short specifications marked as inert, and the
# table counts how many files of each label carry that marker. `kind` names the source and how
# the file came to be. Recall and FPR are at 0.5.
#
# Published results on the same 1,384 files, for reference: a word TF-IDF SVM trained on the
# benchmark's other sources blocks 95.6% of malicious and 62.4% of benign files
# ([arXiv:2608.19901](https://arxiv.org/abs/2608.19901)). Cisco's skill-scanner blocks 7.7% and
# 4.0% with its rules alone, and with an LLM judge sends 66.7% of malicious and 15.4% of benign
# files to review ([its README](https://github.com/cisco-ai-defense/skill-scanner), read November 23, 2026).

# %%
bench = evaluation[evaluation["set"] == "skillbench"]
print(pd.crosstab(bench["kind"], bench["label"], rownames=["source"]).assign(
    marked_inert=bench.groupby("kind")["text"].apply(lambda t: int(t.str.contains("SKILLFORTIFYBENCH:INERT").sum()))).to_string())
by_source = []
for model in scores:
    blocked = scores.loc[bench.index, model] >= thresholds[model]["half"]
    row = {"model": model}
    for source, group in bench.groupby("kind"):
        for label, name in ((1, "recall"), (0, "FPR")):
            part = blocked[group.index[group["label"] == label]]
            if len(part):
                row[f"{source} {name}"] = part.mean()
    fortify = bench[bench["kind"].str.startswith("SkillFortifyBench")]
    row["SkillFortifyBench AUROC"] = roc_auc_score(fortify["label"], scores.loc[fortify.index, model])
    by_source.append(row)
by_source = pd.DataFrame(by_source)
by_source.to_csv(RESULTS / "skillbench-sources.csv", index=False)
by_source.set_index("model").round(3).T

# %% [markdown]
# The malicious SkillHarm file that Jev 1.14 scores lowest, at its highest-scoring chunk:

# %%
harm = bench[bench["kind"].str.startswith("SkillHarm")]
lowest = harm.loc[scores.loc[harm.index, "Jev 1.14"].idxmin()]
piece = max(chunks(lowest["text"]), key=lambda piece: predictors["Jev 1.14"]([piece])[0])
print(f"{lowest['id']} ({lowest['kind']}), score {scores.loc[lowest.name, 'Jev 1.14']:.2f}, {len(lowest['text']):,} characters\n")
print(piece[:700])

# %% [markdown]
# ### Do disputed labels set the threshold?
#
# The 1% rule lets 4 of the 407 benign validation prompts score above the threshold. The rows
# printed above include prompts labeled benign that read as jailbreaks. The check: drop the benign
# validation prompts that both decision models score at 0.5 or above, set every threshold again by
# the same rule, and apply it to the test split and the two connector sets. For the two decision
# models the check is circular, since their own scores choose the rows. For the other four models
# it is not. This was added on November 23, after the results above were seen.

# %%
on_rows = evaluation[on_validation]
disputed = (on_rows["label"] == 0) & (scores.loc[on_rows.index, "Jev 1.14"] >= 0.5) & (scores.loc[on_rows.index, "Gemini Decision 1"] >= 0.5)
kept = on_rows[~disputed]
print(f"{int(disputed.sum())} of {int((on_rows['label'] == 0).sum())} benign validation prompts are scored 0.5 or above by both decision models")
for text in on_rows.loc[disputed, "text"].str[:70].drop_duplicates().head(6):
    print("  ", repr(text))
relabeled = []
for model in scores:
    threshold = block_threshold(scores.loc[kept.index, model].values, kept["label"].values)
    row = {"model": model, "validation threshold": thresholds[model]["validation"], "without the disputed rows": threshold}
    for set_name in ("primary test", "skills", "mcp"):
        rows = EVAL_SETS[set_name][evaluation.index]
        blocked, labels = scores.loc[rows, model] >= threshold, evaluation.loc[rows, "label"]
        row |= {f"{set_name} recall": blocked[labels == 1].mean(), f"{set_name} FPR": blocked[labels == 0].mean()}
    relabeled.append(row)
relabeled = pd.DataFrame(relabeled)
relabeled.to_csv(RESULTS / "disputed-labels.csv", index=False)
relabeled.set_index("model").round(3)

# %% [markdown]
# ### Is the skills result an effect of taking the worst chunk?
#
# A skill file is scored by its highest-scoring chunk, and a median file has 6. If single chunks
# were rarely blocked, the false blocks on whole files would come from taking the maximum. The
# check: the share of single chunks of benign skill files that reach each threshold, beside the
# share of whole files.

# %%
benign_skills = evaluation[(evaluation["set"] == "skills") & (evaluation["label"] == 0)]
pieces = [piece for text in benign_skills["text"] for piece in chunks(text)]
by_chunk = []
for model, predict in predictors.items():
    chunk_scores = np.asarray(predict(pieces), dtype=float)
    for name, threshold in thresholds[model].items():
        by_chunk.append({"model": model, "threshold": name if name == "validation" else "0.5",
                         "benign chunks blocked": (chunk_scores >= threshold).mean(),
                         "benign files blocked": (scores.loc[benign_skills.index, model] >= threshold).mean()})
print(f"{len(pieces):,} chunks from {len(benign_skills)} benign skill files")
by_chunk = pd.DataFrame(by_chunk)
by_chunk.to_csv(RESULTS / "chunks.csv", index=False)
by_chunk.set_index(["model", "threshold"]).round(3)

# %% [markdown]
# ### Is the MCP result a length effect?
#
# In the MCP set a poisoned description is longer than a clean one (the attacker's sentence is
# added to it). A detector could score well there by measuring length. The check: use the number
# of characters as the score and compare its AUROC with the models'.

# %%
for set_name in ("mcp", "skills", "skillbench", "deepset"):
    group = sets[sets["set"] == set_name]
    print(f"{set_name}: AUROC of text length alone = {roc_auc_score(group['label'], group['text'].str.len()):.3f}"
          f"   median chars benign {int(group.loc[group['label'] == 0, 'text'].str.len().median())},"
          f" malicious {int(group.loc[group['label'] == 1, 'text'].str.len().median())}")

# %% [markdown]
# ### Is the skills result a campaign effect?
#
# The skills dataset's card publishes a rule of 13 indicators (campaign names, hosts, an IP
# address, a decoding command) and reports that it separates the set perfectly. MaliciousSkillBench
# (Wang et al., 2026, [arXiv:2608.19901](https://arxiv.org/abs/2608.19901)) keeps this source's
# malicious files and leaves its benign files out of its primary benchmark as weakly labelled.
# The check: apply the card's rule to the cleaned set. If it separates the classes, a high AUROC
# on this set shows that a model recognises these campaigns, and says nothing about a new one.

# %%
import re

CARD_RULE = re.compile("|".join([
    r"openclawcli", r"openclaw-core", r"glot\.io/snippets", r"rentry\.co/openclaw", r"Ddoy233", r"denboss99",
    r"base64\s+-[dD]", r"91\.92\.242\.30", r"openclawcli\.vercel", r"webhook\.site", r"setup-service\.com",
    r"AuthTool", r"pass(?:word)?[:\s]+.{0,10}openclaw"]), re.IGNORECASE)
skill_files = sets[sets["set"] == "skills"]
pd.crosstab(skill_files["label"], skill_files["text"].map(lambda text: bool(CARD_RULE.search(text))),
            rownames=["label"], colnames=["card rule matches"])

# %% [markdown]
# The same rule on the skillbench set, whose sources are not those campaigns:

# %%
skill_files = sets[sets["set"] == "skillbench"]
pd.crosstab(skill_files["label"], skill_files["text"].map(lambda text: bool(CARD_RULE.search(text))),
            rownames=["label"], colnames=["card rule matches"])

# %% [markdown]
# ### A threshold set on connector data
#
# The thresholds above were set on chat prompts and did not transfer. The check: set the threshold
# by the same rule (at most 1% of benign texts blocked) on benign texts of the connector's own
# kind. Each secondary set's benign texts are split in half at random. The threshold is set on one
# half, and recall is measured on all of the set's malicious texts and FPR on the other half. No
# malicious text is used to set it, as in deployment, where a team has its own installed
# connectors and no labeled attacks. The table gives the median over 200 random splits and the
# range that holds 95% of them. Half of a set's benign texts is 111 skill files, 272 skillbench
# files or 162 MCP descriptions, so the 1% rule allows one or two blocked texts and the threshold
# moves with the split. One published rule of thumb asks for about 16 / p benign samples to fix a
# false positive rate p within half its value, which is 1,600 for 1%
# ([Cisco, July 13, 2026](https://blogs.cisco.com/ai/consistent-security-model-deployment-with-fpr-calibration)).
# This was added on November 23, after the results above were seen.

# %%
split = np.random.default_rng(0)
calibrated = []
for set_name in ("skills", "skillbench", "mcp", "deepset"):
    rows = EVAL_SETS[set_name][evaluation.index]
    labels = evaluation.loc[rows, "label"].values
    for model in scores:
        scored = scores.loc[rows, model].values
        benign, malicious = scored[labels == 0], scored[labels == 1]
        runs = []
        for _ in range(200):
            order = split.permutation(len(benign))
            fit, held = benign[order[:len(benign) // 2]], benign[order[len(benign) // 2:]]
            threshold = block_threshold(fit, np.zeros(len(fit)))
            runs.append((threshold, (malicious >= threshold).mean(), (held >= threshold).mean()))
        runs = np.array(runs)
        at_validation = scored >= thresholds[model]["validation"]
        calibrated.append({"set": set_name, "model": model,
                           "recall, validation threshold": at_validation[labels == 1].mean(),
                           "FPR, validation threshold": at_validation[labels == 0].mean(),
                           "threshold set on the set": np.median(runs[:, 0]),
                           "recall": np.median(runs[:, 1]), "recall low": np.percentile(runs[:, 1], 2.5),
                           "recall high": np.percentile(runs[:, 1], 97.5),
                           "FPR on the held half": np.median(runs[:, 2]), "FPR high": np.percentile(runs[:, 2], 97.5)})
calibrated = pd.DataFrame(calibrated)
calibrated.to_csv(RESULTS / "calibrated.csv", index=False)
calibrated.set_index(["set", "model"]).round(3)

# %% [markdown]
# ### Which poisoned descriptions get through
#
# MCPTox labels each poisoned description with the template it was written from (`kind`). The
# table gives each model's recall at 0.5 by template, and the number of descriptions in each.

# %%
poisoned_rows = evaluation[(evaluation["set"] == "mcp") & (evaluation["label"] == 1)]
by_template = pd.DataFrame({model: (scores.loc[poisoned_rows.index, model] >= thresholds[model]["half"]).groupby(poisoned_rows["kind"]).mean()
                            for model in scores}).T
by_template.loc["descriptions"] = poisoned_rows["kind"].value_counts()
by_template.to_csv(RESULTS / "mcp-templates.csv", index_label="model")
by_template.round(3)

# %% [markdown]
# ### What a reviewer would see
#
# Recall and FPR do not say how many blocks a reviewer has to look at. One study confirmed 157
# malicious skills among 98,380 in two public registries
# ([arXiv:2602.06547](https://arxiv.org/abs/2602.06547)). The table applies that rate to 10,000
# skill files and each model's recall and FPR at 0.5 on the two skills sets: the malicious files
# blocked, the benign files blocked, and the number of blocks a reviewer opens for each one that
# is malicious. It assumes the rates measured here hold in that registry, which the difference
# between the two sets shows they may not.

# %%
BASE_RATE = 157 / 98_380
queue = []
for set_name in ("skills", "skillbench"):
    rows = EVAL_SETS[set_name][evaluation.index]
    labels = evaluation.loc[rows, "label"]
    for model in scores:
        blocked = scores.loc[rows, model] >= thresholds[model]["half"]
        caught = 10_000 * BASE_RATE * blocked[labels == 1].mean()
        false = 10_000 * (1 - BASE_RATE) * blocked[labels == 0].mean()
        queue.append({"set": set_name, "model": model, "malicious blocked": caught, "malicious missed": 10_000 * BASE_RATE - caught,
                      "benign blocked": false, "blocks per malicious one": (caught + false) / caught if caught else float("nan")})
queue = pd.DataFrame(queue)
queue.to_csv(RESULTS / "review-queue.csv", index=False)
queue.set_index(["set", "model"]).round(1)

# %% [markdown]
# ### Invisible characters
#
# Instructions can be written in characters no display shows: zero-width and direction marks, and
# the Unicode tag block. The app blocks any item that contains them. The count below is how many
# texts in the three connector sets contain any, by label.

# %%
INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\U000e0000-\U000e007f]")
connector = sets[sets["set"].isin(["skills", "skillbench", "mcp"])]
pd.crosstab([connector["set"], connector["label"]], connector["text"].map(lambda text: bool(INVISIBLE.search(text))),
            colnames=["has invisible characters"])

# %% [markdown]
# ### False blocks per server
#
# A team installs a server, and a server is blocked when any one of its tool descriptions is. The
# 325 clean MCP descriptions come from 43 servers (a median of 6 tools each). The table gives the
# share of clean descriptions blocked beside the share of clean servers with at least one blocked
# description.

# %%
clean = evaluation[(evaluation["set"] == "mcp") & (evaluation["label"] == 0)]
server = clean["id"].str.split("/").str[0]
per_server = []
for model in scores:
    for name, threshold in thresholds[model].items():
        blocked = scores.loc[clean.index, model] >= threshold
        per_server.append({"model": model, "threshold": name if name == "validation" else "0.5",
                           "clean descriptions blocked": blocked.mean(),
                           "clean servers blocked": blocked.groupby(server).any().mean(),
                           "servers": int(blocked.groupby(server).any().sum()), "of": server.nunique()})
per_server = pd.DataFrame(per_server)
per_server.to_csv(RESULTS / "per-server.csv", index=False)
per_server.set_index(["model", "threshold"]).round(3)

# %% [markdown]
# ## 7. Evasion test
#
# The attacker knows a detector reads the text, so they add a note addressed to it. The note
# below is appended to every malicious MCP description, and each model scores the result at its
# unchanged thresholds. A drop in recall is the share of attacks this one note gets through.

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
evasion.set_index("model").round(3)

# %% [markdown]
# ### An attacker who picks among six wrappers
#
# One fixed note is a weak attacker. Nasr et al. ("The Attacker Moves Second", 2025,
# [arXiv:2510.09023](https://arxiv.org/abs/2510.09023)) bypassed 12 published defenses with attacks
# that adapt to the defense. The test below is a small step in that direction and no more. It uses
# six fixed wrappers (`EVASION_VARIANTS` in `gate.py`; the first is the note above, and the other
# five were written before any of the five was scored). The attacker can query the detector and
# keeps, for each description and each model, whichever scores lowest: the description unchanged
# or one of the six wrapped forms. Leaving it unchanged was added as a choice after the first
# run, because some wrappers raise a model's score and an attacker would not use those.

# %%
for number, (prefix, suffix) in enumerate(EVASION_VARIANTS, 1):
    print(number, repr(prefix + "<description>" + suffix))
variants = {model: np.column_stack([score(predict, [prefix + text + suffix for text in poisoned["text"]])
                                    for prefix, suffix in EVASION_VARIANTS]) for model, predict in predictors.items()}
per_wrapper = pd.DataFrame({model: (with_all >= thresholds[model]["half"]).mean(axis=0) for model, with_all in variants.items()},
                           index=pd.RangeIndex(1, len(EVASION_VARIANTS) + 1, name="wrapper")).T
per_wrapper.to_csv(RESULTS / "evasion-wrappers.csv", index_label="model")
lowest = {model: np.minimum(with_all.min(axis=1), scores.loc[poisoned.index, model]) for model, with_all in variants.items()}
best = pd.DataFrame([{"model": model,
                      "recall with the attacker's pick": (low >= thresholds[model]["validation"]).mean(),
                      "recall at 0.5 with the attacker's pick": (low >= thresholds[model]["half"]).mean()}
                     for model, low in lowest.items()])
evasion = evasion.merge(best, on="model")
evasion.to_csv(RESULTS / "evasion.csv", index=False)
print("Recall at 0.5 on the 485 poisoned descriptions, per wrapper:")
display(per_wrapper.round(3))
evasion.set_index("model")[["recall", "recall with the attacker's pick", "recall at 0.5", "recall at 0.5 with the attacker's pick"]].round(3)

# %% [markdown]
# ## 8. Latency and cost
#
# Local models are timed on this machine over 200 MCP descriptions, one at a time, as the gate
# would receive them. Hosted models report the time of each request as recorded by `decision.py`,
# network included. Jev 1.14 is priced at $0.042 per million input tokens, and Gemini Decision 1
# at $0.10 ([Google, November 11, 2026](https://blog.google/innovation-and-ai/technology/developers-tools/gemini-decisions-api/)).
# Neither charges for output.

# %%
sample = list(sets.loc[sets["set"] == "mcp", "text"].head(200))
latency = []
for model in ("logistic regression", "DistilBERT", "ProtectAI v2", "Horizon Guard"):
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
PRICE = {"Jev 1.14": 0.042, "Gemini Decision 1": 0.10}  # USD per million input tokens
latency["USD per 1,000 requests"] = latency["median input tokens"] * latency["model"].map(PRICE) / 1e6 * 1000
latency.to_csv(RESULTS / "latency.csv", index=False)
latency.set_index("model").round(4)

# %% [markdown]
# ## 9. Findings
#
# Numbers are from the tables above (run of November 23, 2026). Findings 4 and 9 to 14 rest on
# models, data and analyses added on November 23, after the first results were seen.
#
# 1. **Both trained models score well on their own test split.** AUROC 0.995 (logistic
#    regression) and 0.997 (DistilBERT), and at the validation threshold each blocks 90% of attacks
#    while blocking under 1% of benign prompts.
# 2. **Neither transfers to the unseen sets.** Their AUROC there is 0.41 to 0.82. At its
#    validation threshold DistilBERT blocks every malicious skill file and also 99.6% of the benign
#    ones. Text length alone ranks MCP descriptions at 0.74 and deepset prompts at 0.81, about the
#    same as DistilBERT (0.74 and 0.82) and above the baseline (0.63 and 0.67).
# 3. **The zero-shot decision models rank the skills, MCP and deepset sets at 0.95 to 0.99**, with
#    no training data. Their weakest of the first four sets is the primary one (0.94 and 0.96).
#    The skills result is narrower than it looks: the 13-indicator rule on the dataset's card
#    matches all 124 malicious files and none of the 223 benign ones, so a high AUROC there shows
#    a model recognises these campaigns and does not measure a new one.
# 4. **On a second skills set from other sources, the decision models rank at 0.73 and 0.85 and
#    block a third of the malicious files.** At 0.5, Jev 1.14 blocks 31.9% of malicious files with
#    6.8% of benign ones, and Gemini Decision 1 blocks 32.8% with 2.0%. The card rule matches 5 of
#    the 839 malicious files. The result differs by source. On SkillFortifyBench, the one source
#    with both labels, the two models reach AUROC 0.98 and 0.99 and block 90% and 80% with no
#    benign file blocked. On the ATR files found in registries they block 6 and 7 of 7. On
#    SkillHarm, 87% of the malicious files, they block 24.7% and 29.5% of the files with an
#    injected instruction and 16.5% and 9.5% of the files with a backdoor in a bundled file. The
#    lowest-scoring SkillHarm file (`ASB04_000540`) describes a seismology library and tells the
#    agent to open a PDF manual shipped beside it. Its text carries no harmful instruction, so no
#    score of the text can find one. How many SkillHarm files are of that kind was not counted.
#    The benchmark's own TF-IDF SVM blocks 95.6% of malicious and 62.4% of benign files on the same
#    split, and Cisco's skill-scanner with an LLM judge sends 66.7% and 15.4% to review, so the
#    models here are in the range purpose-built scanners report. Label and source are confounded
#    in this set outside SkillFortifyBench, so its overall AUROC partly measures how different the
#    sources look.
# 5. **A threshold does not transfer either.** The 1%-on-validation rule puts the decision models'
#    thresholds near 0.98, and at that threshold they block 1% to 23% of malicious texts in the
#    two skills sets and the MCP set. At 0.5, Jev 1.14 blocks 91% of malicious files in the first
#    skills set with 1.8% of benign ones, and 58% of poisoned MCP descriptions with none of the
#    clean ones. The 0.5 threshold was added after seeing the first result.
# 6. **Most of the baseline's heaviest weights (18 of the top 20) are pieces of one phrase**, "I
#    have been PWNED", which attackers had to produce in the HackAPrompt competition (a source of
#    the primary set, per its dataset card). That is a property of the dataset. It does not
#    explain the false blocks on skill files, which do not contain the phrase, and nothing here
#    shows what DistilBERT relies on.
# 7. **One appended note did little; a choice among six wrappers did more.** The note raised the
#    median score of every model, and at 0.5 no model lost recall. When the attacker keeps the
#    lowest-scoring of the unchanged description and six fixed wrappers, recall at 0.5 falls from
#    57.7% to 47.2% for Jev 1.14 and from 64.7% to 0.0% for Gemini Decision 1. Each drop comes
#    almost entirely from one wrapper. For Gemini Decision 1 it is a prefix that presents the
#    description as a quoted documentation example (0.0% alone). For Jev 1.14 it is a suffix that
#    denies each clause of the question (47.8% alone). Horizon Guard goes from 87.8% to 86.2%.
#    At 0.5 the two trained models keep their recall (98.8% and 100%), and they also block 92%
#    and 75% of the clean descriptions.
# 8. **Latency and cost:** the local models answer one MCP description in a median of 24 ms or
#    less (logistic regression 0.3 ms, DistilBERT 7 ms, Horizon Guard 11 ms, ProtectAI v2 24 ms).
#    Jev 1.14 answered one chunk in a median of 58 ms at about $0.024 per 1,000 chunks, and Gemini
#    Decision 1 in 202 ms at about $0.039. A median skill file is 6 chunks.
# 9. **The newer published detector matches the hosted models on MCP descriptions and not on
#    skill files.** Horizon Guard, released in October 2026, ranks MCP descriptions at AUROC 0.98
#    and at 0.5 blocks 88% of poisoned ones with 4.9% of clean ones, which is 8 of 43 clean
#    servers. It runs locally in 11 ms, so no text leaves the company. On the first skills set
#    its AUROC is 0.76, and on the second 0.47. ProtectAI v2, whose card says the project is
#    archived, ranks the same sets at 0.91, 0.76 and 0.38 and at 0.5 blocks 24% of poisoned
#    descriptions. Horizon Guard's results on the primary test split and on deepset are not
#    held-out (section 4).
# 10. **A threshold set on a set's own benign texts helps where the model ranks well and not
#    elsewhere.** With the threshold set on half of the benign texts to block at most 1% of them,
#    the median over 200 splits is 83% of malicious files blocked for Gemini Decision 1 and 44%
#    for Jev 1.14 on the first skills set (23% and 5% at the validation threshold), and 67% and
#    76% on MCP descriptions (9% and 2%). On the second skills set it is 16% and 10%. The range
#    across splits is wide (Jev 1.14 on skills: 44% to 93%), because half a set's benign texts is
#    111 to 272 texts and the published rule of thumb asks for about 1,600.
# 11. **Disputed labels set the decision models' validation thresholds.** 13 of the 407 benign
#    validation prompts are scored 0.5 or above by both decision models, and they read as
#    jailbreak role-play prompts. Without them the 1% rule gives 0.71 for Jev 1.14 and 0.39 for
#    Gemini Decision 1 where it gave 0.98 and 0.97. At those thresholds they block 73% and 89% of
#    malicious files in the first skills set with 0.4% and 1.8% of benign ones, and block 4.1%
#    and 4.9% of benign primary test prompts. The check is circular for those two models, whose
#    scores chose the rows. For Horizon Guard, which did not choose them, the threshold moves
#    from 0.997 to 0.39 and its MCP recall from 44% to 91% with 5.2% of clean descriptions.
# 12. **The false blocks on skill files are not an effect of taking the worst chunk.** DistilBERT
#    blocks 90% of single chunks of benign skill files at its validation threshold and the
#    logistic regression 20%, so one chunk is usually enough. The other candidate, the shift from
#    56-character prompts to 1,000-character chunks, was not isolated.
# 13. **Per server, the trained models block nearly every clean server.** A server is blocked when
#    any of its tools is. At 0.5 the two trained models block 40 of 43 clean servers, Horizon
#    Guard 8, ProtectAI v2 6, and the decision models none.
# 14. **At a registry's base rate the blocks are mostly benign files.** At 157 malicious skills in
#    98,380, 10,000 files hold 16 malicious ones. With the first skills set's rates at 0.5, Jev
#    1.14 blocks 14.5 of them and 179 benign files, 13 blocks for each malicious one, and Gemini
#    Decision 1 blocks 13.5 and 45. With the second set's rates Jev 1.14 blocks 5 and 678, and
#    Gemini Decision 1 blocks 5 and 202.
#
# **Limits.** One primary dataset with noisy labels; one run of DistilBERT per learning rate; the
# MCP positives are synthetic (written for a benchmark) and longer than the clean descriptions;
# the first skills set comes from one registry and two campaigns; in the second, nearly all
# benign files come from one source and nearly all malicious files from another; the hosted
# models can change without notice. The evasion test is six fixed wrappers, not an attacker who
# optimises against the detector: Nasr et al. bypassed 12 published defenses that way
# ([arXiv:2510.09023](https://arxiv.org/abs/2510.09023)), and Pretext evaded a frozen skill
# detector in up to 97% of attempts ([arXiv:2609.39607](https://arxiv.org/abs/2609.39607)). Two
# malicious files in the second skills set contain invisible Unicode characters and no benign
# connector text does, so the app's rule for them is supported by two files. One malicious skill
# file uses an HTML comment. Neither technique is measured beyond that.
#
# **What a text score cannot decide.** Three limits hold for any model in this notebook.
#
# - The gate reads what a connector says. A remote MCP server can change what it does while its
#   tool definition stays identical, a skill can keep its payload in a file it ships with (finding
#   4), and a skill can restore a payload at run time that was absent when it was scanned
#   (packing of that kind passed every one of eight scanners in over 90% of cases,
#   [arXiv:2607.02357](https://arxiv.org/abs/2607.02357)). One study of skill detectors measured
#   recall falling from 99.8% to 87.0% when commands moved into a referenced script
#   ([arXiv:2610.17727](https://arxiv.org/abs/2610.17727)).
# - Malicious connectors are rare (finding 14). A score at these error rates can order a review
#   queue. It cannot be the only control, and the controls under it are the ones that bound
#   damage (section 10).
# - A decision is made per connector, and an MCP server has several tools (finding 13).

# %% [markdown]
# ## 10. Operating the gate
#
# The results put the score in one place in a set of controls. It is not the control that decides.
# Each step below names where the practice is published. Pages were read on November 23, 2026.
#
# 1. **Allow connectors by identity first.** A connector is allowed by its URL or its exact
#    command. GitHub's Copilot allowlists match on those and call matching by name "not a security
#    control" ([August 6, 2026](https://github.blog/changelog/2026-08-06-mcp-allowlists-in-enterprise-managed-settings)),
#    and Claude Code's managed settings say the same of `serverName`
#    ([docs](https://code.claude.com/docs/en/managed-mcp.md)). A connector that is not on the list
#    never reaches the score.
# 2. **Fingerprint what was approved.** The approval is stored with a SHA-256 hash of the whole
#    tool definition, and a tool whose hash changes is withheld until a person reviews it again.
#    This is the control in the
#    [OWASP MCP Security Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html),
#    and ChatGPT's workspace connectors work this way: a frozen snapshot of the tools, with later
#    changes shown to an admin as a diff
#    ([OpenAI help centre](https://help.openai.com/en/articles/12584461-developer-mode-and-mcp-connectors)).
#    The MCP specification of July 28, 2026 changed authorization and added no signing of tool
#    definitions ([release post](https://blog.modelcontextprotocol.io/posts/2026-07-28/)), so the
#    client or a gateway has to do it. MITRE ATLAS lists the attack this stops as AML.T0109, AI
#    Supply Chain Rug Pull, and the one the score addresses as AML.T0110, AI Agent Tool Poisoning.
#    The app shows the fingerprint beside each item. It does not store approvals.
# 3. **Use the score to order review, and log before blocking.** Section 6 gives the number of
#    blocks a reviewer would open for each malicious file at a registry's base rate. A team would
#    run the score in a log-only mode first and read what it would have blocked, the way AWS ships
#    `LOG_ONLY` beside `ENFORCE` for agent policies
#    ([docs](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/policy-enforcement-modes.html)).
# 4. **Set the threshold on the team's own benign connectors.** Section 6 shows what that changes.
#    The connectors a team has already installed are the benign sample, and no labeled attack is
#    needed. The threshold is set again whenever the model changes. A hosted model is called by a
#    fixed version (`jev-1.14.0`), and a fixed set of texts with known scores is rescored on a
#    schedule to notice a change the provider did not announce. That last check is this project's
#    suggestion. No published source for it was found.
# 5. **Watch three numbers after launch.** The share of installs blocked each week, where a fall
#    can mean attacks written to score under the threshold. The score distribution of approved
#    connectors, where a shift means the threshold no longer blocks the share it was set for. The
#    reviewers' verdicts on blocked items, which are the only labels the team will get and the
#    trigger for setting the threshold again.
# 6. **Bound the damage under the gate.** Scoped credentials for each connector and a fixed list of
#    places the agent can send data. Anthropic's guide for security teams calls the outbound limit
#    "your strongest control against prompt injection"
#    ([July 17, 2026](https://claude.com/blog/ciso-guide-to-agentic-ai)). OpenAI states the goal as
#    a system where "the impact of manipulation is constrained, even if it succeeds"
#    ([March 11, 2026](https://openai.com/index/designing-agents-to-resist-prompt-injection/)), and
#    Google writes that "no detector catches every injection attempt"
#    ([November 17, 2026](https://blog.google/security/how-geminis-defense-in-depth-approach-works/)).
#
# **What each failure costs.** A miss is a poisoned connector installed with the agent's
# credentials, limited only by step 6. A false block is a tool a team cannot install until a
# reviewer clears it, so the cost of the gate is the review queue in section 6.
#
# **Published results for reference.** Purpose-built scanners report the same order of error on
# held-out skill files as the models here (section 6). Trail of Bits bypassed ClawHub's detector,
# Cisco's skill scanner and the three scanners on skills.sh with four malicious skills, three of
# which took under an hour to write
# ([June 3, 2026](https://blog.trailofbits.com/2026/06/03/the-sorry-state-of-skill-distribution/)).
# Updates to a tool's code after approval reached 97.6% attack success across seven agent
# frameworks ([arXiv:2611.01976](https://arxiv.org/abs/2611.01976)), which no score of the
# description can see.

# %% [markdown]
# ## 11. Follow-up: what the hosted model is asked
#
# Every result above uses one question, fixed before any result was seen (section 5). This
# section was added on November 23, after the misses in section 6 had been read, and tests a
# second way of asking on Jev 1.14 only. It is a follow-up and does not replace the results above.
#
# **What changes.** Each chunk is sent with the name and stated purpose it came with: for an MCP
# tool, its server, its name and the names of the server's other tools; for a skill file, the name
# and description in its header.
# Three questions are asked in one request (`variants/dev.py`):
#
# - *purpose*: does the text tell the agent to do something a tool or skill with that name and
#   purpose does not need, to act outside its job, or to conceal what it does?
# - *pressure*: does the text present a step as mandatory, urgent or required by policy to make
#   the agent do something unrelated to the stated purpose?
# - *other tools*: does the text tell the agent to change how it uses other tools, or to do
#   something before or after tasks that are not this one?
#
# A file's score is the mean of its highest answers, over all three questions for an MCP tool and
# over the first two for a skill file.
#
# **How it was chosen.** A development set of 250 files was drawn from the three connector sets
# with a fixed seed. Wordings and eight other techniques were compared on it
# (`variants/explore.py`). The questions, the scoring rule and the thresholds were committed, and
# the other 2,291 files were then scored (`variants/heldout.py`). The table is that held-out
# result. Its threshold is the one that blocked 2 benign development files of the same kind.

# %%
variant = pd.read_csv("results/question-variant.csv")
variant = variant[variant["method"].isin(["current", "three"])].replace(
    {"method": {"current": "one question (sections 5 to 9)", "three": "three questions, name and purpose shown"},
     "group": {"mcp": "MCP tool descriptions", "skills": "first skills set", "skillbench": "second skills set"}})
variant.set_index(["group", "method"])[["malicious", "benign", "AUROC", "caught at 0.5", "benign blocked at 0.5",
                                        "threshold from development", "caught at it", "benign blocked at it"]].style.format(
    {"AUROC": "{:.3f}", "threshold from development": "{:.2f}"} | {c: "{:.1%}" for c in ("caught at 0.5", "benign blocked at 0.5", "caught at it", "benign blocked at it")}, na_rep="")

# %% [markdown]
# **Reading.** On held-out MCP tool descriptions the three questions catch 95% at 0.5 with no
# clean description blocked, where the one question caught 59%. On the second skills set recall
# rises from 32% to 56% at 0.5, with benign files blocked going from 6.5% to 8.1%, and from 24%
# to 49% at the development threshold, with 4.8% and 5.1% blocked. On the first skills set recall
# falls: 91% to 76% at 0.5, and 84% to 73% at the development threshold. That malware tells the
# agent to install a named program, which the one question names and the three do not ask about;
# whether that explains the fall was not tested. Half of the malicious files in the second skills
# set are still missed.
#
# On the 45 poisoned tool descriptions of the development set, the purpose question alone was
# also run against the six wrappers of section 7. At 0.5 it caught 84% as written and 78% with
# the attacker's pick of wrapper; the one question caught 47% and 42%. The quoted-example wrapper
# that takes Gemini Decision 1 to zero did not lower it.
#
# **Limits.** One model. Its answers vary between runs: the purpose question's recall on the
# development skills at 0.5 was between 79% and 84% over four runs, which also differed in the
# questions asked beside it. Before the development set was drawn, two rewordings of the question
# had been scored on every connector file (`variants/first_pass.py`), and one of them asked about
# stated purpose, so the held-out files were not unseen in that sense. The MCP result was scored
# twice: the first frame showed arguments, which the data has for clean tools only, and removing
# them left the result at 0.5 unchanged. A poisoned MCPTox tool is an invented tool added to a real
# server, so its name is unlike its neighbors' names. The development and held-out files come
# from the same sources. The name and stated purpose shown to the model are written by the same
# author as the text, so an attacker can choose a name that makes the payload look expected.
#
# **A second model.** Gemini Decision 1 was asked the same three questions on the 250 development
# files only (`variants/gemini.py`, `results/question-variant-gemini.csv`). At 0.5 it catches 93%
# of the poisoned tools, against 58% with the one question, and 72% of the malicious skill files,
# against 47%, with 1 of 70 benign skill files blocked. The quoted-example wrapper of section 7
# still takes it to zero on the 45 poisoned tools. These files are the ones the questions were
# chosen on, with Jev 1.14's answers.
#
# **All models on the same files.** `results/comparison.csv` scores every model at 0.5 on the
# 2,291 files outside the development set. Over all of them Jev 1.14 catches 45% of malicious
# files with the one question and 70% with the three, with 3.7% and 5.0% of benign files blocked.
# Gemini Decision 1 with the one question catches 48% and blocks 1.2%; Horizon Guard 43% and 10%;
# ProtectAI v2 13% and 8%; DistilBERT 100% and 93%.
#
# The app offers this as a second deciding model, `Jev 1.14, asked about purpose`, beside the one
# question.
