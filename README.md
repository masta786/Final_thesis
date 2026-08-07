# Personalised Medical Chatbot — Hybrid vs LLM vs ML

**Mastan Vali Shaik (24226807)** · MSc Artificial Intelligence · National College of Ireland

A medical question-answering assistant built three ways, so the three designs can be
compared on the same questions, with the same patient profile, against the same evidence:

| # | System | Method |
|---|--------|--------|
| 1 | **Hybrid (LSTM + Gemini)** | BiLSTM retrieves grounding passages from MedQuAD → Gemini writes a personalised answer. *The proposed system.* |
| 2 | **Gemini Only** | The LLM answers directly. Fluent and personalised, but nothing anchors it to a source. |
| 3 | **LSTM Only** | Pure retrieval. Returns the closest MedQuAD passage verbatim — faithful to its source, but generic. |

On held-out questions the hybrid produces the best answers of the three, beating pure
retrieval by **+30.3 points (d-z = 1.15, p < 0.001)** and cutting unsupported claims
**16%** below the LLM. Its margin over the LLM alone is not statistically significant,
and the report says so.

---

## Quick start

```bash
pip install -r requirements.txt

cp .env.example .env          # then paste your Gemini API key into .env

python app.py                 # live demo  ->  http://localhost:5000
jupyter notebook medical_chatbot.ipynb    # full research pipeline
```

Get a free Gemini API key at <https://aistudio.google.com/apikey>. Without one the
notebook still runs every offline step (EDA, BiLSTM training, the four-model retrieval
comparison); only the Gemini and Hybrid systems need it.

---

## Project layout

```
working_masthan/
├── submission/                                     # everything the examiner opens
│   ├── MSc_Research_Project_Report_24226807.docx   #   the report (17 pp, NCI template)
│   ├── Configuration_Manual_24226807.docx          #   separate, as the handbook requires
│   ├── Thesis_Presentation.pptx                    #   10 slides for the presentation video
│   └── Presentation_Script.docx                    #   word-for-word script, ~10.4 min
│
├── app.py                    # Flask demo - all three systems side by side
├── medical_chatbot.ipynb     # the research pipeline, Sections 1-10
├── rebuild_evaluation.py     # the three-system evaluation: generate, judge, audit
├── train_retriever.py        # trains the corrected BiLSTM retriever
│
├── overleaf/                 # LaTeX version of the report - zip and upload
│   ├── main.tex              #   generated from the .docx, so it cannot drift
│   ├── references.bib        #   22 entries, cited with natbib
│   └── figures/              #   the seven figures
│
├── tools/                    # document generators - nothing is hand-copied
│   ├── results_io.py         #   single source of truth for every reported number
│   ├── build_figures.py      #   the seven report figures
│   ├── build_report.py       #   the report
│   ├── build_slides.py       #   slides + speaking script
│   ├── build_config_manual.py
│   └── build_latex.py        #   the Overleaf project
│
├── data/medquad.csv          # MedQuAD dataset
├── saved_models/             # bilstm_fixed.pt, vocab_fixed.json, cached embeddings
├── outputs/                  # the seven report figures + every result file
├── templates/index.html      # demo UI
├── user_data/                # demo session log
├── backup/                   # superseded documents, earlier runs, logs, RESULTS_LOG.md
├── requirements.txt
├── README.md
└── .env                      # your API key (never commit this)
```

Every number in the report, slides and manual is read from `outputs/summary_*.json`
through `tools/results_io.py`. Re-running the four `tools/build_*.py` scripts
regenerates all four deliverables in `submission/` from the files on disk.

---

## The live demo

`python app.py` starts a chat interface. Type any medical question and it is answered
by **all three systems at once**, displayed side by side in thesis order — Hybrid first.

Each answer is scored live:

- **Response quality /100** — a blind clinician-style rubric (accuracy /30,
  personalisation /25, helpfulness /25, safety /20). The judge is never told which
  system wrote the answer. This is the headline metric.
- **Hallucination / Grounded** — every sentence is split into a claim and checked
  against the passages the BiLSTM retrieved. Claims whose content words already
  appear in the evidence are marked grounded for free; the rest go to Gemini for a
  supported / unsupported / contradicted verdict.
- **Personalisation** — how much of this patient's own record the answer actually uses,
  ignoring any term the question already contained.
- **Latency**.

Expand *Claim check* under any answer to see each sentence colour-coded, and
*Retrieved evidence* to see the passages everything was scored against.

> **Two scoring notes.** A live question has no gold MedQuAD answer, so the app scores
> against the retrieved passages only, while the notebook also folds in the gold answer —
> the two sets of numbers are not directly comparable. For the same reason the live
> rubric scores **accuracy** against medical knowledge where the notebook scores
> **grounding** against the evidence pool: with no gold answer, grounding would measure
> the retriever rather than the response.

---

## Results

All reported numbers come from `rebuild_evaluation.py` and are read from
`outputs/summary_<tag>.json`; nothing is hand-copied into a document. Both response
quality conditions use the same generator, the same independent judge and the same
**held-out** questions (`--offset 100`, n = 30), disjoint from the questions the
prompts and retrieval settings were tuned on.

### 1. Retrieval — the corrected encoder (n = 600, no API required)

| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| BiLSTM (original, as submitted) | 0.010 | 0.035 | 0.073 | 0.034 |
| TF-IDF + cosine (baseline) | 0.315 | 0.667 | 0.717 | 0.463 |
| BiLSTM (defects corrected) | 0.495 | 0.663 | 0.720 | 0.579 |
| **BiLSTM corrected + TF-IDF rerank** | **0.513** | **0.743** | **0.788** | **0.620** |

Five defects were found in the original retriever — the decisive one being that
out-of-vocabulary words mapped to index 0, which was also `padding_idx`, so the
patient's own medicines (`metformin`, `lisinopril`) were fed to the network as blank
padding. Recall@1 went from **0.010 to 0.513, a 51x improvement**, and from 30x worse
than a TF-IDF baseline to 63% better than it.

### 2. Response quality — full-coverage knowledge base

| Rank | System | Quality /100 | Grounding /30 | Personal. /25 | Helpful. /25 | Safety /20 | Halluc. |
|---|---|---|---|---|---|---|---|
| 1 | **Hybrid (BiLSTM + Gemini)** | **79.27** | **24.47** | 15.87 | 20.07 | 18.87 | 0.767 |
| 2 | Gemini Only (LLM) | 78.23 | 20.87 | **18.20** | 20.37 | 18.80 | 0.913 |
| 3 | ML Only (BiLSTM + rules) | 48.97 | 20.60 | 2.77 | 10.57 | 15.03 | **0.133** |

| Comparison | Mean diff | Holm p | Cohen d-z | Verdict |
|---|---|---|---|---|
| Hybrid vs ML Only | **+30.30** | 2.2e-06 | **1.15** | significant |
| Gemini Only vs ML Only | +29.27 | 2.6e-06 | 1.11 | significant |
| Hybrid vs Gemini Only | +1.03 | 0.547 | 0.11 | **not significant** |

**What this supports.** Retrieval-augmented generation beats pure neural retrieval by
30 points with a large effect size. Against the language model alone, the hybrid buys
**faithfulness rather than fluency**: +3.60 grounding and a hallucination rate 16%
lower, measured by a claim auditor that is independent of the judge.

**What it does not support.** The 1.03-point overall margin over the language model is
not statistically distinguishable from zero. The thesis reports it as such rather than
presenting the ordering as settled.

### 3. Coverage ablation — the finding that qualifies everything above

Same questions and generator, but each question's own gold document is hidden:

| | Full KB | Leave-one-out |
|---|---|---|
| Hybrid overall | 79.27 | 78.10 |
| Gemini Only overall | 78.23 | 78.70 |
| ML Only overall | 48.97 | **29.53** |
| Grounding, Hybrid - Gemini | **+3.60** | +0.20 |
| Hallucination, Hybrid vs Gemini | **-16%** | none |

The hybrid's entire advantage is **contingent on the corpus covering the question**.
Remove the answering document and the grounding gain and the hallucination reduction
both disappear. Retrieval augmentation helps exactly as much as the corpus covers the
query, and no more — which is why the proposed next step is a retrieval-confidence
gate that falls back to plain generation when the corpus has nothing relevant.

### 4. Latency

Median per answer, with request pacing excluded from the measurement: **ML Only 11 ms,
Gemini Only 1,497 ms, Hybrid 1,926 ms.** The hybrid pays roughly one extra model call
for its grounding advantage.

### Evaluation instrument

The original judge rubric ended with a worked example whose scores summed to 78. A
controlled ablation showed the judge copied it: a mediocre answer and a good one both
scored exactly 78, and 78 was a hard ceiling. The earlier "Hybrid 68.1 vs Gemini 66.7"
result was an artefact of how often each system triggered that anchor (paired
p = 0.766, 13 exact ties) and **is withdrawn**. The replacement rubric uses
placeholders and per-band descriptors, and separates the same three answers cleanly.

### Figures

| File | What it shows |
|------|---------------|
| `outputs/fig_response_quality.png` | Response quality by rubric axis and total — **the headline result** |
| `outputs/fig_retrieval_v2.png` | Recall@k and MRR across the four retrievers |
| `outputs/fig_coverage.png` | The coverage effect: full KB vs leave-one-out |
| `outputs/fig_hallucination_v2.png` | Claim-level hallucination rate, judge-independent |
| `outputs/fig_quality_axes.png` | Per-axis rubric scores, grouped by system |
| `outputs/fig_anchoring.png` | The 78-point judge anchor, old rubric vs new |
| `outputs/fig_architecture.png` | System architecture |
| `outputs/summary_holdout_full.json`, `summary_holdout_loo.json` | Every reported number, machine-readable |

> **Not fit for clinical use.** Hallucination rates for all generative configurations
> are high in absolute terms. This is a research prototype.

---

## Reproducing

```bash
# 1. retrieval results - no API key needed
python train_retriever.py

# 2. response quality, both conditions (~35 min each, paced under the request allowance)
GEN_MODEL=gemini-3.5-flash-lite JUDGE_MODEL=gemini-3.1-flash-lite ADJUDICATOR_MODEL=gemini-flash-lite-latest KB_MODE=full python -u rebuild_evaluation.py --stage all --n 30 --offset 100 --tag holdout_full

GEN_MODEL=gemini-3.5-flash-lite JUDGE_MODEL=gemini-3.1-flash-lite ADJUDICATOR_MODEL=gemini-flash-lite-latest KB_MODE=loo  python -u rebuild_evaluation.py --stage all --n 30 --offset 100 --tag holdout_loo

# 3. deliverables, all read from outputs/summary_*.json
python tools/build_figures.py && python tools/build_report.py \
  && python tools/build_slides.py && python tools/build_config_manual.py
```

Each role runs on a **different model** so that the judge is independent of the
generator, and so that the three roles draw on separate request quotas.

**Stages.** `--stage generate` produces answers and checkpoints them to
`outputs/answers_<tag>.csv`; `--stage score` resumes from that file and judges them;
`--stage report` recomputes every table and statistic from scored answers already on
disk and **issues no API call at all**. An interrupted run therefore never pays for
the same answers twice, and the tables can always be rebuilt offline.

**Safety rails.** A run that loses more than 10% of its judge calls aborts with exit
code 2 and writes nothing, so a partial failure can never overwrite good results with
NaNs. Requests are paced under the per-minute allowance; raising `WORKERS` does not
help, because the limit is requests per minute per model rather than concurrency.

**Caches.** `data/medquad.csv` is downloaded once; `saved_models/bilstm_fixed.pt` and
`vocab_fixed.json` come from `train_retriever.py`; `saved_models/kb_emb_*.npy` caches
corpus embeddings and is keyed on the checkpoint timestamp, so it invalidates itself
when the encoder is retrained.

---

*Not medical advice. Consult a qualified healthcare professional for actual medical guidance.*
