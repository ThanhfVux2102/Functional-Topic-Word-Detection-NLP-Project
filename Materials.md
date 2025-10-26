#  Functional & Topic Word Detection NLP Project

##  Project Overview
This project builds an **NLP system** that analyzes an English paragraph and:
1. **Detects functional words** (logical connectors such as *however*, *therefore*, *in contrast*, etc.).
2. **Classifies content words** into **semantic topics** (e.g., *Health*, *Fashion*, *Media*, *Technology*, *Environment*, *Sociology*).

The goal is to train a **medium-sized model (~3–5k sentences, ~50k tokens)** that generalizes well and can be **deployed as a web app** for educational and linguistic purposes.

---

##  Core Objectives
- Build a **custom labeled dataset** for functional and topic-based vocabulary.
- Train and evaluate ML/NLP models to identify functional vs. content tokens.
- Classify content words into 5–6 semantic topics.
- Deploy the model via **FastAPI + React/Next.js** with colored highlights in text.

---

##  Tech Stack

| Component | Technology |
|------------|-------------|
| **Language** | Python 3.10+, TypeScript (for web) |
| **NLP Libraries** | spaCy, scikit-learn, HuggingFace Transformers |
| **Data Handling** | pandas, numpy, json, csv |
| **Modeling** | RandomForest / LogisticRegression (baseline), DistilBERT (advanced) |
| **Evaluation** | seqeval, sklearn.metrics, matplotlib, seaborn |
| **Backend (API)** | FastAPI |
| **Frontend (UI)** | Next.js (React) + Tailwind CSS |
| **Deployment** | Render / Vercel / HuggingFace Spaces |
| **Version Control** | Git + GitHub |
| **Containerization (optional)** | Docker |

---

##  MVP Scope

### Input
A short paragraph or article (English text).

### Output
JSON response and web highlights:
```json
{
  "functional_words": ["However", "Therefore"],
  "content_topics": {
    "Health": ["body", "disorder"],
    "Fashion": ["model", "waistline"],
    "Media": ["magazine"]
  }
}
```

### Minimal Features
- Detect functional connectors with ≥0.85 F1.
- Identify content words and assign to 1 of 5–6 topics.
- REST endpoint `/analyze` that returns structured JSON.
- Web interface highlighting tokens by color (green = functional, red = content).

---

##  System Architecture & Pipeline

```text
                ┌────────────────────────────┐
                │   Raw Text Sources          │
                │  (Wikipedia, BBC, IELTS)    │
                └────────────┬────────────────┘
                             │
                 [1] Data Preparation
                             │
            ┌────────────────┴────────────────┐
            │ Tokenization (spaCy)            │
            │ POS tagging + Lemmatization      │
            └────────────────┬────────────────┘
                             │
                 [2] Auto Labeling
                             │
            ┌────────────────┴────────────────┐
            │ FUNC detection via lexicon       │
            │ CONTENT tagging via POS filtering│
            │ Topic assignment via seed list   │
            └────────────────┬────────────────┘
                             │
                 [3] Dataset (CSV / CoNLL)
                             │
                 [4] Model Training
                             │
      ┌────────────────────────────┬──────────────────────────┐
      │ Functional Detector (Token)│ Topic Classifier (Sentence)│
      │ RandomForest / DistilBERT  │ TF-IDF + LogisticReg.     │
      └────────────────────────────┴──────────────────────────┘
                             │
                 [5] Evaluation
                             │
            ┌────────────────┴────────────────┐
            │ Token-F1 / Span-F1 (seqeval)    │
            │ Topic Macro-F1 (sklearn)        │
            └────────────────┬────────────────┘
                             │
                 [6] Deployment
                             │
        ┌────────────────────┴────────────────────┐
        │ FastAPI REST API (model inference)      │
        │ React/Next.js UI (text highlighting)    │
        └────────────────────────────────────────┘
```

---

##  Dataset Plan

| Type | Size | Description |
|------|------|--------------|
| **Raw Corpus** | 3k–5k sentences | Academic + general texts |
| **Tokens** | ~50k | Token-level labeled data |
| **Labels** | FUNC / CONTENT / OTHER + topic tag |
| **Topics** | Health, Fashion, Media, Tech, Environment, Sociology |
| **Format** | CSV / CoNLL (token, pos, lemma, prev, next, label, topic) |

---

##  Model Details

### Functional Word Detector
- **Input:** token features (`word`, `lemma`, `pos`, `prev`, `next`)  
- **Baseline:** RandomForestClassifier  
- **Advanced:** `DistilBERTForTokenClassification`  
- **Output:** FUNC / CONTENT / OTHER  

### Topic Classifier
- **Input:** sentence or paragraph of content words  
- **Model:** TF-IDF + LogisticRegression (multi-class)  
- **Output:** topic label for content tokens  

---

##  Evaluation Metrics

| Task | Metric | Target |
|------|---------|--------|
| Functional Detection | Token-F1 / Span-F1 | ≥ 0.85 |
| Topic Classification | Macro-F1 | ≥ 0.70 |
| Runtime | < 1s per paragraph | ✅ real-time ready |

---

##  Deployment Plan

**Backend:**  
- FastAPI endpoint `/analyze`  
  - Loads trained models (`joblib` / `from_pretrained`)  
  - Returns JSON with token types and topics  

**Frontend:**  
- Next.js + Tailwind UI  
  - Textarea input  
  - Color-coded highlights  
  - Sidebar summary (topic counts, list of functional connectors)

**Hosting:**  
- API → Render / HuggingFace Space  
- Frontend → Vercel  
- Optional: Dockerfile for local deployment

---

##  Project Folder Structure

```
nlp-functional-topic/
│
├── data/
│   ├── raw_texts/
│   ├── tokens_labeled.csv
│
├── lexicons/
│   ├── functional_list.txt
│   ├── topic_seed.json
│
├── src/
│   ├── preprocess.py
│   ├── train_func_model.py
│   ├── train_topic_model.py
│   ├── evaluate.py
│   ├── api_server.py   # FastAPI
│
├── web/               # Next.js frontend
│   ├── pages/
│   ├── components/
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Model_Eval.ipynb
│
├── requirements.txt
└── README.md
```

---

##  Suggested Timeline (6 Weeks)

| Week | Goal | Deliverables |
|------|------|--------------|
| 1 | Data collection + spaCy preprocessing | Tokenized corpus |
| 2 | Auto-label + manual cleanup | Labeled CSV |
| 3 | Train FUNC model (baseline + eval) | F1 report |
| 4 | Train Topic model + cross-check | Topic F1 report |
| 5 | Build FastAPI + Next.js UI | Working local app |
| 6 | Deploy & write documentation | Online demo + README |

---

##  Expected Outcomes
- A working **web app** that detects logical connectors and classifies content words by topic.  
- A **medium-sized labeled dataset (publicly reusable)**.  
- A complete ML pipeline that demonstrates practical NLP engineering skills.  
- A **strong portfolio project** for internships or junior AI/NLP roles.

---
