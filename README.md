#  Functional & Topic Word Detection

###  Overview
This NLP project automatically **detects functional words** (e.g., *however*, *therefore*, *in contrast*)  
and **classifies content words** into **semantic topics** like *Health*, *Fashion*, *Media*, *Technology*, and *Sociology*.

###  Objectives
- Build a custom **50k-token dataset** labeled with Functional / Content / Topic classes.  
- Train **machine learning models** (RandomForest + DistilBERT) to perform token-level classification.  
- Deploy a real-time **web application** to highlight words by their function or topic.

---

##  Live Demo
🔗 **Web App:** [Coming soon — Deployed on Vercel]  
🔗 **API Docs:** [Coming soon — Hosted via FastAPI on Render]  

---

##  Tech Stack

| Layer | Tools |
|--------|--------|
| **Language** | Python 3.10+, TypeScript |
| **Core NLP** | spaCy, HuggingFace Transformers |
| **ML & Data** | scikit-learn, pandas, numpy |
| **API Backend** | FastAPI |
| **Frontend UI** | Next.js (React) + Tailwind CSS |
| **Deployment** | Render (API), Vercel (UI) |
| **Evaluation** | seqeval, sklearn.metrics |
| **Version Control** | Git + GitHub |

---

## Features
- **Functional Word Detection** → highlights logical connectors.  
- **Topic Classification** → groups content words by theme.  
- **Interactive Web UI** → color-coded highlights and topic summaries.  
- **REST API** → `/analyze` endpoint returning JSON results.

---

## Folder Structure

```
nlp-functional-topic/
│
├── README.md                 ← short summary + links
├── docs/
│   └── PROJECT_PLAN.md       ← full technical documentation
├── data/                     ← raw and labeled corpus
├── lexicons/                 ← functional & topic seed lists
├── src/                      ← preprocessing, model, API code
├── web/                      ← Next.js frontend
├── notebooks/                ← EDA, training, evaluation
└── requirements.txt
```

---

## Quick Start (local)
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/nlp-functional-topic.git
cd nlp-functional-topic

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run FastAPI server
uvicorn src.api_server:app --reload

# 4. (Optional) Run Next.js frontend
cd web
npm install
npm run dev
```

---

## Documentation
See the full project plan, dataset pipeline, and deployment roadmap in:  
📄 [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md)

---

## License
Released under the **MIT License** — for educational and non-commercial use.

---

## Author
**Vũ Minh Thành**  
AI & NLP Enthusiast | HCMUIT Student  

