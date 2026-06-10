<div align="center">

<img src="static/readme_logo.png" alt="Query Prediction Engine" width="140" style="margin-bottom:12px;">

# Query Prediction Engine

**Next-word prediction using N-gram Language Models — transparent, fast, and privacy-aware**

[GitHub](https://github.com/theodhorex/NLP_Query_Prediction_Engine) · [Documentation](#) · [Try Now](#setup-)

---

</div>

### Introduction

Search queries are everywhere. Users type the first few words and expect the system to predict what comes next. Most systems today rely on neural networks trained on billions of parameters, sending user keystrokes to remote servers. But what if we could build something smarter with less?

**Query Prediction Engine** takes a step back and asks: *"Can we predict the next word accurately using lightweight statistical models?"* We believe the answer is yes. Using `n-gram` language models built directly from your corpus, we deliver:
- **Real-time predictions** (latency < 100ms)
- **Full transparency** (every suggestion is traceable to corpus statistics)
- **Zero external calls** (runs entirely locally)
- **Tiny footprint** (model size ~5MB for 30 documents)

> **Key insight:** Probabilistic models let us understand *why* a word was predicted. Compare that to black-box neural networks—here, you get explainability for free.

---

### Features 🔍

✨ **Language Model Statistics**
- Unigram, bigram, and trigram frequency tables
- Probability computation from corpus statistics
- Top-5 suggestions ranked by likelihood

⚡ **Real-time Performance**
- Process 30 documents in ~5 seconds
- Predict next word in < 100ms
- Run on 2GB RAM or less

<p align="center">
    <img src="static/readme_logo.png" alt="Query Prediction Engine" width="140" />
</p>

# Query Prediction Engine

│

Compact, local-first tool to generate next-word suggestions from a curated corpus. Designed for reproducibility, low resource usage, and interpretability.

## Highlights

- Local unigram/bigram/trigram models
- Fast, explainable predictions served via a Flask UI
- Corpus statistics and CSV export for auditability

## Quickstart

1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

2. Start the server

```bash
python app.py
```

Open http://localhost:5000

## Layout

- `app.py` — Flask app and API endpoints
- `templates/` — UI (`index.html`, `stats.html`)
- `scripts/` — preprocessing and model builders
- `corpus/` — source documents
- `data/` — generated model cache
- `static/` — assets (`logo.png`, `favicon.png`, `favicon.ico`)

## How it works (summary)

1. Preprocess corpus (clean, tokenize)
2. Build n-gram frequency tables
3. Compute conditional probabilities and serve top-k

## Development & Testing

- Rebuild models with scripts in `scripts/`.
- Tests under `tests/` (run with `pytest`).

## Contributors

- Aurelio Theodhore Riyanto — Full-stack
- Deo Dewanto — Data & corpus

---

For details, examples, and attribution see the repository on GitHub.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full text.
├── templates/
│   └── index.html              # Web UI (Flask)
│
├── corpus/                     # Input documents (30 Wikipedia articles)
│   ├── Machine_learning.txt
│   ├── Artificial_intelligence.txt
│   ├── Natural_language_processing.txt
│   └── ... (27 more)
│
└── data/
    └── language_model.pkl      # Cached model (auto-generated)
```

**System Flow:**
<div align="center">

![Query Prediction Engine](static/logo.png)

# Query Prediction Engine

Lightweight, explainable next-word prediction using n-gram language models.

Project: https://github.com/theodhorex/NLP_Query_Prediction_Engine

</div>

## Summary

This project implements a transparent next-word prediction system using unigram, bigram, and trigram frequency tables built from a local corpus. It focuses on low resource usage, interpretability, and offline operation (no external APIs required).

## Key Features

- Local n-gram language models (1–3 grams)
- Real-time predictions via a Flask web UI
- Corpus statistics dashboard and CSV export
- Small footprint; easy to inspect and reproduce predictions

## Quickstart

Prerequisites: Python 3.8+ and pip.

1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

2. Run the web server

```bash
python app.py
```

Open http://localhost:5000 in your browser.

## Project Layout

- `app.py` — Flask application and API endpoints
- `templates/` — HTML UI (`index.html`, `stats.html`)
- `scripts/` — preprocessing and language model code
- `corpus/` — plaintext documents used to build the model
- `data/` — generated model cache (`language_model.pkl`)
- `static/` — site assets (`logo.png`, `favicon.png`, `favicon.ico`)

## How It Works (brief)

1. Preprocess documents (normalize, tokenize)
2. Build n-gram frequency counts
3. Compute conditional probabilities for candidate words
4. Serve top-k suggestions via `/api/predict`

## Development

- To rebuild the model, run the preprocessing and language model scripts in `scripts/`.
- Tests are in `tests/` and can be run with `pytest`.

## Contributors

- Aurelio Theodhore Riyanto — Full-stack developer
- Deo Dewanto — Data engineer & corpus collection

## License

Academic project for UKDW course TI9203. See repository for details.
