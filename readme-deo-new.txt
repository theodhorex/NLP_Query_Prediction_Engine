# Query Prediction Engine

**Lightweight, explainable next-word prediction using N-gram Language Models with bilingual TF-IDF document retrieval.**

![Query Prediction Engine](static/logo.png)

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Core Components](#core-components)
  - [1. Document Preprocessor (`preprocess.py`)](#1-document-preprocessor)
  - [2. Language Model (`language_model.py`)](#2-language-model)
  - [3. Search Engine (`search_engine.py`)](#3-search-engine)
- [API Endpoints](#api-endpoints)
- [Algorithms & Methods](#algorithms--methods)
  - [N-gram Prediction Pipeline](#n-gram-prediction-pipeline)
  - [TF-IDF Relevance Scoring](#tf-idf-relevance-scoring)
  - [Additive (Laplace) Smoothing](#additive-laplace-smoothing)
  - [Bilingual Search Strategy](#bilingual-search-strategy)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Development & Testing](#development--testing)
- [Technical Specifications](#technical-specifications)
- [Contributors](#contributors)
- [License](#license)

---

## Overview

**Query Prediction Engine** is a statistical natural language processing system that predicts the next word in a partial query using **n-gram language models** (unigram, bigram, trigram). Unlike neural approaches (GPT, LSTM), this system operates on transparent, interpretable frequency tables built directly from a local corpus. Every prediction is traceable to specific corpus statistics.

The system also includes a **bilingual document search engine** (English and Indonesian) using TF-IDF vectorization and cosine similarity, powered by scikit-learn.

Key design goals:
- **Explainability**: every predicted word includes its frequency count and conditional probability
- **Low latency**: predictions served in under 100ms
- **Privacy**: fully offline, no external API calls
- **Minimal footprint**: model size ~5MB for 30 documents

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Flask Web Server (app.py)                  │
│                                                                     │
│  ┌─────────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  /api/predict    │  │  /api/stats  │  │  /api/search           │ │
│  │  POST            │  │  GET         │  │  POST                  │ │
│  └────────┬────────┘  └──────┬───────┘  └───────────┬────────────┘ │
│           │                  │                       │              │
│  ┌────────▼──────────────────▼───────────────────────▼──────────┐  │
│  │                     Service Layer                             │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │  │
│  │  │ Preprocessor │─▶│ LanguageModel │  │  SearchEngine    │   │  │
│  │  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘   │  │
│  │         │                 │                    │              │  │
│  │    ┌────▼────┐      ┌────▼────┐         ┌─────▼──────┐      │  │
│  │    │ corpus/ │      │  data/  │         │ corpus_id/ │      │  │
│  │    │  .txt   │      │ .pkl    │         │   .txt     │      │  │
│  │    └─────────┘      └─────────┘         └────────────┘      │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

The system follows a three-layer architecture:

1. **Web Layer** — Flask routes that handle HTTP requests, JSON serialization, CSV export
2. **Service Layer** — Three core services: preprocessing, language modeling, search
3. **Data Layer** — Raw corpus text files, serialized model cache (pickle)

---

## Core Components

### 1. Document Preprocessor

**File:** `scripts/preprocess.py`

The `DocumentPreprocessor` class handles text normalization and tokenization:

- **Cleaning**: lowercases text, removes punctuation via regex `[^\w\s]`
- **Tokenization**: uses NLTK's `word_tokenize`, filters to alphabetic tokens only
- **Batch loading**: reads all `.txt` files from a directory, tracks per-file errors
- **Error reporting**: accumulates load errors for debug visibility

```python
preprocessor = DocumentPreprocessor()
tokens = preprocessor.preprocess_document("Hello, World!")
# Result: ["hello", "world"]
```

### 2. Language Model

**File:** `scripts/language_model.py`

The `LanguageModel` class builds and serves n-gram predictions:

| Data Structure     | Type        | Description                         |
|--------------------|-------------|-------------------------------------|
| `unigrams`         | `Counter`   | Single word frequencies             |
| `bigrams`          | `Counter`   | Two-word sequence frequencies       |
| `trigrams`         | `Counter`   | Three-word sequence frequencies     |
| `vocabulary`       | `set`       | Unique tokens across all documents  |
| `document_stats`   | `list[dict]`| Per-document token counts           |

**Prediction algorithm** (three-tier fallback):

```
predict_next_word(query_words):
    1. TRIGRAM: if ≥2 context words available
       → score candidates from trigram table
       → P(w3 | w1, w2) = count(w1,w2,w3) / count(w1,w2)

    2. BIGRAM: fallback using last word
       → P(w2 | w1) = count(w1,w2) / count(w1)

    3. UNIGRAM: final fallback
       → P(w) = count(w) / total_tokens
       → returns most frequent words overall
```

Each step applies **additive (Laplace) smoothing** to handle unseen n-grams:
```
P_smooth(w | context) = (count(context, w) + 1) / (count(context) + |V|)
```

Partial prefix matching is supported: if the last word in the query partially matches known bigram/trigram first words (minimum 2 characters), they are included as candidates.

### 3. Search Engine

**File:** `scripts/search_engine.py`

The `SearchEngine` class provides bilingual document retrieval:

#### CorpusIndex
- Builds a **TF-IDF matrix** using `sklearn.feature_extraction.text.TfidfVectorizer`
- Parameters: `max_df=1.0`, `min_df=1`, `sublinear_tf=True`
- Stop words removed per language (NLTK's English and Indonesian stop word lists)
- Indonesian text optionally stemmed via **Sastrawi** stemmer

#### SearchEngine
- Maintains two `CorpusIndex` instances: English (`corpus/`) and Indonesian (`corpus_id/`)
- Preprocesses queries per language (stop word removal, optional stemming)
- Returns merged, score-sorted results across selected languages
- Each result includes: `score` (cosine similarity), `filename`, `language`, `snippet` (first 150 characters), `rank`

---

## API Endpoints

### `POST /api/predict`
Predict next word(s) from partial query.

**Request:**
```json
{ "query": "machine" }
```

**Response:**
```json
{
  "query": "machine",
  "timestamp": "2026-06-10T12:00:00Z",
  "predictions": [
    { "word": "learning",   "probability": 0.284, "count": 142, "rank": 1 },
    { "word": "vision",    "probability": 0.096, "count": 48,  "rank": 2 },
    { "word": "translation","probability": 0.074, "count": 37,  "rank": 3 }
  ]
}
```

Limits: max 200 characters, max 2 words context. Returns top-5 predictions.

### `GET /api/stats`
Model-level statistics (unigram/bigram/trigram counts, vocabulary size).

### `GET /api/ngram-stats?limit=20`
Top unigrams and bigrams with counts and probabilities.

### `GET /api/corpus-stats?limit=20`
Full corpus statistics including per-document breakdown.

### `POST /api/search`
Bilingual document search.

**Request:**
```json
{ "query": "machine learning", "top_k": 10, "language": "all" }
```

**Response:**
```json
{
  "query": "machine learning",
  "results": [
    { "score": 0.4271, "filename": "Machine_learning.txt", "language": "en", "snippet": "...", "rank": 1 }
  ],
  "total": 1,
  "language_filter": "all"
}
```

Language options: `"all"`, `"en"`, `"id"`. Max `top_k`: 30.

### `GET /api/export-csv?type={latest|history}`
CSV export of prediction data. Columns: `query`, `rank`, `word`, `probability`, `count`, `timestamp`.

### `GET /stats`
HTML page showing corpus statistics with bar charts and tables.

### `GET /search`
HTML page for interactive bilingual document search.

---

## Algorithms & Methods

### N-gram Prediction Pipeline

1. **Capture**: user input received as raw string
2. **Normalize**: lowercased, split into tokens, trimmed to last 2 words
3. **Retrieve**: matching entries looked up in trigram → bigram → unigram tables
4. **Score**: conditional probabilities computed with additive smoothing
5. **Rank**: top-k candidates sorted by descending probability
6. **Deliver**: JSON response with word, probability, count, and rank

### TF-IDF Relevance Scoring

Term Frequency–Inverse Document Frequency (TF-IDF) measures term importance:

```
tf-idf(t, d, D) = tf(t, d) × idf(t, D)
idf(t, D) = log(N / df(t))
```

- `tf(t, d)`: raw count of term `t` in document `d`
- `idf(t, D)`: inverse document frequency across corpus `D`
- `N`: total documents in corpus
- `df(t)`: number of documents containing term `t`

Sublinear term frequency scaling (`sublinear_tf=True`) applies: `tf = 1 + log(tf)`.

Query-document similarity is computed as **cosine similarity** between the query TF-IDF vector and each document TF-IDF vector:

```
cosine(q, d) = (q · d) / (||q|| × ||d||)
```

### Additive (Laplace) Smoothing

To handle n-grams not present in the training corpus, additive smoothing assigns a small non-zero probability:

```
P_smooth(w_n | w_1, ..., w_{n-1}) = (count(w_1, ..., w_n) + 1) / (count(w_1, ..., w_{n-1}) + |V|)
```

Where `|V|` is the vocabulary size. This ensures no candidate is assigned zero probability, improving generalization.

### Bilingual Search Strategy

The search engine maintains separate TF-IDF indices for English and Indonesian corpora:

- **English corpus** (`corpus/`): 30 Wikipedia articles on CS/AI/NLP topics
- **Indonesian corpus** (`corpus_id/`): 15 Wikipedia articles on equivalent topics
- Query preprocessing differs per language:
  - English: lowercased, punctuation removed, stop words filtered
  - Indonesian: same + optional Sastrawi stemming
- Results from both indices are merged and sorted by descending cosine similarity
- Language filter (`language` parameter) allows restricting to a single language

---

## Project Structure

```
├── app.py                          # Flask application, API routes, error handlers
├── scrape_corpus.py                # Wikipedia corpus scraper (BeautifulSoup)
├── requirements.txt                # Production dependencies
├── requirements-dev.txt            # Development dependencies (pytest, pytest-cov)
├── AGENTS.md                       # Agent configuration (empty)
├── LICENSE                         # MIT License
├── README.md                       # Project documentation
│
├── scripts/
│   ├── __init__.py
│   ├── preprocess.py               # DocumentPreprocessor class
│   ├── language_model.py           # LanguageModel class (n-gram)
│   └── search_engine.py            # SearchEngine + CorpusIndex classes
│
├── templates/
│   ├── index.html                  # Main UI: prediction interface + stats
│   ├── stats.html                  # Corpus statistics dashboard
│   └── search.html                 # Bilingual document search UI
│
├── static/
│   ├── logo.png                    # Project logo
│   ├── favicon.png                 # Favicon (256x256)
│   └── favicon.ico                 # Favicon (ICO format)
│
├── corpus/                         # English Wikipedia articles (30 files)
│   ├── Machine_learning.txt
│   ├── Artificial_intelligence.txt
│   ├── Natural_language_processing.txt
│   └── ... (27 more)
│
├── corpus_id/                      # Indonesian Wikipedia articles (15 files)
│   ├── pembelajaran_mesin.txt
│   ├── kecerdasan_buatan.txt
│   ├── pemrosesan_bahasa_alami.txt
│   └── ... (12 more)
│
├── data/
│   └── language_model.pkl          # Serialized model cache (auto-generated)
│
└── tests/
    ├── __init__.py
    ├── conftest.py                 # Test configuration (sys.path setup)
    ├── test_preprocess.py          # Tests for DocumentPreprocessor
    ├── test_language_model.py      # Tests for LanguageModel
    └── test_search_engine.py       # Tests for SearchEngine/CorpusIndex
```

---

## Installation

### Prerequisites

- Python 3.8+
- pip

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/theodhorex/NLP_Query_Prediction_Engine.git
   cd NLP_Query_Prediction_Engine
   ```

2. Install production dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Install Indonesian search dependencies:
   ```bash
   pip install sastrawi scikit-learn
   ```

   > **Note:** `scikit-learn` and `Sastrawi` are required only for the search engine. The core n-gram prediction works without them.

### Dependencies

| Package      | Version | Purpose                                |
|--------------|---------|----------------------------------------|
| Flask        | 2.3.0   | Web framework                          |
| nltk         | 3.8.1   | Tokenization, stop words               |
| Pillow       | 9.6.0   | Image handling (static assets)         |
| scikit-learn | —       | TF-IDF vectorization, cosine similarity|
| Sastrawi     | —       | Indonesian language stemmer            |
| pytest       | 7.4.4   | Test runner (dev)                      |
| pytest-cov   | 4.1.0   | Test coverage (dev)                    |

---

## Usage

### Start the Server

```bash
python app.py
```

On first launch, the system:
1. Scans `corpus/` for `.txt` documents
2. Preprocesses all documents (clean + tokenize)
3. Builds unigram, bigram, and trigram frequency tables
4. Initializes the bilingual search engine (TF-IDF indices)
5. Caches the model to `data/language_model.pkl`

Subsequent launches load the cached model (~5 seconds for 30 documents).

### Access the UI

- **Main Interface:** http://localhost:5000 — next-word prediction with live stats
- **Statistics Dashboard:** http://localhost:5000/stats — corpus metrics, top n-grams, document table
- **Document Search:** http://localhost:5000/search — bilingual EN/ID search with relevance scores

### API Usage

Predict next word:
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"query": "deep"}'
```

Search documents:
```bash
curl -X POST http://localhost:5000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "neural network", "language": "en", "top_k": 5}'
```

Export prediction history as CSV:
```bash
curl http://localhost:5000/api/export-csv?type=history -o predictions.csv
```

---

## Development & Testing

### Running Tests

```bash
pytest
```

With coverage:
```bash
pytest --cov=scripts --cov-report=term-missing
```

### Test Coverage

| Test File                | Tests      | Scope                                   |
|--------------------------|------------|-----------------------------------------|
| `test_preprocess.py`     | 4 tests    | Text cleaning, tokenization, validation |
| `test_language_model.py` | 4 tests    | Build, predict, persistence, edge cases |
| `test_search_engine.py`  | 13 tests   | Index build, search, bilingual, ranking |

### Rebuilding the Model

Delete the cached model and restart:
```bash
Remove-Item -LiteralPath "data/language_model.pkl"
python app.py
```

### Scraping New Corpus Data

```bash
python scrape_corpus.py
```

This fetches Wikipedia articles defined in the script's article list and saves them to `corpus/`.

---

## Technical Specifications

| Metric                    | Value                       |
|---------------------------|-----------------------------|
| Python version            | 3.8+                        |
| Framework                 | Flask 2.3.0                 |
| NLP library               | NLTK 3.8.1                  |
| ML library (search)       | scikit-learn                |
| Model serialization       | pickle                      |
| Language support          | English, Indonesian         |
| Corpus size (EN)          | 30 documents                |
| Corpus size (ID)          | 15 documents                |
| Model build time          | ~5 seconds (30 documents)   |
| Prediction latency        | <100ms                      |
| Model file size           | ~5MB                        |
| UI design system          | Coinbase Design System spec |
| License                   | MIT                         |

---

## Contributors

- **Aurelio Theodhore Riyanto** — Full-stack developer (Flask, UI, language model, search engine)
- **Deo Dewanto** — Data engineer & corpus collection (Wikipedia scraping, Indonesian corpus)

Academic project for course **TI9203** at **Universitas Kristen Duta Wacana (UKDW)**, Yogyakarta.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for full text.
