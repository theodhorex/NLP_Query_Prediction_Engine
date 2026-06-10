# Comprehensive Analysis Report: NLP Query Prediction Engine

**Project:** NLP Query Prediction Engine  
**Course:** TI9203 — Natural Language Processing  
**Institution:** Universitas Kristen Duta Wacana (UKDW), Yogyakarta  
**Authors:** Aurelio Theodhore Riyanto, Deo Dewanto  
**Date:** June 2026

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Web Application Flow](#3-web-application-flow)
4. [Preprocessing Pipeline](#4-preprocessing-pipeline)
5. [Language Model: Training & Prediction](#5-language-model-training--prediction)
6. [N-gram Algorithms & Mathematical Formulations](#6-n-gram-algorithms--mathematical-formulations)
7. [Search Engine: TF-IDF & Cosine Similarity](#7-search-engine-tf-idf--cosine-similarity)
8. [Technologies & Dependencies](#8-technologies--dependencies)
9. [API Specification](#9-api-specification)
10. [User Interface Walkthrough](#10-user-interface-walkthrough)
11. [Testing Strategy](#11-testing-strategy)
12. [Performance Characteristics](#12-performance-characteristics)

---

## 1. Project Overview

The **NLP Query Prediction Engine** is a statistical natural language processing system that performs **next-word prediction** using **n-gram language models** (unigram, bigram, and trigram). It is designed as a transparent, privacy-aware alternative to neural network-based prediction systems. Every prediction is traceable back to corpus frequency counts, providing full explainability.

The system also includes a **bilingual document search engine** (English and Indonesian) using **TF-IDF vectorization** with **cosine similarity**, demonstrating a second NLP application within the same web application.

### Design Principles

| Principle | Implementation |
|---|---|
| **Transparency** | Every prediction includes frequency count and conditional probability |
| **Low latency** | Predictions served in under 100ms |
| **Privacy** | Fully offline operation, zero external API calls |
| **Minimal footprint** | Model size ~5MB for 30 documents |
| **Reproducibility** | Same corpus always produces identical predictions |

---

## 2. System Architecture

The system follows a **three-layer architecture**:

```

┌─────────────────────────────────────────────────────────────────────────┐
│                         WEB LAYER (Flask)                               │
│                                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │  /api/predict│  │  /api/stats  │  │ /api/search  │  │ /export-csv│  │
│  │  POST        │  │  GET         │  │  POST        │  │  GET       │  │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └─────┬──────┘  │
│         │                 │                 │                │          │
│  ┌──────▼─────────────────▼─────────────────▼────────────────▼──────┐  │
│  │                      SERVICE LAYER                                 │  │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐   │  │
│  │  │DocumentPrepro- │  │ LanguageModel  │  │   SearchEngine     │   │  │
│  │  │cessor          │→ │ (n-gram)       │  │  (TF-IDF + CosSim) │   │  │
│  │  └───────┬────────┘  └───────┬────────┘  └─────────┬──────────┘   │  │
│  └──────────┼───────────────────┼──────────────────────┼──────────────┘  │
│             │                   │                      │                 │
│  ┌──────────▼───────────────────▼──────────────────────▼──────────────┐  │
│  │                       DATA LAYER                                    │  │
│  │                                                                     │  │
│  │  ┌──────────────┐    ┌──────────────────────┐    ┌──────────────┐   │  │
│  │  │   corpus/    │    │  data/language_model │    │  corpus_id/  │   │  │
│  │  │  30 EN .txt  │    │    .pkl (cache)      │    │  15 ID .txt  │   │  │
│  │  └──────────────┘    └──────────────────────┘    └──────────────┘   │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────────────┘
```

### Layer Breakdown

**1. Web Layer (`app.py`)** — Flask application handling HTTP routing, request validation, JSON serialization, error handling, and CSV export. Serves both RESTful API endpoints and HTML templates.

**2. Service Layer (`scripts/`)** — Three core services:
- `DocumentPreprocessor`: Text cleaning and tokenization
- `LanguageModel`: N-gram frequency table building and next-word prediction
- `SearchEngine`: TF-IDF indexing and bilingual document retrieval

**3. Data Layer** — Raw corpus text files, cached serialized model, static assets.

---

## 3. Web Application Flow

### 3.1 Application Startup Sequence

When `python app.py` is executed, the following sequence occurs:

```
Start
  │
  ├─► load_or_build_model()
  │     │
  │     ├─► Check: does data/language_model.pkl exist?
  │     │     │
  │     │     ├─► YES → Load pickle → Deserialize → model ready
  │     │     │
  │     │     └─► NO  → Load .txt from corpus/
  │     │               │
  │     │               ├─► For each .txt file:
  │     │               │     ├─► Read raw text (UTF-8)
  │     │               │     ├─► clean_text() → lower + remove punctuation
  │     │               │     ├─► tokenize() → NLTK word_tokenize
  │     │               │     └─► Append {filename, tokens, text}
  │     │               │
  │     │               ├─► model.build_model(documents)
  │     │               │     ├─► Build unigram Counter
  │     │               │     ├─► Build bigram Counter (n-2 sliding window)
  │     │               │     ├─► Build trigram Counter (n-3 sliding window)
  │     │               │     └─► Build vocabulary set
  │     │               │
  │     │               └─► model.save_model(MODEL_PATH) → pickle dump
  │     │
  │     └─► Initialize SearchEngine
  │           ├─► Load corpus/ → build English TF-IDF index
  │           └─► Load corpus_id/ → build Indonesian TF-IDF index
  │
  └─► Start Flask on port 5000
```

### 3.2 Prediction Request Flow

When a user types a query and the frontend calls `POST /api/predict`:

```
User Input ("machine learning")
  │
  ├─► Frontend: debounce 300ms → fetch POST /api/predict
  │
  ├─► Flask: predict()
  │     ├─► Validate request JSON
  │     ├─► Strip query, limit to 200 chars
  │     ├─► Check: empty? → 400 error
  │     ├─► Split into words, take last 2
  │     │
  │     ├─► model.predict_next_word(["machine", "learning"])
  │     │     │
  │     │     ├─► TRIGRAM phase (since ≥2 context words)
  │     │     │     ├─► Find all trigrams where (w1, w2) match context
  │     │     │     ├─► Compute P(w3 | w1, w2) with Laplace smoothing
  │     │     │     ├─► Include partial prefix matches (w1 matches, w2 starts with input)
  │     │     │     └─► Sort by probability descending → return top-10
  │     │     │
  │     │     ├─► BIGRAM phase (fallback if trigram yields nothing)
  │     │     │     ├─► Match bigrams where w1 == last_word
  │     │     │     ├─► Compute P(w2 | w1) with Laplace smoothing
  │     │     │     └─► Sort descending → return top-10
  │     │     │
  │     │     └─► UNIGRAM phase (final fallback)
  │     │           ├─► Take most common words globally
  │     │           └─► P(w) = count(w) / total_tokens
  │     │
  │     ├─► Build response JSON with word, probability, count, rank
  │     ├─► Append to prediction_history (in-memory, max 200)
  │     └─► Return JSON response
  │
  └─► Frontend: render predictions as clickable badges
```

### 3.3 Search Request Flow

```
User Input ("neural network")
  │
  ├─► Frontend: fetch POST /api/search {query, language, top_k}
  │
  ├─► Flask: search()
  │     ├─► Validate request
  │     ├─► Determine language filter (all/en/id)
  │     │
  │     ├─► SearchEngine.search(query, top_k, language)
  │     │     │
  │     │     ├─► If language = all → search both EN and ID indices
  │     │     │
  │     │     ├─► For each active index:
  │     │     │     ├─► preprocess_query()
  │     │     │     │     ├─► Lowercase, remove punctuation
  │     │     │     │     ├─► Tokenize
  │     │     │     │     ├─► Remove stopwords (per language)
  │     │     │     │     └─► If ID: optional Sastrawi stemming
  │     │     │     │
  │     │     │     ├─► vectorizer.transform([cleaned_query])
  │     │     │     ├─► cosine_similarity(query_vec, tfidf_matrix)
  │     │     │     └─► Collect (doc_index, score) pairs
  │     │     │
  │     │     ├─► Merge results from all active indices
  │     │     ├─► Sort by score descending
  │     │     ├─► Filter scores > 0.001
  │     │     ├─► Assign ranks
  │     │     └─► Return top_k results
  │     │
  │     └─► Return JSON with results, total count, language filter
  │
  └─► Frontend: render ranked results with progress bars
```

---

## 4. Preprocessing Pipeline

### 4.1 Text Cleaning (`clean_text`)

```
Input:  "Hello, World! Machine learning is fun."
Output: "hello world machine learning is fun"
```

**Operations performed:**
1. **Lowercasing**: All characters converted to lowercase
2. **Punctuation removal**: Regex `[^\w\s]` removes all non-word, non-whitespace characters

**Formula:**
```
clean(text) = lower(replace(r'[^\w\s]', '', text))
```

### 4.2 Tokenization (`tokenize`)

**Method:** NLTK `word_tokenize` (Punkt tokenizer)

**Post-filtering:**
- Only alphabetic tokens retained: `token.isalpha()` — filters out numbers and punctuation remnants

```
Input:  "hello world machine learning"
Output: ["hello", "world", "machine", "learning"]
```

### 4.3 Complete Preprocessing Chain

```
Raw Text
  │
  ├─► clean_text():
  │     ├─► .lower()
  │     └─► re.sub(r'[^\w\s]', '', text)
  │
  ├─► tokenize(cleaned_text):
  │     ├─► word_tokenize(text)
  │     └─► [t for t in tokens if t.isalpha()]
  │
  └─► Result: List[str] of cleaned tokens
```

### 4.4 Batch Document Loading (`load_documents_from_folder`)

- Scans a directory for `.txt` files
- For each file:
  - Reads with UTF-8 encoding
  - Runs full preprocessing chain
  - Appends to list as `{filename: str, tokens: List[str], text: str}`
- Accumulates errors per file (does not halt on single file failure)
- Returns empty list with detailed error messages if all files fail

---

## 5. Language Model: Training & Prediction

### 5.1 Model Training (`build_model`)

The `LanguageModel` class builds frequency tables from preprocessed documents.

**Data structures used:**

| Structure | Type | Description | Example Entry |
|---|---|---|---|
| `unigrams` | `Counter` | Single word frequencies | `"learning": 142` |
| `bigrams` | `Counter` | Two-word sequence frequencies | `("machine", "learning"): 89` |
| `trigrams` | `Counter` | Three-word sequence frequencies | `("machine", "learning", "algorithm"): 12` |
| `vocabulary` | `set` | Unique tokens across corpus | `{"machine", "learning", ...}` |

**Training algorithm (per document):**

For each document with token list `T = [t₁, t₂, ..., tₙ]`:

```python
# Unigrams: count each token
for token in T:
    unigrams[token] += 1

# Bigrams: sliding window of size 2
for i in range(len(T) - 1):
    bigrams[(T[i], T[i+1])] += 1

# Trigrams: sliding window of size 3
for i in range(len(T) - 2):
    trigrams[(T[i], T[i+1], T[i+2])] += 1
```

Built using NLTK's `ngrams()` utility function which generates sliding-window n-gram sequences:

```python
from nltk.util import ngrams
bigram_list = list(ngrams(tokens, 2))    # n=2
trigram_list = list(ngrams(tokens, 3))   # n=3
```

### 5.2 Model Persistence

- **Serialization format:** Python `pickle`
- **Cache location:** `data/language_model.pkl`
- **Cached data:** `{unigrams, bigrams, trigrams, vocabulary, document_stats, total_tokens, total_documents}`
- **Cache strategy:** On subsequent launches, the pickle is deserialized instead of rebuilding — reduces startup from ~5s to <1s

### 5.3 Prediction Algorithm

The prediction uses a **three-tier hierarchical fallback strategy**:

```
predict_next_word(query_words: List[str]) → List[Dict]
```

#### Tier 1: Trigram (context = 2 words)

**Condition:** `len(query_words) >= 2`

**Process:**
1. Let `(w₁, w₂)` be the last two words of the query
2. Search trigram table for all entries `(w₁, w₂, w₃)` where first two words match
3. Also include partial prefix matches: `w₁` matches AND `w₂` starts with the input `w₂` (minimum 2 characters match)

**Probability formula (with Laplace smoothing):**
```
P(w₃ | w₁, w₂) = (count(w₁, w₂, w₃) + 1) / (count(w₁, w₂) + |V|)

Where:
- count(w₁, w₂, w₃) = trigram frequency
- count(w₁, w₂) = bigram frequency of context
- |V| = vocabulary size (unique tokens)
```

#### Tier 2: Bigram (context = 1 word)

**Condition:** Tier 1 yields no candidates

**Process:**
1. Let `last_word` be the final word of the query
2. Search bigram table for all entries `(w₁, w₂)` where `w₁ == last_word`
3. Also include partial prefix matches: `w₁` starts with `last_word` (min 2 chars)

**Probability formula (with Laplace smoothing):**
```
P(w₂ | w₁) = (count(w₁, w₂) + 1) / (count(w₁) + |V|)

Where:
- count(w₁, w₂) = bigram frequency
- count(w₁) = unigram frequency of context word
- |V| = vocabulary size
```

#### Tier 3: Unigram (no context)

**Condition:** Tier 2 yields no candidates (or query is empty)

**Process:**
Return the most frequent words from the entire corpus.

**Probability formula:**
```
P(w) = count(w) / total_tokens

Where:
- count(w) = unigram frequency of word w
- total_tokens = sum of all unigram counts
```

### 5.4 Complete Prediction Flow Diagram

```
User query: "machine"
  │
  ├─► query_words = ["machine"]  (1 word)
  │
  ├─► Tier 1 (Trigram): SKIP — need ≥2 words
  │
  ├─► Tier 2 (Bigram):
  │     ├─► Find bigrams (machine, *)
  │     │     ├─► ("machine", "learning"): count=89
  │     │     ├─► ("machine", "translation"): count=37
  │     │     └─► ("machine", "vision"): count=12
  │     │
  │     ├─► Compute probabilities:
  │     │     ├─► P(learning | machine) = (89+1) / (500+12980) = 0.0067
  │     │     ├─► P(translation | machine) = (37+1) / (500+12980) = 0.0028
  │     │     └─► P(vision | machine) = (12+1) / (500+12980) = 0.0009
  │     │
  │     └─► Return top-10 sorted by probability
  │
  └─► Response: [{"word": "learning", "prob": 0.0067, "count": 89, "rank": 1}, ...]
```

---

## 6. N-gram Algorithms & Mathematical Formulations

### 6.1 N-gram Language Model

An **n-gram** is a contiguous sequence of `n` items from a text corpus. The n-gram language model estimates the probability of a word given its preceding context.

**General n-gram probability:**
```
P(wₙ | w₁, ..., wₙ₋₁) = count(w₁, ..., wₙ) / count(w₁, ..., wₙ₋₁)
```

### 6.2 Models Used in This Project

| Model | n | Context Size | Probability Formula |
|---|---|---|---|
| Unigram | 1 | 0 words | P(w) = count(w) / N |
| Bigram | 2 | 1 word | P(w₂ | w₁) = count(w₁, w₂) / count(w₁) |
| Trigram | 3 | 2 words | P(w₃ | w₁, w₂) = count(w₁, w₂, w₃) / count(w₁, w₂) |

Where `N` = total number of tokens in the corpus.

### 6.3 Additive (Laplace) Smoothing

**Problem:** Unseen n-grams receive zero probability, making them impossible to predict.

**Solution:** Add 1 to all n-gram counts (pseudocount).

**Smoothed probability formula:**
```
P_smooth(wₙ | w₁, ..., wₙ₋₁) = (count(w₁, ..., wₙ) + 1) / (count(w₁, ..., wₙ₋₁) + |V|)
```

Where:
- `|V|` = vocabulary size (number of unique tokens)
- `+1` in numerator: Laplace (add-one) smoothing
- `+ |V|` in denominator: ensures probabilities sum to 1

**Example:**
```
Without smoothing:
  P(quantum | machine) = count(machine, quantum) / count(machine)
                        = 0 / 500 = 0

With Laplace smoothing:
  P_smooth(quantum | machine) = (0 + 1) / (500 + 12980) = 1/13480 ≈ 0.000074
```

### 6.4 Partial Prefix Matching

The system includes a heuristic for handling incomplete input words (e.g., typing "learn" instead of "learning"):

**Condition:** If `input_word` is ≥2 characters AND `matched_word` starts with `input_word` AND `matched_word ≠ input_word`:

```
Examples:
  Input: "mach" → matches "machine" (mach starts with "mach")
  Input: "l" → NO MATCH (less than 2 chars)
  Input: "machine" → EXACT MATCH (standard lookup)
```

This provides **fuzzy/prefix completion** within the n-gram framework.

### 6.5 Ranking & Selection

- All candidate words are scored using the appropriate tier's probability formula
- Candidates are sorted by probability in **descending order**
- Top-10 candidates are returned
- Rank (1-indexed) is assigned in the response
- Probabilities are reported as raw floats (not percentages)

---

## 7. Search Engine: TF-IDF & Cosine Similarity

### 7.1 TF-IDF (Term Frequency — Inverse Document Frequency)

TF-IDF is a numerical statistic that reflects the importance of a term to a document within a corpus.

#### Term Frequency (TF)

Raw count of term `t` in document `d`:

```
tf(t, d) = raw count of t in d
```

With **sublinear scaling** (`sublinear_tf=True` in scikit-learn):

```
tf(t, d) = 1 + log(raw_count)  if raw_count > 0
tf(t, d) = 0                    otherwise
```

#### Inverse Document Frequency (IDF)

```
idf(t, D) = log(N / df(t))

Where:
- N = total number of documents in corpus
- df(t) = number of documents containing term t
```

#### TF-IDF Score

```
tf-idf(t, d, D) = tf(t, d) × idf(t, D)
```

### 7.2 Cosine Similarity

The similarity between a query and a document is computed as the cosine of the angle between their TF-IDF vectors:

```
cosine(q, d) = (q · d) / (||q|| × ||d||)
```

Where:
- `q · d` = dot product of query vector and document vector
- `||q||` = Euclidean norm (magnitude) of query vector
- `||d||` = Euclidean norm of document vector

**Range:** [0, 1] — 0 = no similarity, 1 = identical

### 7.3 Search Engine Architecture

```
SearchEngine
  │
  ├─► en_index: CorpusIndex (English)
  │     ├─► Corpus: 30 Wikipedia articles from corpus/
  │     ├─► Stop words: NLTK English stopwords
  │     └─► TF-IDF parameters: max_df=1.0, min_df=1, sublinear_tf=True
  │
  └─► id_index: CorpusIndex (Indonesian)
        ├─► Corpus: 15 Wikipedia articles from corpus_id/
        ├─► Stop words: NLTK Indonesian stopwords
        ├─► Stemming: Sastrawi (optional, if installed)
        └─► TF-IDF parameters: same as English
```

### 7.4 Bilingual Search Strategy

1. **Query preprocessing differs per language:**
   - **English:** lowercase → remove punctuation → tokenize → remove English stopwords
   - **Indonesian:** lowercase → remove punctuation → tokenize → remove Indonesian stopwords → Sastrawi stem

2. **Search execution:**
   - If `language=all`: search both indices, merge results
   - If `language=en`: search English index only
   - If `language=id`: search Indonesian index only

3. **Result merging:**
   - Combined results sorted by cosine similarity score (descending)
   - Filter: scores ≤ 0.001 are excluded
   - Top-k results returned with assigned ranks

4. **Each result contains:**
   - `score`: cosine similarity (0-1, rounded to 4 decimal places)
   - `filename`: source document name
   - `language`: "en" or "id"
   - `snippet`: first 150 characters of document text
   - `rank`: position in sorted results

---

## 8. Technologies & Dependencies

### 8.1 Production Dependencies

| Package | Version | Purpose | Usage Location |
|---|---|---|---|
| **Flask** | 2.3.0 | Web framework (routing, request handling, templating) | `app.py` |
| **nltk** | 3.8.1 | Natural Language Toolkit (tokenization, stopwords, n-grams) | `preprocess.py`, `language_model.py`, `search_engine.py` |
| **Pillow** | 9.6.0 | Python Imaging Library (serving static image assets) | `static/` (logo, favicon) |
| **scikit-learn** | — | Machine learning library (TF-IDF vectorization, cosine similarity) | `search_engine.py` |
| **Sastrawi** | — | Indonesian language stemmer | `search_engine.py` (optional) |
| **BeautifulSoup4** | — | HTML parsing for Wikipedia scraper | `scrape_corpus.py` |
| **requests** | — | HTTP client for Wikipedia scraper | `scrape_corpus.py` |

### 8.2 Development Dependencies

| Package | Version | Purpose |
|---|---|---|
| **pytest** | 7.4.4 | Test framework |
| **pytest-cov** | 4.1.0 | Test coverage reporting |

### 8.3 Frontend Technologies

| Technology | Purpose |
|---|---|
| **Tailwind CSS** (CDN) | Utility-first CSS framework for UI styling |
| **Google Fonts (Inter + JetBrains Mono)** | Typography |
| **Vanilla JavaScript** | Client-side interactivity, API calls, localStorage |
| **IntersectionObserver API** | Scroll-triggered reveal animations |

### 8.4 Key Python Libraries Used

**NLTK (`nltk`):**
- `nltk.tokenize.word_tokenize`: Punkt tokenizer for word boundary detection
- `nltk.corpus.stopwords`: Stop word lists for English and Indonesian
- `nltk.util.ngrams`: Sliding-window n-gram generation

**scikit-learn (`sklearn`):**
- `sklearn.feature_extraction.text.TfidfVectorizer`: TF-IDF matrix construction
- `sklearn.metrics.pairwise.cosine_similarity`: Vector similarity computation

---

## 9. API Specification

### 9.1 `POST /api/predict`

Predict next word(s) from a partial user query.

**Request:**
```json
{ "query": "machine learning" }
```

**Response:**
```json
{
  "query": "machine learning",
  "timestamp": "2026-06-10T12:00:00.000000+00:00",
  "predictions": [
    { "word": "algorithm",   "probability": 0.067, "count": 12, "rank": 1 },
    { "word": "model",       "probability": 0.051, "count": 9,  "rank": 2 },
    { "word": "technique",   "probability": 0.039, "count": 7,  "rank": 3 }
  ]
}
```

**Validation rules:**
| Rule | Limit | HTTP Code |
|---|---|---|
| Max query length | 200 characters | 400 |
| Max context words | 2 words (older words truncated) | N/A (truncated silently) |
| Empty query | Not allowed | 400 |

### 9.2 `GET /api/stats`

Model-level statistics.

**Response:**
```json
{
  "unigrams_count": 148210,
  "bigrams_count": 512884,
  "trigrams_count": 1024332,
  "vocabulary_size": 12980
}
```

### 9.3 `GET /api/ngram-stats?limit=20`

Top unigrams and bigrams with counts and probabilities.

### 9.4 `GET /api/corpus-stats?limit=20`

Full corpus statistics, including per-document breakdown, total tokens, unique words, average tokens per document.

### 9.5 `POST /api/search`

Bilingual document search across English and Indonesian corpora.

**Request:**
```json
{
  "query": "machine learning",
  "top_k": 10,
  "language": "all"
}
```

**Languages:** `"all"` (default), `"en"`, `"id"`  
**Max top_k:** 30

### 9.6 `GET /api/export-csv?type=latest|history`

CSV export of prediction data. Columns: `query, rank, word, probability, count, timestamp`.

### 9.7 `GET /` — Main HTML interface
### 9.8 `GET /stats` — Statistics dashboard
### 9.9 `GET /search` — Search interface

---

## 10. User Interface Walkthrough

### 10.1 Main Interface (`index.html`)

The main interface is a single-page application with these sections:

1. **Hero Section** — Dark-themed header showing model snapshot (unigram/bigram/trigram counts, vocabulary size)
2. **How It Works** — Three cards explaining: Corpus Indexing, Probability Scoring, Top-k Suggestions
3. **Pipeline Lifecycle** — Four-step diagram: Capture → Normalize → Retrieve → Deliver
4. **Live Prediction Interface:**
   - Statistics bar showing live unigram/bigram/trigram/vocabulary counts (fetched from `/api/stats`)
   - Text input field with 300ms debounce
   - Prediction badges showing top-5 predicted words with probability percentages
   - Clicking a prediction appends it and triggers re-prediction
   - Recent searches panel (localStorage, paginated, max 50 entries)
   - Most frequent queries panel
   - CSV export buttons (latest prediction or full history)

**Frontend Prediction Flow:**
```
keyup → debounce(300ms) → POST /api/predict → render badges
  │
  ├─► Click badge → append word → re-predict
  ├─► Click history entry → set query → re-predict
  ├─► Save to localStorage → render history panels
  └─► Update statistics display
```

### 10.2 Statistics Dashboard (`stats.html`)

- Total documents, total tokens, unique words, avg tokens per document
- Top unigrams with horizontal bar charts
- Top bigrams with horizontal bar charts
- Corpus overview table (filename, token count, unique words per document)

### 10.3 Search Interface (`search.html`)

- Search input with language filter dropdown (All/English/Indonesian)
- Results displayed as ranked cards with:
  - Rank number badge
  - Document filename
  - Language label (EN/ID)
  - Similarity score as progress bar + percentage
  - Relevance label (High/Medium/Low based on score thresholds)
  - Text snippet (first 150 chars)

---

## 11. Testing Strategy

### 11.1 Test Framework

- **pytest** 7.4.4 with `pytest-cov` 4.1.0 for coverage reporting

### 11.2 Test Files

#### `test_preprocess.py` (4 tests)

| Test | Scenario | Expected |
|---|---|---|
| `test_clean_text` | `"Hello, World!"` → cleaned | `"hello world"` |
| `test_tokenize` | `"machine learning"` → tokens | `["machine", "learning"]` |
| `test_empty_document` | Empty string → ValueError | Raises `ValueError` |
| `test_invalid_text_type` | Integer input → ValueError | Raises `ValueError` |

#### `test_language_model.py` (4 tests)

| Test | Scenario | Expected |
|---|---|---|
| `test_build_model` | Build from 1 doc | Correct unigram/bigram/trigram counts |
| `test_predict_next_word` | Predict after `["machine"]` | First prediction is `"learning"` |
| `test_empty_query` | `predict_next_word([])` | Empty list |
| `test_model_persistence` | Save → load → compare | Loaded model matches original |

#### `test_search_engine.py` (13 tests)

| Test | Scenario | Expected |
|---|---|---|
| `test_build_empty_documents` | Build with empty list | No index created, search returns empty |
| `test_build_and_search_basic` | 2 docs, search query | Most relevant doc ranked first |
| `test_search_before_build` | Search without building | Empty list |
| `test_search_result_fields` | Single doc search | Correct score, filename, language, snippet |
| `test_search_top_k` | 10 docs, top_k=3 | ≤3 results returned |
| `test_search_ranks_by_score` | Match vs non-match | Match ranked first |
| `test_search_empty_query` | Empty/whitespace query | Empty list |
| `test_search_english_only` | Language filter "en" | All results have language="en" |
| `test_search_indonesian_only` | Language filter "id" | All results have language="id" |
| `test_search_all_languages_bilingual` | "all" filter | Returns results from both languages |
| `test_search_all_merges_and_sorts` | Merged bilingual results | Sorted descending by score |
| `test_search_returns_rank` | Single result | `rank` field = 1 |
| `test_search_snippet_is_first_150_chars` | Long document | Snippet ≤ 150 characters |
| `test_search_corpus_id_not_found_logs_warning` | Missing ID corpus | Warning logged |
| `test_search_without_index` | Search without building | Empty list |

### 11.3 Test Coverage (as designed)

| Module | Lines | Coverage Target |
|---|---|---|
| `preprocess.py` | 97 | Core logic: clean, tokenize, preprocess_document |
| `language_model.py` | 247 | Build, predict, save, load, statistics |
| `search_engine.py` | 154 | Index build, search, bilingual merging |

---

## 12. Performance Characteristics

### 12.1 Measured Benchmarks

| Metric | Value |
|---|---|
| Model build time (30 documents) | ~5 seconds |
| Prediction latency | <100ms |
| Model file size (pickle cache) | ~5MB |
| Server memory usage | ~100-200MB |
| Startup time (cached) | <1 second |
| Startup time (first build) | ~5-10 seconds |
| Prediction history (in-memory limit) | 200 entries |

### 12.2 Corpus Statistics (as configured)

| Metric | English (corpus/) | Indonesian (corpus_id/) |
|---|---|---|
| Documents | 30 | 15 |
| Topics | CS, AI, NLP, Statistics | CS, AI, NLP (Indonesian) |
| Source | Wikipedia | Wikipedia |
| Acquisition | `scrape_corpus.py` (BeautifulSoup) | Manual collection |

### 12.3 Scalability Considerations

- **Time complexity of prediction:** O(V) worst where V = vocabulary size (linear scan of n-gram tables)
- **Time complexity of building:** O(N × L) where N = number of documents, L = average document length
- **Space complexity of model:** O(U + B + T) where U/B/T = unique unigram/bigram/trigram types
- **Limitation:** Current implementation keeps all n-gram counts in memory (RAM-bound)
- **Limitation:** Prediction searches are linear scans; no index structures (e.g., hash maps on prefixes) are used beyond Python's built-in `Counter` dictionary lookup

---

## Appendix A: Complete File Listing

```
├── app.py                        # Flask web application (253 lines)
├── scrape_corpus.py              # Wikipedia scraper (125 lines)
├── requirements.txt              # Production dependencies
├── requirements-dev.txt          # Development dependencies
├── AGENTS.md                     # Agent configuration (empty)
├── LICENSE                       # MIT License
├── README.md                     # Project documentation (188 lines)
├── readme-deo-new.txt            # Extended documentation (489 lines)
├── .gitignore                    # Git ignore rules
│
├── scripts/
│   ├── __init__.py
│   ├── preprocess.py             # DocumentPreprocessor (97 lines)
│   ├── language_model.py         # LanguageModel (247 lines)
│   └── search_engine.py          # SearchEngine + CorpusIndex (154 lines)
│
├── templates/
│   ├── index.html                # Main prediction UI (757 lines)
│   ├── stats.html                # Statistics dashboard (256 lines)
│   └── search.html               # Bilingual search UI (257 lines)
│
├── static/
│   ├── styles.css                # Custom CSS (99 lines)
│   ├── logo.png
│   ├── logo2.png
│   ├── favicon.png
│   └── favicon.ico
│
├── corpus/                       # 30 English Wikipedia articles (EN)
├── corpus_id/                    # 15 Indonesian Wikipedia articles (ID)
├── data/
│   └── language_model.pkl        # Serialized model cache
│
└── tests/
    ├── __init__.py
    ├── conftest.py               # sys.path configuration
    ├── test_preprocess.py        # 4 tests
    ├── test_language_model.py    # 4 tests
    └── test_search_engine.py     # 13 tests
```

## Appendix B: Mathematical Notation Summary

| Symbol | Meaning |
|---|---|
| P(w) | Probability of word w |
| P(wₙ | w₁, ..., wₙ₋₁) | Conditional probability of wₙ given preceding words |
| count(sequence) | Frequency of n-gram in corpus |
| |V| | Vocabulary size (number of unique tokens) |
| N | Total number of tokens in corpus |
| tf(t, d) | Term frequency of term t in document d |
| idf(t, D) | Inverse document frequency of term t across corpus D |
| df(t) | Document frequency of term t |
| cos(q, d) | Cosine similarity between query q and document d |
| q · d | Dot product of vectors q and d |
| ||q|| | Euclidean norm of vector q |
