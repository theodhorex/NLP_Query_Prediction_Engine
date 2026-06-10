---
# LAPORAN ANALISIS LENGKAP PROYEK: NLP QUERY PREDICTION ENGINE
**Tanggal Analisis:** 11 Juni 2026
**Analis:** AI Technical Analyst

---

## 1. PROJECT OVERVIEW

### Temuan

**Nama Proyek:** NLP Query Prediction Engine  
**Tujuan Utama:** Sistem prediksi kata berikutnya (next-word prediction) berbasis n-gram language model yang transparan, cepat, dan berjalan sepenuhnya secara lokal. Proyek ini juga mencakup mesin pencari dokumen bilingual (Inggris-Indonesia) menggunakan TF-IDF dan cosine similarity.

**Problem yang Diselesaikan:**
- Kebutuhan prediksi kata selanjutnya saat pengguna mengetik query pencarian
- Ketergantungan pada sistem neural network besar (GPT, LSTM) yang membutuhkan sumber daya komputasi tinggi dan koneksi internet
- Kurangnya transparansi pada model prediksi konvensional (black-box)
- Kebutuhan akan sistem yang dapat diaudit dan dilacak asal-usul prediksinya

**Target Pengguna:**
- Mahasiswa dan peneliti NLP yang mempelajari language model statistik
- Pengembang yang membutuhkan sistem prediksi lightweight dan offline
- Akademisi UKDW untuk kebutuhan perkuliahan TI9203

**Stack Teknologi:**

| Layer | Teknologi |
|---|---|
| Backend Framework | Flask 2.3.0 |
| NLP Library | NLTK 3.8.1 |
| ML Library | scikit-learn (TF-IDF) |
| Frontend | Vanilla JavaScript + Tailwind CSS (CDN) |
| Frontend Fonts | Inter, JetBrains Mono (Google Fonts) |
| Template Engine | Jinja2 (Flask default) |
| Serialization | Python pickle |
| Indonesian NLP | Sastrawi (opsional) |
| Testing | pytest 7.4.4, pytest-cov 4.1.0 |

**Skala Kompleksitas:** Skala kecil hingga menengah. Total ~2,500 baris kode (Python + JS + HTML + CSS). Model n-gram dapat menangani 30-45 dokumen dengan prediksi <100ms.

### Penilaian
**Baik** — Proyek fokus, terdefinisi dengan jelas, dan sesuai dengan tujuan akademisnya.

### Rekomendasi
- Pertimbangkan menambahkan `pyproject.toml` untuk standardisasi build modern
- Dokumentasikan roadmap pengembangan ke depan

---

## 2. FEATURE ANALYSIS

### Temuan

**Fitur Utama (Core Features):**

| Fitur | Status | Prioritas | Keterangan |
|---|---|---|---|
| Prediksi n-gram (unigram/bigram/trigram) | ✅ Selesai | Wajib | 3-tier fallback dengan Laplace smoothing |
| Prefix/fuzzy matching pada input | ✅ Selesai | Wajib | Partial word matching (≥2 karakter) |
| API REST `/api/predict` | ✅ Selesai | Wajib | POST endpoint untuk prediksi |
| Live UI prediksi (debounce 300ms) | ✅ Selesai | Wajib | Input + badge hasil prediksi |
| Click-to-append pada prediksi | ✅ Selesai | Pendukung | Klik kata prediksi untuk menambahkan |
| Riwayat pencarian (localStorage) | ✅ Selesai | Pendukung | 50 entry, pagination |
| CSV export (latest/history) | ✅ Selesai | Pendukung | Download hasil prediksi |
| Statistik model (`/api/stats`) | ✅ Selesai | Pendukung | Unigram/bigram/trigram/vocab count |
| Statistik korpus (`/api/corpus-stats`) | ✅ Selesai | Pendukung | Per-document breakdown, top n-gram |
| Dashboard statistik (`/stats`) | ✅ Selesai | Pendukung | Bar chart + tabel dokumen |
| Search engine bilingual (`/api/search`) | ✅ Selesai | Wajib | TF-IDF + cosine similarity |
| Search UI bilingual (`/search`) | ✅ Selesai | Pendukung | Filter bahasa EN/ID |
| Scraping korpus Wikipedia | ✅ Selesai | Pendukung | `scrape_corpus.py` |
| Unit testing | ✅ Selesai | Wajib | 21 tests total |
| Caching model (pickle) | ✅ Selesai | Wajib | `data/language_model.pkl` |

**Fitur yang Tidak Ada / Belum Diimplementasikan:**

| Fitur | Keterangan | Prioritas |
|---|---|---|
| Autentikasi pengguna | Tidak diperlukan untuk scope akademis | Low |
| Multiple model versioning | Hanya 1 model aktif | Low |
| A/B testing | Tidak diperlukan | Low |
| Real-time websocket | Menggunakan HTTP polling | Low |
| Dark mode toggle | Hanya static theme | Medium |
| Data persistence (database) | Riwayat di localStorage (client-side) | Medium |
| Export model metrics | Hanya via API JSON | Low |
| User settings/preferences | Tidak ada | Low |
| Multi-language support | Hanya EN + ID | Low |

### Penilaian
**Baik** — Semua fitur inti selesai dan berfungsi. Fitur pendukung cukup lengkap untuk proyek akademis.

### Rekomendasi
- Tambahkan fitur rekomendasi query lengkap (auto-complete) di search page
- Implementasikan export statistik korpus ke CSV (saat ini hanya prediction yang bisa di-export)
- Pertimbangkan menambahkan text completion (bukan hanya next-word) untuk use case yang lebih luas

---

## 3. METHOD & ALGORITHM ANALYSIS

### Temuan

**Algoritma dan Metode yang Digunakan:**

#### A. N-gram Language Model
- **Jenis:** Unigram, Bigram, Trigram
- **Implementasi:** NLTK `ngrams()` dengan sliding window
- **Smoothing:** Additive (Laplace) smoothing: `P = (count + 1) / (count(context) + |V|)`
- **Fallback hierarchy:** Trigram → Bigram → Unigram (backoff tanpa interpolation)

#### B. Next-Word Prediction
- **Input:** 1-2 kata terakhir dari query
- **Matching:** Exact match + partial prefix match (≥2 karakter)
- **Ranking:** Sortir descending berdasarkan conditional probability
- **Output:** Top-10 kandidat dengan word, probability, count, rank

#### C. TF-IDF Vectorization
- **Library:** scikit-learn `TfidfVectorizer`
- **Parameter:** `max_df=1.0, min_df=1, sublinear_tf=True`
- **Sublinear TF:** `tf = 1 + log(raw_count)` untuk mengurangi dominasi term yang sangat frequent

#### D. Cosine Similarity
- **Metode:** `sklearn.metrics.pairwise.cosine_similarity`
- **Formula:** `cos(q, d) = (q · d) / (||q|| × ||d||)`
- **Threshold:** Score > 0.001 untuk filtering hasil tidak relevan

#### E. Indonesian Text Stemming
- **Library:** Sastrawi (opsional, try/except import)
- **Fallback:** Jika Sastrawi tidak terinstall, lewati stemming

#### F. Partial Prefix Matching
- **Heuristic:** Cocokkan prefix kata input (min 2 karakter) dengan kata dalam n-gram table
- **Use case:** Input "mach" → match "machine"

**Justifikasi Pemilihan Metode:**
- **N-gram dipilih** karena: sederhana, transparan, cepat, cocok untuk kebutuhan akademis dan demonstrasi konsep language model
- **Laplace smoothing** dipilih karena: sederhana, mudah diimplementasikan, dan cukup efektif untuk korpus kecil
- **TF-IDF** dipilih karena: standar industri untuk information retrieval, lightweight, dan explainable
- **Backoff (bukan interpolation)** dipilih karena: lebih sederhana, komputasi lebih ringan

**Alternatif Metode yang Bisa Dipertimbangkan:**
- **Kneser-Ney smoothing** — lebih akurat untuk sparse data (korpus kecil)
- **Stupid backoff** — lebih sederhana, tanpa normalisasi probabilitas
- **Interpolated n-gram** — weighted average dari semua level n-gram
- **Word embeddings (Word2Vec/GloVe)** — representasi semantik yang lebih kaya, tapi membutuhkan lebih banyak data
- **Neural language models (LSTM/Transformer)** — lebih akurat tapi resource-intensive

### Penilaian
**Baik** — Metode yang dipilih sesuai dengan tujuan proyek (transparan, lightweight, edukatif). Pilihan smoothing dan backoff sudah tepat untuk skala proyek.

### Rekomendasi
- Implementasikan **Kneser-Ney smoothing** sebagai opsi alternatif untuk perbandingan
- Tambahkan interpolation (weighted trigram + bigram + unigram) sebagai pengganti backoff keras
- Dokumentasikan perbandingan performa antara backoff vs interpolation dengan data aktual

---

## 4. WORKFLOW ANALYSIS

### Temuan

#### A. Alur Kerja Startup Sistem

```
Start
  ├─► Cek file pickle: data/language_model.pkl
  │     ├─► ADA → load_model() → deserialize dari pickle
  │     └─► TIDAK ADA → load_documents_from_folder(corpus/)
  │           ├─► For setiap .txt:
  │           │     ├─► Baca file (UTF-8)
  │           │     ├─► clean_text() → tokenize()
  │           │     └─► Simpan {filename, tokens, text}
  │           ├─► build_model(documents)
  │           │     ├─► Hitung unigram Counter
  │           │     ├─► Hitung bigram Counter
  │           │     ├─► Hitung trigram Counter
  │           │     └──► Bentuk vocabulary set
  │           └─► save_model() → pickle dump
  │
  ├─► inisialisasi SearchEngine
  │     ├─► Index EN (30 dokumen)
  │     └─► Index ID (15 dokumen, jika folder ada)
  │
  └─► Flask app.run(port=5000)
```

**Masalah teridentifikasi:** Jika folder `corpus_id/` tidak ada (seperti di kode), warning tercetak tapi tidak fatal. Search engine tetap berjalan dengan index EN saja.

#### B. Alur Kerja Prediksi

```
Input: "machine learning"
  │
  ├─► /api/predict menerima POST request
  │     ├─► Validasi: empty? → 400
  │     ├─► Validasi: >200 chars? → 400
  │     ├─► Split spasi, ambil 2 kata terakhir
  │     └─► query_words = ["machine", "learning"]
  │
  ├─► predict_next_word(["machine", "learning"])
  │     ├─► TRIGRAM PHASE (karena ≥2 kata)
  │     │     ├─► Cari (machine, learning, *) di trigram table
  │     │     ├─► Hitung P(w3 | machine, learning)
  │     │     ├─► Partial match: (machine, learn*) → cocok
  │     │     └─► Jika ada kandidat → return top-10
  │     │
  │     ├─► BIGRAM PHASE (fallback jika trigram kosong)
  │     │     ├─► Cari (learning, *) di bigram table
  │     │     ├─► Hitung P(w2 | learning)
  │     │     └─► Jika ada kandidat → return top-10
  │     │
  │     └─► UNIGRAM PHASE (final fallback)
  │           └─► Ambil kata paling frequent
  │
  ├─► Format response JSON
  ├─► Simpan ke prediction_history (list in-memory, max 200)
  └─► Return JSON
```

**Error Handling:**
- Query kosong → 400 Bad Request
- Query >200 chars → 400 Bad Request
- Model error → 500 Internal Server Error
- ValueError → 400 Bad Request
- Endpoint tidak dikenal → 404 Not Found

#### C. Alur Kerja Search

```
Input: {query: "neural network", language: "all", top_k: 10}
  │
  ├─► /api/search menerima POST request
  │     ├─► Validasi input
  │     └─► Tentukan indeks yang akan dicari
  │
  ├─► SearchEngine.search()
  │     ├─► Jika language = "all":
  │     │     ├─► Preprocess query untuk EN (lowercase, tokenize, stopword removal)
  │     │     ├─► cosine_similarity(query_vec, tfidf_matrix_en)
  │     │     ├─► Preprocess query untuk ID (+ stemming jika Sastrawi ada)
  │     │     └─► cosine_similarity(query_vec, tfidf_matrix_id)
  │     │
  │     ├─► Merge + sortir hasil (descending by score)
  │     ├─► Filter score ≤ 0.001
  │     ├─► Assign rank
  │     └─► Return top_k
  │
  └─► Return JSON response
```

**Interaksi Antar Komponen:**

```
app.py
  ├─► DocumentPreprocessor (load_documents_from_folder)
  │     └─► NLTK (word_tokenize, stopwords)
  │
  ├─► LanguageModel (build_model, predict_next_word, save/load_model)
  │     └─► NLTK (ngrams)
  │     └─► pickle (serialization)
  │
  └─► SearchEngine (search)
        ├─► CorpusIndex (build, search)
        │     └─► scikit-learn (TfidfVectorizer, cosine_similarity)
        │     └─► NLTK (word_tokenize, stopwords)
        └─► Sastrawi (ID stemming, opsional)
```

### Penilaian
**Baik** — Alur kerja jelas, modular, dan memiliki error handling yang memadai. Backoff mechanism dan fallback sudah terstruktur dengan baik.

### Rekomendasi
- Tambahkan logging terstruktur (bukan `print()`) menggunakan modul `logging`
- Implementasikan retry mechanism untuk kegagalan loading dokumen
- Tambahkan health check endpoint (`/api/health`) untuk monitoring
- Pertimbangkan background task untuk rebuild model tanpa restart server

---

## 5. USER FLOW ANALYSIS

### Temuan

#### A. User Journey Utama

**Journey 1: Memprediksi Kata Berikutnya**

```
Halaman Utama (/) 
  │
  ├─► Melihat model snapshot (unigram, bigram, trigram, vocabulary count)
  ├─► Melihat penjelasan "How It Works" (3 cards)
  ├─► Melihat pipeline lifecycle (4 steps)
  │
  ├─► Mengetik di input field
  │     ├─► 300ms debounce → API call → tampil prediksi
  │     ├─► Jika >2 kata → error "Maximum 2 words allowed"
  │     ├─► Jika kosong → tidak ada prediksi
  │     └─► Jika error → tampil pesan error
  │
  ├─► Melihat top-5 prediksi sebagai badge (word + probability %)
  ├─► Klik badge → append word → re-predict (infinite loop)
  │
  ├─► Melihat recent searches panel (pagination 5/page)
  ├─► Klik recent search → set query → re-predict
  ├─► Clear history → hapus localStorage
  │
  ├─► Melihat most frequent queries panel
  │
  ├─► Download Latest CSV → export prediksi terakhir
  └─► Download History CSV → export semua riwayat
```

**Journey 2: Melihat Statistik Korpus**

```
Halaman Stats (/stats)
  │
  ├─► Melihat overview cards (total docs, tokens, unique words, avg/doc)
  ├─► Melihat top unigrams (bar chart horizontal)
  ├─► Melihat top bigrams (bar chart horizontal)
  └─► Melihat tabel corpus (filename, token count, unique words)
```

**Journey 3: Mencari Dokumen**

```
Halaman Search (/search)
  │
  ├─► Memilih filter bahasa (All / English / Indonesian)
  ├─► Mengetik query pencarian
  ├─► Klik "Cari" atau tekan Enter
  │     ├─► Loading spinner
  │     ├─► Hasil berupa card: rank, filename, language badge, score bar, snippet
  │     └─► Score color-coded (Hijau=Tinggi, Kuning=Sedang, Biru=Rendah)
  └─► Scroll hasil
```

**Edge Cases:**
- **Input >2 words:** Frontend memblokir dengan pesan error. Backend memotong ke 2 kata terakhir.
- **Input kosong:** Frontend tidak mengirim request. Backend return 400.
- **Input sangat panjang (>200 chars):** Frontend mengirim, backend return 400.
- **Tidak ada koneksi:** Frontend catch error, tampilkan pesan.
- **Hasil prediksi kosong:** Frontend menampilkan "∅ No predictions".
- **Belum ada riwayat:** Panel menampilkan "No searches yet".
- **Model belum dibangun:** Backend return 500, frontend tampil error.

**Friction Points Teridentifikasi:**
1. **Tidak ada feedback loading saat startup** — Pengguna mungkin melihat layar kosong saat model pertama kali dibangun (~5 detik). Frontend tidak menampilkan loading indicator di halaman utama saat fetch `/api/stats`.
2. **Bahasa UI campuran** — UI menggunakan campuran Bahasa Indonesia (label tombol, placeholder) dan Bahasa Inggris (deskripsi fitur). Tidak konsisten.
3. **Tidak ada tombol reset/clear input** — Pengguna harus menghapus manual input.
4. **Pagination history tidak menyimpan state** — Setiap re-render kembali ke page 1.
5. **Tidak ada keyboard shortcuts** — Hanya tab + enter dasar.

### Penilaian
**Baik** — User flow cukup lancar untuk proyek akademis. Edge cases sudah di-handle dengan baik.

### Rekomendasi
- Seragamkan bahasa UI (pilih salah satu: EN atau ID)
- Tambahkan loading indicator di halaman utama saat startup
- Tambahkan tombol clear input (X icon di dalam input field)
- Implementasikan keyboard shortcuts (↑↓ untuk navigasi prediksi, Escape untuk clear)
- Simpan state pagination history di sessionStorage

---

## 6. UI/UX ANALYSIS

### Temuan

#### A. Struktur Halaman

**Halaman Utama (`index.html`):**
1. **Navigation Bar** — Logo + Query Engine + menu links (Beranda, Cara Kerja, Lifecycle, Interface, Statistik, Pencarian)
2. **Hero Section** — Dark band dengan headline besar + model snapshot card
3. **Primitives Section** — 3 cards: Corpus Indexing, Probability Scoring, Top-k Suggestions
4. **Lifecycle Section** — Dark band dengan 4-step pipeline diagram
5. **Prediction Interface** — Stats counter + input field + prediction badges + recent history + most frequent + CSV export
6. **Footer** — Links (Interface, Lifecycle, Statistik, Pencarian, GitHub) + About (UKDW) + Copyright (Theo & Deo)

**Halaman Stats (`stats.html`):**
1. **Navigation Bar** — Sederhana
2. **Header** — "Corpus statistics"
3. **Overview Cards** — 4 card metrics
4. **Top Unigrams** — Bar chart
5. **Top Bigrams** — Bar chart
6. **Corpus Table** — Filename, token count, unique words

**Halaman Search (`search.html`):**
1. **Navigation Bar** — Sederhana
2. **Header** — "Pencarian Dokumen Bilingual"
3. **Search Box** — Input + filter bahasa + "Cari" button
4. **Results** — Ranked cards dengan score progress bar

#### B. Konsistensi Desain

**Positif:**
- Menggunakan design system Coinbase (warna, spacing, typography yang terdefinisi)
- Tailwind CSS dengan custom config yang konsisten di semua halaman
- Font Inter untuk body/display, JetBrains Mono untuk data/kode
- Border radius, shadow, spacing konsisten
- Dark band (surface-dark) dan light band (surface-soft/canvas) bergantian

**Visual Design Tokens:**
```
colors: {
  primary: '#0052ff' (biru),
  ink: '#0a0b0d' (hitam teks),
  body: '#5b616e' (abu teks),
  canvas: '#ffffff' (putih),
  surface-dark: '#0a0b0d' (hitam),
  semantic-up: '#05b169' (hijau),
  semantic-down: '#cf202f' (merah),
}
```

#### C. Aksesibilitas

**Sudah diimplementasikan:**
- `skip-link` untuk keyboard navigation
- `aria-label` pada interactive elements
- `aria-current="page"` pada navigasi aktif
- `role="button"` pada elemen klik
- `sr-only` class untuk screen reader
- Semantic HTML (`<nav>`, `<main>`, `<section>`, `<footer>`)
- `prefers-reduced-motion` support
- Fokus visible (`focus-visible:outline`)

**Belum diimplementasikan:**
- Color contrast testing (belum diverifikasi)
- Keyboard trap pada modal/popup (tidak ada modal)
- ARIA live regions untuk dynamic content
- focus management setelah klik prediksi

#### D. Responsivitas

| Breakpoint | Layout |
|---|---|
| Mobile (<768px) | Stack vertical, hamburger menu (icon saja, belum berfungsi penuh) |
| Tablet (768-1024px) | Grid 2 kolom untuk cards |
| Desktop (>1024px) | Grid 3-5 kolom, layout lengkap |

**Masalah Responsivitas:**
- Hamburger menu icon ada tapi **tidak memiliki fungsi toggle** (hanya dekoratif)
- Prediksi badges: 5 kolom di desktop → 2 kolom di mobile (baik)
- Lifecycle steps: border styling complex di berbagai breakpoint

#### E. Penilaian User Experience

**Positif:**
- Debounce 300ms mengurangi API calls berlebihan
- Click-to-append memungkinkan iterasi cepat
- Hover effects pada cards dan buttons
- Scroll reveal animations (IntersectionObserver)
- Loading spinner untuk async operations

**Negatif:**
- Tidak ada shortcut keyboard (↑↓ untuk pilih prediksi)
- Input field tidak auto-focus saat halaman dimuat
- Tidak ada visual feedback saat copy (tidak ada fitur copy)
- CSV download tanpa konfirmasi

### Penilaian
**Cukup** — UI dirancang dengan baik secara visual dan memiliki aksesibilitas dasar, namun ada beberapa masalah fungsional (hamburger tidak berfungsi, bahasa campuran).

### Rekomendasi
- **Fix hamburger menu** — implementasi toggle function untuk mobile
- **Seragamkan bahasa** — pilih Bahasa Inggris atau Indonesia, jangan campur
- **Auto-focus input** — tambahkan `autofocus` pada input prediksi
- **Keyboard navigasi** — ↑↓ untuk memilih prediksi, Enter untuk mengkonfirmasi
- **Tambahkan empty state yang lebih baik** — ilustrasi atau copywriting yang membantu

---

## 7. FRONTEND ANALYSIS

### Temuan

#### A. Framework dan Library

| Library | Penggunaan | Masalah |
|---|---|---|
| Tailwind CSS (CDN) | Utility-first styling | **CDN version = ~3MB** — tidak optimal untuk production. Sebaiknya gunakan build tool (Vite/PostCSS) untuk tree-shaking |
| Google Fonts (CDN) | Inter + JetBrains Mono | Tambahan ~200KB, fallback sudah ada |
| Vanilla JavaScript | Semua interaktivitas | Tanpa framework — cocok untuk scope ini |

#### B. Struktur Komponen

Tidak ada komponen framework (React/Vue). Semua interaktivitas ditulis inline dalam `<script>` tag di masing-masing HTML file.

**Fungsi JavaScript utama:**
- `predictQuery()` — debounced API call ke `/api/predict`
- `renderHistory()` — render recent + top queries dengan pagination
- `exportCsv(type)` — build + download CSV
- `buildCsv(rows)` / `csvEscape(value)` — CSV formatting
- `escapeHtml(text)` — sanitasi input untuk mencegah XSS
- `debounce(func, delay)` — rate limiting API calls
- `formatTimeAgo(ts)` — human-readable timestamp
- Set `renderTopList()` dan `renderDocumentTable()` di stats page

#### C. State Management

| State | Lokasi | Inisialisasi |
|---|---|---|
| Query history | `localStorage` key `queryHistory` | Empty array |
| Last prediction | In-memory variabel `lastPrediction` | `null` |
| Stats counters | DOM elements (di-update via fetch) | `0` |
| Pagination state | In-memory variables (reset setiap render) | Page 1 |

**Masalah:** State management tidak terpusat. History menggunakan localStorage, prediksi terakhir in-memory. Tidak ada state management library — ini OK untuk skala proyek.

#### D. Routing

Routing dilakukan server-side oleh Flask:
- `/` → index.html (prediction interface)
- `/stats` → stats.html (statistics dashboard)
- `/search` → search.html (search interface)

Client-side routing: **tidak ada**. Semua navigasi adalah full page reload.

#### E. Asset Management

| Asset | Format | Ukuran (estimasi) |
|---|---|---|
| `logo.png` | PNG | ~10-20KB |
| `logo2.png` | PNG | ~10-20KB |
| `favicon.png` | PNG 256x256 | ~5KB |
| `favicon.ico` | ICO | ~5KB |
| `styles.css` | CSS | ~2KB |

**Masalah:** `logo2.png` tidak digunakan di template manapun (dead asset).

#### F. Kualitas Kode Frontend

**Positif:**
- XSS protection via `escapeHtml()` untuk semua user-generated content
- Debounce pattern untuk API calls
- Semantic event delegation
- Graceful degradation (IntersectionObserver check)

**Negatif:**
- Semua JS inline dalam HTML — tidak ada pemisahan concerns
- Tidak ada module bundling (Vite/Webpack)
- Tidak ada minification
- Banyak magic numbers dalam kode (300ms delay, 200 char limit, 5 items/page)
- Duplikasi fungsi `escapeHtml()` di ketiga file HTML
- CSS selector menggunakan ID dan class campuran

### Penilaian
**Cukup** — Fungsional dan aman, tapi tidak mengikuti best practice modern frontend development.

### Rekomendasi
- Pisahkan JavaScript ke file `.js` eksternal
- Gunakan Vite atau Webpack untuk build pipeline
- Ekstrak magic numbers ke constants di awal file
- Buat shared utility module untuk `escapeHtml`, `debounce`, `formatTimeAgo`
- Hapus `logo2.png` jika tidak digunakan

---

## 8. BACKEND ANALYSIS

### Temuan

#### A. Framework dan Arsitektur

- **Framework:** Flask 2.3.0
- **Pattern:** Monolithic — semua logic dalam satu aplikasi Flask
- **Template Engine:** Jinja2 (bawaan Flask)
- **Port:** 5000 (hardcoded)

#### B. Desain API

| Method | Endpoint | Input | Output | Fungsi |
|---|---|---|---|---|
| POST | `/api/predict` | `{query: string}` | `{query, timestamp, predictions[]}` | Prediksi kata |
| GET | `/api/stats` | — | `{unigrams_count, bigrams_count, ...}` | Statistik model |
| GET | `/api/ngram-stats?limit=N` | Query param | `{top_unigrams[], top_bigrams[]}` | Top n-gram |
| GET | `/api/corpus-stats?limit=N` | Query param | `{total_documents, ...}` | Statistik korpus |
| POST | `/api/search` | `{query, top_k, language}` | `{query, results[], total}` | Cari dokumen |
| GET | `/api/export-csv?type=latest\|history` | Query param | CSV file | Export prediksi |

**Desain REST:** Sederhana dan sesuai. Tidak ada versioning (`/api/v1/...`).

#### C. Autentikasi dan Otorisasi

**Tidak ada.** Semua endpoint publik tanpa autentikasi. Untuk proyek akademis yang berjalan di localhost, ini dapat diterima. Namun jika akan di-deploy ke internet, ini adalah **security issue serius**.

#### D. Business Logic Layer

**Terdiri dari 3 service class:**

1. **DocumentPreprocessor** (`scripts/preprocess.py`)
   - `clean_text(text)` → string
   - `tokenize(text)` → List[str]
   - `preprocess_document(text)` → List[str]
   - `load_documents_from_folder(path)` → List[Dict]
   - `get_errors()` → List[str]

2. **LanguageModel** (`scripts/language_model.py`)
   - `build_model(documents)` → None
   - `predict_next_word(query_words)` → List[Dict]
   - `save_model(path)` → None
   - `load_model(path)` → None
   - `get_statistics()` → Dict
   - `get_ngram_stats(limit)` → Dict
   - `get_corpus_stats(limit)` → Dict

3. **SearchEngine** (`scripts/search_engine.py`)
   - `search(query, top_k, language)` → List[Dict]
   - `_build_indexes()` → None
   - `_preprocess_query(query, language)` → string

#### E. Middleware dan Error Handlers

```python
@app.errorhandler(404)  # → {"error": "Endpoint not found"}
@app.errorhandler(500)  # → {"error": "Internal server error"}
```

Tidak ada middleware kustom (untuk logging, rate limiting, CORS, dll).

#### F. Security Analysis

**Vulnerabilities Teridentifikasi:**

| Issue | Severity | Lokasi | Detail |
|---|---|---|---|
| No authentication | HIGH | Semua endpoint | Siapa pun bisa akses semua API |
| No rate limiting | MEDIUM | `/api/predict` | Bisa di-DDoS dengan request berulang |
| No CORS configuration | LOW | `app.py` | Flask default mengizinkan same-origin |
| Debug mode aktif | MEDIUM | `app.run(debug=True)` | Debug mode mengekspos stack trace |
| Pickle deserialization | MEDIUM | `load_model()` | Pickle bisa mengeksekusi kode berbahaya jika file korup |
| No HTTPS | MEDIUM | Seluruh app | Tidak ada SSL/TLS |

### Penilaian
**Cukup** — Backend fungsional dan terstruktur dengan baik, tapi memiliki beberapa security concerns yang perlu diatasi terutama jika akan di-deploy.

### Rekomendasi
- **Remove `debug=True`** di production — ganti dengan environment variable `FLASK_ENV`
- **Tambahkan rate limiting** menggunakan `flask-limiter`
- **Validasi input lebih ketat** — pastikan tipe data sesuai sebelum diproses
- **Gunakan environment variables** untuk konfigurasi (port, debug mode, path)
- **Pertimbangkan CORS** jika frontend akan dipisah

---

## 9. DATA PREPROCESSING ANALYSIS

### Temuan

#### A. Pipeline Preprocessing

**Text Cleaning:**
```python
def clean_text(self, text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Hapus semua non-word, non-space
    return text
```

**Tokenization:**
```python
def tokenize(self, text):
    tokens = word_tokenize(text)          # NLTK Punkt tokenizer
    tokens = [t for t in tokens if t.isalpha()]  # Filter hanya alphabetic
    return tokens
```

#### B. Teknik Pembersihan Data

| Langkah | Operasi | Contoh |
|---|---|---|
| Lowercasing | `text.lower()` | `"Hello"` → `"hello"` |
| Punctuation removal | `re.sub(r'[^\w\s]', '', text)` | `"Hello, World!"` → `"hello world"` |
| Tokenization | `nltk.word_tokenize(text)` | `"hello world"` → `["hello", "world"]` |
| Alphabetic filter | `token.isalpha()` | `"123"` → dihapus, `"hello"` → retained |

#### C. Yang TIDAK Dilakukan

| Teknik | Status | Dampak |
|---|---|---|
| Stopword removal | ❌ Tidak | Kata seperti "the", "is", "and" tetap masuk ke model n-gram → meningkatkan noise |
| Stemming/Lemmatization | ❌ Tidak | "learning", "learned", "learns" dihitung sebagai token terpisah → sparse data |
| Number removal | ✅ Ya (via `isalpha`) | Angka dan token numerik dihapus |
| Whitespace normalization | ✅ Ya (via regex dan tokenizer) | Multiple spasi dihapus |
| URL/HTML tag removal | ❌ Tidak | Tapi data dari Wikipedia sudah dibersihkan oleh scraper |
| Low-frequency word removal | ❌ Tidak | Semua kata dipertahankan → vocabulary besar |

#### D. Kualitas Pipeline

**Positif:**
- Error handling per dokumen (satu file gagal tidak menghentikan proses)
- Error reporting (accumulated errors + print)
- Encoding handling (UTF-8 explicit)
- Graceful handling file kosong

**Negatif:**
- **Tidak ada stopword removal** untuk model n-gram. Stopwords seperti "the", "a", "is" akan muncul sebagai unigram frequent dan mengkontaminasi prediksi.
- **Tidak ada stemming** → variasi morfologis kata tidak dihandle
- **Regex `[^\w\s]`** — Menghapus underscore (\w termasuk underscore). Untuk korpus Wikipedia, ini bisa menghilangkan informasi dari link dan referensi.
- **Sequential processing** — Dokumen diproses satu per satu, tidak ada parallel processing

### Penilaian
**Cukup** — Pipeline preprocessing berfungsi dan stabil, tapi melewatkan dua langkah penting (stopword removal, stemming) yang bisa meningkatkan kualitas model secara signifikan.

### Rekomendasi
- **Tambahkan stopword removal** setelah tokenization: `[t for t in tokens if t not in stopwords.words('english')]`
- **Tambahkan opsi stemming** (PorterStemmer atau SnowballStemmer dari NLTK) untuk mereduksi variasi morfologis
- **Gunakan `re.sub(r'[^\w\s]', '', text)` dengan hati-hati** — pertimbangkan untuk mempertahankan apostrof (contoh: "don't")
- **Tambahkan config parameter** untuk preprocessing options (stopword removal on/off, stemming on/off)
- **Dokumentasikan preprocessing assumptions** dalam docstring

---

## 10. MODEL TRAINING ANALYSIS

### Temuan

#### A. Arsitektur Model

Model yang digunakan adalah **n-gram frequency tables** — bukan model neural network atau ML tradisional. Tidak ada proses training dalam arti sebenarnya (iterative optimization). Yang dilakukan adalah:

1. **Counting** — menghitung frekuensi kemunculan setiap n-gram
2. **Storage** — menyimpan frekuensi dalam Counter dictionaries
3. **Smoothing** — Laplace (add-one) smoothing saat inference

#### B. Dataset

| Aspek | Detail |
|---|---|
| **Sumber** | Wikipedia (Bahasa Inggris + Bahasa Indonesia) |
| **Scraper** | `scrape_corpus.py` menggunakan BeautifulSoup |
| **Jumlah dokumen EN** | 30 artikel |
| **Jumlah dokumen ID** | 15 artikel |
| **Topik** | CS, AI, NLP, Statistics, Machine Learning |
| **Ukuran per dokumen** | Bervariasi, ~500-5000+ karakter |
| **Kualitas** | Wikipedia → teks terstruktur, grammar baik |
| **Bias** | Fokus ke STEM (Science, Technology, Engineering, Mathematics) |

**Masalah Dataset:**
- **Domain terbatas** — Hanya topik CS/AI. Model tidak akan berfungsi baik untuk domain lain (medis, hukum, dll.)
- **Ukuran kecil** — 30 dokumen EN + 15 ID. Untuk language model, ini sangat kecil.
- **Tidak ada balanced representation** — Beberapa topik mungkin mendominasi.

#### C. Training Configuration

Tidak ada training configuration traditionil karena ini adalah model statistik counting-based. Parameter yang ada:

| Parameter | Nilai | Keterangan |
|---|---|---|
| n-gram range | 1-3 (unigram, bigram, trigram) | Hardcoded |
| Smoothing | Additive (Laplace) | Alpha = 1 (hardcoded) |
| Vocabulary | Semua unique tokens | Tanpa threshold |
| Context window | 2 words max | Sesuai trigram |

#### D. Evaluation Metrics

**Tidak ada evaluation metrics yang built-in.** Model tidak memiliki:
- Perplexity score
- Accuracy
- Precision/Recall
- Cross-validation
- Train/test split

Ini adalah **keterbatasan signifikan** yang perlu dicatat. Evaluasi hanya bisa dilakukan secara manual melalui UI.

#### E. Overfitting / Underfitting

**Overfitting:** Risiko rendah karena model adalah counting-based (tidak ada parameter yang di-optimalkan). Namun, dengan dataset kecil, model akan memiliki **generalization yang buruk** pada domain di luar korpus.

**Underfitting:** Model hanya menangkap pola permukaan (surface patterns) dari teks. Tidak bisa menangkap:
- Semantic relationships
- Long-range dependencies (>3 words)
- Syntactic structures

#### F. Model Saving dan Versioning

| Aspek | Detail |
|---|---|
| Format | `pickle` |
| Lokasi | `data/language_model.pkl` |
| Versioning | Tidak ada (file di-overwrite setiap rebuild) |
| Backup | Tidak ada |
| Git-tracked | ❌ (di `.gitignore`: `data/*.pkl`) |

**Masalah:** Pickle adalah format tidak aman dan tidak portable antar versi Python yang berbeda. Untuk proyek yang lebih serius, pertimbangkan format portable seperti JSON atau format biner yang lebih stabil.

#### G. Deployment Strategy

Model di-load saat startup aplikasi Flask. Strategi:
1. Cek file pickle → jika ada, load
2. Jika tidak ada → build dari korpus → save ke pickle
3. Selama runtime, model tetap di memory (tidak di-reload)

**Tidak ada:** hot-reload, A/B testing, canary deployment, atau model registry.

### Penilaian
**Cukup** — Untuk proyek akademis yang mendemonstrasikan konsep n-gram language model, pendekatan ini sudah sesuai. Namun, jika ingin berfungsi sebagai sistem production-ready, diperlukan peningkatan signifikan.

### Rekomendasi
- **Tambahkan perplexity score** sebagai evaluation metric built-in
- **Implementasikan train/test split** — pisahkan dokumen untuk validasi
- **Tambahkan threshold vocabulary** — buang kata dengan frekuensi < 2 atau < 3 untuk mengurangi noise
- **Gunakan format JSON untuk model serialization** sebagai alternatif pickle
- **Tambahkan experiment logging** (parameter, timestamp, corpus stats) setiap kali model dibangun

---

## 11. SYSTEM ARCHITECTURE

### Temuan

#### A. Arsitektur Keseluruhan

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CLIENT (Browser)                                 │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  HTML (Jinja2)  │  CSS (Tailwind)  │  JS (Vanilla)                │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ HTTP (localhost:5000)
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        SERVER (Flask 2.3.0)                            │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Route Handlers (app.py)                                         │   │
│  │  ├─► / → index.html                                              │   │
│  │  ├─► /stats → stats.html                                         │   │
│  │  ├─► /search → search.html                                       │   │
│  │  ├─► /api/predict → LanguageModel.predict_next_word()            │   │
│  │  ├─► /api/stats → LanguageModel.get_statistics()                 │   │
│  │  ├─► /api/ngram-stats → LanguageModel.get_ngram_stats()          │   │
│  │  ├─► /api/corpus-stats → LanguageModel.get_corpus_stats()        │   │
│  │  ├─► /api/search → SearchEngine.search()                         │   │
│  │  └─► /api/export-csv → CSV builder                               │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Service Layer                                                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │   │
│  │  │ Preprocessor │  │ LanguageModel│  │    SearchEngine      │   │   │
│  │  └──────┬───────┘  └──────┬───────┘  └─────────┬────────────┘   │   │
│  │         │                 │                      │                │   │
│  │         ▼                 ▼                      ▼                │   │
│  │  ┌─────────┐      ┌──────────────┐      ┌─────────────────┐     │   │
│  │  │ NLTK    │      │ NLTK + pickle│      │ scikit-learn +  │     │   │
│  │  │         │      │              │      │ NLTK + Sastrawi │     │   │
│  │  └─────────┘      └──────────────┘      └─────────────────┘     │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  Data Layer                                                      │   │
│  │  ├─► corpus/       → 30 file .txt (EN Wikipedia)                │   │
│  │  ├─► corpus_id/    → 15 file .txt (ID Wikipedia)                │   │
│  │  ├─► data/         → language_model.pkl                         │   │
│  │  └─► static/       → logo, favicon, CSS                         │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

#### B. Integrasi Antar Layanan

| Service | Dependensi | Dipanggil Oleh |
|---|---|---|
| `DocumentPreprocessor` | NLTK (`word_tokenize`, `stopwords`) | `LanguageModel` (indirect via `app.py`), `SearchEngine` |
| `LanguageModel` | NLTK (`ngrams`), `pickle` | `app.py` (route handlers) |
| `SearchEngine` | scikit-learn (`TfidfVectorizer`, `cosine_similarity`), NLTK, Sastrawi | `app.py` (route handlers) |

**Tingkat kopling:** **Low coupling** — Setiap service class independen dan hanya berinteraksi melalui `app.py`. Baik untuk maintainability.

#### C. Skalabilitas

| Aspek | Kapasitas Saat Ini | Keterbatasan |
|---|---|---|
| Prediksi | <100ms untuk 30 dokumen | Linear scan of n-gram tables → O(V) |
| Model size | ~5MB untuk 30 dokumen | Growth O(N×L) dengan jumlah dokumen |
| Concurrent users | Single-threaded (Flask default) | Multiple requests terblokir selama prediction |
| Memory | In-memory model + history (200 entries) | Growth O(V²) untuk bigram/trigram tables |
| Horizontal scaling | Tidak support (in-memory state) | Perlu shared cache (Redis) dan session management |

**Single Points of Failure:**
1. **In-memory model** — Jika server restart, model dibangun ulang (atau load dari pickle)
2. **In-memory history** — Prediction history hilang saat restart
3. **Single process** — Semua request diproses oleh satu thread

#### D. Arsitektur secara Keseluruhan

**Pattern:** Monolithic Flask app dengan service layer pattern

**Positif:**
- Modular (3 service classes terpisah)
- Separation of concerns jelas
- Template rendering server-side (SEO friendly)
- Cukup untuk skala proyek

**Negatif:**
- Tidak bisa diskalakan secara horizontal tanpa refactor
- Tidak ada message queue untuk background jobs
- Tidak ada caching layer (Redis/Memcached)

### Penilaian
**Baik** — Arsitektur sesuai untuk proyek akademis skala kecil. Mudah dipahami dan dimodifikasi.

### Rekomendasi
- Untuk production deployment, migrasi ke WSGI production server (Gunicorn/Waitress) — Flask dev server tidak untuk production
- Pisahkan model inference ke service terpisah jika skalabilitas menjadi kebutuhan
- Implementasikan caching untuk frequent queries (contoh: cache.predictions[query] = result)
- Dokumentasikan arsitektur dengan diagram yang lebih formal (C4 atau UML)

---

## 12. DEPENDENCIES & CONFIGURATION

### Temuan

#### A. Production Dependencies (`requirements.txt`)

| Package | Version | Ukuran | Fungsi | Critical? |
|---|---|---|---|---|
| Flask | 2.3.0 | ~1MB | Web framework | ✅ Ya |
| nltk | 3.8.1 | ~10MB+ | NLP toolkit | ✅ Ya |
| Pillow | 9.6.0 | ~2MB | Image handling | ❌ Tidak (hanya untuk static assets) |

**Masalah:** `requirements.txt` hanya mencantumkan 3 package. Tidak tercantum:
- `scikit-learn` — dibutuhkan oleh `search_engine.py` (TF-IDF, cosine_similarity)
- `Sastrawi` — dibutuhkan untuk Indonesian stemming
- `requests` + `beautifulsoup4` — dibutuhkan oleh `scrape_corpus.py`
- `flask` — core framework

Ini menyebabkan `pip install -r requirements.txt` **tidak menginstall semua dependensi yang diperlukan** untuk search engine. Search engine akan gagal dengan ImportError jika scikit-learn tidak terinstall.

#### B. Development Dependencies (`requirements-dev.txt`)

| Package | Version | Fungsi |
|---|---|---|
| pytest | 7.4.4 | Test runner |
| pytest-cov | 4.1.0 | Test coverage |

#### C. Kompatibilitas

| Package | Python Version | Catatan |
|---|---|---|
| Flask 2.3.0 | 3.8+ | ✅ Compatible |
| NLTK 3.8.1 | 3.7+ | ✅ Compatible |
| scikit-learn | 3.8+ | ✅ Compatible |
| Sastrawi | 3.6+ | ✅ Compatible |
| Pillow 9.6.0 | 3.7+ | ✅ Compatible |

**Masalah Potensial:** NLTK 3.8.1 sudah cukup lama (dirilis 2023). NLTK 3.9+ memiliki beberapa perubahan API. Tidak ada pinning versi untuk scikit-learn, Sastrawi, requests, beautifulsoup4.

#### D. Environment Configuration

| Konfigurasi | Nilai | Lokasi |
|---|---|---|
| Port | 5000 (hardcoded) | `app.py:250` |
| Debug mode | `True` (hardcoded) | `app.py:250` |
| Max query length | 200 chars (hardcoded) | `app.py:86` |
| Max history | 200 entries (hardcoded) | `app.py:21` |
| Model path | `data/language_model.pkl` (hardcoded) | `app.py:25` |
| Corpus path | `corpus/` (hardcoded) | `app.py:23` |

**Tidak ada environment variables** — semua konfigurasi hardcoded. Ini menyulitkan deployment di environment yang berbeda.

#### E. Secrets Management

**Tidak ada secrets** — proyek tidak menggunakan API keys, database credentials, atau token. Aman untuk open source.

#### F. Containerization

**Tidak ada** — Tidak ada Dockerfile, docker-compose.yml, atau container configuration.

### Penilaian
**Perlu Perbaikan** — `requirements.txt` tidak lengkap dan semua konfigurasi hardcoded.

### Rekomendasi
- **Lengkapi `requirements.txt`** dengan semua dependensi: `scikit-learn`, `requests`, `beautifulsoup4`, `Sastrawi`
- **Gunakan environment variables** untuk konfigurasi: `FLASK_PORT`, `FLASK_DEBUG`, `CORPUS_PATH`, `MODEL_PATH`
- **Pertimbangkan `pyproject.toml`** sebagai pengganti `requirements.txt` untuk standardisasi modern
- **Tambahkan Dockerfile** untuk reproducible deployment
- **Pin versi scikit-learn dan Sastrawi** untuk reproducibility

---

## 13. CODE QUALITY ASSESSMENT

### Temuan

#### A. Struktur dan Organisasi Kode

```
root/
  ├── app.py              # 253 lines — Flask routes + startup + error handlers
  ├── scrape_corpus.py    # 125 lines — Wikipedia scraper (standalone)
  │
  ├── scripts/            # Service layer
  │   ├── preprocess.py   # 97 lines — DocumentPreprocessor
  │   ├── language_model.py # 247 lines — LanguageModel
  │   └── search_engine.py  # 154 lines — SearchEngine + CorpusIndex
  │
  ├── templates/          # UI layer
  │   ├── index.html      # 757 lines — Main UI (HTML + inline JS + Tailwind config)
  │   ├── stats.html      # 256 lines — Stats UI
  │   └── search.html     # 257 lines — Search UI
  │
  ├── static/
  │   └── styles.css      # 99 lines — Custom CSS
  │
  └── tests/
      ├── conftest.py     # 4 lines — sys.path configuration
      ├── test_preprocess.py  # 26 lines — 4 tests
      ├── test_language_model.py # 46 lines — 4 tests
      └── test_search_engine.py  # 208 lines — 13 tests
```

**Total:** ~2,300 baris kode (termasuk test, template, CSS)

#### B. Naming Conventions

| Aspek | Konsistensi | Contoh |
|---|---|---|
| Python classes | ✅ CamelCase | `LanguageModel`, `DocumentPreprocessor`, `SearchEngine` |
| Python functions | ✅ snake_case | `predict_next_word`, `load_documents_from_folder` |
| Python variables | ✅ snake_case | `query_words`, `prediction_history` |
| File names | ✅ snake_case | `language_model.py`, `search_engine.py` |
| API endpoints | ✅ kebab-case | `/api/predict`, `/api/ngram-stats` |
| HTML IDs | ✅ camelCase | `predictionsList`, `searchInput` |
| CSS classes | ✅ BEM-like | `lifecycle-step`, `reveal-delay-1` |

#### C. Documentation dan Komentar

| Aspek | Kualitas |
|---|---|
| Docstrings (class) | ✅ Ada untuk major classes |
| Docstrings (method) | ⚠️ Sebagian besar method tidak memiliki docstring detail |
| Inline comments | ⚠️ Minimal (hanya untuk kode kompleks) |
| README.md | ✅ ✅ Sangat detail |
| README-deo-new.txt | ✅ ✅ Dokumentasi teknis lengkap |
| API documentation | ✅ Ada di README |

**Masalah:** Beberapa method tidak memiliki docstring. Contoh:
- `LanguageModel.get_statistics()` — tidak ada docstring
- `LanguageModel.get_ngram_stats()` — tidak ada docstring
- `LanguageModel.get_corpus_stats()` — tidak ada docstring
- `SearchEngine._build_indexes()` — tidak ada docstring
- Semua route function di `app.py` — tidak ada docstring

#### D. Test Coverage

| Test File | Tests | Lines of Test | Coverage (estimated) |
|---|---|---|---|
| `test_preprocess.py` | 4 | 26 | ~70% dari preprocess.py |
| `test_language_model.py` | 4 | 46 | ~60% dari language_model.py |
| `test_search_engine.py` | 13 | 208 | ~85% dari search_engine.py |

**Total: 21 tests**

**Test gaps:**
- Tidak ada test untuk `load_documents_from_folder` dengan folder tidak ada
- Tidak ada test untuk empty corpus
- Tidak ada test untuk partial prefix matching
- Tidak ada test untuk error handling (ValueError, exception cases)
- Tidak ada test untuk API endpoint integration (Flask test client)
- Tidak ada test untuk UI components

#### E. Technical Debt

| Item | Severity | Effort to Fix |
|---|---|---|
| `debug=True` di production | Medium | 5 menit |
| Pickle untuk serialization | Low | 2 jam (migrasi ke JSON) |
| Inline JS di HTML | Medium | 4 jam (ekstrak ke .js files) |
| Magic numbers | Low | 1 jam (constants) |
| Tidak ada type hints | Low | 2 jam (tambahkan typing) |
| `print()` statt logging | Low | 30 menit (ganti ke logging) |
| CDN dependencies tanpa fallback | Medium | 1 jam (local fallback) |
| Dead asset (logo2.png) | Low | 5 menit (hapus) |

#### F. Duplikasi Kode

| Duplikasi | Lokasi | Detail |
|---|---|---|
| `escapeHtml()` function | index.html, stats.html, search.html | 3x duplikasi |
| Tailwind config | index.html, stats.html, search.html | 3x duplikasi (60+ lines each) |
| Navigation bar HTML | index.html, stats.html, search.html | 3x duplikasi (30+ lines each) |

### Penilaian
**Baik** — Kode terorganisir dengan baik, naming konsisten, test coverage cukup untuk scope proyek. Area yang perlu diperbaiki: duplikasi frontend, magic numbers, dan dokumentasi method.

### Rekomendasi
- **Ekstrak shared JavaScript** ke file `static/app.js` — `escapeHtml`, `debounce`, `formatTimeAgo`
- **Gunakan Jinja2 template inheritance** — buat `base.html` untuk navbar, footer, Tailwind config
- **Tambahkan type hints** untuk semua method parameter dan return types
- **Ganti `print()` dengan `logging` module** — lebih terstruktur
- **Gunakan constants** untuk magic numbers (MAX_QUERY_LENGTH, HISTORY_LIMIT, dll.)

---

## 14. PERFORMANCE ANALYSIS

### Temuan

#### A. Bottleneck Teridentifikasi

| Bottleneck | Lokasi | Dampak | Severity |
|---|---|---|---|
| **Linear scan n-gram tables** | `predict_next_word()` | O(V) — scan semua trigram/bigram untuk setiap prediksi | **High** |
| **Single-threaded Flask** | `app.py` | Request blocking — 1 request menghalangi yang lain | **High** |
| **In-memory model** | `LanguageModel` | Semua data di RAM — tidak bisa scale ke korpus besar | **Medium** |
| **CDN dependencies** | HTML templates | Blocking render — menunggu Tailwind + Google Fonts | **Medium** |
| **Pickle serialization** | `save_model()`/`load_model()` | I/O bound untuk model besar | **Low** |
| **No caching** | Tidak ada | Setiap request yang sama diproses ulang | **Medium** |

#### B. Query Optimization

**Saat ini:**
- Prediksi: loop linear melalui semua trigram → filter by context → sort
- Search: TF-IDF vectorization + cosine similarity (scikit-learn sudah optimized)

**Potensi Optimasi:**
- **Hash map index** — Gunakan dictionary prefix-based untuk akses O(1) ke n-gram dengan context tertentu
- **Precompute common predictions** — Cache hasil prediksi untuk query yang sering muncul
- **Batch TF-IDF** — scikit-learn sudah menggunakan sparse matrix + optimized linear algebra

#### C. Caching Strategy

**Saat ini:**
- **Model cache:** Pickle file — di-load sekali saat startup
- **Prediction history:** In-memory list (max 200 entries)
- **No query cache:** Setiap request diproses dari awal

**Potensi Caching:**
- **LRU cache untuk prediksi** — `functools.lru_cache` untuk query yang sama
- **Redis/Memcached** — Untuk shared cache di multi-process deployment
- **Precompute top unigrams** — Top-10 unigrams bisa di-cache karena tidak berubah

#### D. Response Time Estimasi

| Endpoint | Estimasi | Faktor |
|---|---|---|
| `GET /` | <50ms | Template rendering + stats fetch |
| `POST /api/predict` | <100ms | Linear scan + sort |
| `GET /api/stats` | <10ms | Dictionary lookup |
| `POST /api/search` | <200ms | TF-IDF transform + cosine similarity |
| `GET /api/export-csv` | <50ms | CSV build (memory) |
| `GET /stats` | <100ms | Template + API fetch |
| `GET /search` | <50ms | Template render |

**Catatan:** Waktu response sangat tergantung pada ukuran korpus. Dengan 30 dokumen, prediksi <100ms. Dengan 1,000 dokumen, waktu bisa meningkat ke detik.

#### E. Memory Usage Estimasi

| Komponen | Estimasi |
|---|---|
| Unigrams Counter | ~500KB (untuk 12,980 tokens) |
| Bigrams Counter | ~5MB (untuk 512,884 bigrams) |
| Trigrams Counter | ~15MB (untuk 1,024,332 trigrams) |
| Vocabulary set | ~500KB |
| Search TF-IDF matrices | ~2MB each |
| Flask + NLTK overhead | ~50-100MB |
| **Total** | **~75-120MB** |

### Penilaian
**Cukup** — Performa memadai untuk skala saat ini (30 dokumen), tapi memiliki bottleneck yang akan muncul dengan dataset yang lebih besar.

### Rekomendasi
- **Implementasikan prefix dictionary** — Group trigrams by `(w1, w2)` key untuk akses O(1):
  ```python
  self.trigram_index = defaultdict(list)
  for (w1, w2, w3), count in self.trigrams.items():
      self.trigram_index[(w1, w2)].append((w3, count))
  ```
- **Tambahkan `functools.lru_cache`** untuk `predict_next_word()` — cache hasil query yang sama
- **Gunakan Gunicorn/Waitress** dengan multiple workers untuk concurrency
- **Load Google Fonts + Tailwind secara async** — `media="print" onload="this.media='all'"`

---

## 15. SECURITY ANALYSIS

### Temuan

#### A. Input Validation

| Endpoint | Validation | Status |
|---|---|---|
| `POST /api/predict` | Empty check, max length (200), JSON valid | ✅ Cukup |
| `POST /api/search` | Empty check, max length (200), JSON valid, lang filter | ✅ Cukup |
| `GET /api/export-csv` | Type parameter (latest/history) | ⚠️ Minimal |
| Other GET endpoints | No input validation needed | ✅ OK |

**Kerentanan:**
- **No input sanitization untuk XSS** — Input query tidak di-sanitize di backend (hanya di frontend via `escapeHtml`). Namun, karena output dikembalikan sebagai JSON (bukan HTML), risiko XSS di backend minimal.
- **No SQL injection** — Tidak ada database, jadi tidak relevan.

#### B. Authentication & Authorization

**Tidak ada.** Semua endpoint publik. Analisis risiko:
- **Localhost deployment:** Risiko sangat rendah — hanya akses lokal
- **Public deployment:** Risiko sangat tinggi — siapa pun bisa akses semua API

#### C. Pickle Deserialization

**Masalah Keamanan:** Pickle adalah format serialisasi tidak aman. Jika file `language_model.pkl` dimodifikasi oleh pihak jahat, bisa mengeksekusi arbitrary code saat di-load.

**Dampak:** **Critical** jika pickle di-load dari sumber tidak terpercaya.  
**Mitigasi:** File pickle dibuat dan di-load dari sistem yang sama. Risiko rendah jika tidak ada akses write ke file system.

#### D. Flask Debug Mode

**`app.run(debug=True)`** — Debug mode Flask menyediakan:
- **Werkzeug debugger** — Bisa mengeksekusi kode Python arbitrer melalui browser
- **Stack trace exposure** — Informasi sensitif bisa bocor

**Severity:** **HIGH** jika diakses dari jaringan luar.  
**Mitigasi:** Set `debug=False` di production.

#### E. Exposed Secrets / Credentials

**Tidak ada secrets yang terekspos.** Proyek ini tidak menggunakan:
- API keys
- Database credentials
- OAuth tokens
- SSH keys

Ini adalah **nilai plus** untuk keamanan.

#### F. Additional Security Issues

| Issue | Severity | Detail |
|---|---|---|
| No HTTPS | Medium | Semua komunikasi plaintext |
| No rate limiting | Medium | API bisa di-DDoS |
| No CORS policy | Low | Mengizinkan origin lain (tidak dikonfigurasi) |
| No Content Security Policy | Low | Tidak ada CSP headers |
| Server version exposure | Low | Flask header default menampilkan versi |
| No session management | Low | Tidak ada user sessions |

### Penilaian
**Cukup** — Untuk proyek lokal/akademis, security sudah memadai. Untuk deployment publik, perlu perbaikan signifikan.

### Rekomendasi
- **Set `debug=False`** menggunakan environment variable `FLASK_DEBUG=0`
- **Tambahkan rate limiting** — `pip install flask-limiter`:
  ```python
  from flask_limiter import Limiter
  limiter = Limiter(app, key_func=lambda: request.remote_addr)
  limiter.limit("30/minute")(predict)
  ```
- **Jika di-deploy publik:** tambahkan basic auth atau API key
- **Tambahkan security headers:** `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`
- **Validasi pickle integrity** — gunakan checksum atau signature sebelum load

---

## RINGKASAN EKSEKUTIF

### Kekuatan Utama Proyek

| Kekuatan | Detail |
|---|---|
| **Kode bersih dan terstruktur** | Modular service layer, konsistensi naming, dokumentasi README yang sangat baik |
| **Arsitektur tepat untuk scope** | Monolithic Flask dengan service pattern — ideal untuk proyek akademis |
| **Transparansi penuh** | Setiap prediksi traceable ke frequency counts — sesuai dengan tujuan edukasi |
| **Testing cukup** | 21 tests mencakup core functionality |
| **UI/UX baik** | Design system konsisten, aksesibilitas dasar, animasi smooth |
| **Privacy-first** | 100% offline, tidak ada data dikirim ke eksternal |

### Kelemahan Kritis yang Harus Segera Diperbaiki

| Prioritas | Kelemahan | Dampak | Solusi |
|---|---|---|---|
| **🔴 HIGH** | `debug=True` di production | Remote code execution via debugger | Ganti ke env variable |
| **🔴 HIGH** | `requirements.txt` tidak lengkap | Search engine gagal (missing scikit-learn) | Tambahkan semua dependensi |
| **🟡 MEDIUM** | Linear scan n-gram tables | Performa turun drastis dengan korpus besar | Prefix dictionary index |
| **🟡 MEDIUM** | Tidak ada stopword removal | Prediksi terkontaminasi kata umum | Tambahkan preprocessing |
| **🟡 MEDIUM** | Campuran bahasa UI (EN + ID) | User experience tidak konsisten | Seragamkan bahasa |
| **🟡 MEDIUM** | CDN Tailwind CSS | Render blocking + 3MB download | Build local CSS |
| **🟢 LOW** | Pickle serialization | Tidak aman, tidak portable | Alternatif JSON |
| **🟢 LOW** | Duplikasi kode frontend | Maintenance burden | Template inheritance |

### Prioritas Perbaikan

| Prioritas | Item | Effort |
|---|---|---|
| **P0 (Critical)** | Fix `debug=True` | 5 menit |
| **P0 (Critical)** | Lengkapi `requirements.txt` | 10 menit |
| **P1 (High)** | Tambahkan prefix dictionary index | 2-3 jam |
| **P1 (High)** | Tambahkan stopword removal | 30 menit |
| **P1 (High)** | Ekstrak shared frontend code | 2 jam |
| **P2 (Medium)** | Gunakan Jinja2 template inheritance | 1 jam |
| **P2 (Medium)** | Environment variables untuk konfigurasi | 30 menit |
| **P2 (Medium)** | Ganti `print()` ke `logging` | 30 menit |
| **P3 (Low)** | Tambahkan type hints | 2 jam |
| **P3 (Low)** | Migrasi dari pickle ke JSON | 2 jam |
| **P3 (Low)** | Tambahkan perplexity evaluation | 3 jam |

### Estimasi Effort untuk Perbaikan Keseluruhan

| Kategori | Effort |
|---|---|
| Critical (P0) | 15 menit |
| High (P1) | 4-6 jam |
| Medium (P2) | 2-3 jam |
| Low (P3) | 7-8 jam |
| **Total** | **~14-17 jam** |

---

## ROADMAP PERBAIKAN

### Fase 1: Quick Wins (Hari 1) — 1 jam

| # | Task | Effort | Dampak |
|---|---|---|---|
| 1 | Set `debug=False` via env variable | 5 menit | 🔴 Security critical |
| 2 | Lengkapi `requirements.txt` | 10 menit | 🔴 Search engine fixes |
| 3 | Hapus `logo2.png` (dead asset) | 2 menit | 🟢 Cleanup |
| 4 | Tambahkan `.env` file untuk konfigurasi | 15 menit | 🟡 Config management |
| 5 | Tambahkan logging statt `print()` | 20 menit | 🟡 Observability |

### Fase 2: Performance & Quality (Hari 2-3) — 6 jam

| # | Task | Effort | Dampak |
|---|---|---|---|
| 6 | Implementasi prefix dictionary index | 3 jam | 🔴 Performance critical |
| 7 | Tambahkan stopword removal ke preprocessing | 30 menit | 🟡 Prediction quality |
| 8 | Tambahkan `functools.lru_cache` untuk prediksi | 15 menit | 🟡 Performance |
| 9 | Tambahkan add-on smoothing option (Kneser-Ney) | 2 jam | 🟡 Algorithm improvement |

### Fase 3: Frontend Refactor (Hari 3-4) — 4 jam

| # | Task | Effort | Dampak |
|---|---|---|---|
| 10 | Buat `static/app.js` untuk shared functions | 1 jam | 🟡 Code quality |
| 11 | Buat `templates/base.html` untuk layout | 1 jam | 🟡 Code quality |
| 12 | Refactor 3 halaman pakai template inheritance | 1 jam | 🟡 Code quality |
| 13 | Build Tailwind CSS lokal (PostCSS/Vite) | 1 jam | 🟡 Performance |

### Fase 4: Advanced Features (Hari 5-7) — 8 jam

| # | Task | Effort | Dampak |
|---|---|---|---|
| 14 | Tambahkan perplexity evaluation | 2 jam | 🟡 Evaluation |
| 15 | Tambahkan interpolation smoothing (weighted) | 2 jam | 🟡 Algorithm |
| 16 | Migrasi serialization ke JSON | 2 jam | 🟢 Portability |
| 17 | Tambahkan Gunicorn/Waitress untuk production | 1 jam | 🟡 Deployment |
| 18 | Tambahkan security headers + rate limiting | 1 jam | 🟡 Security |

### Fase 5: Polish (Opsional) — 4 jam

| # | Task | Effort | Dampak |
|---|---|---|---|
| 19 | Tambahkan keyboard shortcuts (↑↓) | 1 jam | 🟢 UX |
| 20 | Auto-focus input field | 5 menit | 🟢 UX |
| 21 | Implementasi hamburger menu mobile | 30 menit | 🟢 Responsiveness |
| 22 | Seragamkan bahasa UI | 1 jam | 🟢 Konsistensi |
| 23 | Tambahkan Dockerfile | 1 jam | 🟢 Deployment |

---

**Akhir Laporan Analisis**
