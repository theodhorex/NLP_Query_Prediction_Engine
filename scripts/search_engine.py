import os
import re
import logging
from typing import Any

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    _has_sastrawi = True
except ImportError:
    stemmer = None
    _has_sastrawi = False

logger = logging.getLogger(__name__)

_en_stopwords = set(stopwords.words('english'))
_id_stopwords = set(stopwords.words('indonesian'))


class CorpusIndex:
    def __init__(self):
        self.vectorizer: TfidfVectorizer = None
        self.tfidf_matrix: Any = None
        self.documents: list[dict[str, Any]] = []
        self.language: str = ''

    def build(self, documents: list[dict[str, Any]], language: str) -> None:
        self.language = language
        self.documents = documents

        if not documents:
            logger.warning("No documents to index for language: %s", language)
            return

        cleaned_docs = [
            self._clean_for_tfidf(doc['text'])
            for doc in documents
        ]

        self.vectorizer = TfidfVectorizer(
            max_df=1.0,
            min_df=1,
            sublinear_tf=True
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(cleaned_docs)

        logger.info(
            "Indexed %s corpus: %d docs, %d terms",
            language, len(documents), self.tfidf_matrix.shape[1]
        )

    def _clean_for_tfidf(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = word_tokenize(text)
        if self.language == 'en':
            tokens = [t for t in tokens if t.isalpha() and t not in _en_stopwords]
        else:
            tokens = [t for t in tokens if t.isalpha() and t not in _id_stopwords]
        return ' '.join(tokens)

    def search(self, query_clean: str, top_k: int = 10) -> list[dict[str, Any]]:
        if not self.vectorizer or self.tfidf_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query_clean])
        similarity = cosine_similarity(query_vec, self.tfidf_matrix)
        results = list(enumerate(similarity[0]))
        results.sort(key=lambda x: x[1], reverse=True)

        output = []
        for idx, score in results:
            if score <= 0.001:
                continue
            doc = self.documents[idx]
            output.append({
                'score': float(round(score, 4)),
                'filename': doc['filename'],
                'language': self.language,
                'snippet': doc['text'][:150].replace('\n', ' ').strip()
            })
            if len(output) >= top_k:
                break

        return output


class SearchEngine:
    def __init__(self, corpus_en_path: str, corpus_id_path: str, preprocessor) -> None:
        self.corpus_en_path = corpus_en_path
        self.corpus_id_path = corpus_id_path
        self.preprocessor = preprocessor

        self.en_index = CorpusIndex()
        self.id_index = CorpusIndex()

        self._build_indexes()

    def _build_indexes(self) -> None:
        en_docs = self.preprocessor.load_documents_from_folder(self.corpus_en_path)
        self.en_index.build(en_docs, 'en')

        if os.path.exists(self.corpus_id_path):
            id_docs = self.preprocessor.load_documents_from_folder(self.corpus_id_path)
            self.id_index.build(id_docs, 'id')
        else:
            logger.warning("Corpus ID path not found: %s", self.corpus_id_path)

    def _preprocess_query(self, query: str, language: str) -> str:
        query = query.lower()
        query = re.sub(r'[^\w\s]', ' ', query)
        tokens = word_tokenize(query)
        tokens = [t for t in tokens if t.isalpha()]

        if language == 'en':
            tokens = [t for t in tokens if t not in _en_stopwords]
        elif language == 'id':
            tokens = [t for t in tokens if t not in _id_stopwords]
            if _has_sastrawi:
                tokens = [stemmer.stem(t) for t in tokens]

        return ' '.join(tokens)

    def search(self, query: str, top_k: int = 10, language: str = 'all') -> list[dict[str, Any]]:
        if not query or not query.strip():
            return []

        all_results = []

        if language in ('all', 'en'):
            q_en = self._preprocess_query(query, 'en')
            if q_en.strip():
                all_results.extend(self.en_index.search(q_en, top_k))

        if language in ('all', 'id'):
            q_id = self._preprocess_query(query, 'id')
            if q_id.strip():
                all_results.extend(self.id_index.search(q_id, top_k))

        all_results.sort(key=lambda x: x['score'], reverse=True)
        for i, r in enumerate(all_results[:top_k], start=1):
            r['rank'] = i

        return all_results[:top_k]
