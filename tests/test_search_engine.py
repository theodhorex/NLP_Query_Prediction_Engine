"""Tests for CorpusIndex and SearchEngine"""
import os
import tempfile
import pytest
from scripts.search_engine import CorpusIndex, SearchEngine
from scripts.preprocess import DocumentPreprocessor


class TestCorpusIndex:
    def test_build_empty(self):
        idx = CorpusIndex()
        idx.build([], 'en')
        assert idx.vectorizer is None
        assert idx.tfidf_matrix is None
        assert idx.search("test") == []

    def test_build_and_search_basic(self):
        idx = CorpusIndex()
        docs = [
            {"filename": "doc1.txt", "text": "machine learning is fun", "tokens": []},
            {"filename": "doc2.txt", "text": "deep learning is amazing", "tokens": []},
        ]
        idx.build(docs, 'en')
        assert idx.vectorizer is not None
        assert idx.tfidf_matrix is not None

        results = idx.search("machine learning")
        assert len(results) > 0
        assert results[0]["filename"] == "doc1.txt"
        assert results[0]["language"] == "en"
        assert "score" in results[0]

    def test_search_before_build(self):
        idx = CorpusIndex()
        assert idx.search("anything") == []

    def test_result_fields(self):
        idx = CorpusIndex()
        docs = [{"filename": "test.txt", "text": "artificial intelligence is transforming the world", "tokens": []}]
        idx.build(docs, "en")
        results = idx.search("artificial intelligence")
        assert len(results) >= 1
        r = results[0]
        assert isinstance(r["score"], float)
        assert r["filename"] == "test.txt"
        assert r["language"] == "en"
        assert isinstance(r["snippet"], str)

    def test_top_k_limit(self):
        idx = CorpusIndex()
        docs = [{"filename": f"doc{i}.txt", "text": f"unique content about topic {i}", "tokens": []} for i in range(10)]
        idx.build(docs, "en")
        results = idx.search("unique content", top_k=3)
        assert len(results) <= 3

    def test_ranks_by_score(self):
        idx = CorpusIndex()
        docs = [
            {"filename": "match.txt", "text": "machine learning deep learning neural network", "tokens": []},
            {"filename": "random.txt", "text": "the weather is nice today", "tokens": []},
        ]
        idx.build(docs, "en")
        results = idx.search("machine learning")
        assert len(results) >= 1
        assert results[0]["filename"] == "match.txt"

    def test_score_threshold(self):
        idx = CorpusIndex()
        docs = [
            {"filename": "d1.txt", "text": "machine learning is fun", "tokens": []},
            {"filename": "d2.txt", "text": "completely unrelated topic here", "tokens": []},
        ]
        idx.build(docs, "en")
        results = idx.search("machine learning")
        for r in results:
            assert r["score"] > 0.001

    def test_snippet_is_first_150_chars(self):
        idx = CorpusIndex()
        text = "hello world " * 50
        docs = [{"filename": "long.txt", "text": text, "tokens": []}]
        idx.build(docs, "en")
        results = idx.search("hello", top_k=1)
        assert len(results) >= 1
        assert len(results[0]["snippet"]) <= 150

    def test_build_indonesian_stopwords(self):
        idx = CorpusIndex()
        docs = [{"filename": "d1.txt", "text": "pembelajaran mesin adalah cabang dari kecerdasan buatan", "tokens": []}]
        idx.build(docs, "id")
        results = idx.search("pembelajaran mesin")
        assert isinstance(results, list)


class TestSearchEngine:
    def test_english_only(self):
        en = {"ml.txt": "Machine learning is a subset of artificial intelligence"}
        with tempfile.TemporaryDirectory() as en_dir, tempfile.TemporaryDirectory() as id_dir:
            for n, c in en.items():
                with open(os.path.join(en_dir, n), "w", encoding="utf-8") as f:
                    f.write(c)
            open(os.path.join(id_dir, "dummy.txt"), "w", encoding="utf-8").close()
            p = DocumentPreprocessor()
            engine = SearchEngine(en_dir, id_dir, p)
            results = engine.search("machine learning", language="en")
            assert len(results) > 0
            for r in results:
                assert r["language"] == "en"

    def test_indonesian_only(self):
        with tempfile.TemporaryDirectory() as en_dir, tempfile.TemporaryDirectory() as id_dir:
            open(os.path.join(en_dir, "dummy.txt"), "w", encoding="utf-8").close()
            with open(os.path.join(id_dir, "ml_id.txt"), "w", encoding="utf-8") as f:
                f.write("pembelajaran mesin adalah cabang dari kecerdasan buatan")
            p = DocumentPreprocessor()
            engine = SearchEngine(en_dir, id_dir, p)
            results = engine.search("pembelajaran mesin", language="id")
            assert len(results) > 0
            for r in results:
                assert r["language"] == "id"

    def test_all_languages_bilingual(self):
        en_docs = {"en.txt": "machine learning is transforming technology"}
        id_docs = {"id.txt": "pembelajaran mesin mengubah teknologi"}
        with tempfile.TemporaryDirectory() as en_dir, tempfile.TemporaryDirectory() as id_dir:
            for n, c in en_docs.items():
                with open(os.path.join(en_dir, n), "w", encoding="utf-8") as f:
                    f.write(c)
            for n, c in id_docs.items():
                with open(os.path.join(id_dir, n), "w", encoding="utf-8") as f:
                    f.write(c)
            p = DocumentPreprocessor()
            engine = SearchEngine(en_dir, id_dir, p)
            results = engine.search("machine learning", language="all")
            assert len(results) >= 1

    def test_merges_and_sorts(self):
        en_docs = {"en_match.txt": "artificial intelligence robots", "en_other.txt": "cooking recipes"}
        id_docs = {"id_match.txt": "kecerdasan buatan robot"}
        with tempfile.TemporaryDirectory() as en_dir, tempfile.TemporaryDirectory() as id_dir:
            for n, c in en_docs.items():
                with open(os.path.join(en_dir, n), "w", encoding="utf-8") as f:
                    f.write(c)
            for n, c in id_docs.items():
                with open(os.path.join(id_dir, n), "w", encoding="utf-8") as f:
                    f.write(c)
            p = DocumentPreprocessor()
            engine = SearchEngine(en_dir, id_dir, p)
            results = engine.search("artificial intelligence", language="all", top_k=5)
            assert len(results) >= 1
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True)

    def test_returns_rank(self):
        with tempfile.TemporaryDirectory() as en_dir:
            with open(os.path.join(en_dir, "doc.txt"), "w", encoding="utf-8") as f:
                f.write("machine learning is fun")
            open(os.path.join(en_dir, "doc2.txt"), "w", encoding="utf-8").close()
            p = DocumentPreprocessor()
            engine = SearchEngine(en_dir, "C:\\nonexistent_id", p)
            results = engine.search("machine learning", top_k=5)
            if results:
                assert "rank" in results[0]
                assert results[0]["rank"] == 1

    def test_empty_query(self):
        p = DocumentPreprocessor()
        with tempfile.TemporaryDirectory() as en_dir:
            open(os.path.join(en_dir, "dummy.txt"), "w", encoding="utf-8").close()
            engine = SearchEngine(en_dir, "C:\\nonexistent_id", p)
            assert engine.search("") == []
            assert engine.search("   ") == []

    def test_corpus_id_not_found_logs_warning(self, caplog):
        import logging
        caplog.set_level(logging.WARNING)
        with tempfile.TemporaryDirectory() as en_dir:
            with open(os.path.join(en_dir, "doc.txt"), "w", encoding="utf-8") as f:
                f.write("test content")
            p = DocumentPreprocessor()
            engine = SearchEngine(en_dir, "C:\\path\\does\\not\\exist", p)
            assert any("not found" in record.message.lower() for record in caplog.records)

    def test_without_index_returns_empty(self):
        idx = CorpusIndex()
        assert idx.search("anything") == []
