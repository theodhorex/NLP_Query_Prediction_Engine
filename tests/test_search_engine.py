import os
import tempfile

from scripts.preprocess import DocumentPreprocessor
from scripts.search_engine import CorpusIndex, SearchEngine


def test_build_empty_documents():
    index = CorpusIndex()
    index.build([], 'en')
    assert index.vectorizer is None
    assert index.tfidf_matrix is None
    assert index.search("test") == []


def test_build_and_search_basic():
    index = CorpusIndex()
    docs = [
        {"filename": "doc1.txt", "text": "machine learning is fun", "tokens": []},
        {"filename": "doc2.txt", "text": "deep learning is amazing", "tokens": []},
    ]
    index.build(docs, 'en')
    assert index.vectorizer is not None
    assert index.tfidf_matrix is not None

    results = index.search("machine learning")
    assert len(results) > 0
    assert results[0]['filename'] == "doc1.txt"
    assert results[0]['language'] == 'en'
    assert 'score' in results[0]
    assert 'snippet' in results[0]


def test_search_before_build_returns_empty():
    index = CorpusIndex()
    assert index.search("anything") == []


def test_search_result_fields():
    index = CorpusIndex()
    docs = [
        {"filename": "test.txt", "text": "artificial intelligence is transforming the world", "tokens": []},
    ]
    index.build(docs, 'en')
    results = index.search("artificial intelligence")
    assert len(results) == 1

    r = results[0]
    assert isinstance(r['score'], float)
    assert r['filename'] == "test.txt"
    assert r['language'] == 'en'
    assert isinstance(r['snippet'], str)
    assert len(r['snippet']) > 0


def test_search_top_k():
    index = CorpusIndex()
    docs = [
        {"filename": f"doc{i}.txt", "text": f"unique content about topic {i}", "tokens": []}
        for i in range(10)
    ]
    index.build(docs, 'en')
    results = index.search("unique content", top_k=3)
    assert len(results) <= 3


def test_search_ranks_by_score():
    index = CorpusIndex()
    docs = [
        {"filename": "match.txt", "text": "machine learning deep learning neural network", "tokens": []},
        {"filename": "random.txt", "text": "the weather is nice today", "tokens": []},
    ]
    index.build(docs, 'en')
    results = index.search("machine learning")
    assert len(results) >= 1
    assert results[0]['filename'] == "match.txt"


def test_search_empty_query():
    preprocessor = DocumentPreprocessor()
    with tempfile.TemporaryDirectory() as en_dir:
        with open(os.path.join(en_dir, "dummy.txt"), "w", encoding="utf-8") as f:
            f.write("test content")

        engine = SearchEngine(en_dir, "/nonexistent", preprocessor)
        assert engine.search("") == []
        assert engine.search("   ") == []


def test_search_english_only():
    en_docs = {
        "ml.txt": "Machine learning is a subset of artificial intelligence",
        "nlp.txt": "Natural language processing is a branch of AI",
    }

    with tempfile.TemporaryDirectory() as en_dir:
        for name, content in en_docs.items():
            with open(os.path.join(en_dir, name), "w", encoding="utf-8") as f:
                f.write(content)

        preprocessor = DocumentPreprocessor()
        engine = SearchEngine(en_dir, "/nonexistent", preprocessor)

        results = engine.search("machine learning", language='en')
        assert len(results) > 0
        for r in results:
            assert r['language'] == 'en'


def test_search_indonesian_only():
    with tempfile.TemporaryDirectory() as en_dir, tempfile.TemporaryDirectory() as id_dir:
        with open(os.path.join(en_dir, "dummy.txt"), "w", encoding="utf-8") as f:
            f.write("some english content")

        with open(os.path.join(id_dir, "ml_id.txt"), "w", encoding="utf-8") as f:
            f.write("pembelajaran mesin adalah cabang dari kecerdasan buatan")

        preprocessor = DocumentPreprocessor()
        engine = SearchEngine(en_dir, id_dir, preprocessor)

        results = engine.search("pembelajaran mesin", language='id')
        assert len(results) > 0
        for r in results:
            assert r['language'] == 'id'


def test_search_all_languages_bilingual():
    en_docs = {"en.txt": "machine learning is transforming technology"}
    id_docs = {"id.txt": "pembelajaran mesin mengubah teknologi"}

    with tempfile.TemporaryDirectory() as en_dir, tempfile.TemporaryDirectory() as id_dir:
        for name, content in en_docs.items():
            with open(os.path.join(en_dir, name), "w", encoding="utf-8") as f:
                f.write(content)
        for name, content in id_docs.items():
            with open(os.path.join(id_dir, name), "w", encoding="utf-8") as f:
                f.write(content)

        preprocessor = DocumentPreprocessor()
        engine = SearchEngine(en_dir, id_dir, preprocessor)

        results = engine.search("machine learning", language='all')
        assert len(results) >= 1


def test_search_all_merges_and_sorts():
    en_docs = {"en_match.txt": "artificial intelligence robots", "en_other.txt": "cooking recipes"}
    id_docs = {"id_match.txt": "kecerdasan buatan robot"}

    with tempfile.TemporaryDirectory() as en_dir, tempfile.TemporaryDirectory() as id_dir:
        for name, content in en_docs.items():
            with open(os.path.join(en_dir, name), "w", encoding="utf-8") as f:
                f.write(content)
        for name, content in id_docs.items():
            with open(os.path.join(id_dir, name), "w", encoding="utf-8") as f:
                f.write(content)

        preprocessor = DocumentPreprocessor()
        engine = SearchEngine(en_dir, id_dir, preprocessor)

        results = engine.search("artificial intelligence", language='all', top_k=5)
        assert len(results) >= 1
        scores = [r['score'] for r in results]
        assert scores == sorted(scores, reverse=True)


def test_search_returns_rank():
    with tempfile.TemporaryDirectory() as en_dir:
        with open(os.path.join(en_dir, "doc.txt"), "w", encoding="utf-8") as f:
            f.write("machine learning is fun")

        preprocessor = DocumentPreprocessor()
        engine = SearchEngine(en_dir, "/nonexistent", preprocessor)

        results = engine.search("machine learning", top_k=5)
        if results:
            assert 'rank' in results[0]
            assert results[0]['rank'] == 1


def test_search_snippet_is_first_150_chars():
    index = CorpusIndex()
    text = "hello world " * 50
    docs = [{"filename": "long.txt", "text": text, "tokens": []}]
    index.build(docs, 'en')

    results = index.search("hello", top_k=1)
    assert len(results) == 1
    assert len(results[0]['snippet']) <= 150


def test_search_corpus_id_not_found_logs_warning(caplog):
    import logging
    caplog.set_level(logging.WARNING)

    with tempfile.TemporaryDirectory() as en_dir:
        with open(os.path.join(en_dir, "doc.txt"), "w", encoding="utf-8") as f:
            f.write("test content")

        preprocessor = DocumentPreprocessor()
        engine = SearchEngine(en_dir, "/path/does/not/exist", preprocessor)

        assert any("Corpus ID path not found" in record.message for record in caplog.records)


def test_search_without_index():
    index = CorpusIndex()
    assert index.search("anything") == []
