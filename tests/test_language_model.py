"""Tests for LanguageModel (add-1 smoothing, no configurable alpha)"""
import pytest
from scripts.language_model import LanguageModel


class TestBuildModel:
    def test_single_document(self):
        model = LanguageModel()
        docs = [{"filename": "doc1", "tokens": ["machine", "learning", "is", "fun"]}]
        model.build_model(docs)
        assert model.unigrams["machine"] == 1
        assert model.bigrams[("machine", "learning")] == 1
        assert model.trigrams[("machine", "learning", "is")] == 1

    def test_multiple_documents_aggregate(self):
        model = LanguageModel()
        docs = [
            {"filename": "doc1", "tokens": ["a", "b"]},
            {"filename": "doc2", "tokens": ["a", "c"]},
        ]
        model.build_model(docs)
        assert model.unigrams["a"] == 2
        assert model.total_documents == 2
        assert model.total_tokens == 4

    def test_empty_docs_list(self):
        model = LanguageModel()
        with pytest.raises(ValueError, match="No documents provided"):
            model.build_model([])

    def test_missing_tokens_key(self):
        model = LanguageModel()
        with pytest.raises(ValueError, match="no tokens"):
            model.build_model([{"filename": "doc1"}])

    def test_empty_tokens_list(self):
        model = LanguageModel()
        with pytest.raises(ValueError, match="no tokens"):
            model.build_model([{"filename": "doc1", "tokens": []}])


class TestPredictNextWord:
    def test_basic_bigram(self):
        model = LanguageModel()
        model.build_model([{"filename": "d1", "tokens": ["machine", "learning", "is", "fun"]}])
        preds = model.predict_next_word(["machine"])
        assert preds
        assert preds[0]["word"] == "learning"

    def test_trigram_preferred(self):
        model = LanguageModel()
        model.build_model([{"filename": "d1", "tokens": ["machine", "learning", "is", "fun", "machine", "learning", "rocks"]}])
        preds = model.predict_next_word(["machine", "learning"])
        assert preds
        assert preds[0]["word"] in ("is", "rocks")

    def test_prefix_match_bigram(self):
        model = LanguageModel()
        model.build_model([{"filename": "d1", "tokens": ["machine", "learning", "machine", "vision"]}])
        preds = model.predict_next_word(["mac"])
        assert preds
        words = [p["word"] for p in preds]
        assert "learning" in words
        assert "vision" in words

    def test_prefix_match_trigram(self):
        model = LanguageModel()
        model.build_model([{"filename": "d1", "tokens": ["machine", "learning", "is", "fun", "machine", "learning", "rocks"]}])
        preds = model.predict_next_word(["machine", "lear"])
        assert preds
        words = [p["word"] for p in preds]
        assert any(w in words for w in ("is", "rocks"))

    def test_unigram_fallback(self):
        model = LanguageModel()
        model.build_model([{"filename": "d1", "tokens": ["python", "java", "python", "rust", "python", "go"]}])
        preds = model.predict_next_word(["kotlin"])
        assert preds
        words = [p["word"] for p in preds]
        assert "python" in words

    def test_empty_query(self):
        model = LanguageModel()
        assert model.predict_next_word([]) == []

    def test_none_query(self):
        model = LanguageModel()
        assert model.predict_next_word(None) == []

    def test_max_10_results(self):
        model = LanguageModel()
        docs = [{"filename": "d1", "tokens": ["a"] + [str(i) for i in range(30)]}]
        model.build_model(docs)
        preds = model.predict_next_word(["a"])
        assert len(preds) <= 10

    def test_sorted_by_probability(self):
        model = LanguageModel()
        model.build_model([{"filename": "d1", "tokens": ["a", "b", "a", "c", "a", "b"]}])
        preds = model.predict_next_word(["a"])
        probs = [p["probability"] for p in preds]
        assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))

    def test_repeated_context(self):
        model = LanguageModel()
        model.build_model([{"filename": "d1", "tokens": ["the", "the", "the", "the"]}])
        preds = model.predict_next_word(["the"])
        assert preds
        assert preds[0]["word"] == "the"

    def test_single_word_corpus(self):
        model = LanguageModel()
        model.build_model([{"filename": "d1", "tokens": ["hello"]}])
        preds = model.predict_next_word(["hello"])
        assert preds

    def test_no_match_still_returns_unigrams(self):
        model = LanguageModel()
        model.build_model([{"filename": "d1", "tokens": ["hello", "world", "foo", "bar"]}])
        preds = model.predict_next_word(["xyznonexistent"])
        assert preds

    def test_add1_smoothing(self):
        model = LanguageModel()
        model.build_model([{"filename": "d1", "tokens": ["cat", "dog", "cat", "bird"]}])
        preds = model.predict_next_word(["cat"])
        assert preds
        for p in preds:
            assert p["probability"] > 0


class TestSaveLoad:
    def test_persistence(self, tmp_path):
        model = LanguageModel()
        model.build_model([{"filename": "d1", "tokens": ["data", "science", "rocks"]}])
        path = tmp_path / "model.pkl"
        model.save_model(str(path))
        loaded = LanguageModel()
        loaded.load_model(str(path))
        assert loaded.unigrams == model.unigrams
        assert loaded.bigrams == model.bigrams
        assert loaded.trigrams == model.trigrams

    def test_save_empty_raises(self, tmp_path):
        model = LanguageModel()
        with pytest.raises(ValueError, match="Model is empty"):
            model.save_model(str(tmp_path / "empty.pkl"))

    def test_load_nonexistent(self, tmp_path):
        model = LanguageModel()
        with pytest.raises(FileNotFoundError):
            model.load_model(str(tmp_path / "nonexistent.pkl"))

    def test_load_corrupted(self, tmp_path):
        path = tmp_path / "corrupt.pkl"
        with open(path, "wb") as f:
            f.write(b"not pickle data")
        model = LanguageModel()
        with pytest.raises(Exception):
            model.load_model(str(path))


class TestStatistics:
    def test_get_statistics(self):
        model = LanguageModel()
        model.build_model([{"filename": "d1", "tokens": ["hello", "world"]}])
        s = model.get_statistics()
        assert s["unigrams_count"] == 2
        assert s["bigrams_count"] == 1
        assert s["trigrams_count"] == 0
        assert s["vocabulary_size"] == 2

    def test_get_statistics_empty(self):
        s = LanguageModel().get_statistics()
        assert s["unigrams_count"] == 0

    def test_get_ngram_stats(self):
        model = LanguageModel()
        model.build_model([{"filename": "d1", "tokens": ["a", "b", "c"]}])
        s = model.get_ngram_stats(limit=5)
        assert "top_unigrams" in s
        assert "top_bigrams" in s

    def test_get_corpus_stats(self):
        model = LanguageModel()
        model.build_model([
            {"filename": "d1", "tokens": ["hello", "world"]},
            {"filename": "d2", "tokens": ["foo", "bar", "baz"]}
        ])
        s = model.get_corpus_stats(limit=5)
        assert s["total_documents"] == 2
        assert s["total_tokens"] == 5
        assert s["unique_words"] == 5
        assert "top_unigrams" in s
        assert "top_bigrams" in s
        assert len(s["documents"]) == 2
