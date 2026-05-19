from scripts.language_model import LanguageModel


def test_build_model():
    model = LanguageModel()
    docs = [
        {"filename": "doc1", "tokens": ["machine", "learning", "is", "fun"]}
    ]
    model.build_model(docs)
    assert model.unigrams["machine"] == 1
    assert model.bigrams[("machine", "learning")] == 1
    assert model.trigrams[("machine", "learning", "is")] == 1


def test_predict_next_word():
    model = LanguageModel()
    docs = [
        {"filename": "doc1", "tokens": ["machine", "learning", "is", "fun"]}
    ]
    model.build_model(docs)
    predictions = model.predict_next_word(["machine"])
    assert predictions
    assert predictions[0]["word"] == "learning"
    assert predictions[0]["count"] == 1


def test_empty_query():
    model = LanguageModel()
    assert model.predict_next_word([]) == []


def test_model_persistence(tmp_path):
    model = LanguageModel()
    docs = [
        {"filename": "doc1", "tokens": ["data", "science", "rocks"]}
    ]
    model.build_model(docs)
    filepath = tmp_path / "model.pkl"
    model.save_model(str(filepath))

    loaded = LanguageModel()
    loaded.load_model(str(filepath))

    assert loaded.unigrams == model.unigrams
    assert loaded.bigrams == model.bigrams
    assert loaded.trigrams == model.trigrams
