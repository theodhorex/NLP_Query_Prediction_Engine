import pytest
from scripts.preprocess import DocumentPreprocessor


def test_clean_text():
    preprocessor = DocumentPreprocessor()
    text = "Hello, World!"
    assert preprocessor.clean_text(text) == "hello world"


def test_tokenize():
    preprocessor = DocumentPreprocessor()
    tokens = preprocessor.tokenize("machine learning")
    assert tokens == ["machine", "learning"]


def test_empty_document():
    preprocessor = DocumentPreprocessor()
    with pytest.raises(ValueError):
        preprocessor.preprocess_document("")


def test_invalid_text_type():
    preprocessor = DocumentPreprocessor()
    with pytest.raises(ValueError):
        preprocessor.clean_text(123)
