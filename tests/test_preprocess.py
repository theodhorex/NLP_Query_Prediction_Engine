"""Tests for DocumentPreprocessor"""
import os
import tempfile
import pytest
from scripts.preprocess import DocumentPreprocessor


class TestCleanText:
    def test_basic(self):
        p = DocumentPreprocessor()
        assert p.clean_text("Hello, World!") == "hello world"

    def test_numbers_only(self):
        p = DocumentPreprocessor()
        assert p.clean_text("12345 6789") == ""

    def test_special_chars_only(self):
        p = DocumentPreprocessor()
        assert p.clean_text("@#$%^&*()") == ""

    def test_unicode(self):
        p = DocumentPreprocessor()
        assert "caf" in p.clean_text("café naïve")

    def test_whitespace_only(self):
        p = DocumentPreprocessor()
        assert p.clean_text("   \n  \t  ") == ""

    def test_invalid_input(self):
        p = DocumentPreprocessor()
        with pytest.raises(ValueError, match="must be a string"):
            p.clean_text(123)
        with pytest.raises(ValueError, match="must be a string"):
            p.clean_text(None)


class TestTokenize:
    def test_basic(self):
        p = DocumentPreprocessor()
        assert p.tokenize("machine learning") == ["machine", "learning"]

    def test_numbers_filtered(self):
        p = DocumentPreprocessor()
        assert p.tokenize("machine 123 learning 456") == ["machine", "learning"]

    def test_punctuation_filtered(self):
        p = DocumentPreprocessor()
        assert p.tokenize("machine, learning! is? fun.") == ["machine", "learning", "is", "fun"]

    def test_empty_text(self):
        p = DocumentPreprocessor()
        with pytest.raises(ValueError, match="cannot be empty"):
            p.tokenize("")

    def test_whitespace_only(self):
        p = DocumentPreprocessor()
        with pytest.raises(ValueError, match="cannot be empty"):
            p.tokenize("   \n  ")

    def test_numbers_only_no_alpha(self):
        p = DocumentPreprocessor()
        with pytest.raises(ValueError, match="No valid tokens"):
            p.tokenize("12345 6789")


class TestPreprocessDocument:
    def test_basic(self):
        p = DocumentPreprocessor()
        tokens = p.preprocess_document("Machine Learning is Fun!")
        assert "machine" in tokens
        assert "learning" in tokens
        assert "fun" in tokens

    def test_empty_document(self):
        p = DocumentPreprocessor()
        with pytest.raises(ValueError, match="Invalid document"):
            p.preprocess_document("")

    def test_none_document(self):
        p = DocumentPreprocessor()
        with pytest.raises(ValueError, match="Invalid document"):
            p.preprocess_document(None)

    def test_numbers_only_doc(self):
        p = DocumentPreprocessor()
        with pytest.raises(ValueError):
            p.preprocess_document("12345 6789")


class TestLoadDocumentsFromFolder:
    def test_nonexistent_folder(self):
        p = DocumentPreprocessor()
        docs = p.load_documents_from_folder(r"C:\nonexistent_folder_xyz")
        assert docs == []
        assert any("Folder not found" in e for e in p.get_errors())

    def test_empty_folder(self):
        p = DocumentPreprocessor()
        with tempfile.TemporaryDirectory() as tmp:
            docs = p.load_documents_from_folder(tmp)
        assert docs == []
        assert any("No .txt files found" in e for e in p.get_errors())

    def test_non_txt_files_only(self):
        p = DocumentPreprocessor()
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("test.csv", "test.md", "test.py"):
                open(os.path.join(tmp, name), "w").close()
            docs = p.load_documents_from_folder(tmp)
        assert docs == []
        assert any("No .txt files" in e for e in p.get_errors())

    def test_single_document(self):
        p = DocumentPreprocessor()
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "test.txt"), "w", encoding="utf-8") as f:
                f.write("Hello world, this is a test document.")
            docs = p.load_documents_from_folder(tmp)
        assert len(docs) == 1
        assert docs[0]["filename"] == "test.txt"
        assert "hello" in docs[0]["tokens"]

    def test_multiple_documents(self):
        p = DocumentPreprocessor()
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                with open(os.path.join(tmp, f"doc{i}.txt"), "w", encoding="utf-8") as f:
                    f.write(f"This is document number {i}.")
            docs = p.load_documents_from_folder(tmp)
        assert len(docs) == 3

    def test_empty_file_skipped(self):
        p = DocumentPreprocessor()
        with tempfile.TemporaryDirectory() as tmp:
            open(os.path.join(tmp, "empty.txt"), "w", encoding="utf-8").close()
            with open(os.path.join(tmp, "valid.txt"), "w", encoding="utf-8") as f:
                f.write("Valid document.")
            docs = p.load_documents_from_folder(tmp)
        assert len(docs) == 1
        assert docs[0]["filename"] == "valid.txt"

    def test_encoding_error_skipped(self):
        p = DocumentPreprocessor()
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "bad.txt"), "wb") as f:
                f.write(b"\xff\xfe\x00\x01")
            docs = p.load_documents_from_folder(tmp)
        assert docs == []

    def test_get_errors(self):
        p = DocumentPreprocessor()
        p.load_documents_from_folder(r"C:\nonexistent_folder_xyz")
        assert isinstance(p.get_errors(), list)
        assert len(p.get_errors()) > 0
