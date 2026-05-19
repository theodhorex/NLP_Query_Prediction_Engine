import os
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

class DocumentPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.load_errors = []

    def clean_text(self, text):
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def tokenize(self, text):
        if not text.strip():
            raise ValueError("Text cannot be empty")
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]
        if not tokens:
            raise ValueError("No valid tokens found after filtering")
        return tokens

    def preprocess_document(self, text):
        if not text or not isinstance(text, str):
            raise ValueError("Invalid document content")
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return tokens

    def load_documents_from_folder(self, folder_path):
        documents = []
        self.load_errors = []

        if not os.path.exists(folder_path):
            error_msg = f"Folder not found: {folder_path}"
            print(f"ERROR: {error_msg}")
            self.load_errors.append(error_msg)
            return documents

        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
        if not txt_files:
            error_msg = f"No .txt files found in {folder_path}"
            print(f"WARNING: {error_msg}")
            self.load_errors.append(error_msg)
            return documents

        print(f"Found {len(txt_files)} documents to process...")

        for filename in txt_files:
            filepath = os.path.join(folder_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                if not content.strip():
                    raise ValueError("Document is empty")

                tokens = self.preprocess_document(content)
                documents.append({
                    'filename': filename,
                    'tokens': tokens,
                    'text': content
                })
                print(f"✓ Loaded: {filename} ({len(tokens)} tokens)")

            except UnicodeDecodeError as e:
                error_msg = f"{filename}: Encoding error (try UTF-8). {str(e)}"
                print(f"✗ {error_msg}")
                self.load_errors.append(error_msg)
            except ValueError as e:
                error_msg = f"{filename}: {str(e)}"
                print(f"✗ {error_msg}")
                self.load_errors.append(error_msg)
            except Exception as e:
                error_msg = f"{filename}: {str(e)}"
                print(f"✗ ERROR: {error_msg}")
                self.load_errors.append(error_msg)

        if not documents and self.load_errors:
            print(f"\nCRITICAL: No documents loaded successfully!")
            print("Errors encountered:")
            for err in self.load_errors:
                print(f"  - {err}")

        return documents

    def get_errors(self):
        return self.load_errors
