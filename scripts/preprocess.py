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

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def tokenize(self, text):
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.isalpha()]
        return tokens

    def preprocess_document(self, text):
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return tokens

    def load_documents_from_folder(self, folder_path):
        documents = []
        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} tidak ditemukan!")
            return documents

        for filename in os.listdir(folder_path):
            if filename.endswith('.txt'):
                filepath = os.path.join(folder_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        tokens = self.preprocess_document(content)
                        documents.append({
                            'filename': filename,
                            'tokens': tokens,
                            'text': content
                        })
                        print(f"Loaded: {filename} ({len(tokens)} tokens)")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        return documents
