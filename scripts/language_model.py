import json
import pickle
import os
from collections import defaultdict, Counter
from nltk.util import ngrams

class LanguageModel:
    def __init__(self):
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.vocabulary = set()
        self.document_stats = []
        self.total_tokens = 0
        self.total_documents = 0
        self.build_errors = []

    def build_model(self, documents):
        """Build unigram, bigram, trigram from documents"""
        if not documents:
            raise ValueError("No documents provided to build model")

        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.vocabulary = set()
        self.document_stats = []
        self.total_tokens = 0
        self.total_documents = len(documents)

        all_tokens = []
        for doc in documents:
            if not doc.get('tokens'):
                raise ValueError(f"Document {doc.get('filename', 'unknown')} has no tokens")

            tokens = doc['tokens']
            token_count = len(tokens)
            unique_count = len(set(tokens))
            all_tokens.extend(tokens)

            self.document_stats.append({
                'filename': doc.get('filename', 'unknown'),
                'token_count': token_count,
                'unique_words': unique_count
            })
            self.total_tokens += token_count

            self.unigrams.update(tokens)

            bigram_list = list(ngrams(tokens, 2))
            self.bigrams.update(bigram_list)

            trigram_list = list(ngrams(tokens, 3))
            self.trigrams.update(trigram_list)

        if not all_tokens:
            raise ValueError("No tokens generated from documents")

        self.vocabulary = set(all_tokens)
        print(f"✓ Model built: {len(self.unigrams)} unigrams, {len(self.bigrams)} bigrams, {len(self.trigrams)} trigrams")

    def predict_next_word(self, query_words):
        """Predict next word based on query using Language Model"""
        try:
            if not query_words:
                return []

            if not isinstance(query_words, (list, tuple)):
                raise ValueError("query_words must be a list or tuple")

            query_words = [w.lower() for w in query_words if isinstance(w, str)]

            if not query_words:
                raise ValueError("No valid query words after filtering")

            vocab_size = len(self.vocabulary) or 1

            # Step 1: coba trigram jika tersedia 2 kata konteks
            if len(query_words) >= 2:
                w1, w2 = query_words[-2], query_words[-1]

                if w1 and w2:
                    candidates = {}
                    total_context = self.bigrams.get((w1, w2), 0)

                    for (tw1, tw2, tw3), count in self.trigrams.items():
                        if tw1 == w1 and tw2 == w2:
                            prob = (count + 1) / (total_context + vocab_size) if total_context > 0 else 0
                            candidates[tw3] = (prob, count)
                        elif tw1 == w1 and tw2.startswith(w2) and len(w2) >= 2 and tw2 != w2:
                            prob = (count + 1) / (total_context + vocab_size) if total_context > 0 else 0
                            existing = candidates.get(tw3)
                            if existing is None or prob > existing[0]:
                                candidates[tw3] = (prob, count)

                    if candidates:
                        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
                        return [
                            {'word': w, 'probability': prob, 'count': cnt}
                            for w, (prob, cnt) in sorted_candidates[:10]
                        ]

            # Step 2: coba bigram dengan kata terakhir sebagai konteks
            last_word = query_words[-1]
            if last_word:
                candidates = {}
                total_context = self.unigrams.get(last_word, 0)

                for (w1, w2), count in self.bigrams.items():
                    if w1 == last_word:
                        prob = (count + 1) / (total_context + vocab_size) if total_context > 0 else 0
                        candidates[w2] = (prob, count)
                    elif w1.startswith(last_word) and len(last_word) >= 2 and w1 != last_word:
                        prob = (count + 1) / (self.unigrams.get(w1, 0) + vocab_size) if self.unigrams.get(w1, 0) > 0 else 0
                        existing = candidates.get(w2)
                        if existing is None or prob > existing[0]:
                            candidates[w2] = (prob, count)

                if candidates:
                    sorted_candidates = sorted(candidates.items(), key=lambda x: x[1][0], reverse=True)
                    return [
                        {'word': w, 'probability': prob, 'count': cnt}
                        for w, (prob, cnt) in sorted_candidates[:10]
                    ]

            # Step 3: fallback unigram — ambil kata paling frequent
            total_all = sum(self.unigrams.values()) or 1
            predictions = [
                {'word': w, 'probability': c / total_all, 'count': c}
                for w, c in self.unigrams.most_common(10)
            ]
            return predictions

        except ValueError as e:
            print(f"Validation Error in predict_next_word: {str(e)}")
            return []
        except Exception as e:
            print(f"Unexpected error in predict_next_word: {str(e)}")
            return []

    def save_model(self, filepath):
        """Save model to pickle file"""
        try:
            if not self.unigrams or not self.bigrams:
                raise ValueError("Model is empty, cannot save")

            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            model_data = {
                'unigrams': self.unigrams,
                'bigrams': self.bigrams,
                'trigrams': self.trigrams,
                'vocabulary': self.vocabulary,
                'document_stats': self.document_stats,
                'total_tokens': self.total_tokens,
                'total_documents': self.total_documents
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            print(f"✓ Model saved to {filepath}")

        except Exception as e:
            print(f"✗ Error saving model: {str(e)}")
            raise

    def load_model(self, filepath):
        """Load model from pickle file"""
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")

            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)

            if not all(key in model_data for key in ['unigrams', 'bigrams', 'trigrams', 'vocabulary']):
                raise ValueError("Model file is corrupted or incomplete")

            self.unigrams = model_data['unigrams']
            self.bigrams = model_data['bigrams']
            self.trigrams = model_data['trigrams']
            self.vocabulary = model_data['vocabulary']
            self.document_stats = model_data.get('document_stats', [])
            self.total_documents = model_data.get('total_documents', len(self.document_stats))
            self.total_tokens = model_data.get(
                'total_tokens',
                sum(doc.get('token_count', 0) for doc in self.document_stats) or sum(self.unigrams.values())
            )
            print(f"✓ Model loaded from {filepath}")

        except FileNotFoundError as e:
            print(f"✗ {str(e)}")
            raise
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            raise

    def get_statistics(self):
        """Get model statistics"""
        return {
            'unigrams_count': len(self.unigrams),
            'bigrams_count': len(self.bigrams),
            'trigrams_count': len(self.trigrams),
            'vocabulary_size': len(self.vocabulary)
        }

    def get_ngram_stats(self, limit=20):
        total_unigrams = sum(self.unigrams.values()) or 1
        total_bigrams = sum(self.bigrams.values()) or 1

        top_unigrams = [
            {
                'word': word,
                'count': count,
                'prob': count / total_unigrams
            }
            for word, count in self.unigrams.most_common(limit)
        ]

        top_bigrams = [
            {
                'word': f"{w1} {w2}",
                'count': count,
                'prob': count / total_bigrams
            }
            for (w1, w2), count in self.bigrams.most_common(limit)
        ]

        return {
            'top_unigrams': top_unigrams,
            'top_bigrams': top_bigrams
        }

    def get_corpus_stats(self, limit=20):
        ngram_stats = self.get_ngram_stats(limit=limit)
        total_documents = self.total_documents or len(self.document_stats)
        total_tokens = self.total_tokens
        avg_tokens = (total_tokens / total_documents) if total_documents else 0

        return {
            'total_documents': total_documents,
            'total_tokens': total_tokens,
            'unique_words': len(self.vocabulary),
            'avg_tokens_per_doc': avg_tokens,
            'top_unigrams': ngram_stats['top_unigrams'],
            'top_bigrams': ngram_stats['top_bigrams'],
            'documents': self.document_stats
        }
