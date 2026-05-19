import json
import pickle
from collections import defaultdict, Counter
from nltk.util import ngrams

class LanguageModel:
    def __init__(self):
        self.unigrams = Counter()
        self.bigrams = Counter()
        self.trigrams = Counter()
        self.vocabulary = set()

    def build_model(self, documents):
        """Build unigram, bigram, trigram from documents"""
        all_tokens = []

        for doc in documents:
            tokens = doc['tokens']
            all_tokens.extend(tokens)

            self.unigrams.update(tokens)

            bigram_list = list(ngrams(tokens, 2))
            self.bigrams.update(bigram_list)

            trigram_list = list(ngrams(tokens, 3))
            self.trigrams.update(trigram_list)

        self.vocabulary = set(all_tokens)
        print(f"Model built: {len(self.unigrams)} unigrams, {len(self.bigrams)} bigrams, {len(self.trigrams)} trigrams")

    def predict_next_word(self, query_words):
        """Predict next word based on query using Language Model"""
        query_words = [w.lower() for w in query_words]
        predictions = []

        if len(query_words) == 1:
            word = query_words[0]
            candidates = defaultdict(float)

            for (w1, w2), count in self.bigrams.items():
                if w1 == word:
                    prob = count / self.unigrams[w1] if self.unigrams[w1] > 0 else 0
                    candidates[w2] = prob

            sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
            predictions = [(word, prob) for word, prob in sorted_candidates[:10]]

        elif len(query_words) >= 2:
            w1, w2 = query_words[-2], query_words[-1]
            candidates = defaultdict(float)

            for (tw1, tw2, tw3), count in self.trigrams.items():
                if tw1 == w1 and tw2 == w2:
                    prob = count / self.bigrams.get((w1, w2), 1) if self.bigrams.get((w1, w2), 0) > 0 else 0
                    candidates[tw3] = prob

            sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
            predictions = [(word, prob) for word, prob in sorted_candidates[:10]]

            if not predictions:
                for (tw1, tw2), count in self.bigrams.items():
                    if tw1 == w2:
                        prob = count / self.unigrams[w2] if self.unigrams[w2] > 0 else 0
                        candidates[tw2] = prob

                sorted_candidates = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
                predictions = [(word, prob) for word, prob in sorted_candidates[:10]]

        return predictions

    def save_model(self, filepath):
        """Save model to pickle file"""
        model_data = {
            'unigrams': self.unigrams,
            'bigrams': self.bigrams,
            'trigrams': self.trigrams,
            'vocabulary': self.vocabulary
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model from pickle file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.unigrams = model_data['unigrams']
        self.bigrams = model_data['bigrams']
        self.trigrams = model_data['trigrams']
        self.vocabulary = model_data['vocabulary']
        print(f"Model loaded from {filepath}")

    def get_statistics(self):
        """Get model statistics"""
        return {
            'unigrams_count': len(self.unigrams),
            'bigrams_count': len(self.bigrams),
            'trigrams_count': len(self.trigrams),
            'vocabulary_size': len(self.vocabulary)
        }
