import unittest

import app


class AppRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app.load_or_build_model()
        cls.client = app.app.test_client()

    def test_stats_endpoint_loads_model_data(self):
        response = self.client.get('/api/stats')

        self.assertEqual(response.status_code, 200)
        self.assertGreater(response.json['unigrams_count'], 0)
        self.assertGreater(response.json['bigrams_count'], 0)
        self.assertGreater(response.json['trigrams_count'], 0)
        self.assertGreater(response.json['vocabulary_size'], 0)

    def test_predict_rejects_more_than_two_words(self):
        response = self.client.post('/api/predict', json={'query': 'one two three'})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json, {'error': 'Please enter maximum 2 words'})

    def test_predict_accepts_partial_single_word_queries(self):
        response = self.client.post('/api/predict', json={'query': 'tes'})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['query'], 'tes')
        self.assertIsInstance(response.json['predictions'], list)


if __name__ == '__main__':
    unittest.main()
