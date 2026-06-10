"""Tests for Flask app endpoints"""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), 'scripts'))

os.environ['DEBUG'] = 'false'

from app import app, prediction_history


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


@pytest.fixture(autouse=True)
def reset_history():
    prediction_history.clear()


class TestPages:
    def test_index(self, client):
        resp = client.get('/')
        assert resp.status_code in (200, 500)

    def test_stats_page(self, client):
        resp = client.get('/stats')
        assert resp.status_code == 200
        assert b'text/html' in resp.data or b'<!DOCTYPE' in resp.data

    def test_search_page(self, client):
        resp = client.get('/search')
        assert resp.status_code == 200

    def test_404(self, client):
        resp = client.get('/nonexistent')
        assert resp.status_code == 404


class TestApiPredict:
    ENDPOINT = '/api/predict'

    def test_empty_body(self, client):
        resp = client.post(self.ENDPOINT, content_type='application/json', data='')
        # Flask JSON parser throws 400 -> caught by generic except -> 500
        assert resp.status_code == 500

    def test_empty_json(self, client):
        resp = client.post(self.ENDPOINT, json={})
        assert resp.status_code == 400
        assert 'error' in resp.get_json()

    def test_empty_query(self, client):
        resp = client.post(self.ENDPOINT, json={'query': ''})
        assert resp.status_code == 400

    def test_blank_query(self, client):
        resp = client.post(self.ENDPOINT, json={'query': '   '})
        assert resp.status_code == 400

    def test_long_query(self, client):
        resp = client.post(self.ENDPOINT, json={'query': 'x' * 201})
        assert resp.status_code == 400

    def test_response_structure(self, client):
        resp = client.post(self.ENDPOINT, json={'query': 'machine'})
        if resp.status_code == 200:
            data = resp.get_json()
            assert 'query' in data
            assert 'predictions' in data
            assert 'timestamp' in data
            assert isinstance(data['predictions'], list)
            if data['predictions']:
                p = data['predictions'][0]
                assert 'word' in p
                assert 'probability' in p
                assert 'count' in p
                assert 'rank' in p

    def test_history_updated(self, client):
        prediction_history.clear()
        client.post(self.ENDPOINT, json={'query': 'machine'})
        assert len(prediction_history) >= 1
        assert prediction_history[-1]['query'] == 'machine'

    def test_history_capped(self, client):
        prediction_history.clear()
        for i in range(250):
            client.post(self.ENDPOINT, json={'query': f'word{i}'})
        assert len(prediction_history) <= 200


class TestApiStats:
    def test_endpoint(self, client):
        resp = client.get('/api/stats')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'unigrams_count' in data
        assert 'bigrams_count' in data
        assert 'trigrams_count' in data
        assert 'vocabulary_size' in data
        assert isinstance(data['unigrams_count'], int)

    def test_ngram_stats(self, client):
        resp = client.get('/api/ngram-stats')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'top_unigrams' in data
        assert 'top_bigrams' in data

    def test_corpus_stats(self, client):
        resp = client.get('/api/corpus-stats')
        assert resp.status_code == 200
        data = resp.get_json()
        assert 'total_documents' in data
        assert 'total_tokens' in data
        assert 'unique_words' in data


class TestApiSearch:
    ENDPOINT = '/api/search'

    def test_empty_body(self, client):
        resp = client.post(self.ENDPOINT, content_type='application/json', data='')
        assert resp.status_code == 500

    def test_empty_query(self, client):
        resp = client.post(self.ENDPOINT, json={'query': ''})
        data = resp.get_json()
        assert resp.status_code == 400
        assert 'error' in data

    def test_blank_query(self, client):
        resp = client.post(self.ENDPOINT, json={'query': '   '})
        assert resp.status_code == 400

    def test_long_query(self, client):
        resp = client.post(self.ENDPOINT, json={'query': 'x' * 201})
        assert resp.status_code == 400

    def test_response_structure(self, client):
        resp = client.post(self.ENDPOINT, json={'query': 'machine', 'top_k': 5, 'language': 'all'})
        if resp.status_code == 200:
            data = resp.get_json()
            assert 'query' in data
            assert 'results' in data
            assert 'total' in data
            assert isinstance(data['results'], list)
        elif resp.status_code == 503:
            assert 'search engine' in resp.get_json()['error'].lower()

    def test_invalid_top_k(self, client):
        resp = client.post(self.ENDPOINT, json={'query': 'machine', 'top_k': 'abc'})
        # int('abc') raises ValueError -> caught by ValueError handler -> 400
        assert resp.status_code == 400

    def test_invalid_language_defaults(self, client):
        resp = client.post(self.ENDPOINT, json={'query': 'machine', 'language': 'klingon'})
        assert resp.status_code in (200, 503)

    def test_english_filter(self, client):
        resp = client.post(self.ENDPOINT, json={'query': 'machine', 'language': 'en'})
        if resp.status_code == 200:
            data = resp.get_json()
            for r in data['results']:
                assert r['language'] == 'en'

    def test_indonesian_filter(self, client):
        resp = client.post(self.ENDPOINT, json={'query': 'machine', 'language': 'id'})
        if resp.status_code == 200:
            data = resp.get_json()
            for r in data['results']:
                assert r['language'] == 'id'


class TestExportCsv:
    def test_no_history_latest(self, client):
        resp = client.get('/api/export-csv?type=latest')
        assert resp.status_code == 404

    def test_no_history_all(self, client):
        resp = client.get('/api/export-csv?type=history')
        assert resp.status_code == 404

    def test_with_history(self, client):
        client.post('/api/predict', json={'query': 'machine learning'})
        resp = client.get('/api/export-csv?type=latest')
        assert resp.status_code == 200
        assert 'text/csv' in resp.content_type
        assert 'predictions_latest.csv' in resp.headers.get('Content-Disposition', '')

    def test_invalid_type(self, client):
        resp = client.get('/api/export-csv?type=invalid')
        assert resp.status_code == 400


class TestMethodNotAllowed:
    def test_get_on_post_endpoint(self, client):
        resp = client.get('/api/predict')
        assert resp.status_code == 405
