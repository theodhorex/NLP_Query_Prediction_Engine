from flask import Flask, render_template, request, jsonify, Response
from datetime import datetime, timezone
import csv
import io
import os
import sys

scripts_path = os.path.join(os.path.dirname(__file__), 'scripts')
sys.path.insert(0, scripts_path)

from preprocess import DocumentPreprocessor
from language_model import LanguageModel
from search_engine import SearchEngine

app = Flask(__name__)

preprocessor = DocumentPreprocessor()
model = LanguageModel()

prediction_history = []
MAX_HISTORY = 200

CORPUS_PATH = os.path.join(os.path.dirname(__file__), 'corpus')
CORPUS_ID_PATH = os.path.join(os.path.dirname(__file__), 'corpus_id')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'language_model.pkl')

search_engine = None

def load_or_build_model():
    global search_engine
    try:
        if os.path.exists(MODEL_PATH):
            model.load_model(MODEL_PATH)
            print("✓ Model loaded from cache")
        else:
            print("Building model from corpus...")
            documents = preprocessor.load_documents_from_folder(CORPUS_PATH)

            if not documents:
                errors = preprocessor.get_errors()
                print("ERROR: No documents loaded!")
                if errors:
                    print("Details:")
                    for err in errors:
                        print(f"  - {err}")
                raise RuntimeError("Failed to load any documents from corpus")

            print(f"Building n-gram model from {len(documents)} documents...")
            model.build_model(documents)
            model.save_model(MODEL_PATH)

        try:
            search_engine = SearchEngine(CORPUS_PATH, CORPUS_ID_PATH, preprocessor)
            print("✓ Search engine initialized")
        except Exception as e:
            print(f"WARNING: Search engine failed to initialize: {str(e)}")
            search_engine = None

        return model

    except Exception as e:
        print(f"CRITICAL ERROR during model loading: {str(e)}")
        raise

@app.route('/')
def index():
    try:
        stats = model.get_statistics()
        return render_template('index.html', stats=stats)
    except Exception as e:
        return jsonify({'error': f'Failed to load page: {str(e)}'}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        if not data:
            return jsonify({'error': 'Request body is empty'}), 400

        query = data.get('query', '').strip()

        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400

        if len(query) > 200:
            return jsonify({'error': 'Query too long (max 200 characters)'}), 400

        query_words = query.split()

        if len(query_words) > 2:
            query_words = query_words[-2:]

        predictions = model.predict_next_word(query_words)
        timestamp = datetime.now(timezone.utc).isoformat()

        response_predictions = []
        for idx, prediction in enumerate(predictions, start=1):
            response_predictions.append({
                'word': prediction['word'],
                'probability': float(prediction['probability']),
                'count': int(prediction.get('count', 0)),
                'rank': idx
            })

        prediction_history.append({
            'query': query,
            'predictions': response_predictions,
            'timestamp': timestamp
        })

        if len(prediction_history) > MAX_HISTORY:
            prediction_history.pop(0)

        return jsonify({
            'query': query,
            'timestamp': timestamp,
            'predictions': response_predictions
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        print(f"Error in /api/predict: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/stats', methods=['GET'])
def stats():
    try:
        stats_data = model.get_statistics()
        return jsonify(stats_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-csv', methods=['GET'])
def export_csv():
    try:
        export_type = request.args.get('type', 'latest').lower()
        if export_type not in ['latest', 'history']:
            return jsonify({'error': 'Invalid export type'}), 400

        if not prediction_history:
            return jsonify({'error': 'No predictions available'}), 404

        entries = [prediction_history[-1]] if export_type == 'latest' else prediction_history

        output = io.StringIO()
        fieldnames = ['query', 'rank', 'word', 'probability', 'count', 'timestamp']
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for entry in entries:
            query = entry.get('query', '')
            timestamp = entry.get('timestamp', '')
            for idx, pred in enumerate(entry.get('predictions', []), start=1):
                writer.writerow({
                    'query': query,
                    'rank': pred.get('rank', idx),
                    'word': pred.get('word', ''),
                    'probability': pred.get('probability', 0),
                    'count': pred.get('count', 0),
                    'timestamp': timestamp
                })

        csv_data = output.getvalue()
        filename = f'predictions_{export_type}.csv'
        headers = {'Content-Disposition': f'attachment; filename={filename}'}
        return Response(csv_data, mimetype='text/csv', headers=headers)

    except Exception as e:
        print(f"Error in /api/export-csv: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/ngram-stats', methods=['GET'])
def ngram_stats():
    try:
        data = model.get_ngram_stats(limit=20)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/corpus-stats', methods=['GET'])
def corpus_stats():
    try:
        data = model.get_corpus_stats(limit=20)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def stats_page():
    return render_template('stats.html')

@app.route('/search')
def search_page():
    return render_template('search.html')

@app.route('/api/search', methods=['POST'])
def search():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Request body is empty'}), 400

        query = data.get('query', '').strip()
        if not query:
            return jsonify({'error': 'Query tidak boleh kosong'}), 400

        if len(query) > 200:
            return jsonify({'error': 'Query terlalu panjang (max 200 karakter)'}), 400

        top_k = min(int(data.get('top_k', 10)), 30)
        language = data.get('language', 'all').lower()
        if language not in ('all', 'en', 'id'):
            language = 'all'

        if search_engine is None:
            return jsonify({'error': 'Search engine tidak tersedia'}), 503

        results = search_engine.search(query, top_k=top_k, language=language)

        return jsonify({
            'query': query,
            'results': results,
            'total': len(results),
            'language_filter': language
        })

    except ValueError as e:
        return jsonify({'error': f'Input tidak valid: {str(e)}'}), 400
    except Exception as e:
        print(f"Error in /api/search: {str(e)}")
        return jsonify({'error': 'Terjadi kesalahan internal server'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# Static assets (logo, favicon) are served from the `static/` folder by Flask.

if __name__ == '__main__':
    try:
        load_or_build_model()
        print("\n✓ Server starting on http://localhost:5000")
        app.run(debug=True, port=5000)
    except Exception as e:
        print(f"\nFAILED TO START: {str(e)}")
        sys.exit(1)
