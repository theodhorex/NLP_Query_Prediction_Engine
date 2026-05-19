from flask import Flask, render_template, request, jsonify
import os
import sys

scripts_path = os.path.join(os.path.dirname(__file__), 'scripts')
sys.path.insert(0, scripts_path)

from preprocess import DocumentPreprocessor
from language_model import LanguageModel

app = Flask(__name__)

preprocessor = DocumentPreprocessor()
model = LanguageModel()

CORPUS_PATH = os.path.join(os.path.dirname(__file__), 'corpus')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'language_model.pkl')

def load_or_build_model():
    global model
    if model is None:
        model = LanguageModel()
    if os.path.exists(MODEL_PATH):
        model.load_model(MODEL_PATH)
        print("Model loaded from cache")
    else:
        documents = preprocessor.load_documents_from_folder(CORPUS_PATH)
        if documents:
            model.build_model(documents)
            model.save_model(MODEL_PATH)
        else:
            print("No documents found!")
    return model

def ensure_model_loaded():
    if model is None or not model.is_loaded():
        load_or_build_model()

@app.route('/')
def index():
    ensure_model_loaded()
    stats = model.get_statistics()
    return render_template('index.html', stats=stats)

@app.route('/api/predict', methods=['POST'])
def predict():
    ensure_model_loaded()
    data = request.get_json(silent=True) or {}
    query = data.get('query', '').strip()

    if not query:
        return jsonify({'error': 'Query cannot be empty'}), 400

    query_words = query.split()
    if len(query_words) > 2:
        return jsonify({'error': 'Please enter maximum 2 words'}), 400
    predictions = model.predict_next_word(query_words)

    return jsonify({
        'query': query,
        'predictions': [{'word': word, 'probability': float(prob)} for word, prob in predictions]
    })

@app.route('/api/stats', methods=['GET'])
def stats():
    ensure_model_loaded()
    return jsonify(model.get_statistics())

if __name__ == '__main__':
    load_or_build_model()
    app.run(debug=True, port=5000)
