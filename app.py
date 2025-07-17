from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import os
import logging
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_DIR = r"" # model Address

tokenizer = None
model = None
model_id2label = {}

try:
    if not os.path.exists(MODEL_DIR):
        raise FileNotFoundError(f"The model directory at '{MODEL_DIR}' was not found. Please ensure the model directory and its files are in the correct path.")

    logging.info(f"Loading tokenizer from: {MODEL_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    logging.info("Tokenizer loaded successfully.")

    logging.info(f"Loading model from: {MODEL_DIR}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    
    if hasattr(model.config, 'id2label'):
        model_id2label = model.config.id2label
        logging.info(f"Model ID to Label mapping found: {model_id2label}")
    else:
        logging.warning("No 'id2label' mapping found in model config. Output labels might be generic (e.g., LABEL_0). Please ensure config.json has this mapping.")

    logging.info("Model loaded successfully.")

except FileNotFoundError as fnfe:
    logging.error(f"Error loading model: {fnfe}")
    logging.error("Please ensure the model directory and its files (model.safetensors, config.json, tokenizer.json) are in the correct path.")
except Exception as e:
    logging.error(f"General error loading model: {e}", exc_info=True)
    logging.error("Sentiment analysis model could not be loaded. Please check the errors above.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/analyze', methods=['POST'])
def analyze_sentiment():
    if tokenizer is None or model is None:
        logging.error("Analysis request received, but tokenizer or model is not loaded.")
        return jsonify({"error": "Sentiment analysis model components are not loaded. Please check server logs."}), 500

    try:
        data = request.get_json()
        if not data:
            logging.warning("Request received without JSON content.")
            return jsonify({"error": "Request must contain valid JSON."}), 400
        
        text = data.get('text', '')
        text = text.strip()

        if not text:
            logging.warning("Request received with empty text.")
            return jsonify({"error": "Please provide text for analysis."}), 400

    except Exception as e:
        logging.error(f"Error processing input JSON: {e}", exc_info=True)
        return jsonify({"error": "Invalid input JSON format."}), 400

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        logits = outputs.logits
        
        probabilities = torch.softmax(logits, dim=1)
        
        predicted_id = torch.argmax(probabilities, dim=1).item()
        
        sentiment_score = probabilities[0][predicted_id].item()

        sentiment_name = model_id2label.get(predicted_id, f"LABEL_{predicted_id}")

        persian_label = sentiment_name
        if "positive" in sentiment_name.lower():
            persian_label = "مثبت"
        elif "negative" in sentiment_name.lower():
            persian_label = "منفی"
        elif "neutral" in sentiment_name.lower():
            persian_label = "خنثی"
        elif "star 1" in sentiment_name.lower():
            persian_label = "بسیار منفی"
        elif "star 2" in sentiment_name.lower():
            persian_label = "منفی"
        elif "star 3" in sentiment_name.lower():
            persian_label = "خنثی"
        elif "star 4" in sentiment_name.lower():
            persian_label = "مثبت"
        elif "star 5" in sentiment_name.lower():
            persian_label = "بسیار مثبت"
            
        return jsonify({
            "text": text,
            "sentiment": persian_label,
            "score": round(float(sentiment_score), 4)
        })

    except Exception as e:
        logging.error(f"Error analyzing sentiment for text '{text}': {e}", exc_info=True)
        return jsonify({"error": f"Error in sentiment analysis: {str(e)}. Please contact the system administrator."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

