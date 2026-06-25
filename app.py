import os
import pickle
from flask import Flask, request, jsonify
from utils import clean_text

app = Flask(__name__)

# Paths
MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pkl")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "vectorizer.pkl")

# Load models at startup
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        
        # Load best model info if exists
        info_path = os.path.join(MODELS_DIR, "best_model_info.txt")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                model_name = f.read().strip()
        else:
            model_name = "Selected Model"
        print(f"Loaded {model_name} and Vectorizer successfully.")
    else:
        model = None
        vectorizer = None
        model_name = None
        print("Warning: Model files not found. Please run training and evaluation first.")
except Exception as e:
    model = None
    vectorizer = None
    model_name = None
    print(f"Error loading model assets: {e}")
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "running",
        "model": model_name,
        "usage": "POST /predict with JSON body: {\"text\": \"your message here\"}"
    })
@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model has not been trained or loaded. Please train the model first."}), 500
        
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Invalid request. JSON body must contain a 'text' key."}), 400
        
    text = data["text"]
    cleaned = clean_text(text)
    
    # Vectorize and Predict
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    
    # Predict probability if available
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(vec)[0][pred])
        
    return jsonify({
        "prediction": "spam" if pred == 1 else "ham",
        "confidence": prob,
        "model_used": model_name
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
