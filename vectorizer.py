import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

MODELS_DIR = "models"

def build_vectorizer(max_features=5000):
    return TfidfVectorizer(max_features=max_features)

def train_and_save_vectorizer(texts, save_path=None):
    if save_path is None:
        if not os.path.exists(MODELS_DIR):
            os.makedirs(MODELS_DIR)
        save_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
        
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = build_vectorizer()
    X_vec = vectorizer.fit_transform(texts)
    
    print(f"Saving vectorizer to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    return X_vec, vectorizer

def load_vectorizer(path=None):
    if path is None:
        path = os.path.join(MODELS_DIR, "vectorizer.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Vectorizer not found at {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)
