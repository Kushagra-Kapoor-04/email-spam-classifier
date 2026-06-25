import os
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from data import get_train_test_data
from vectorizer import train_and_save_vectorizer

MODELS_DIR = "models"

def train_models():
    # 1. Load data
    print("Loading data...")
    X_train, X_test, y_train, y_test = get_train_test_data()
    
    # 2. Vectorize X_train and save vectorizer
    X_train_vec, vectorizer = train_and_save_vectorizer(X_train)
    
    # 3. Vectorize X_test
    X_test_vec = vectorizer.transform(X_test)
    
    # 4. Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_vec, y_train)
    
    # Save Logistic Regression model
    lr_path = os.path.join(MODELS_DIR, "lr_model.pkl")
    print(f"Saving Logistic Regression to {lr_path}...")
    with open(lr_path, 'wb') as f:
        pickle.dump(lr_model, f)
        
    # 5. Train Naive Bayes
    print("Training Multinomial Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    
    # Save Naive Bayes model
    nb_path = os.path.join(MODELS_DIR, "nb_model.pkl")
    print(f"Saving Naive Bayes to {nb_path}...")
    with open(nb_path, 'wb') as f:
        pickle.dump(nb_model, f)
        
    # Save test data to run evaluation on
    test_data_path = os.path.join(MODELS_DIR, "test_data.pkl")
    with open(test_data_path, 'wb') as f:
        pickle.dump((X_test_vec, y_test), f)
        
    print("Training completed successfully.")

if __name__ == "__main__":
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    train_models()
