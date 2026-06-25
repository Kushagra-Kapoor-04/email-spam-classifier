from model import train_models
from evaluate import evaluate_models

if __name__ == "__main__":
    print("Step 1/2 — Training models...")
    train_models()
    print("\nStep 2/2 — Evaluating and generating reports...")
    evaluate_models()
    print("\nDone. Run 'python app.py' to start the API.")