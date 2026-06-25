import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    precision_recall_curve, 
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

MODELS_DIR = "models"
REPORTS_DIR = "reports"

def load_saved_objects():
    lr_path = os.path.join(MODELS_DIR, "lr_model.pkl")
    nb_path = os.path.join(MODELS_DIR, "nb_model.pkl")
    test_data_path = os.path.join(MODELS_DIR, "test_data.pkl")
    
    with open(lr_path, 'rb') as f:
        lr_model = pickle.load(f)
    with open(nb_path, 'rb') as f:
        nb_model = pickle.load(f)
    with open(test_data_path, 'rb') as f:
        X_test_vec, y_test = pickle.load(f)
        
    return lr_model, nb_model, X_test_vec, y_test

def evaluate_models():
    if not os.path.exists(REPORTS_DIR):
        os.makedirs(REPORTS_DIR)
        
    lr, nb, X_test, y_test = load_saved_objects()
    
    # 1. Predictions
    y_pred_lr = lr.predict(X_test)
    y_pred_nb = nb.predict(X_test)
    
    # Predict probabilities for precision-recall curves
    y_prob_lr = lr.predict_proba(X_test)[:, 1]
    y_prob_nb = nb.predict_proba(X_test)[:, 1]
    
    # 2. Performance Metrics
    metrics = {
        'Logistic Regression': {
            'Accuracy': accuracy_score(y_test, y_pred_lr),
            'Precision': precision_score(y_test, y_pred_lr),
            'Recall': recall_score(y_test, y_pred_lr),
            'F1': f1_score(y_test, y_pred_lr),
        },
        'Naive Bayes': {
            'Accuracy': accuracy_score(y_test, y_pred_nb),
            'Precision': precision_score(y_test, y_pred_nb),
            'Recall': recall_score(y_test, y_pred_nb),
            'F1': f1_score(y_test, y_pred_nb),
        }
    }
    
    print("\n" + "="*40)
    print("LOGISTIC REGRESSION REPORT:")
    print("="*40)
    print(classification_report(y_test, y_pred_lr))
    
    print("\n" + "="*40)
    print("NAIVE BAYES REPORT:")
    print("="*40)
    print(classification_report(y_test, y_pred_nb))
    
    # 3. Plots
    # Confusion Matrix Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm_lr = confusion_matrix(y_test, y_pred_lr)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    axes[0].set_title('Logistic Regression Confusion Matrix')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
    axes[1].set_title('Naive Bayes Confusion Matrix')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrices.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion Matrix saved to {cm_path}")
    
    # Precision-Recall Curve Plot
    precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_prob_lr)
    precision_nb, recall_nb, _ = precision_recall_curve(y_test, y_prob_nb)
    
    ap_lr = average_precision_score(y_test, y_prob_lr)
    ap_nb = average_precision_score(y_test, y_prob_nb)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_lr, precision_lr, label=f'Logistic Regression (AP = {ap_lr:.3f})', color='blue')
    plt.plot(recall_nb, precision_nb, label=f'Naive Bayes (AP = {ap_nb:.3f})', color='green')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    
    pr_path = os.path.join(REPORTS_DIR, "precision_recall_curves.png")
    plt.savefig(pr_path)
    plt.close()
    print(f"Precision-Recall Curve saved to {pr_path}")
    
    # Determine and save best model
    best_model_name = "Logistic Regression" if metrics['Logistic Regression']['F1'] >= metrics['Naive Bayes']['F1'] else "Naive Bayes"
    best_model_file = "lr_model.pkl" if best_model_name == "Logistic Regression" else "nb_model.pkl"
    
    # Link or copy best model as models/best_model.pkl
    best_model_src = os.path.join(MODELS_DIR, best_model_file)
    best_model_dest = os.path.join(MODELS_DIR, "best_model.pkl")
    
    # Save a record of the best model name
    with open(os.path.join(MODELS_DIR, "best_model_info.txt"), "w") as f:
        f.write(best_model_name)
        
    with open(best_model_src, 'rb') as src_f:
        best_model_obj = pickle.load(src_f)
    with open(best_model_dest, 'wb') as dest_f:
        pickle.dump(best_model_obj, dest_f)
        
    print(f"\nBest Model identified by F1 score: {best_model_name}")
    print(f"Saved best model copy to {best_model_dest}")

if __name__ == "__main__":
    evaluate_models()
