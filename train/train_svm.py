"""
train_svm.py
------------
Trains a tuned Support Vector Machine (SVM) classifier to recognize user intents 
for the BookMate Hotel Booking Chatbot using GridSearchCV and class weighting.
"""
import sys
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)
from sklearn.svm import SVC

# PROJECT PATH
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
from chatbot.preprocessing import preprocess_text

# PATH CONFIGURATION
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "bitext-hospitality-llm-chatbot-training-dataset.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "svm.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "svm_vectorizer.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "svm_label_encoder.pkl")
CM_PATH = os.path.join(MODEL_DIR, "svm_confusion_matrix.png")

def main():
    print("=" * 60)
    print("Loading Bitext Hospitality Dataset & Training Optimized SVM")
    print("=" * 60)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found:\n{DATASET_PATH}")
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = pd.read_csv(DATASET_PATH).dropna(subset=["instruction", "intent"]).copy()
    
    X_raw = df["instruction"].astype(str)
    y_raw = df["intent"].astype(str)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Applying NLP Preprocessing...")
    X_train_cleaned = X_train_raw.apply(preprocess_text)
    X_test_cleaned = X_test_raw.apply(preprocess_text)

    # 1. Build the Pipeline with CalibratedClassifierCV
    base_svm = SVC(kernel="linear", random_state=42, class_weight="balanced")
    calibrated_svm = CalibratedClassifierCV(estimator=base_svm, ensemble=False)

    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)),
        ('classifier', calibrated_svm)
    ])

    # 2. Define Hyperparameter Grid
    param_grid = {
        'classifier__estimator__C': [0.1, 1.0, 10.0]
    }

    # 3. GridSearchCV for Optimization
    print("\nRunning GridSearchCV (This may take a moment)...")
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train_cleaned, y_train)

    print(f"Best Parameters: {grid_search.best_params_}")
    
    # Extract the best model and vectorizer from the winning pipeline
    best_pipeline = grid_search.best_estimator_
    best_vectorizer = best_pipeline.named_steps['vectorizer']
    best_classifier = best_pipeline.named_steps['classifier']

    print("\nGenerating predictions on Test Set...")
    X_test = best_vectorizer.transform(X_test_cleaned)
    y_pred = best_classifier.predict(X_test)

    # Evaluation
    print("\n" + "=" * 60)
    print("SVM Evaluation Results")
    print("=" * 60)
    print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision : {precision_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"Recall    : {recall_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 Score  : {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")

    print("\nSaving Optimized SVM Models...")
    joblib.dump(best_classifier, MODEL_PATH)
    joblib.dump(best_vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    print("SVM training completed successfully!")

    print(
        "Generating confusion matrix..."
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(14, 12)
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    disp.plot(
        ax=ax,
        xticks_rotation=90,
        cmap="Blues",
        colorbar=False
    )

    plt.title(
        "SVM Intent Classification - Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CM_PATH,
        dpi=200
    )

    plt.close()

    print(
        f"Confusion matrix saved to:\n"
        f"{CM_PATH}"
    )

if __name__ == "__main__":
    main()