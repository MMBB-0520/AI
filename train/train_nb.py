"""
train_model.py
----------------
Trains a Multinomial Naive Bayes classifier to recognise user intents
for the Hotel Booking Chatbot.
"""

import sys
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# Setup paths dynamically so it doesn't break depending on where you run it
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import preprocessing (with fallback in case they updated the preprocessing script structure)
try:
    from chatbot.preprocessing import preprocess_text
except ImportError:
    from chatbot.preprocessing import TextPreprocessor
    preprocessor = TextPreprocessor()
    preprocess_text = preprocessor.process

# Model save paths
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "nb.pkl")
VECTORIZER_PATH = os.path.join(PROJECT_ROOT, "models", "nb_vectorizer.pkl")
ENCODER_PATH = os.path.join(PROJECT_ROOT, "models", "nb_label_encoder.pkl")
CM_PATH = os.path.join(PROJECT_ROOT, "models", "nb_confusion_matrix.png")


def main():
    print("========== Loading Dataset (Naive Bayes) ==========")
    
    # 1. Dynamic Dataset Loading (Bulletproof against name changes)
    dataset_dir = os.path.join(PROJECT_ROOT, "dataset")
    possible_datasets = [
        "bitext-hospitality-llm-chatbot-training-dataset.csv",
        "hotel_booking.csv",
        "intents.csv"
    ]
    
    dataset_path = None
    for file in possible_datasets:
        path = os.path.join(dataset_dir, file)
        if os.path.exists(path):
            dataset_path = path
            break
            
    if not dataset_path:
        raise FileNotFoundError("Could not find any dataset in the /dataset/ folder!")
        
    print(f"Loaded dataset: {os.path.basename(dataset_path)}")
    df = pd.read_csv(dataset_path)

    print(f"Total dataset rows: {len(df)}")
    
    # 2. Handle column name changes safely ('text' vs 'utterance')
    text_col = 'instruction'
    intent_col = 'intent'

    print(f"Unique intents: {df[intent_col].nunique()}")

    # Apply preprocessing pipeline
    print("Applying NLP preprocessing (normalization, tokenization, lemmatization)...")
    X_raw = df[text_col].astype(str)
    X_cleaned = X_raw.apply(preprocess_text)
    y_raw = df[intent_col]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(X_cleaned)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print("Training Multinomial Naive Bayes Model...")
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train, y_train)

    # 5-fold Stratified Cross Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(MultinomialNB(alpha=0.1), X, y, cv=skf)
    print(f"5-fold CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    y_pred = nb_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print("\n========== Evaluation Results (Naive Bayes) ==========")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\n========== Classification Report ==========")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues")
    plt.title("Naive Bayes Intent Classification - Confusion Matrix")
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(CM_PATH), exist_ok=True)
    plt.savefig(CM_PATH)

    print("\nSaving trained Naive Bayes model & preprocessors...")
    joblib.dump(nb_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    print("Naive Bayes Model trained and saved successfully!")

if __name__ == "__main__":
    main()
