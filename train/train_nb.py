"""
train_model.py
----------------
Trains a Multinomial Naive Bayes classifier to recognise user intents
for the Hotel Booking Chatbot.

Pipeline:
    1. Load intents.json (patterns -> tag)
    2. Text pre-processing (lowercasing, punctuation removal, TF-IDF vectorisation)
    3. Train/test split (stratified, 80/20)
    4. Train MultinomialNB
    5. Evaluate: accuracy, precision, recall, F1-score (macro average), confusion matrix
    6. Save the trained model + vectorizer for use by chatbot.py
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

# Add root directory to sys.path to allow imports from chatbot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chatbot.preprocessing import preprocess_text

DATASET_PATH = "dataset/intents.csv"
MODEL_PATH = "models/nb.pkl"
VECTORIZER_PATH = "models/nb_vectorizer.pkl"
ENCODER_PATH = "models/nb_label_encoder.pkl"
CM_PATH = "models/nb_confusion_matrix.png"


def main():
    print("========== Loading Dataset (Naive Bayes) ==========")
    df = pd.read_csv(DATASET_PATH)

    print(f"Total dataset rows: {len(df)}")
    print(f"Unique intents: {df['intent'].nunique()}")

    # Apply preprocessing pipeline
    print("Applying NLP preprocessing (normalization, tokenization, lemmatization)...")
    X_raw = df["text"]
    X_cleaned = X_raw.apply(preprocess_text)
    y_raw = df["intent"]

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
    plt.savefig(CM_PATH)

    print("\nSaving trained Naive Bayes model & preprocessors...")
    joblib.dump(nb_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    print("Naive Bayes Model trained and saved successfully!")


if __name__ == "__main__":
    main()