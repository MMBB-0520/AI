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

import json
import re
import string
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

DATA_PATH = "dataset/intents.csv"
MODEL_PATH = "models/naive_bayes.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"
CONFUSION_MATRIX_PATH = "models/confusion_matrix.png"


def clean_text(text: str) -> str:
    """Lowercase and strip punctuation from a sentence."""
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts, labels = [], []
    for intent in data["intents"]:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            texts.append(clean_text(pattern))
            labels.append(tag)
    return texts, labels


def main():
    print("Loading dataset...")
    texts, labels = load_dataset(DATA_PATH)
    print(f"Total training examples: {len(texts)}")
    print(f"Number of intents: {len(set(labels))}")

    # 80/20 stratified split so every intent appears in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Bag-of-words vectorisation with unigrams + bigrams
    # (CountVectorizer outperformed TF-IDF for this dataset during tuning,
    #  which fits Naive Bayes' assumption of raw term frequencies)
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Naive Bayes (small alpha since the dataset/vocabulary is compact)
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)

    # 5-fold stratified cross-validation on the full dataset gives a more
    # reliable estimate of performance than a single small train/test split
    X_all_vec = vectorizer.transform(texts)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(MultinomialNB(alpha=0.1), X_all_vec, labels, cv=skf)
    print(f"\n5-fold Cross-Validation Accuracy: {cv_scores.mean():.4f} "
          f"(+/- {cv_scores.std():.4f})")

    # Evaluate on the held-out test split
    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    print("\n===== Evaluation Results (Naive Bayes) =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro)   : {recall:.4f}")
    print(f"F1-score (macro) : {f1:.4f}")
    print("\nDetailed classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix plot
    labels_sorted = sorted(set(labels))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_sorted)
    disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
    plt.title("Naive Bayes Intent Classification - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    print(f"\nConfusion matrix saved to {CONFUSION_MATRIX_PATH}")

    # Save model + vectorizer
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Vectorizer saved to {VECTORIZER_PATH}")