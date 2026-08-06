"""
train_lr.py
-----------
Trains a Logistic Regression classifier to recognize user intents
for the BookMate Hotel Booking Chatbot.
"""

import sys
import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Add root directory to sys.path to allow imports from chatbot
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chatbot.preprocessing import preprocess_text

DATASET_PATH = "dataset/intents.csv"
MODEL_PATH = "models/logistic_regression.pkl"
VECTORIZER_PATH = "models/lr_vectorizer.pkl"
ENCODER_PATH = "models/lr_label_encoder.pkl"

def main():
    print("\n========== Loading Dataset (Logistic Regression) ==========")
    df = pd.read_csv(DATASET_PATH)

    print(f"Total dataset rows: {len(df)}")
    print(f"Unique intents: {df['intent'].nunique()}")

    print("\nApplying NLP preprocessing (normalization, tokenization, lemmatization)...")
    df["clean_text"] = df["text"].apply(preprocess_text)

    X = df["clean_text"]
    y = df["intent"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=None,
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining Logistic Regression Model (Higher C for sharper confidence)...")
    lr_model = LogisticRegression(
        C=20.0,
        random_state=42,
        max_iter=1000
    )

    lr_model.fit(X_train, y_train)

    y_pred = lr_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n========== Evaluation Results (Logistic Regression) ==========")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\n========== Classification Report ==========")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    cm = confusion_matrix(y_test, y_pred)
    print("\n========== Confusion Matrix ==========")
    print(cm)

    print("\nSaving trained Logistic Regression model & preprocessors...")
    joblib.dump(lr_model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)

    print("Logistic Regression model trained and saved successfully!")

if __name__ == "__main__":
    main()
