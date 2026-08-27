"""
train_lr.py
-----------
Trains a Logistic Regression (LR) classifier to recognize user intents
for the BookMate Hotel Booking Chatbot.

Dataset:
Bitext Hospitality LLM Chatbot Training Dataset

Input:
    instruction

Target:
    intent
"""

import sys
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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
    classification_report,
    ConfusionMatrixDisplay
)


# PROJECT PATH
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


from chatbot.preprocessing import preprocess_text


# PATH CONFIGURATION
DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "dataset",
    "bitext-hospitality-llm-chatbot-training-dataset.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "logistic_regression.pkl"
)

VECTORIZER_PATH = os.path.join(
    MODEL_DIR,
    "lr_vectorizer.pkl"
)

ENCODER_PATH = os.path.join(
    MODEL_DIR,
    "lr_label_encoder.pkl"
)

CM_PATH = os.path.join(
    MODEL_DIR,
    "lr_confusion_matrix.png"
)


# MAIN
def main():

    print("=" * 60)
    print("Loading Bitext Hospitality Dataset for Logistic Regression")
    print("=" * 60)

    # Check dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    # Load dataset
    df = pd.read_csv(
        DATASET_PATH
    )

    print(f"Dataset path : {DATASET_PATH}")
    print(f"Total rows   : {len(df)}")
    print(f"Columns      : {list(df.columns)}")

    # Validate required columns
    required_columns = [
        "instruction",
        "intent"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # Remove rows with missing training data
    df = df.dropna(
        subset=["instruction", "intent"]
    ).copy()

    print(f"Rows after cleaning: {len(df)}")
    print(f"Number of intents: {df['intent'].nunique()}")

    print("\nIntent distribution:")
    print(
        df["intent"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    # PREPARE DATA
    print("\n" + "=" * 60)
    print("Preparing Training Data")
    print("=" * 60)

    X_raw = df["instruction"].astype(str)
    y_raw = df["intent"].astype(str)

    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(
        y_raw
    )

    print(f"Number of classes: {len(label_encoder.classes_)}")

    # TRAIN / TEST SPLIT
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"\nTraining samples: {len(X_train_raw)}")
    print(f"Testing samples : {len(X_test_raw)}")

    # NLP PREPROCESSING
    print("\n" + "=" * 60)
    print("Applying NLP Preprocessing")
    print("=" * 60)

    print("Processing training samples...")
    X_train_cleaned = X_train_raw.apply(
        preprocess_text
    )

    print("Processing testing samples...")
    X_test_cleaned = X_test_raw.apply(
        preprocess_text
    )

    print("NLP preprocessing completed.")

    # TF-IDF
    print("\n" + "=" * 60)
    print("Building TF-IDF Features")
    print("=" * 60)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    # Fit ONLY on training data
    X_train = vectorizer.fit_transform(
        X_train_cleaned
    )

    # Test data is only transformed
    X_test = vectorizer.transform(
        X_test_cleaned
    )

    print(f"TF-IDF training shape: {X_train.shape}")
    print(f"TF-IDF testing shape : {X_test.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # TRAIN LOGISTIC REGRESSION
    print("\n" + "=" * 60)
    print("Training Logistic Regression Model")
    print("=" * 60)

    lr_model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        random_state=42
    )

    lr_model.fit(
        X_train,
        y_train
    )

    print("Logistic Regression training completed.")

    # PREDICTION
    print("\nGenerating predictions...")
    y_pred = lr_model.predict(
        X_test
    )

    # EVALUATION
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0
    )

    print("\n" + "=" * 60)
    print("Logistic Regression Evaluation Results")
    print("=" * 60)

    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # CLASSIFICATION REPORT
    print("\n" + "=" * 60)
    print("Classification Report")
    print("=" * 60)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    # CONFUSION MATRIX
    print("Generating confusion matrix...")
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

    plt.title("Logistic Regression Intent Classification - Confusion Matrix")
    plt.tight_layout()

    plt.savefig(
        CM_PATH,
        dpi=200
    )

    plt.close()

    print(f"Confusion matrix saved to:\n{CM_PATH}")

    # SAVE MODEL
    print("\n" + "=" * 60)
    print("Saving Logistic Regression Model")
    print("=" * 60)

    joblib.dump(
        lr_model,
        MODEL_PATH
    )

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )

    joblib.dump(
        label_encoder,
        ENCODER_PATH
    )

    print(f"Model saved       : {MODEL_PATH}")
    print(f"Vectorizer saved  : {VECTORIZER_PATH}")
    print(f"Label encoder     : {ENCODER_PATH}")

    print("\nLogistic Regression training completed successfully!")


if __name__ == "__main__":
    main()
