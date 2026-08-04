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

DATASET_PATH = "dataset/intents.csv"
MODEL_PATH = "models/logistic_regression.pkl"
VECTORIZER_PATH = "models/lr_vectorizer.pkl"
ENCODER_PATH = "models/lr_label_encoder.pkl"

def main():
        
    # Load chatbot dataset
    df = pd.read_csv(DATASET_PATH)

    print("\n========== Dataset Preview ==========")
    print(df.head())

    print("\n========== Dataset Information ==========")
    print(df.info())

    print("\n========== Intent Distribution ==========")
    print(df["intent"].value_counts())

    # Convert text labels into numeric values
    X = df["text"]
    y = df["intent"]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Create Logistic Regression classifier
    lr_model = LogisticRegression(
        random_state=42,
        max_iter=1000
    )

    lr_model.fit(
        X_train,
        y_train
    )

    y_pred = lr_model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )
    print(f"\nAccuracy : {accuracy:.4f}")

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
    print(f"Precision : {precision:.4f}")

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
    print(f"Recall : {recall:.4f}")

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )
    print(f"F1 Score : {f1:.4f}")

    print("\n========== Classification Report ==========")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(f"\n========== Confusion Matrix ==========")
    print(cm)

    print("\nSaving trained model...")
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

    print("Model saved successfully!")

if __name__ == "__main__":
    main()
