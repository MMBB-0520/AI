"""
BookMate Chatbot
Intent Prediction Module (SVM / Naive Bayes / Logistic Regression)

Purpose:
Load trained ML models and predict user intent using the NLP preprocessing pipeline.
"""

import sys
import os
import joblib

# Ensure chatbot package can be imported regardless of working directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chatbot.preprocessing import preprocess_text


class IntentPredictor:
    """
    Predict user intent using trained Machine Learning models (SVM, Naive Bayes, Logistic Regression).
    """

    def __init__(self, model_name: str = "Support Vector Machine"):
        """Load trained model, vectorizer, and label encoder."""
        self.model_name = model_name

        if model_name == "Support Vector Machine":
            self.model = joblib.load("models/svm.pkl")
            self.vectorizer = joblib.load("models/svm_vectorizer.pkl")
            self.label_encoder = joblib.load("models/svm_label_encoder.pkl")

        elif model_name == "Naive Bayes":
            self.model = joblib.load("models/nb.pkl")
            self.vectorizer = joblib.load("models/nb_vectorizer.pkl")
            self.label_encoder = joblib.load("models/nb_label_encoder.pkl")

        elif model_name == "Logistic Regression":
            self.model = joblib.load("models/lr.pkl")
            self.vectorizer = joblib.load("models/lr_vectorizer.pkl")
            self.label_encoder = joblib.load("models/lr_label_encoder.pkl")

        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def predict(self, user_message: str, preprocessed_text: str = None) -> dict:
        """
        Predict intent of user message.

        Args:
            user_message (str): Raw user input.
            preprocessed_text (str, optional): Pre-cleaned/lemmatized text to avoid duplicate preprocessing.

        Returns:
            dict: {"intent": str, "confidence": float, "cleaned_text": str}
        """
        # Use provided preprocessed text if available, otherwise apply NLP preprocessing pipeline
        if preprocessed_text is not None:
            cleaned_text = preprocessed_text
        else:
            cleaned_text = preprocess_text(user_message)

        # Convert text into TF-IDF / Count vector
        text_vector = self.vectorizer.transform([cleaned_text])

        # Predict intent label index
        prediction = self.model.predict(text_vector)

        # Predict confidence score
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(text_vector)
            confidence = float(probabilities.max())
        else:
            confidence = 1.0

        # Convert numeric label back to string tag
        intent = self.label_encoder.inverse_transform(prediction)[0]

        return {
            "intent": intent,
            "confidence": confidence,
            "cleaned_text": cleaned_text
        }


# Test predictor
if __name__ == "__main__":
    print("=== Testing IntentPredictor with Preprocessing ===")
    test_queries = [
        "I want to book a deluxe room for 2 nights please",
        "What time is check in and check out?",
        "Do you have free parking available?",
        "How much is a single room per night?"
    ]
    for m_name in ["Support Vector Machine", "Naive Bayes", "Logistic Regression"]:
        predictor = IntentPredictor(m_name)
        print(f"\n--- Model: {m_name} ---")
        for q in test_queries:
            res = predictor.predict(q)
            print(f"Query: '{q}' -> Intent: {res['intent']} (Confidence: {res['confidence']:.2%})")