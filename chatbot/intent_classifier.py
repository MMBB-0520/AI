"""
BookMate Hotel Booking Chatbot
================================

Intent Classification Module

Purpose:
    Load the trained SVM intent classification model and
    predict the user's intent.

Machine Learning Pipeline:
    User Input
        ↓
    NLP Preprocessing
        ↓
    TF-IDF Vectorization
        ↓
    SVM Classifier
        ↓
    Intent Prediction
        ↓
    Confidence Check

Supported Dataset:
    Bitext Hospitality LLM Chatbot Training Dataset

Number of intents:
    25
"""

import os
import sys
import joblib

# PROJECT PATH
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# IMPORT PREPROCESSING
from chatbot.preprocessing import preprocess_text

# MODEL DIRECTORY
MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "models"
)

# INTENT PREDICTOR
class IntentPredictor:
    """
    Predict user intent using the trained Machine Learning model (SVM / Logistic Regression).

    This class performs inference only.
    It does not train the model.

    Components:
        - ML classifier
        - TF-IDF vectorizer
        - Label encoder
        - NLP preprocessing pipeline
    """

    def __init__(
        self,
        model_name: str = "Support Vector Machine",
        confidence_threshold: float = 0.50
    ):
        """
        Initialize the intent predictor.

        Args:
            model_name:
                Name of the trained model to use ("Support Vector Machine", "Logistic Regression", or "Naive Bayes").
            confidence_threshold:
                Minimum confidence required for accepting
                the predicted intent.
        """

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        if "logistic" in model_name.lower():
            model_file = os.path.join(MODEL_DIR, "logistic_regression.pkl")
            vectorizer_file = os.path.join(MODEL_DIR, "lr_vectorizer.pkl")
            encoder_file = os.path.join(MODEL_DIR, "lr_label_encoder.pkl")
        elif "naive" in model_name.lower():
            model_file = os.path.join(MODEL_DIR, "naive_bayes.pkl")
            vectorizer_file = os.path.join(MODEL_DIR, "nb_vectorizer.pkl")
            encoder_file = os.path.join(MODEL_DIR, "nb_label_encoder.pkl")
        else:
            model_file = os.path.join(MODEL_DIR, "svm.pkl")
            vectorizer_file = os.path.join(MODEL_DIR, "svm_vectorizer.pkl")
            encoder_file = os.path.join(MODEL_DIR, "svm_label_encoder.pkl")

        # Check required model files
        required_files = {
            f"{model_name} model": model_file,
            "TF-IDF vectorizer": vectorizer_file,
            "Label encoder": encoder_file
        }

        missing_files = [
            f"{name}: {path}"
            for name, path in required_files.items()
            if not os.path.exists(path)
        ]

        if missing_files:
            raise FileNotFoundError(
                "Required trained model files are missing:\n"
                + "\n".join(missing_files)
            )

        # Load trained components
        print(f"Loading {model_name} intent classification model...")

        self.model = joblib.load(
            model_file
        )

        self.vectorizer = joblib.load(
            vectorizer_file
        )

        self.label_encoder = joblib.load(
            encoder_file
        )

        print(f"{model_name} loaded successfully.")

        # Display model information
        self.intents = list(
            self.label_encoder.classes_
        )

    # TEXT PREPARATION
    def _prepare_text(
        self,
        user_message: str,
        preprocessed_text: str | None = None
    ) -> str:
        """
        Prepare user input for the trained SVM model.

        If preprocessed_text is supplied, it is used directly.
        Otherwise, the standard preprocessing pipeline is run.
        """

        if preprocessed_text is not None:
            return preprocessed_text

        if not user_message:
            return ""

        if not user_message.strip():
            return ""

        return preprocess_text(
            user_message
        )

    # CONFIDENCE CALCULATION
    def _get_confidence(
        self,
        text_vector
    ) -> float:
        """
        Calculate the confidence score of the SVM prediction.

        The SVM was trained with probability=True, therefore
        predict_proba() is used when available.

        Note:
            This value is treated as a confidence score,
            not as a guaranteed real-world probability.
        """

        # Preferred method: predict_proba()
        if hasattr(
            self.model,
            "predict_proba"
        ):
            probabilities = self.model.predict_proba(
                text_vector
            )

            return float(
                probabilities.max()
            )

        # Fallback: decision_function()
        if hasattr(
            self.model,
            "decision_function"
        ):
            import numpy as np

            scores = self.model.decision_function(
                text_vector
            )

            scores = np.asarray(scores)

            # Binary classification
            if scores.ndim == 1:
                score = abs(float(scores[0]))

                confidence = 1.0 / (
                    1.0 + np.exp(-score)
                )

                return float(confidence)

            # Multiclass classification
            scores = scores[0]

            exp_scores = np.exp(
                scores - np.max(scores)
            )

            probabilities = (
                exp_scores /
                exp_scores.sum()
            )

            return float(
                probabilities.max()
            )

        return 0.0

    # MAIN PREDICTION
    def predict(
        self,
        user_message: str,
        preprocessed_text: str | None = None
    ) -> dict:
        """
        Predict the user's intent.

        Returns:
            {
                "intent": "book_hotel",
                "predicted_intent": "book_hotel",
                "confidence": 0.95,
                "status": "confident",
                "cleaned_text": "want book hotel"
            }
        """

        # Prepare input
        cleaned_text = self._prepare_text(
            user_message,
            preprocessed_text
        )

        # Empty input
        if not cleaned_text:

            return {
                "intent": "unknown",
                "predicted_intent": None,
                "confidence": 0.0,
                "status": "empty_input",
                "cleaned_text": ""
            }

        # TF-IDF transformation
        text_vector = self.vectorizer.transform(
            [cleaned_text]
        )

        # SVM prediction
        prediction = self.model.predict(
            text_vector
        )

        # Convert numerical label → intent name
        intent = self.label_encoder.inverse_transform(
            prediction
        )[0]

        # Confidence
        confidence = self._get_confidence(
            text_vector
        )

        # Confidence threshold
        if confidence < self.confidence_threshold:

            return {
                "intent": "unknown",
                "predicted_intent": intent,
                "confidence": confidence,
                "status": "low_confidence",
                "cleaned_text": cleaned_text
            }

        # Confident prediction
        return {
            "intent": intent,
            "predicted_intent": intent,
            "confidence": confidence,
            "status": "confident",
            "cleaned_text": cleaned_text
        }

    # TOP PREDICTIONS
    def predict_top(
        self,
        user_message: str,
        preprocessed_text: str | None = None,
        top_k: int = 3
    ) -> dict:
        """
        Return the top-k predicted intents.

        Useful for debugging and model analysis.
        """

        cleaned_text = self._prepare_text(
            user_message,
            preprocessed_text
        )

        if not cleaned_text:

            return {
                "intent": "unknown",
                "confidence": 0.0,
                "top_predictions": [],
                "cleaned_text": ""
            }

        # Vectorize
        text_vector = self.vectorizer.transform(
            [cleaned_text]
        )

        # Probability-based prediction
        if hasattr(
            self.model,
            "predict_proba"
        ):

            probabilities = self.model.predict_proba(
                text_vector
            )[0]

            classes = self.model.classes_

            ranked_indices = probabilities.argsort()[::-1]

            ranked_indices = ranked_indices[:top_k]

            top_predictions = []

            for index in ranked_indices:

                encoded_label = [
                    classes[index]
                ]

                intent = self.label_encoder.inverse_transform(
                    encoded_label
                )[0]

                top_predictions.append({
                    "intent": intent,
                    "confidence": float(
                        probabilities[index]
                    )
                })

        else:

            prediction = self.model.predict(
                text_vector
            )

            intent = self.label_encoder.inverse_transform(
                prediction
            )[0]

            confidence = self._get_confidence(
                text_vector
            )

            top_predictions = [
                {
                    "intent": intent,
                    "confidence": confidence
                }
            ]

        # Final intent
        best = top_predictions[0]

        if (
            best["confidence"]
            < self.confidence_threshold
        ):
            final_intent = "unknown"
        else:
            final_intent = best["intent"]

        return {
            "intent": final_intent,
            "confidence": best["confidence"],
            "top_predictions": top_predictions,
            "cleaned_text": cleaned_text
        }


# DEFAULT PREDICTOR
_default_predictor = None


def predict_intent(
    text: str
) -> dict:
    """
    Convenience function for intent prediction.

    Example:
        result = predict_intent(
            "I want to book a hotel"
        )
    """

    global _default_predictor

    if _default_predictor is None:

        _default_predictor = IntentPredictor()

    return _default_predictor.predict(
        text
    )

# TEST
if __name__ == "__main__":

    print()
    print("=" * 60)
    print("BookMate SVM Intent Predictor Test")
    print("=" * 60)

    test_queries = [

        "I want to book a hotel for two nights",

        "Can I cancel my hotel reservation?",

        "I need to change my reservation",

        "What time is check in?",

        "What time is check out?",

        "How much does a hotel room cost?",

        "Do you have free parking?",

        "Can I bring my pet?",

        "Do you have a swimming pool?",

        "Where can I find my booking?",

        "I want to complain about my stay",

        "I need to speak to a human",

        "Do you provide shuttle service?",

        "Can I store my luggage?",

        "Where can I find my invoices?"
    ]

    try:

        predictor = IntentPredictor()

        print()
        print("Available intents:")
        print("-" * 60)

        for intent in predictor.intents:
            print(f"  - {intent}")

        print()
        print("=" * 60)
        print("Prediction Tests")
        print("=" * 60)

        for query in test_queries:

            result = predictor.predict(
                query
            )

            print()
            print(f"User       : {query}")
            print(
                f"Intent     : {result['intent']}"
            )
            print(
                f"Confidence : "
                f"{result['confidence']:.2%}"
            )
            print(
                f"Status     : {result['status']}"
            )
            print(
                f"Cleaned    : "
                f"{result['cleaned_text']}"
            )

            # Show top predictions
            top_result = predictor.predict_top(
                query,
                top_k=3
            )

            print("Top predictions:")

            for item in top_result["top_predictions"]:

                print(
                    f"   {item['intent']}: "
                    f"{item['confidence']:.2%}"
                )

    except Exception as e:

        print()
        print("ERROR:")
        print(e)

        import traceback
        traceback.print_exc()