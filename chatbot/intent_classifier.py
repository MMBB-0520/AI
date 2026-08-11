"""
BookMate Chatbot
Intent Prediction Module

Purpose:
Load trained ML models and predict user intent using
the BookMate NLP preprocessing pipeline.

Supported Models:

* Support Vector Machine
* Naive Bayes
* Logistic Regression
  """

import os
import sys
import joblib

# IMPORT PREPROCESSOR
# Ensure project root is available for imports

PROJECT_ROOT = os.path.abspath(
os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from chatbot.preprocessing import preprocess_text

# PATH CONFIGURATION
MODEL_DIR = os.path.join(
PROJECT_ROOT,
"models"
)

# INTENT PREDICTOR
class IntentPredictor:
    """
    Predict user intent using a trained ML model.
    
    The predictor does NOT train models.
    It only loads existing trained models and performs inference.
    """

    SUPPORTED_MODELS = {
        "Support Vector Machine": "svm",
        "Naive Bayes": "nb",
        "Logistic Regression": "lr"
    }

    def __init__(
        self,
        model_name: str = "Support Vector Machine",
        confidence_threshold: float = 0.55
    ):
        """
        Initialize IntentPredictor.

        Args:
            model_name:
                Name of the trained model to use.

            confidence_threshold:
                Minimum confidence required to accept
                the predicted intent.
        """

        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model name: {model_name}. "
                f"Supported models: "
                f"{list(self.SUPPORTED_MODELS.keys())}"
            )

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold

        model_prefix = self.SUPPORTED_MODELS[
            model_name
        ]

        # Build model file paths
        model_path = os.path.join(
            MODEL_DIR,
            f"{model_prefix}.pkl"
        )

        vectorizer_path = os.path.join(
            MODEL_DIR,
            f"{model_prefix}_vectorizer.pkl"
        )

        label_encoder_path = os.path.join(
            MODEL_DIR,
            f"{model_prefix}_label_encoder.pkl"
        )

        # Check files before loading
        required_files = {
            "model": model_path,
            "vectorizer": vectorizer_path,
            "label_encoder": label_encoder_path
        }

        missing_files = [
            path
            for path in required_files.values()
            if not os.path.exists(path)
        ]

        if missing_files:
            raise FileNotFoundError(
                "Required model files are missing:\n"
                + "\n".join(missing_files)
            )

        # Load trained components
        self.model = joblib.load(
            model_path
        )

        self.vectorizer = joblib.load(
            vectorizer_path
        )

        self.label_encoder = joblib.load(
            label_encoder_path
        )

    # PREPROCESS
    def _prepare_text(
        self,
        user_message: str,
        preprocessed_text: str | None = None
    ) -> str:
        """
        Prepare input text.

        If preprocessed_text is provided, use it directly.
        Otherwise run the standard preprocessing pipeline.
        """

        if preprocessed_text is not None:
            return preprocessed_text

        if not user_message or not user_message.strip():
            return ""

        return preprocess_text(
            user_message
        )

    # CONFIDENCE
    def _get_confidence(
        self,
        text_vector
    ) -> float:
        """
        Calculate prediction confidence.

        Priority:
        1. predict_proba()
        2. decision_function()
        3. fallback 0.0

        Note:
        decision_function scores are not true probabilities.
        """

        # Models supporting probability prediction
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

        # SVM models may expose decision_function()
        if hasattr(
            self.model,
            "decision_function"
        ):
            scores = self.model.decision_function(
                text_vector
            )

            # Binary classification
            if getattr(
                scores,
                "ndim",
                1
            ) == 1:

                # Convert decision score into
                # a rough 0-1 confidence.
                import numpy as np

                confidence = 1 / (
                    1 + np.exp(-abs(float(scores[0])))
                )

                return float(confidence)

            # Multiclass classification
            import numpy as np

            scores = np.asarray(
                scores[0]
            )

            # Softmax-like normalization
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

        # Unknown confidence
        return 0.0

    # PREDICT
    def predict(
        self,
        user_message: str,
        preprocessed_text: str | None = None
    ) -> dict:
        """
        Predict user intent.

        Returns:

        {
            "intent": "book_room",
            "confidence": 0.87,
            "status": "confident",
            "cleaned_text": "want book deluxe room"
        }
        """

        cleaned_text = self._prepare_text(
            user_message,
            preprocessed_text
        )

        # Empty input
        if not cleaned_text:

            return {
                "intent": "unknown",
                "confidence": 0.0,
                "status": "empty_input",
                "cleaned_text": ""
            }

        # Vectorize
        text_vector = self.vectorizer.transform(
            [cleaned_text]
        )

        # Predict encoded intent
        prediction = self.model.predict(
            text_vector
        )

        # Convert encoded label back to intent name
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
        Return the top predicted intents.

        Mainly useful for debugging and model evaluation.
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

        text_vector = self.vectorizer.transform(
            [cleaned_text]
        )

        # Probability-based models
        if hasattr(
            self.model,
            "predict_proba"
        ):

            probabilities = self.model.predict_proba(
                text_vector
            )[0]

            classes = self.model.classes_

            ranked_indices = probabilities.argsort()[
                ::-1
            ][:top_k]

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

            # For SVM models without probability support,
            # return the main prediction only.
            prediction = self.model.predict(
                text_vector
            )

            intent = self.label_encoder.inverse_transform(
                prediction
            )[0]

            confidence = self._get_confidence(
                text_vector
            )

            top_predictions = [{
                "intent": intent,
                "confidence": confidence
            }]

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

# CONVENIENCE FUNCTION
_default_predictor = None


def predict_intent(
    text: str,
    model_name: str = "Support Vector Machine"
) -> dict:
    """
    Convenience function for intent prediction.
    """

    global _default_predictor

    if (
        _default_predictor is None
        or _default_predictor.model_name != model_name
    ):
        _default_predictor = IntentPredictor(
            model_name=model_name
        )

    return _default_predictor.predict(
        text
    )


# TEST
if __name__ == "__main__":
    print(
        "=== BookMate Intent Predictor Test ==="
    )

    test_queries = [
        "I want to book a deluxe room for 2 nights please",
        "What time is check in and check out?",
        "Do you have free parking available?",
        "How much is a single room per night?",
        "Can I cancel my reservation?",
        "Where is the resort located?",
        "Do you have breakfast?",
        "Thank you goodbye"
    ]

    model_names = [
        "Support Vector Machine",
        "Naive Bayes",
        "Logistic Regression"
    ]

    for model_name in model_names:

        print(
            f"\n{'=' * 60}"
        )

        print(
            f"Model: {model_name}"
        )

        print(
            f"{'=' * 60}"
        )

        try:

            predictor = IntentPredictor(
                model_name=model_name
            )

            for query in test_queries:

                result = predictor.predict(
                    query
                )

                print(
                    f"\nQuery: {query}"
                )

                print(
                    f"Intent: "
                    f"{result['intent']}"
                )

                print(
                    f"Confidence: "
                    f"{result['confidence']:.2%}"
                )

                print(
                    f"Status: "
                    f"{result['status']}"
                )

                print(
                    f"Cleaned: "
                    f"{result['cleaned_text']}"
                )

        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            import traceback
            traceback.print_exc()

