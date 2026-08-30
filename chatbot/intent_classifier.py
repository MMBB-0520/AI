"""
BookMate Hotel Booking Chatbot
================================

Intent Classification Module

Purpose:
    Load the trained SVM / Logistic Regression / Naive Bayes intent classification model and
    predict the user's intent with confidence estimation, margin ambiguity detection,
    short-keyword boosting, and batch inference.

Machine Learning Pipeline:
    User Input
        ↓
    NLP Preprocessing
        ↓
    TF-IDF Vectorization
        ↓
    Classifier (SVM / LR / NB)
        ↓
    Confidence & Margin Calculation
        ↓
    Short Keyword Protection
        ↓
    Intent Prediction
"""

import os
import sys
import joblib
import numpy as np

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

# High-frequency domain keyword boosts for short sparse inputs
SHORT_KEYWORD_BOOSTS = {
    "wifi": "check_hotel_facilities",
    "wi-fi": "check_hotel_facilities",
    "internet": "check_hotel_facilities",
    "pool": "check_hotel_facilities",
    "swimming": "check_hotel_facilities",
    "gym": "check_hotel_facilities",
    "fitness": "check_hotel_facilities",
    "spa": "check_hotel_facilities",
    "massage": "check_hotel_facilities",
    "park": "book_parking_space",
    "parking": "book_parking_space",
    "car park": "book_parking_space",
    "breakfast": "check_menu",
    "menu": "check_menu",
    "food": "check_menu",
    "dinner": "check_menu",
    "lunch": "check_menu",
    "price": "check_hotel_prices",
    "prices": "check_hotel_prices",
    "rate": "check_hotel_prices",
    "rates": "check_hotel_prices",
    "cost": "check_hotel_prices",
    "pet": "bring_pets",
    "pets": "bring_pets",
    "dog": "bring_pets",
    "cat": "bring_pets",
    "cancel": "cancel_hotel_reservation",
    "cancellation": "cancellation_fees",
    "book": "book_hotel",
    "booking": "book_hotel",
    "reserve": "book_hotel",
    "reservation": "book_hotel",
    "modify": "change_hotel_reservation",
    "change": "change_hotel_reservation",
    "checkin": "check_in",
    "check-in": "check_in",
    "checkout": "check_out",
    "check-out": "check_out",
    "status": "check_hotel_reservation",
    "invoice": "invoices",
    "invoices": "invoices",
    "receipt": "invoices",
    "bill": "invoices",
    "shuttle": "shuttle_service",
    "transport": "shuttle_service",
    "luggage": "store_luggage",
    "baggage": "store_luggage",
    "human": "human_agent",
    "agent": "human_agent",
    "complain": "file_complaint",
    "complaint": "file_complaint",
    "refund": "get_refund"
}


class IntentPredictor:
    """
    Predict user intent using the trained Machine Learning model (SVM / Logistic Regression / Naive Bayes).

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
                Minimum confidence required for accepting the predicted intent.
        """
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.intents = []

        self._load_model_files(model_name)

    def _load_model_files(self, model_name: str):
        """Load model, vectorizer, and label encoder from disk."""
        self.model_name = model_name

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

        print(f"Loading {model_name} intent classification model...")
        self.model = joblib.load(model_file)
        self.vectorizer = joblib.load(vectorizer_file)
        self.label_encoder = joblib.load(encoder_file)
        self.intents = list(self.label_encoder.classes_)
        print(f"{model_name} loaded successfully.")

    def switch_model(self, new_model_name: str):
        """
        Dynamically switch active ML model (e.g. 'Support Vector Machine', 'Logistic Regression', 'Naive Bayes').
        """
        if new_model_name == self.model_name:
            return
        self._load_model_files(new_model_name)

    def get_model_info(self) -> dict:
        """
        Return model metadata, feature count, and active threshold.
        """
        vocab_size = len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, "vocabulary_") else 0
        return {
            "model_name": self.model_name,
            "algorithm": type(self.model).__name__,
            "total_intents": len(self.intents),
            "intents": self.intents,
            "vocabulary_size": vocab_size,
            "ngram_range": getattr(self.vectorizer, "ngram_range", (1, 2)),
            "confidence_threshold": self.confidence_threshold
        }

    def _prepare_text(
        self,
        user_message: str,
        preprocessed_text: str | None = None
    ) -> str:
        """
        Prepare user input for the trained model.
        """
        if preprocessed_text is not None:
            return preprocessed_text
        if not user_message or not user_message.strip():
            return ""
        return preprocess_text(user_message)

    def _get_probabilities(self, text_vector) -> np.ndarray:
        """
        Calculate class probabilities for a given vector or batch of vectors.
        """
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(text_vector)

        if hasattr(self.model, "decision_function"):
            scores = self.model.decision_function(text_vector)
            scores = np.asarray(scores)
            if scores.ndim == 1:
                prob = 1.0 / (1.0 + np.exp(-abs(scores)))
                return np.column_stack([1.0 - prob, prob])

            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / exp_scores.sum(axis=1, keepdims=True)

        return np.zeros((text_vector.shape[0], len(self.intents)))

    def predict(
        self,
        user_message: str,
        preprocessed_text: str | None = None
    ) -> dict:
        """
        Predict the user's intent with confidence, margin, and short-keyword boost.
        """
        cleaned_text = self._prepare_text(user_message, preprocessed_text)

        if not cleaned_text:
            return {
                "intent": "unknown",
                "predicted_intent": None,
                "confidence": 0.0,
                "margin": 0.0,
                "is_ambiguous": True,
                "status": "empty_input",
                "cleaned_text": ""
            }

        text_vector = self.vectorizer.transform([cleaned_text])
        probabilities = self._get_probabilities(text_vector)[0]
        classes = self.model.classes_

        # Rank indices
        ranked_indices = probabilities.argsort()[::-1]
        top1_idx = ranked_indices[0]
        top2_idx = ranked_indices[1] if len(ranked_indices) > 1 else top1_idx

        top1_intent = self.label_encoder.inverse_transform([classes[top1_idx]])[0]
        top1_conf = float(probabilities[top1_idx])
        top2_conf = float(probabilities[top2_idx]) if len(ranked_indices) > 1 else 0.0

        margin = top1_conf - top2_conf

        # Short text keyword boost protection
        raw_clean = user_message.strip().lower()
        if raw_clean in SHORT_KEYWORD_BOOSTS:
            boost_intent = SHORT_KEYWORD_BOOSTS[raw_clean]
            top1_intent = boost_intent
            top1_conf = max(0.95, top1_conf)

        # 1. Zero-feature protection (No vocabulary features matched in input)
        elif text_vector.nnz == 0:
            return {
                "intent": "unknown",
                "predicted_intent": None,
                "confidence": 0.0,
                "margin": 0.0,
                "is_ambiguous": True,
                "status": "zero_features",
                "cleaned_text": cleaned_text
            }

        # 2. Informative Domain Feature Check
        # If all matched n-grams consist entirely of generic functional stop-words without any domain terms
        else:
            functional_words = {
                "what", "who", "where", "when", "why", "how", "which",
                "be", "is", "are", "am", "was", "were", "been", "being",
                "you", "your", "yours", "i", "me", "my", "mine", "we", "our",
                "it", "its", "they", "them", "their", "do", "does", "did",
                "to", "the", "a", "an", "of", "for", "in", "on", "at", "by", "with",
                "good", "fine", "hello", "hi", "hey"
            }
            matched_features = [self.vectorizer.get_feature_names_out()[i] for i in text_vector.indices]
            has_domain_word = any(
                not all(w in functional_words for w in f.split())
                for f in matched_features
            )
            if not has_domain_word:
                return {
                    "intent": "unknown",
                    "predicted_intent": top1_intent,
                    "confidence": 0.10,
                    "margin": 0.0,
                    "is_ambiguous": False,
                    "status": "non_domain_sparse",
                    "cleaned_text": cleaned_text
                }

        is_ambiguous = (margin < 0.15)

        if top1_conf < self.confidence_threshold:
            status = "low_confidence"
            final_intent = "unknown"
        else:
            status = "confident"
            final_intent = top1_intent

        return {
            "intent": final_intent,
            "predicted_intent": top1_intent,
            "confidence": top1_conf,
            "margin": margin,
            "is_ambiguous": is_ambiguous,
            "status": status,
            "cleaned_text": cleaned_text
        }

    def predict_top(
        self,
        user_message: str,
        preprocessed_text: str | None = None,
        top_k: int = 3
    ) -> dict:
        """
        Return the top-k predicted intents with confidence margin.
        """
        cleaned_text = self._prepare_text(user_message, preprocessed_text)

        if not cleaned_text:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "margin": 0.0,
                "is_ambiguous": True,
                "top_predictions": [],
                "cleaned_text": ""
            }

        text_vector = self.vectorizer.transform([cleaned_text])
        probabilities = self._get_probabilities(text_vector)[0]
        classes = self.model.classes_

        ranked_indices = probabilities.argsort()[::-1][:top_k]
        top_predictions = []

        for index in ranked_indices:
            intent = self.label_encoder.inverse_transform([classes[index]])[0]
            top_predictions.append({
                "intent": intent,
                "confidence": float(probabilities[index])
            })

        best = top_predictions[0]
        second_best_conf = top_predictions[1]["confidence"] if len(top_predictions) > 1 else 0.0
        margin = best["confidence"] - second_best_conf
        is_ambiguous = (margin < 0.15)

        # Short text boost
        raw_clean = user_message.strip().lower()
        if raw_clean in SHORT_KEYWORD_BOOSTS:
            boost_intent = SHORT_KEYWORD_BOOSTS[raw_clean]
            best["intent"] = boost_intent
            best["confidence"] = max(0.95, best["confidence"])

        if best["confidence"] < self.confidence_threshold:
            final_intent = "unknown"
        else:
            final_intent = best["intent"]

        return {
            "intent": final_intent,
            "confidence": best["confidence"],
            "margin": margin,
            "is_ambiguous": is_ambiguous,
            "top_predictions": top_predictions,
            "cleaned_text": cleaned_text
        }

    def predict_batch(self, messages: list[str]) -> list[dict]:
        """
        Perform high-throughput matrix batch prediction for a list of messages.
        """
        if not messages:
            return []

        cleaned_texts = [self._prepare_text(m) for m in messages]
        valid_indices = [i for i, t in enumerate(cleaned_texts) if t]
        results = [None] * len(messages)

        for i, t in enumerate(cleaned_texts):
            if not t:
                results[i] = {
                    "intent": "unknown",
                    "predicted_intent": None,
                    "confidence": 0.0,
                    "margin": 0.0,
                    "is_ambiguous": True,
                    "status": "empty_input",
                    "cleaned_text": ""
                }

        if valid_indices:
            valid_cleaned = [cleaned_texts[i] for i in valid_indices]
            vectors = self.vectorizer.transform(valid_cleaned)
            probs_matrix = self._get_probabilities(vectors)
            classes = self.model.classes_

            for row_idx, original_idx in enumerate(valid_indices):
                probs = probs_matrix[row_idx]
                ranked = probs.argsort()[::-1]
                top1_idx = ranked[0]
                top2_idx = ranked[1] if len(ranked) > 1 else top1_idx

                top1_intent = self.label_encoder.inverse_transform([classes[top1_idx]])[0]
                top1_conf = float(probs[top1_idx])
                top2_conf = float(probs[top2_idx]) if len(ranked) > 1 else 0.0
                margin = top1_conf - top2_conf

                raw = messages[original_idx].strip().lower()
                if raw in SHORT_KEYWORD_BOOSTS:
                    boost_intent = SHORT_KEYWORD_BOOSTS[raw]
                    top1_intent = boost_intent
                    top1_conf = max(0.95, top1_conf)

                is_ambiguous = (margin < 0.15)
                final_intent = "unknown" if top1_conf < self.confidence_threshold else top1_intent
                status = "low_confidence" if final_intent == "unknown" else "confident"

                results[original_idx] = {
                    "intent": final_intent,
                    "predicted_intent": top1_intent,
                    "confidence": top1_conf,
                    "margin": margin,
                    "is_ambiguous": is_ambiguous,
                    "status": status,
                    "cleaned_text": cleaned_texts[original_idx]
                }

        return results


# DEFAULT PREDICTOR
_default_predictor = None


def predict_intent(text: str) -> dict:
    """
    Convenience function for intent prediction.
    """
    global _default_predictor

    if _default_predictor is None:
        _default_predictor = IntentPredictor()

    return _default_predictor.predict(text)
