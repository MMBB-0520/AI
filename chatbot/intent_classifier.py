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
import re
import sys
import json
from datetime import datetime
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
            model_file = os.path.join(MODEL_DIR, "nb.pkl") if os.path.exists(os.path.join(MODEL_DIR, "nb.pkl")) else os.path.join(MODEL_DIR, "naive_bayes.pkl")
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

    def _match_keyword_boost(self, message: str) -> str | None:
        """Check if message or stripped message matches a domain keyword boost."""
        if not message:
            return None

        raw_clean = message.strip().lower()
        clean_no_id = re.sub(r"\b#?bk[-\s_]?\d+\b|\b(?:\+?60|0)1\d{7,9}\b|\b\d{8,12}\b", "", raw_clean).strip()
        clean_no_id = re.sub(r"\s+", " ", clean_no_id).strip(" .,:;!?")

        if any(clean_no_id.startswith(p) for p in ["check booking", "check my booking", "check reservation", "check my reservation", "view booking", "view my booking", "booking status"]):
            return "check_hotel_reservation"
        if any(clean_no_id.startswith(p) for p in ["get invoice", "check invoice", "my invoice", "view invoice", "official invoice", "tax invoice"]):
            return "invoices"
        if any(clean_no_id.startswith(p) for p in ["cancel booking", "cancel reservation", "cancel my booking", "cancel my reservation"]):
            return "cancel_hotel_reservation"

        # Domain term regex boosts
        if re.search(r"\b(?:breakfast|dining|menu|buffet|restaurant|food|dinner|lunch)\b", clean_no_id):
            return "check_menu"
        if re.search(r"\b(?:wifi|wi-fi|swimming pool|pool|fitness gym|gym|spa|massage)\b", clean_no_id):
            return "check_hotel_facilities"
        if re.search(r"\b(?:pet|pets|dog|dogs|cat|cats)\b", clean_no_id):
            return "bring_pets"
        if re.search(r"\b(?:parking|car park|valet)\b", clean_no_id):
            return "book_parking_space"
        if re.search(r"\b(?:shuttle|airport transfer)\b", clean_no_id):
            return "shuttle_service"
        if re.search(r"\b(?:luggage|baggage)\b", clean_no_id):
            return "store_luggage"
        if re.search(r"\b(?:price|prices|rate|rates|room rate|room rates|cost)\b", clean_no_id):
            return "check_hotel_prices"

        return None

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
        boost_match = self._match_keyword_boost(user_message)
        if boost_match:
            top1_intent = boost_match
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
        boost_match = self._match_keyword_boost(user_message)
        if boost_match:
            best["intent"] = boost_match
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

                boost_match = self._match_keyword_boost(messages[original_idx])
                if boost_match:
                    top1_intent = boost_match
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
                    "cleaned_text": valid_cleaned[row_idx]
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
