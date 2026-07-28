"""
BookMate Chatbot
Intent Prediction using Support Vector Machine (SVM)

Purpose:
Load the trained SVM model and predict the user's intent.
"""

import joblib

from chatbot.response import get_response


class IntentPredictor:
    """
    Predict user intent using the trained SVM model.
    """

    def __init__(self, model_name):
        """Load trained model and preprocessing tools."""

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

    def predict(self, user_message):
        """
        Predict the intent of a user message.

        Args:
            user_message (str): User input.

        Returns:
            str: Predicted intent.
        """

        # Convert text into TF-IDF vector
        text_vector = self.vectorizer.transform([user_message])

        # Predict intent
        prediction = self.model.predict(text_vector)
        
        # Predict confidence
        probabilities = self.model.predict_proba(text_vector)
        confidence = probabilities.max()
        
        # Convert numeric label back to text
        intent = self.label_encoder.inverse_transform(prediction)

        return {
            "intent": intent[0],
            "confidence": confidence
        }


# Test predictor
if __name__ == "__main__":

    predictor = IntentPredictor()

    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        predicted_intent = predictor.predict(user_input)

        response = get_response(predicted_intent)

        print("\nBot:", response)