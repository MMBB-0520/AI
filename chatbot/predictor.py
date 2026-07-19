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

    def __init__(self):
        """Load trained model and preprocessing tools."""

        self.model = joblib.load("models/svm.pkl")
        self.vectorizer = joblib.load("models/vectorizer.pkl")
        self.label_encoder = joblib.load("models/label_encoder.pkl")

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

        # Convert numeric label back to text
        intent = self.label_encoder.inverse_transform(prediction)

        return intent[0]


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