import streamlit as st
import random
import json
import os

from chatbot.predictor import IntentPredictor
from chatbot.response import get_response
from chatbot.hotel_info import HOTEL_NAME
from chatbot.booking import process_booking
from chatbot.preprocessing import process_input

# Minimum confidence threshold for intent classification
CONFIDENCE_THRESHOLD = 0.50

st.set_page_config(
    page_title="BookMate",
    page_icon="🏨",
    layout="wide"
)

st.title("🏨 BookMate")
st.subheader("Your Smart Hotel Booking Assistant")

st.write(f"Welcome to **{HOTEL_NAME}**!")

with st.sidebar:
    st.header(f"🏨 {HOTEL_NAME}")
    st.write("BookMate Chatbot")

    st.divider()

    model = st.selectbox(
        "Machine Learning Model",
        [
            "Support Vector Machine",
            "Naive Bayes",
            "Logistic Regression"
        ]
    )

    st.divider()

    st.write("Supported Services")
    st.write("""
    ✅ Book Room

    ✅ Check Availability

    ✅ Room Price

    ✅ Breakfast

    ✅ Parking

    ✅ Hotel Facilities

    ✅ Payment

    ✅ Contact

    ✅ Check-in / Check-out
    """)

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.session_state.booking = {
            "active": False,
            "step": 0,
            "name": "",
            "checkin": "",
            "checkout": "",
            "guests": "",
            "room": ""
        }
        if "last_prediction" in st.session_state:
            del st.session_state.last_prediction
        st.rerun()

# Load Predictor based on selected model
if "predictor" not in st.session_state or st.session_state.get("current_model") != model:
    st.session_state.predictor = IntentPredictor(model)
    st.session_state.current_model = model

predictor = st.session_state.predictor

if "booking" not in st.session_state:
    st.session_state.booking = {
        "active": False,
        "step": 0,
        "name": "",
        "checkin": "",
        "checkout": "",
        "guests": "",
        "room": ""
    }

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "prediction" in message:
            pred = message["prediction"]
            used_model = pred.get("model", model)
            st.caption(
                f"🧠 Model: **{used_model}** | Intent: **{pred['intent']}** | Confidence: **{pred['confidence']:.2%}**"
            )

# Handle user input
user_input = st.chat_input("Type your message...")

if user_input:
    # Append user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    booking = st.session_state.booking
    prediction_info = None

    if booking["active"]:
        bot_reply = process_booking(
            booking,
            user_input,
            None
        )
    else:
        # Preprocessing & Intent Prediction
        nlp_details = process_input(user_input)
        result = predictor.predict(user_input, preprocessed_text=nlp_details["preprocessed_text"])

        intent = result["intent"]
        confidence = result["confidence"]
        prediction_info = {
            "model": model,
            "intent": intent,
            "confidence": confidence,
            "cleaned_text": nlp_details["preprocessed_text"],
            "detected_pii": nlp_details["detected_pii"]
        }

        # Fallback handling for low confidence predictions
        if confidence < CONFIDENCE_THRESHOLD:
            bot_reply = (
                "Sorry, I didn't quite understand that. "
                "Could you please rephrase your request? "
                "You can ask about room booking, prices, facilities, or check-in/out times!"
            )
        else:
            booking_reply = process_booking(
                booking,
                user_input,
                intent
            )

            if booking_reply is not None:
                bot_reply = booking_reply
            else:
                bot_reply = get_response(intent)

    # Append assistant reply
    msg_obj = {
        "role": "assistant",
        "content": bot_reply
    }
    if prediction_info:
        msg_obj["prediction"] = prediction_info

    st.session_state.messages.append(msg_obj)
    st.rerun()
