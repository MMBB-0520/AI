
import streamlit as st
import random
import json
import os

from chatbot.predictor import IntentPredictor
from chatbot.response import get_response
from chatbot.hotel_info import HOTEL_NAME

from chatbot.booking import process_booking

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

if st.sidebar.button("🗑 Clear Chat"):

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
    
    st.rerun()

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

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

user_input = st.chat_input(
    "Type your message..."
)

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    booking = st.session_state.booking

    if booking["active"]:

        bot_reply = process_booking(
            booking,
            user_input,
            None
        )

    else:

        result = predictor.predict(user_input)

        intent = result["intent"]
        confidence = result["confidence"]

        booking_reply = process_booking(
            booking,
            user_input,
            intent
        )

        if booking_reply is not None:
            bot_reply = booking_reply
        else:
            bot_reply = get_response(intent)

        st.session_state.prediction = {
            "intent": intent,
            "confidence": confidence
        }    
        
    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

        if "prediction" in st.session_state:
            st.caption(
                f"Intent: {st.session_state.prediction['intent']} | "
                f"Confidence: {st.session_state.prediction['confidence']:.2%}"
            )

    st.rerun()

