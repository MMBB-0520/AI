import streamlit as st

from chatbot.predictor import IntentPredictor
from chatbot.response import get_response
from chatbot.hotel_info import HOTEL_NAME

st.set_page_config(
    page_title="BookMate",
    page_icon="🏨",
    layout="wide"
)
st.title("🏨 BookMate")
st.subheader("Your Smart Hotel Booking Assistant")

st.write(f"Welcome to **{HOTEL_NAME}**!")

predictor = IntentPredictor()

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

    intent = predictor.predict(user_input)

    bot_reply = get_response(intent)

    st.session_state.messages.append({
        "role": "assistant",
        "content": bot_reply
    })

    st.rerun()

with st.sidebar:

    st.header("🏨 Oriented Resort")

    st.write("BookMate Chatbot")

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

    st.rerun()