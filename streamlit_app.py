import streamlit as st

from chatbot.hotel_info import HOTEL_NAME
from chatbot.preprocessing import process_input
from chatbot.entity_extractor import extract_entities
from chatbot.intent_classifier import IntentPredictor
from chatbot.dialogue_manager import DialogueManager
from chatbot.response import get_response


# CONFIGURATION
CONFIDENCE_THRESHOLD = 0.50

st.set_page_config(
    page_title="BookMate",
    page_icon="🏨",
    layout="wide"
)

# PAGE HEADER
st.title("🏨 BookMate")
st.subheader("Your Smart Hotel Booking Assistant")

st.write(f"Welcome to **{HOTEL_NAME}**!")

# SIDEBAR
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

    ✅ Booking Status

    ✅ Cancel Booking

    ✅ Modify Booking
    """)

    st.divider()

    if st.button("🗑 Clear Chat", use_container_width=True):

        st.session_state.messages = []

        # Reset Dialogue Manager
        if "dialogue_manager" in st.session_state:
            st.session_state.dialogue_manager.reset()

        if "last_prediction" in st.session_state:
            del st.session_state.last_prediction

        st.rerun()


# INITIALIZE SESSION STATE
if "messages" not in st.session_state:
    st.session_state.messages = []

# LOAD INTENT PREDICTOR
if (
    "predictor" not in st.session_state
    or st.session_state.get("current_model") != model
):

    st.session_state.predictor = IntentPredictor(model)
    st.session_state.current_model = model

predictor = st.session_state.predictor

# LOAD DIALOGUE MANAGER
if "dialogue_manager" not in st.session_state:
    st.session_state.dialogue_manager = DialogueManager()


dialogue_manager = st.session_state.dialogue_manager

# DISPLAY MESSAGE HISTORY
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Display ML prediction information
        if (
            message["role"] == "assistant"
            and "prediction" in message
        ):

            pred = message["prediction"]

            used_model = pred.get("model", model)

            caption_text = (
                f"🧠 Model: **{used_model}** | "
                f"Intent: **{pred['intent']}** | "
                f"Confidence: **{pred['confidence']:.2%}**"
            )

            if pred.get("entities"):
                caption_text += (
                    f" | 🏷️ Entities: `{pred['entities']}`"
                )

            st.caption(caption_text)

# HANDLE USER INPUT
user_input = st.chat_input("Type your message...")

if user_input:

    # 1. Display user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })


    # 2. NLP PREPROCESSING
    nlp_details = process_input(user_input)

    preprocessed_text = nlp_details["preprocessed_text"]


    # 3. ENTITY EXTRACTION
    extracted_data = extract_entities(user_input)

    extracted_entities = extracted_data["entities_found"]


    # 4. INTENT PREDICTION
    result = predictor.predict(
        user_input,
        preprocessed_text=preprocessed_text
    )

    intent = result["intent"]
    confidence = result["confidence"]


    # 5. CHECK CURRENT DIALOGUE STATE
    dialogue_state = dialogue_manager.state

    booking_active = dialogue_state.get("active", False)

    current_action = dialogue_state.get("action")

    # 6. LOW CONFIDENCE HANDLING
    if (
        confidence < CONFIDENCE_THRESHOLD
        and not booking_active
    ):

        bot_reply = (
            "Sorry, I didn't quite understand that. "
            "Could you please rephrase your request?\n\n"
            "You can ask me about room booking, room prices, "
            "facilities, breakfast, parking, payment, "
            "or check-in/check-out."
        )

    else:

        # 7. DIALOGUE MANAGER
        """
        If the conversation is already inside a booking flow,
        the Dialogue Manager controls the next step.
        Otherwise, it uses the predicted intent.
        """

        if booking_active:

            manager_intent = None

            # Mark intent nicely during active booking wizard step
            if confidence < CONFIDENCE_THRESHOLD:
                intent = "book_hotel (wizard step)"
                confidence = 1.0

        else:

            manager_intent = intent


        bot_reply = dialogue_manager.handle_message(
            user_input=user_input,
            intent=manager_intent,
            extracted_entities=extracted_entities
        )


        # 8. GENERAL INTENT RESPONSE
        # Dialogue Manager returns None when the message is not related to a booking workflow.
        if bot_reply is None:

            bot_reply = get_response(intent, entities=extracted_entities)

    # 9. SAVE PREDICTION INFORMATION
    prediction_info = {
        "model": model,
        "intent": intent,
        "confidence": confidence,
        "cleaned_text": preprocessed_text,
        "detected_pii": nlp_details["detected_pii"],
        "entities": extracted_entities,
        "dialogue_action": current_action
    }

    # 10. DISPLAY BOT RESPONSE
    message_object = {
        "role": "assistant",
        "content": bot_reply,
        "prediction": prediction_info
    }

    st.session_state.messages.append(message_object)

    # Refresh Streamlit UI
    st.rerun()
