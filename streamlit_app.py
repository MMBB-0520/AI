import os
import json
import streamlit as st

from chatbot.hotel_info import HOTEL_NAME
from chatbot.preprocessing import process_input
from chatbot.sentiment_analyzer import analyze_sentiment
from chatbot.entity_extractor import extract_entities
from chatbot.intent_classifier import IntentPredictor
from chatbot.dialogue_manager import DialogueManager
from chatbot.response import get_response, generate_fallback_response


# CONFIGURATION
CONFIDENCE_THRESHOLD = 0.50
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="BookMate",
    page_icon="🏨",
    layout="wide"
)

# HELPER: LOAD BOOKINGS FOR LIVE MOBILE PASS
def get_all_bookings():
    booking_file = os.path.join(PROJECT_ROOT, "data", "booking.json")
    if os.path.exists(booking_file):
        try:
            with open(booking_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

# PAGE HEADER
st.title("🏨 BookMate")
st.subheader("Your Smart Hotel Booking Assistant")

st.write(f"Welcome to **{HOTEL_NAME}**!")

# SIDEBAR
with st.sidebar:

    st.header(f"🏨 {HOTEL_NAME}")

    st.write("BookMate Chatbot")
    
    model = "Support Vector Machine"
    
    st.divider()

    st.markdown("**✨ Supported Services**")
    
    with st.expander("🛏️ Reservations", expanded=True):
        st.markdown("""
        - Book Room
        - Check Booking Status
        - Cancel Booking
        - Invoices & Receipts
        """)
        
    with st.expander("🏨 Hotel Info & Services", expanded=True):
        st.markdown("""
        - Room Prices & Payment
        - Facilities & Parking
        - Breakfast & Dining
        - Check-in / Check-out
        """)

        
    if st.button("🗑 Clear Chat", use_container_width=True):

        st.session_state.messages = []

        # Reset Dialogue Manager
        if "dialogue_manager" in st.session_state:
            st.session_state.dialogue_manager.reset()

        if "last_prediction" in st.session_state:
            del st.session_state.last_prediction

        st.rerun()

    st.divider()
    st.markdown("**✨ Demo Features**")
    
    # LIVE MOBILE ASSISTANT (CONTEXTUAL DEPOSIT PAYMENT & CANCELLATION 2FA)
    dm = st.session_state.get("dialogue_manager")
    dm_state = dm.state if dm else {}
    is_cancel_active = (
        dm_state.get("active") and
        dm_state.get("action") == "confirm_cancel" and
        dm_state.get("mobile_2fa_active", False)
    )
    cancel_booking_id = dm_state.get("target_booking_id")

    # Check if there is an active booking awaiting deposit payment
    is_deposit_pending = (
        dm_state.get("active") and
        dm_state.get("action") == "book" and
        dm_state.get("room") is not None and
        (dm_state.get("step") == "awaiting_deposit" or not dm_state.get("deposit_paid"))
    )

    # Find active cancellation booking details
    cancel_booking = None
    if is_cancel_active and cancel_booking_id:
        all_b = get_all_bookings()
        for b in all_b:
            if b.get("booking_id") == cancel_booking_id:
                cancel_booking = b
                break

    with st.expander("📱 Live Mobile Assistant & Passes", expanded=True):
        if is_deposit_pending:
            pricing = dm._calculate_booking_price()
            st.markdown(
                f"""
                <div style="background-color: #1e293b; color: #f8fafc; padding: 12px; border-radius: 10px; border: 1px solid #38bdf8; font-size: 13px;">
                    <div style="color: #38bdf8; font-weight: bold; margin-bottom: 6px;">📱 Mobile Push: 1-Night Deposit Payment</div>
                    <div><b>👤 Guest:</b> {dm_state.get('name')}</div>
                    <div><b>🛏️ Room:</b> {dm_state.get('room')} ({pricing['rate_str']}/night)</div>
                    <div><b>📅 Dates:</b> {dm_state.get('checkin')} ~ {dm_state.get('checkout')} ({pricing['nights']} night{'s' if pricing['nights'] > 1 else ''})</div>
                    <div><b>👥 Guests:</b> {dm_state.get('guests')} pax</div>
                    <div style="margin-top: 6px; padding-top: 6px; border-top: 1px dashed #475569;">
                        <div><b>💳 1-Night Deposit:</b> <span style="color: #4ade80; font-size: 14px; font-weight: bold;">RM{pricing['deposit_total']}</span> <span style="font-size: 11px; color: #94a3b8;">(Payable Now)</span></div>
                        <div><b>💵 Pay at Check-in:</b> <span style="color: #f8fafc;">RM{pricing['remaining_balance']}</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write("")
            col_pay, col_cancel = st.columns(2)
            with col_pay:
                if st.button("💳 Pay Deposit", key="mobile_pay_deposit", use_container_width=True):
                    confirm_reply = dm._create_booking()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": confirm_reply,
                        "prediction": {"model": model, "intent": "book_hotel", "confidence": 1.0}
                    })
                    st.rerun()
            with col_cancel:
                if st.button("❌ Cancel Reservation", key="mobile_cancel_deposit", use_container_width=True):
                    dm.reset()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "Reservation cancelled. No deposit was charged. Feel free to book again anytime!",
                        "prediction": {"model": model, "intent": "book_hotel", "confidence": 1.0}
                    })
                    st.rerun()

        elif cancel_booking:
            guest_name = cancel_booking.get("name", "Guest")
            guest_phone = cancel_booking.get("phone", "+60 12-345 6789")
            
            st.markdown(
                f"""
                <div style="background-color: #1e293b; color: #f8fafc; padding: 12px; border-radius: 10px; border: 1px solid #f59e0b; font-size: 13px;">
                    <div style="color: #f59e0b; font-weight: bold; margin-bottom: 6px;">🔔 SMS Alert: Cancellation Request</div>
                    <div><b>📱 Phone:</b> <code>{guest_phone}</code></div>
                    <div><b>🏨 Booking ID:</b> <code style="color: #facc15;">{cancel_booking['booking_id']}</code></div>
                    <div><b>👤 Guest Name:</b> {guest_name}</div>
                    <div><b>🛏️ Room:</b> {cancel_booking['room']}</div>
                    <div><b>📅 Stay Dates:</b> {cancel_booking['check_in']} ~ {cancel_booking['check_out']}</div>
                    <div style="margin-top: 6px; color: #f87171; font-size: 12px;">⚠️ <i>Verification requested from chat. Click below to approve or reject on your mobile.</i></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write("")
            col_approve, col_reject = st.columns(2)
            with col_approve:
                if st.button("✅ Confirm Cancel", key=f"mobile_confirm_cancel_{cancel_booking['booking_id']}", use_container_width=True):
                    all_b = get_all_bookings()
                    for b in all_b:
                        if b.get("booking_id") == cancel_booking_id:
                            b["status"] = "Cancelled"
                    with open(os.path.join(PROJECT_ROOT, "data", "booking.json"), "w", encoding="utf-8") as f:
                        json.dump(all_b, f, indent=4)
                    
                    if dm:
                        dm.reset()

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": (
                            f"❌ **Booking {cancel_booking_id} Cancelled Successfully**\n\n"
                            f"- **Guest Name**      : {guest_name}\n"
                            f"- **Verification**    : ✅ Approved via Mobile ({guest_phone})\n"
                            f"- **Status**          : Cancelled\n"
                            f"- **Refund Details**  : A full refund is being processed to your original payment method (5-7 business days).\n\n"
                            f"We're sorry to see you go! Feel free to book with **{HOTEL_NAME}** again anytime."
                        ),
                        "prediction": {"model": model, "intent": "cancel_hotel_reservation", "confidence": 1.0}
                    })
                    st.rerun()

            with col_reject:
                if st.button("🛡️ Keep Booking", key=f"mobile_reject_cancel_{cancel_booking['booking_id']}", use_container_width=True):
                    if dm:
                        dm.reset()
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Cancellation aborted on mobile. Your reservation **{cancel_booking_id}** remains active and confirmed!",
                        "prediction": {"model": model, "intent": "cancel_hotel_reservation", "confidence": 1.0}
                    })
                    st.rerun()

        else:
            st.markdown(
                """
                <div style="background-color: #0f172a; color: #94a3b8; padding: 12px; border-radius: 8px; border: 1px dashed #334155; font-size: 12px; text-align: center;">
                    📱 <b>Simulated Mobile Device</b><br>
                    <i>(Standby Mode)</i><br><br>
                    When reviewing a deposit payment or requesting a cancellation in the chat, interactive mobile passes with 1-click actions will appear here.
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    show_debug = st.toggle(
        "🔍 Developer Debug Mode",
        value=False,
        help="Show ML model, predicted intent, confidence %, and extracted entities."
    )

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

        # Display ML prediction information when Developer Debug Mode is enabled
        if (
            show_debug
            and message["role"] == "assistant"
            and "prediction" in message
        ):

            pred = message["prediction"]

            used_model = pred.get("model", model)

            caption_text = (
                f"🧠 Model: **{used_model}** | "
                f"Intent: **{pred['intent']}** | "
                f"Confidence: **{pred['confidence']:.2%}**"
            )

            if pred.get("sentiment"):
                sent = pred["sentiment"]
                sent_label = sent.get("sentiment", "neutral").capitalize()
                sent_emoji = "😊" if sent_label == "Positive" else ("😡" if sent_label == "Negative" else "😐")
                caption_text += f" | {sent_emoji} Sentiment: **{sent_label}** (`{sent.get('score', 0):+.2f}`)"

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


    # 2. NLP PREPROCESSING & SENTIMENT ANALYSIS (SUPPORTING COMPONENT)
    nlp_details = process_input(user_input)
    preprocessed_text = nlp_details["preprocessed_text"]
    sentiment_info = analyze_sentiment(user_input)

    # 3. INTENT PREDICTION (MACHINE LEARNING MODEL)
    prediction_result = predictor.predict(
        user_input,
        preprocessed_text=preprocessed_text
    )

    intent = prediction_result["intent"]
    confidence = prediction_result["confidence"]

    # 4. ENTITY EXTRACTION
    extracted_entities = extract_entities(user_input).get("entities_found", {})

    # 5. DIALOGUE STATE
    dialogue_state = dialogue_manager.state
    booking_active = dialogue_state.get("active", False)
    current_action = dialogue_state.get("action")

    # 5.5 MULTI-TURN CONTEXT INHERITANCE & BOOKING ID EXPRESS RULE
    pending_intent = dialogue_state.get("pending_intent")
    awaiting_slot = dialogue_state.get("awaiting_slot")

    if pending_intent and awaiting_slot and extracted_entities.get(awaiting_slot):
        intent = pending_intent
        confidence = 1.0
        dialogue_state["pending_intent"] = None
        dialogue_state["awaiting_slot"] = None
    elif extracted_entities.get("booking_id"):
        # Standalone Booking ID Express Bypass
        intent = pending_intent or "check_hotel_reservation"
        confidence = 1.0

    # 6. LOW CONFIDENCE HANDLING
    if (
        confidence < CONFIDENCE_THRESHOLD
        and not booking_active
    ):

        bot_reply = generate_fallback_response(
            user_input,
            predictor=predictor,
            confidence=confidence
        )

    else:

        # 7. DIALOGUE MANAGER
        """
        The Dialogue Manager coordinates multi-turn workflows (booking, status, cancellation),
        supports conversational FAQ interruptions, and manages state.
        """
        bot_reply = dialogue_manager.handle_message(
            user_input=user_input,
            intent=intent,
            extracted_entities=extracted_entities
        )


        # 8. GENERAL INTENT RESPONSE
        # Dialogue Manager returns None when the message is not related to a booking workflow.
        if bot_reply is None:

            bot_reply = get_response(
                intent,
                entities=extracted_entities,
                sentiment=sentiment_info
            )

    # 9. SAVE PREDICTION INFORMATION
    prediction_info = {
        "model": model,
        "intent": intent,
        "confidence": confidence,
        "cleaned_text": preprocessed_text,
        "detected_pii": nlp_details["detected_pii"],
        "entities": extracted_entities,
        "sentiment": sentiment_info,
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
