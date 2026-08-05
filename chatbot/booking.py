import json
import os
import random
from chatbot.hotel_info import HOTEL_NAME
from chatbot.entity_extractor import extract_entities

BOOKING_FILE = "data/booking.json"


def load_bookings():
    """Load all bookings from data/booking.json."""
    if os.path.exists(BOOKING_FILE):
        try:
            with open(BOOKING_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_bookings(bookings):
    """Save bookings list to data/booking.json."""
    os.makedirs(os.path.dirname(BOOKING_FILE), exist_ok=True)
    with open(BOOKING_FILE, "w") as f:
        json.dump(bookings, f, indent=4)


def save_booking(data):
    """Save a single booking record."""
    bookings = load_bookings()
    bookings.append(data)
    save_bookings(bookings)


def get_booking_by_id(booking_id: str):
    """Find booking record by ID."""
    bookings = load_bookings()
    for b in bookings:
        if b.get("booking_id", "").upper() == booking_id.upper():
            return b
    return None


def cancel_booking_by_id(booking_id: str) -> tuple[bool, str]:
    """
    Cancel booking by ID.
    Returns (success: bool, message: str)
    """
    bookings = load_bookings()
    for b in bookings:
        if b.get("booking_id", "").upper() == booking_id.upper():
            if b.get("status") == "Cancelled":
                return False, f"Booking **{booking_id.upper()}** is already cancelled."
            b["status"] = "Cancelled"
            save_bookings(bookings)
            return True, f"❌ Booking **{booking_id.upper()}** has been successfully cancelled."
    return False, f"Could not find any reservation with Booking ID: **{booking_id.upper()}**. Please check your ID."


def check_booking_status(booking_id: str) -> str:
    """Check and format booking status info by ID."""
    b = get_booking_by_id(booking_id)
    if not b:
        return f"No reservation found with Booking ID: **{booking_id.upper()}**. Please verify your booking ID."

    status_emoji = "✅" if b.get("status", "Confirmed") == "Confirmed" else "❌"
    return f"""📋 **Booking Details ({b.get('booking_id')})**

• **Guest Name**: {b.get('name', 'N/A')}
• **Room Type**: {b.get('room', 'N/A')}
• **Check-in**: {b.get('check_in', 'N/A')}
• **Check-out**: {b.get('check_out', 'N/A')}
• **Guests**: {b.get('guests', 'N/A')}
• **Status**: {status_emoji} {b.get('status', 'Confirmed')}"""


def _get_next_question_or_confirm(booking_state):
    """Determine the next step/question based on missing booking fields."""
    if not booking_state.get("name"):
        booking_state["step"] = 1
        return "Sure! Let me help you book a room.\n\nMay I have your name?"

    if not booking_state.get("checkin"):
        booking_state["step"] = 2
        return "What is your check-in date?"

    if not booking_state.get("checkout"):
        booking_state["step"] = 3
        return "What is your check-out date?"

    if not booking_state.get("guests"):
        booking_state["step"] = 4
        return "How many guests?"

    if not booking_state.get("room"):
        booking_state["step"] = 5
        return "Which room type would you like?\n(Standard Room / Deluxe Room / Family Suite / Ocean Villa)"

    # All fields present! Generate confirmation
    booking_id = f"BK{random.randint(1000, 9999)}"
    booking_record = {
        "booking_id": booking_id,
        "name": booking_state["name"],
        "room": booking_state["room"],
        "check_in": booking_state["checkin"],
        "check_out": booking_state["checkout"],
        "guests": booking_state["guests"],
        "status": "Confirmed"
    }
    save_booking(booking_record)

    bot_reply = f"""✅ **Booking Confirmed!**

• **Booking ID**: {booking_id}
• **Name**: {booking_state['name']}
• **Room**: {booking_state['room']}
• **Check-in**: {booking_state['checkin']}
• **Check-out**: {booking_state['checkout']}
• **Guests**: {booking_state['guests']}

Thank you for choosing **{HOTEL_NAME}**!"""

    booking_state["active"] = False
    booking_state["step"] = 0
    booking_state["action"] = None
    return bot_reply


def process_booking(booking_state, user_input, intent=None, extracted_entities=None):
    """
    Process booking creation, status check, cancellation, and modification flows.
    Modifies booking_state in-place and returns the bot's reply.
    """
    if extracted_entities is None:
        extracted = extract_entities(user_input).get("entities_found", {})
    else:
        extracted = extracted_entities

    booking_id = extracted.get("booking_id")

    # 1. Handle Cancel Booking Intent
    if intent == "cancel_booking":
        if booking_id:
            _, reply = cancel_booking_by_id(booking_id)
            return reply
        else:
            booking_state["action"] = "cancel"
            booking_state["active"] = True
            return "Sure! Please provide your **Booking ID** (e.g. BK1234) so I can cancel your reservation."

    # 2. Handle Booking Status Inquiry Intent
    if intent == "booking_status":
        if booking_id:
            return check_booking_status(booking_id)
        else:
            booking_state["action"] = "status"
            booking_state["active"] = True
            return "I'd be happy to check your reservation. Please enter your **Booking ID** (e.g. BK1234)."

    # 3. Handle Modify Booking Intent
    if intent == "modify_booking":
        if booking_id:
            b = get_booking_by_id(booking_id)
            if b:
                return f"Found your booking **{booking_id}**! To modify your dates or room type, please cancel this booking or contact our front desk at +60 4-987 8888."
            return f"Could not find Booking ID **{booking_id}**. Please check your reference number."
        else:
            booking_state["action"] = "modify"
            booking_state["active"] = True
            return "Sure! Please provide your **Booking ID** (e.g. BK1234) to update your reservation."

    # Handle active multi-turn waiting state (e.g. waiting for booking_id after prompt)
    action = booking_state.get("action")
    if booking_state["active"] and action in ["cancel", "status", "modify"]:
        bid = booking_id or user_input.strip().upper()
        if bid.startswith("BK") or len(bid) >= 4:
            booking_state["active"] = False
            booking_state["action"] = None
            if action == "cancel":
                _, reply = cancel_booking_by_id(bid)
                return reply
            elif action == "status":
                return check_booking_status(bid)
            elif action == "modify":
                return f"Found your booking **{bid}**! To modify your dates or room type, please cancel this booking or contact our front desk at +60 4-987 8888."

    # 4. Handle Create New Booking Intent (book_room)
    if intent == "book_room" and not booking_state["active"]:
        booking_state["active"] = True
        booking_state["action"] = "book"

        if extracted.get("name"):
            booking_state["name"] = extracted["name"]
        if extracted.get("check_in"):
            booking_state["checkin"] = extracted["check_in"]
        if extracted.get("check_out"):
            booking_state["checkout"] = extracted["check_out"]
        if extracted.get("guests"):
            booking_state["guests"] = extracted["guests"]
        if extracted.get("room_type"):
            booking_state["room"] = extracted["room_type"]

        return _get_next_question_or_confirm(booking_state)

    if not booking_state["active"]:
        return None

    # Step-by-step processing when new booking creation is active
    step = booking_state.get("step", 0)

    if step == 1:
        booking_state["name"] = extracted.get("name") or user_input.strip()
    elif step == 2:
        booking_state["checkin"] = extracted.get("check_in") or user_input.strip()
    elif step == 3:
        booking_state["checkout"] = extracted.get("check_out") or user_input.strip()
    elif step == 4:
        booking_state["guests"] = extracted.get("guests") or user_input.strip()
    elif step == 5:
        booking_state["room"] = extracted.get("room_type") or user_input.strip()

    if extracted.get("check_in") and not booking_state.get("checkin"):
        booking_state["checkin"] = extracted["check_in"]
    if extracted.get("check_out") and not booking_state.get("checkout"):
        booking_state["checkout"] = extracted["check_out"]
    if extracted.get("guests") and not booking_state.get("guests"):
        booking_state["guests"] = extracted["guests"]
    if extracted.get("room_type") and not booking_state.get("room"):
        booking_state["room"] = extracted["room_type"]

    return _get_next_question_or_confirm(booking_state)


