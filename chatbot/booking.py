import json
import os
import random
import re
from chatbot.hotel_info import HOTEL_NAME

BOOKING_FILE = "data/booking.json"

VALID_ROOMS = [
    "standard", "deluxe", "family suite", "ocean villa",
    "room type a", "room type b", "room type c", "room type d", "room type e", "room type f", "room type g",
    "type a", "type b", "type c", "type d", "type e", "type f", "type g",
    "a", "b", "c", "d", "e", "f", "g"
]

CANCEL_COMMANDS = ["cancel", "exit", "stop", "quit", "nevermind", "restart", "abort"]

def save_booking(data):
    file = BOOKING_FILE
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            try:
                bookings = json.load(f)
            except Exception:
                bookings = []
    else:
        bookings = []

    bookings.append(data)

    with open(file, "w", encoding="utf-8") as f:
        json.dump(bookings, f, indent=4)

def search_booking(user_input):
    """
    Search data/booking.json for matching Booking ID, Guest Name, or Email.
    Uses strict word boundaries to avoid partial word collisions (e.g. 'booking' matching 'King').
    """
    clean_input = user_input.strip().lower()
    if not os.path.exists(BOOKING_FILE):
        return None

    # Ignore general queries that don't contain specific names/IDs
    IGNORE_KEYWORDS = ["i", "want", "to", "cancel", "my", "booking", "modify", "check", "status", "room", "hotel", "resort", "forgot", "id", "number", "the", "a", "an", "is", "can", "please", "help"]
    words = [w.strip() for w in re.findall(r'\b[a-z0-9@.]+\b', clean_input)]
    
    # Check if there is any word outside generic keywords
    specific_tokens = [w for w in words if w not in IGNORE_KEYWORDS]
    if not specific_tokens:
        return None

    try:
        with open(BOOKING_FILE, "r", encoding="utf-8") as f:
            bookings = json.load(f)
    except Exception:
        return None

    matched = []
    for b in bookings:
        b_id = str(b.get("booking_id", "")).lower()
        b_name = str(b.get("name", "")).lower()
        b_email = str(b.get("email", "")).lower()

        # 1. Match Booking ID pattern (e.g., BK1005 or exact 1005)
        if b_id:
            if re.search(r'\b' + re.escape(b_id) + r'\b', clean_input):
                matched.append(b)
                break
            num_part = b_id.replace("bk", "")
            if num_part and re.search(r'\b' + re.escape(num_part) + r'\b', clean_input) and "bk" in clean_input:
                matched.append(b)
                break

        # 2. Match Email (exact word match)
        if b_email and b_email in clean_input:
            matched.append(b)
            break

        # 3. Match Name (strict word boundary match for names >= 3 chars, avoiding generic keywords)
        name_parts = [p for p in b_name.split() if len(p) >= 3 and p not in IGNORE_KEYWORDS]
        for part in name_parts:
            if re.search(r'\b' + re.escape(part) + r'\b', clean_input):
                matched.append(b)
                break
        if matched:
            break

    if matched:
        b = matched[0]
        status_str = b.get("status", "Confirmed")
        return f"""🔍 **Reservation Details Found!**

- **Booking ID**: `{b.get('booking_id', 'N/A')}`
- **Guest Name**: {b.get('name', 'N/A')}
- **Room Type**: {b.get('room', 'N/A')}
- **Check-in Date**: {b.get('check_in', 'N/A')}
- **Check-out Date**: {b.get('check_out', 'N/A')}
- **Guests**: {b.get('guests', '1')}
- **Status**: {status_str}

*How else may I assist you with your reservation?*"""

    return None

def cancel_reservation(user_input):
    """
    Search data/booking.json and mark matching reservation status as Cancelled.
    """
    clean_input = user_input.strip().lower()
    if not os.path.exists(BOOKING_FILE):
        return None

    IGNORE_KEYWORDS = ["i", "want", "to", "cancel", "my", "booking", "modify", "check", "status", "room", "hotel", "resort", "forgot", "id", "number", "the", "a", "an", "is", "can", "please", "help"]
    words = [w.strip() for w in re.findall(r'\b[a-z0-9@.]+\b', clean_input)]
    specific_tokens = [w for w in words if w not in IGNORE_KEYWORDS]
    if not specific_tokens:
        return None

    try:
        with open(BOOKING_FILE, "r", encoding="utf-8") as f:
            bookings = json.load(f)
    except Exception:
        return None

    matched_idx = -1
    for idx, b in enumerate(bookings):
        b_id = str(b.get("booking_id", "")).lower()
        b_name = str(b.get("name", "")).lower()
        b_email = str(b.get("email", "")).lower()

        if b_id and (b_id in clean_input or b_id.replace("bk", "") in clean_input):
            matched_idx = idx
            break

        name_parts = [p for p in b_name.split() if len(p) >= 3 and p not in IGNORE_KEYWORDS]
        if any(re.search(r'\b' + re.escape(part) + r'\b', clean_input) for part in name_parts):
            matched_idx = idx
            break

        if b_email and b_email in clean_input:
            matched_idx = idx
            break

    if matched_idx != -1:
        b = bookings[matched_idx]
        b["status"] = "Cancelled"
        with open(BOOKING_FILE, "w", encoding="utf-8") as f:
            json.dump(bookings, f, indent=4)

        return f"""❌ **Reservation Cancelled!**

- **Booking ID**: `{b.get('booking_id')}`
- **Guest Name**: {b.get('name')}
- **Room**: {b.get('room')}
- **Status**: Cancelled

Your reservation has been successfully cancelled. Let us know if you need anything else!"""

    return None

def process_booking(booking_state, user_input, intent=None):
    """
    Process the booking flow with strict input validation and cancellation support.
    """
    clean_input = user_input.strip().lower()

    # Check for cancellation
    if booking_state["active"] and clean_input in CANCEL_COMMANDS:
        booking_state["active"] = False
        booking_state["step"] = 0
        return "❌ Booking cancelled. How else may I assist you today?"

    if intent == "book_room" and not booking_state["active"]:
        booking_state["active"] = True
        booking_state["step"] = 1
        return "Sure! Let's book a room for you.\n\nMay I have your full name?"

    if not booking_state["active"]:
        return None

    # Step 1: Name Validation
    if booking_state["step"] == 1:
        if len(user_input.strip()) < 2 or user_input.strip().isdigit():
            return "Please provide a valid name (e.g. John Smith)."
        booking_state["name"] = user_input.strip().title()
        booking_state["step"] = 2
        return f"Nice to meet you, {booking_state['name']}! What is your check-in date? (e.g., 25/08/2026)"

    # Step 2: Check-in Date Validation
    elif booking_state["step"] == 2:
        if len(user_input.strip()) < 2:
            return "Please enter a valid check-in date (e.g., 25/08/2026)."
        booking_state["checkin"] = user_input.strip()
        booking_state["step"] = 3
        return "What is your check-out date? (e.g., 28/08/2026)"

    # Step 3: Check-out Date Validation
    elif booking_state["step"] == 3:
        if len(user_input.strip()) < 2:
            return "Please enter a valid check-out date (e.g., 28/08/2026)."
        booking_state["checkout"] = user_input.strip()
        booking_state["step"] = 4
        return "How many guests will be staying? (e.g., 1, 2, 4)"

    # Step 4: Guests Validation
    elif booking_state["step"] == 4:
        # Extract digits from input if available
        digits = re.findall(r'\d+', user_input)
        if not digits or int(digits[0]) <= 0 or int(digits[0]) > 20:
            return "Please enter a valid number of guests (e.g., 1, 2, 4)."
        booking_state["guests"] = digits[0]
        booking_state["step"] = 5
        return "Which room type would you like?\n(Options: Standard, Deluxe, Family Suite, Ocean Villa, or Room Type A-G)"

    # Step 5: Room Type Validation
    elif booking_state["step"] == 5:
        matched_room = None
        for room_option in VALID_ROOMS:
            if room_option in clean_input:
                matched_room = room_option.title()
                break

        if not matched_room:
            return "❌ Invalid room type entered!\n\nPlease choose one of the available room types:\n- Standard\n- Deluxe\n- Family Suite\n- Ocean Villa\n- Room Type A / B / C / D / E / F / G"

        booking_state["room"] = matched_room
        booking_id = f"BK{random.randint(1000, 9999)}"

        booking_record = {
            "booking_id": booking_id,
            "name": booking_state["name"],
            "room": booking_state["room"],
            "check_in": booking_state["checkin"],
            "check_out": booking_state["checkout"],
            "guests": booking_state["guests"]
        }

        save_booking(booking_record)

        bot_reply = f"""
✅ **Booking Confirmed!**

- **Booking ID**: `{booking_id}`
- **Name**: {booking_state['name']}
- **Room**: {booking_state['room']}
- **Check-in**: {booking_state['checkin']}
- **Check-out**: {booking_state['checkout']}
- **Guests**: {booking_state['guests']}

Thank you for choosing {HOTEL_NAME}! You can cancel or check your reservation anytime.
"""

        booking_state["active"] = False
        booking_state["step"] = 0

        return bot_reply
