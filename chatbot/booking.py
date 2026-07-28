import json
import os
import random
from chatbot.hotel_info import HOTEL_NAME

BOOKING_FILE = "data/booking.json"

def save_booking(data):

    file = BOOKING_FILE

    if os.path.exists(file):

        with open(file, "r") as f:
            bookings = json.load(f)

    else:
        bookings = []


    bookings.append(data)


    with open(file, "w") as f:
        json.dump(bookings, f, indent=4)

def process_booking(booking_state, user_input, intent=None):
    """
    Process the booking flow based on the current state.
    Modifies booking_state in-place and returns the bot's reply.
    If the booking flow is not active and the intent is not book_room, returns None.
    """
    if intent == "book_room" and not booking_state["active"]:
        booking_state["active"] = True
        booking_state["step"] = 1
        return "Sure! Let's book a room.\n\nMay I have your name?"

    if not booking_state["active"]:
        return None

    if booking_state["step"] == 1:
        booking_state["name"] = user_input
        booking_state["step"] = 2
        return "What is your check-in date?"

    elif booking_state["step"] == 2:
        booking_state["checkin"] = user_input
        booking_state["step"] = 3
        return "What is your check-out date?"

    elif booking_state["step"] == 3:
        booking_state["checkout"] = user_input
        booking_state["step"] = 4
        return "How many guests?"

    elif booking_state["step"] == 4:
        booking_state["guests"] = user_input
        booking_state["step"] = 5
        return "Which room type would you like?\n(Standard / Deluxe / Family Suite / Ocean Villa)"

    elif booking_state["step"] == 5:
        booking_state["room"] = user_input
        booking_id = f"BK{random.randint(1000,9999)}"

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
        ✅ Booking Confirmed!

        Booking ID: {booking_id}

        Name: {booking_state['name']}
        Room: {booking_state['room']}
        Check-in: {booking_state['checkin']}
        Check-out: {booking_state['checkout']}
        Guests: {booking_state['guests']}

        Thank you for choosing {HOTEL_NAME}!
        """

        booking_state["active"] = False
        booking_state["step"] = 0

        return bot_reply
