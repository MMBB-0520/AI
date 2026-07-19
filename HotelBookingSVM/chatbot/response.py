"""
BookMate Chatbot
Response Dictionary

Purpose:
Store chatbot responses for each detected intent.
"""

from chatbot.hotel_info import *
import random

responses = {

    # Greeting
    "greeting": [
        "Hello! Welcome to Oriented Resort. How may I assist you today?",
        "Hi! I'm BookMate. How can I help you?",
        "Welcome to Oriented Resort! Feel free to ask me anything."
    ],

    # Room Availability
    "availability": [
        "Sure! Please tell me your check-in and check-out dates.",
        "I'd be happy to check room availability for you. When would you like to stay?",
        "Please let me know your travel dates so I can check available rooms."
    ],

    # Book Room
    "book_room": [
        "I'd be happy to help you book a room. What are your check-in and check-out dates?",
        "Sure! Which room type would you like to reserve?",
        "Let's make your reservation. May I know your preferred dates?"
    ],

    # Modify Booking
    "modify_booking": [
        "Sure! Please provide your booking ID so I can help modify your reservation.",
        "I'd be happy to update your booking. What would you like to change?",
        "Please tell me your booking details and the changes you would like to make."
    ],

    # Cancel Booking
    "cancel_booking": [
        "I'm sorry to hear that. Please provide your booking ID to cancel your reservation.",
        "Sure! I can help you cancel your booking. May I have your booking number?",
        "Please share your reservation ID so I can process your cancellation."
    ],

    # Booking Status
    "booking_status": [
        "Please provide your booking ID so I can check your reservation status.",
        "I'd be happy to help. What is your booking reference number?",
        "Kindly provide your booking number for verification."
    ],

    # Room Price
    "room_price": [
        "Our Standard Room starts from RM180 per night.",
        "Our room rates range from RM180 to RM780 per night depending on the room type.",
        "Which room type would you like to know the price for?",
        "Our room rates are:\n" + "\n".join(f"{room}: {price}" for room, price in ROOM_PRICES.items())
    ],

    # Breakfast
    "breakfast": [
        "Breakfast is included for Deluxe Room, Family Suite and Ocean Villa.",
        "Standard Room guests can add breakfast for RM30 per person.",
        f"Breakfast is served daily from {BREAKFAST_TIME}."
    ],

    # Hotel Facilities
    "hotel_facilities": [
        "Our resort offers a swimming pool, gym, spa, restaurant, free WiFi and beach access.",
        "Guests can enjoy our swimming pool, kids playground, fitness centre and spa.",
        "We provide many facilities including free WiFi, restaurant and beach access.",
        "Our facilities include: " + ", ".join(FACILITIES) + "."
    ],

    # Parking
    "parking": [
        "Free parking is available for all hotel guests.",
        "Yes! We provide complimentary parking throughout your stay.",
        "Parking is free and available 24 hours."
    ],

    # Check-in / Check-out
    "checkin_checkout": [
        f"Check-in starts at {CHECK_IN} and check-out is before {CHECK_OUT}.",
        f"Our standard check-in time is {CHECK_IN}, while check-out is at {CHECK_OUT}.",
        "Early check-in and late check-out are subject to availability."
    ],

    # Payment
    "payment": [
        f"We accept {', '.join(PAYMENT_METHODS)}.",
        "You can pay using credit card, online banking or DuitNow QR.",
        "Payment can be made during booking or upon arrival depending on your reservation."
    ],

    # Location
    "location":[
        f"{HOTEL_NAME} is located at {ADDRESS}.",
        f"You can visit us at {ADDRESS}.",
        f"Our resort is located at {ADDRESS}."
    ],

    # Contact
    "contact": [
        "You can contact us at +60 4-987 8888.",
        "Our email address is booking@orientedresort.com.",
        "Feel free to call us or email us anytime for assistance."
    ],

    # Goodbye
    "goodbye": [
        "Thank you for choosing Oriented Resort. Have a wonderful day!",
        "Goodbye! We hope to welcome you soon.",
        "Take care and have a pleasant day!"
    ]

}


def get_response(intent):
    """
    Return a random response based on the predicted intent.
    """

    return random.choice(
        responses.get(
            intent,
            ["I'm sorry, I didn't understand your question."]
        )
    )