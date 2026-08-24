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
        f"Hello! Welcome to {HOTEL_NAME}. How may I assist you today?",
        "Hi! I'm BookMate. How can I help you?",
        f"Welcome to {HOTEL_NAME}! Feel free to ask me anything."
    ],

    # Help & Capabilities
    "help": [
        f"Hi! I'm BookMate, your smart assistant for **{HOTEL_NAME}**. Here is what I can do for you:\n\n• 🏨 **Book a Room** (Interactive reservation)\n• 📅 **Check Room Availability**\n• 💰 **Room Prices & Rates**\n• 🍳 **Breakfast Info**\n• 🚗 **Free Parking Info**\n• 🏊 **Resort Facilities** (Pool, Gym, Spa, WiFi)\n• 💳 **Payment Methods**\n• ⏰ **Check-in / Check-out Times**\n• 📞 **Contact Information**\n\nHow may I assist you today?",
        f"I'm BookMate! I can assist you with:\n\n1. 🏨 **Room Booking & Reservations**\n2. 💰 **Room Prices & Availability**\n3. 🏊 **Resort Facilities & Amenities**\n4. 🍳 **Breakfast & Dining Info**\n5. 🚗 **Parking & Check-in/out Times**\n6. 💳 **Payment Methods & Contact Info**\n\nFeel free to type any question to get started!"
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
        "Sure! To modify your booking or change your room type, please provide your **Booking ID** (e.g. BK1005) or Guest Name so we can update your reservation.",
        "I'd be happy to help modify your reservation! Please share your **Booking ID** or Name, along with the new room type or dates you prefer.",
        "You can easily change your room type or dates! Please tell me your **Booking ID** or Name."
    ],

    # Cancel Booking
    "cancel_booking": [
        "I'm sorry to hear that you wish to cancel. Please provide your **Booking ID** (e.g. BK1005) or Full Name so I can process your cancellation.",
        "Sure! I can help you cancel your booking. If you forgot your **Booking ID**, simply tell me your Name or Email!",
        "Please share your **Booking ID** or Full Name so I can assist with your cancellation."
    ],

    # Booking Status
    "booking_status": [
        "I'd be happy to check your reservation status! Please provide your **Booking ID** (e.g. BK1005) or your **Full Name & Check-in Date** (if you forgot your Booking ID).",
        "You can check your booking anytime! Simply share your **Booking ID** or your **Full Name & Check-in Date** to view your reservation details.",
        "Please tell me your **Booking ID** (or your **Full Name** and **Check-in Date**) so I can look up your reservation for you!"
    ],

    # Room Price & Recommendation
    "room_price": [
        "Here are our room types, rates, and guest capacities:\n\n• **Standard Room**: RM180/night (Max 2 guests)\n• **Deluxe Room**: RM280/night (Max 3 guests)\n• **Family Suite**: RM450/night (Max 4-5 guests — **Recommended for family of 4!**)\n• **Ocean Villa**: RM780/night (Max 6 guests)\n\nFor a family of 4 members, our **Family Suite** (RM450/night) is the perfect choice!",
        "Our room recommendations and capacities:\n\n• **1-2 Guests**: Standard Room (RM180)\n• **2-3 Guests**: Deluxe Room (RM280)\n• **4-5 Guests / Family of 4**: Family Suite (RM450)\n• **5-6 Guests**: Ocean Villa (RM780)\n\nWhich room type would you like to reserve?"
    ],

    # Breakfast
    "breakfast": [
        f"No, lunch and dinner are not provided. We only serve daily breakfast ({BREAKFAST_TIME})!",
        f"We only provide daily breakfast ({BREAKFAST_TIME}). Lunch and dinner are not served at the resort.",
        f"Breakfast is served daily from {BREAKFAST_TIME}. Please note that lunch and dinner are not provided."
    ],

    # Hotel Facilities
    "hotel_facilities": [
        "Our resort offers a swimming pool, gym, spa, restaurant, free WiFi and beach access. No booking needed—simply show your room card!",
        "No reservation required! All resort facilities are complimentary for staying guests—just show your room card to enter.",
        "You don't need to book for facilities! Guests can freely enjoy our swimming pool, gym, spa and beach access by presenting your room card."
    ],

    # Parking
    "parking": [
        "Free 24-hour parking is available for all hotel guests! We accommodate car parking spaces based on your reservation requirements.",
        "Yes! We provide complimentary parking spaces for your vehicles throughout your stay.",
        "Parking is free and available 24 hours for guest cars!"
    ],

    # Check-in / Check-out
    "checkin_checkout": [
        f"Check-in starts at {CHECK_IN} and check-out is before {CHECK_OUT}. To check in upon arrival, simply present your Booking ID or IC/Passport at the resort front desk!",
        f"Our standard check-in time is {CHECK_IN}, while check-out is at {CHECK_OUT}. You can check in directly at our main lobby reception.",
        f"Check-in is from {CHECK_IN} onwards. Early check-in and late check-out (after {CHECK_OUT}) are subject to room availability."
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
            ["I'm sorry, I didn't understand your question. Could you please rephrase it?"]
        )
    )