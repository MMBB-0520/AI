"""
BookMate Chatbot
Response Dictionary

Purpose:
Store chatbot responses for each detected intent (supporting all 25 Bitext Hospitality intents).
"""

from chatbot.hotel_info import *
import random

responses = {

    # 1. Book Hotel (Replacing book_room)
    "book_hotel": [
        "I'd be happy to help you book a room. What are your check-in and check-out dates?",
        "Sure! Which room type would you like to reserve?",
        "Let's make your reservation. May I know your preferred dates?"
    ],

    # 2. Cancel Hotel Reservation (Replacing cancel_booking)
    "cancel_hotel_reservation": [
        "I'm sorry to hear that. Please provide your booking ID to cancel your reservation.",
        "Sure! I can help you cancel your booking. May I have your booking number?",
        "Please share your reservation ID so I can process your cancellation."
    ],

    # 3. Change Hotel Reservation (Replacing modify_booking)
    "change_hotel_reservation": [
        "Sure! Please provide your booking ID so I can help modify your reservation.",
        "I'd be happy to update your booking. What would you like to change?",
        "Please tell me your booking details and the changes you would like to make."
    ],

    # 4. Check Hotel Reservation (Replacing booking_status)
    "check_hotel_reservation": [
        "Please provide your booking ID so I can check your reservation status.",
        "I'd be happy to help. What is your booking reference number?",
        "Kindly provide your booking number for verification."
    ],

    # 5. Check Hotel Prices (Replacing room_price)
    "check_hotel_prices": [
        "Our Standard Room starts from RM180 per night.",
        "Our room rates range from RM180 to RM780 per night depending on the room type.",
        "Which room type would you like to know the price for?",
        "Our room rates are:\n" + "\n".join(f"- {room}: {price}" for room, price in ROOM_PRICES.items())
    ],

    # 6. Check Hotel Facilities (Replacing hotel_facilities)
    "check_hotel_facilities": [
        f"Our resort offers: {', '.join(FACILITIES)}.",
        "Guests can enjoy our swimming pool, fitness centre, spa, and beach access.",
        "We provide top facilities including free WiFi, outdoor pool, spa and restaurant."
    ],

    # 7. Check Hotel Offers
    "check_hotel_offers": [
        f"We currently offer special packages for weekend stays and seasonal discounts at {HOTEL_NAME}!",
        "Enjoy up to 20% off when booking directly with BookMate!",
        "Check our seasonal promotions for free breakfast upgrades and spa vouchers!"
    ],

    # 8. Check In (Replacing checkin_checkout)
    "check_in": [
        f"Check-in starts at {CHECK_IN}. Early check-in is subject to availability.",
        f"Our standard check-in time is {CHECK_IN} at the main reception.",
        f"You can check in starting from {CHECK_IN}. Please have your IC or Passport ready!"
    ],

    # 9. Check Out
    "check_out": [
        f"Check-out is before {CHECK_OUT}. Late check-out can be requested at the reception.",
        f"Our standard check-out time is {CHECK_OUT}.",
        f"Please check out by {CHECK_OUT} on your departure date."
    ],
    "checkin_checkout": [
        f"Check-in starts at {CHECK_IN} and check-out is before {CHECK_OUT}.",
        f"Our standard check-in time is {CHECK_IN}, while check-out is at {CHECK_OUT}."
    ],

    # 10. Book Parking Space (Replacing parking)
    "book_parking_space": [
        "Free parking is available for all hotel guests.",
        "Yes! We provide complimentary parking throughout your stay.",
        f"Parking is free and available 24 hours. {PARKING}"
    ],

    # 11. Bring Pets
    "bring_pets": [
        f"Pets are welcome in designated pet-friendly rooms at {HOTEL_NAME}. Please inform us in advance!",
        "We allow small pets in specific suites. Additional cleaning fees may apply."
    ],

    # 12. Check Menu (Replacing breakfast)
    "check_menu": [
        f"Breakfast is served daily from {BREAKFAST_TIME}. We offer buffet breakfast, room service, and dinner menus.",
        "Breakfast is included for Deluxe Room, Family Suite and Ocean Villa.",
        "Our restaurant serves local and international dining options all day."
    ],

    # 13. Invoices
    "invoices": [
        "You can view and download your invoices from your booking confirmation or request a copy at checkout.",
        "To obtain your invoice, please provide your booking ID and email address.",
        "Our front desk team can issue an official tax invoice upon check-out."
    ],

    # 14. Cancellation Fees
    "cancellation_fees": [
        "Cancellations made at least 48 hours prior to check-in are completely free of charge.",
        "Free cancellation is available up to 2 days before arrival. Late cancellations may incur a 1-night room charge."
    ],

    # 15. Customer Service
    "customer_service": [
        f"Our customer support team is available 24/7. You can contact us at {PHONE} or email {EMAIL}.",
        f"Feel free to reach out to us at {PHONE} or email {EMAIL} for any assistance!"
    ],

    # 16. Human Agent
    "human_agent": [
        f"Connecting you to a human agent... You can also call our front desk directly at {PHONE}.",
        f"A customer service agent will assist you shortly. Phone: {PHONE}."
    ],

    # 17. Host Event
    "host_event": [
        f"{HOTEL_NAME} offers banquet halls and meeting rooms for events, weddings, and conferences. Contact us at {EMAIL}!",
        "We provide event spaces equipped with audiovisual gear for meetings and private celebrations."
    ],

    # 18. File Complaint
    "file_complaint": [
        "We apologize for any dissatisfaction. Please share your feedback or booking ID so our manager can assist immediately.",
        f"We take complaints seriously. Please detail your issue or email our management directly at {EMAIL}."
    ],

    # 19. Leave Review
    "leave_review": [
        "Thank you for your feedback! You can leave a review on our website or Google review page.",
        f"We appreciate your stay at {HOTEL_NAME}! We'd love to hear your review of our service."
    ],

    # 20. Store Luggage
    "store_luggage": [
        "Yes! Complimentary luggage storage is available at reception before check-in or after check-out.",
        "We offer free 24/7 luggage storage at the front desk for our guests."
    ],

    # 21. Add Night
    "add_night": [
        "I'd be happy to help extend your stay! Please provide your booking ID and how many extra nights you'd like to add.",
        "To extend your stay, please share your reservation ID."
    ],

    # 22. Redeem Points
    "redeem_points": [
        "You can redeem your loyalty reward points for room upgrades, free breakfast, or discount vouchers during booking.",
        "BookMate rewards points can be applied toward your current or future reservations."
    ],

    # 23. Get Refund
    "get_refund": [
        "Refunds for eligible cancellations are processed within 5-7 business days to your original payment method.",
        "Please provide your booking ID so our finance team can verify and issue your refund."
    ],

    # 24. Shuttle Service
    "shuttle_service": [
        f"{HOTEL_NAME} provides airport shuttle and local transport services upon request. Please arrange with reception in advance.",
        "We offer hourly complimentary shuttle buses to nearby attractions and shopping districts."
    ],

    # 25. Search Hotel
    "search_hotel": [
        f"Welcome to {HOTEL_NAME}! We are located at {ADDRESS}, offering standard rooms, deluxe suites, and ocean villas.",
        f"Looking for a relaxing stay? {HOTEL_NAME} offers beachfront luxury and top-tier hospitality."
    ],

    # Legacy & General Intents
    "greeting": [
        f"Hello! Welcome to {HOTEL_NAME}. How may I assist you today?",
        "Hi! I'm BookMate. How can I help you?",
        f"Welcome to {HOTEL_NAME}! Feel free to ask me anything."
    ],
    "goodbye": [
        f"Thank you for choosing {HOTEL_NAME}. Have a wonderful day!",
        f"Goodbye! We hope to welcome you to {HOTEL_NAME} soon.",
        "Take care and have a pleasant day!"
    ],
    "location": [
        f"{HOTEL_NAME} is located at {ADDRESS}.",
        f"You can visit us at {ADDRESS}."
    ],
    "contact": [
        f"You can contact us at {PHONE} or email us at {EMAIL}.",
        f"Feel free to call us at {PHONE} or email us at {EMAIL} for assistance."
    ],
    "payment": [
        f"We accept {', '.join(PAYMENT_METHODS)}.",
        "You can pay using credit card, online banking or DuitNow QR."
    ],
    "availability": [
        "Sure! Please tell me your check-in and check-out dates.",
        "I'd be happy to check room availability for you. When would you like to stay?"
    ]
}


HUMAN_INTENT_NAMES = {
    "book_hotel": "🏨 Book a Room",
    "cancel_hotel_reservation": "❌ Cancel Reservation",
    "change_hotel_reservation": "✏️ Modify Reservation",
    "check_hotel_reservation": "🔍 Check Booking Status",
    "check_hotel_prices": "💰 Room Prices & Rates",
    "check_hotel_facilities": "🏊 Resort Facilities",
    "check_hotel_offers": "🎁 Special Offers & Discounts",
    "check_in": "⏰ Check-in Time",
    "check_out": "⏰ Check-out Time",
    "checkin_checkout": "⏰ Check-in / Check-out Times",
    "book_parking_space": "🚗 Free Parking Info",
    "bring_pets": "🐾 Pet Policy",
    "check_menu": "🍳 Breakfast & Dining",
    "invoices": "🧾 Invoices & Receipts",
    "cancellation_fees": "💳 Cancellation Policy",
    "customer_service": "📞 Customer Support",
    "human_agent": "🎧 Connect to Human Agent",
    "host_event": "🎉 Event & Banquet Halls",
    "file_complaint": "💬 Complaints & Feedback",
    "leave_review": "⭐ Reviews & Ratings",
    "store_luggage": "🧳 Luggage Storage",
    "add_night": "📅 Extend Stay / Add Night",
    "redeem_points": "🎁 Reward Points",
    "get_refund": "💵 Refund Status",
    "shuttle_service": "🚌 Shuttle Transport",
    "search_hotel": "🏨 Hotel Overview",
    "greeting": "👋 Greeting",
    "goodbye": "👋 Goodbye",
    "location": "📍 Location & Address",
    "contact": "📞 Contact Details",
    "payment": "💳 Payment Options",
    "availability": "📅 Check Availability"
}


def generate_fallback_response(user_query, predictor, confidence=0.0):
    """
    Generate dynamic Top-K candidate recommendation or general help menu for low confidence input.
    """
    if confidence < 0.20:
        return (
            "Sorry, I am not quite sure what you mean.\n\n"
            "BookMate currently assists with hotel services. You can ask about:\n"
            "• 🏨 **Room Booking & Reservations**\n"
            "• 💰 **Room Prices & Availability**\n"
            "• 🏊 **Resort Facilities & Breakfast**\n"
            "• ⏰ **Check-in / Check-out Times**\n"
            "• 🚗 **Free Parking & Contact Details**"
        )
    
    top_result = predictor.predict_top(user_query, top_k=3)
    top_predictions = top_result.get("top_predictions", [])

    candidates = []
    for item in top_predictions:
        intent_tag = item["intent"]
        display_name = HUMAN_INTENT_NAMES.get(intent_tag, intent_tag.replace("_", " ").title())
        conf_pct = item["confidence"]
        candidates.append(f"• **{display_name}** ({conf_pct:.1%})")

    if candidates:
        return (
            "Sorry, I am not quite sure what you mean. Were you looking for one of these options?\n\n"
            + "\n".join(candidates) + "\n\n"
            "Please feel free to rephrase your question or select a topic above!"
        )

    return (
        "Sorry, I didn't quite understand that. Could you please rephrase your request?\n\n"
        "You can ask me about room booking, prices, facilities, breakfast, parking, or check-in/out times."
    )


def get_response(intent, entities=None):
    """
    Return a response based on the predicted intent and extracted entities.
    """
    if entities and isinstance(entities, dict):
        room_type = entities.get("room_type")
        if room_type and intent in ["check_hotel_prices", "room_price"]:
            if room_type in ROOM_PRICES:
                return f"The rate for **{room_type}** is **{ROOM_PRICES[room_type]}** per night."

    return random.choice(
        responses.get(
            intent,
            ["I'm sorry, I didn't understand your question. Could you please rephrase it?"]
        )
    )

