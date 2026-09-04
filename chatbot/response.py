"""
BookMate Chatbot
Response Generator & Knowledge Binding

Purpose:
Generate context-aware responses bound to hotel_info knowledge base
(supporting all 25 Bitext Hospitality intents, rich markdown formatting, and dynamic slot filling).
"""

from chatbot.hotel_info import (
    HOTEL_NAME, PHONE, EMAIL, ADDRESS,
    CHECK_IN, CHECK_OUT, BREAKFAST_TIME, PARKING,
    PAYMENT_METHODS, FACILITIES, FACILITY_DETAILS,
    ROOM_DETAILS, ROOM_PRICES, ROOM_TYPES, HOTEL_POLICIES
)
import random

def get_room_price_table() -> str:
    """Generate a clean Markdown price & accommodation table."""
    table = (
        f"🏨 **{HOTEL_NAME} Room Rates & Accommodations:**\n\n"
        "| Room Type | Rate / Night | Capacity | Bed & View |\n"
        "| :--- | :---: | :---: | :--- |\n"
    )
    for room, d in ROOM_DETAILS.items():
        table += f"| **{room}** | **{d['price']}** | {d['capacity']} | {d['bed']} ({d['view']}) |\n"
    table += "\n*Rates exclude 6% SST. All rooms include complimentary high-speed WiFi.*"
    return table


def get_single_room_card(room_name: str) -> str:
    """Generate a detailed markdown specification card for a specific room."""
    d = ROOM_DETAILS.get(room_name)
    if not d:
        return f"The rate for **{room_name}** is **{ROOM_PRICES.get(room_name, 'available on request')}**."

    features_str = ", ".join(d.get("features", []))
    return (
        f"🛏️ **{room_name} Details:**\n\n"
        f"- **Rate**     : **{d['price']}** / night (excl. 6% SST)\n"
        f"- **Capacity** : {d['capacity']} (Size: {d.get('size', 'N/A')})\n"
        f"- **Bedding**  : {d['bed']}\n"
        f"- **View**     : {d['view']}\n"
        f"- **Features** : {features_str}"
    )


def get_facilities_summary() -> str:
    """Generate a summary of all resort facilities and their operating hours."""
    lines = [f"🏊 **{HOTEL_NAME} Resort Facilities & Operating Hours:**\n"]
    for name, d in FACILITY_DETAILS.items():
        lines.append(f"- **{name}** (`{d['hours']}`): {d['description']}")
    return "\n".join(lines)


responses = {

    # 1. Book Hotel
    "book_hotel": [
        "I'd be happy to help you book a room. What are your check-in and check-out dates?",
        "Sure! Which room type would you like to reserve? We offer Standard Room, Deluxe Room, Family Suite, and Ocean Villa.",
        "Let's make your reservation. May I have your name and preferred stay dates?"
    ],

    # 2. Cancel Hotel Reservation
    "cancel_hotel_reservation": [
        "I can certainly help you cancel your reservation. Please provide your **Booking ID** (e.g. BK1234) so I can verify your booking.",
        "Sure! I can help you process your cancellation. May I have your booking reference number?",
        "Please share your Booking ID so we can verify your booking and process your cancellation."
    ],

    # 3. Change Hotel Reservation
    "change_hotel_reservation": [
        f"✏️ **Need to modify your reservation dates or room?**\n\n"
        f"Due to room availability adjustments and daily rate differences, stay modifications are handled directly by our reservations desk:\n"
        f"- 📞 **Reservations Hotline**: **{PHONE}**\n"
        f"- ✉️ **Email Support**: **{EMAIL}**\n\n"
        f"💡 *Tip: Because **{HOTEL_NAME}** offers free cancellation up to {HOTEL_POLICIES['cancellation_window_hours']} hours before check-in, you can also cancel your current reservation here anytime and make a fresh booking with your updated dates!*",
        f"To change your stay dates or room category, please contact our 24-hour reservations team at **{PHONE}** or email **{EMAIL}**.\n\n"
        f"Alternatively, you can cancel your existing booking for free and reserve your preferred room again!"
    ],

    # 4. Check Hotel Reservation
    "check_hotel_reservation": [
        "Please provide your **Booking ID** so I can check your current reservation status and details.",
        "I'd be happy to look up your reservation. What is your booking reference number?",
        "Kindly provide your Booking ID for status verification."
    ],

    # 5. Check Hotel Prices
    "check_hotel_prices": [
        get_room_price_table()
    ],

    # 6. Check Hotel Facilities
    "check_hotel_facilities": [
        get_facilities_summary()
    ],

    # 7. Check Hotel Offers
    "check_hotel_offers": [
        f"🎁 We currently offer seasonal packages and direct booking perks at {HOTEL_NAME}!\n\n"
        "- **Direct Booking Bonus**: Enjoy complimentary buffet breakfast upgrade.\n"
        "- **Long Stay Offer**: Stay 4 nights or more and receive RM100 spa voucher.\n"
        "- **Early Bird Discount**: Book 14 days in advance for 15% off regular room rates."
    ],

    # 8. Check In
    "check_in": [
        f"Check-in starts at **{CHECK_IN}**. {HOTEL_POLICIES['early_checkin']}",
        f"Our standard check-in time is **{CHECK_IN}** at the main reception lobby. Please have your ID / Passport ready.",
        f"You can check in starting from **{CHECK_IN}**. If you arrive early, complimentary luggage storage is available!"
    ],

    # 9. Check Out
    "check_out": [
        f"Check-out is before **{CHECK_OUT}**. Late check-out requests can be made at the reception subject to availability.",
        f"Our standard check-out time is **{CHECK_OUT}**.",
        f"Please check out by **{CHECK_OUT}** on your departure date."
    ],

    # 10. Book Parking Space
    "book_parking_space": [
        f"🚗 {PARKING} We have 24-hour secured on-site parking with EV charging stations available.",
        "Yes! Complimentary parking is provided for all registered guests throughout their stay."
    ],

    # 11. Bring Pets
    "bring_pets": [
        f"🐾 {HOTEL_POLICIES['pet_policy']}",
        "We welcome small pets in designated pet-friendly Ocean Villas! Please inform our front desk prior to arrival."
    ],

    # 12. Check Menu
    "check_menu": [
        f"🍳 Breakfast is served daily from **{BREAKFAST_TIME}** at The Orient Bistro. We offer buffet spreads, a la carte dining, and 24-hour room service.",
        "Breakfast is complimentary for Deluxe Room, Family Suite, and Ocean Villa guests.",
        "Our restaurant offers authentic Malaysian specialties, fresh seafood, and Western favorites from 6:30 AM to 10:30 PM."
    ],

    # 13. Invoices
    "invoices": [
        "🧾 You can view and download official invoices with your Booking ID, or request a printed tax invoice upon check-out.",
        "To obtain your invoice details, please provide your **Booking ID**."
    ],

    # 14. Cancellation Fees
    "cancellation_fees": [
        f"💳 **Cancellation Policy:**\n- {HOTEL_POLICIES['cancellation_policy']}\n- {HOTEL_POLICIES['late_cancellation_fee']}",
        f"Free cancellation is available up to {HOTEL_POLICIES['cancellation_window_hours']} hours before check-in. Late cancellations incur a 1-night charge."
    ],

    # 15. Customer Service
    "customer_service": [
        f"📞 Our guest support team is available 24/7. You can contact us at **{PHONE}** or email **{EMAIL}**.",
        f"Feel free to reach out anytime at **{PHONE}** or email **{EMAIL}** for personalized assistance!"
    ],

    # 16. Human Agent
    "human_agent": [
        f"🎧 Connecting you to our front desk team... You can also call us directly at **{PHONE}** for urgent assistance.",
        f"A guest service representative will assist you shortly. Front desk hotline: **{PHONE}**."
    ],

    # 17. Host Event
    "host_event": [
        f"🎉 **{HOTEL_NAME}** offers grand banquet halls, beachfront lawn pavilions, and meeting rooms for weddings, corporate retreats, and private parties. Contact our events team at **{EMAIL}**!"
    ],

    # 18. File Complaint
    "file_complaint": [
        f"💬 We sincerely apologize for any inconvenience. Please provide your booking details or contact management directly at **{EMAIL}** or **{PHONE}** so we can resolve this immediately.",
        f"We take guest feedback very seriously. Please let us know your concern or email **{EMAIL}**."
    ],

    # 19. Leave Review
    "leave_review": [
        f"⭐ Thank you for your feedback! We would love to hear about your experience at **{HOTEL_NAME}**. You can share your review on Google Reviews or TripAdvisor."
    ],

    # 20. Store Luggage
    "store_luggage": [
        f"🧳 {HOTEL_POLICIES['luggage_storage']}",
        "Yes! Complimentary luggage storage is available before check-in or after check-out at the reception."
    ],

    # 21. Add Night
    "add_night": [
        "📅 I'd be happy to help extend your stay! Please provide your **Booking ID** and the dates you wish to add.",
        "To extend your stay, please provide your Booking ID so I can check room availability."
    ],

    # 22. Redeem Points
    "redeem_points": [
        "🎁 You can redeem BookMate loyalty reward points for room rate discounts, breakfast upgrades, or late check-out perks during reservation."
    ],

    # 23. Get Refund
    "get_refund": [
        f"💵 Refunds for eligible cancelled reservations are processed within **{HOTEL_POLICIES['refund_timeframe']}**.",
        f"Please provide your Booking ID so our finance desk can verify your refund status ({HOTEL_POLICIES['refund_timeframe']})."
    ],

    # 24. Shuttle Service
    "shuttle_service": [
        f"🚌 **Shuttle Transport**: {HOTEL_POLICIES['shuttle_service']}. Please reserve with the concierge desk.",
        f"We offer hourly scheduled shuttle bus transfers between {HOTEL_NAME}, Langkawi Airport (LGK), and local shopping hubs."
    ],

    # 25. Search Hotel
    "search_hotel": [
        f"🏨 Welcome to **{HOTEL_NAME}**! Located at **{ADDRESS}**, offering beachfront accommodations, luxury suites, and full resort amenities.",
        f"**{HOTEL_NAME}** is a premier beach resort located at {ADDRESS} with direct sea access, infinity pool, and world-class hospitality."
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
    "search_hotel": "🏨 Hotel Overview"
}


def is_pure_greeting(text: str) -> bool:
    """Check if the user utterance is purely a greeting without other domain requests."""
    if not text:
        return False
    import re
    clean = re.sub(r"[^\w\s]", "", text.strip().lower())
    words = clean.split()
    greeting_words = {
        "hi", "hello", "hey", "helo", "hllo", "hallo", "hiii", "hiya", "howdy",
        "greetings", "good", "morning", "afternoon", "evening", "day", "there", "yo"
    }
    return len(words) >= 1 and len(words) <= 3 and all(w in greeting_words for w in words)


def has_greeting_prefix(text: str) -> bool:
    """Check if the utterance begins with a friendly greeting."""
    if not text:
        return False
    import re
    return bool(re.search(
        r"^(?:hi|hello|hey|helo|hiii|hiya|howdy|greetings|good\s+(?:morning|afternoon|evening))\b",
        text.strip().lower()
    ))


def get_greeting_response() -> str:
    """Generate a warm, friendly welcome message and feature directory."""
    return (
        f"👋 **Hello! Welcome to {HOTEL_NAME}.**\n\n"
        f"I'm **BookMate**, your personal hotel concierge. How may I assist you today?\n\n"
        f"You can ask me to:\n"
        f"- 🛏️ **Book a Room** *(e.g. 'I want to book a Deluxe Room')*\n"
        f"- 💰 **Check Room Rates & Pricing** *(e.g. 'What are your room rates?')*\n"
        f"- 🔍 **Check Booking Status & Invoices** *(e.g. 'Check booking BK1021' or by phone)*\n"
        f"- 🏊 **Explore Resort Facilities** *(e.g. 'What facilities do you have?')*\n"
        f"- 🍳 **Breakfast, Dining & Pet Policies**\n"
        f"- ❌ **Cancel or Modify Reservations**"
    )


def generate_fallback_response(user_query, predictor, confidence=0.0):
    """
    Generate single explicit intent confirmation or domain-scoped help menu.
    """
    # 1. Medium Confidence (0.28 <= confidence < 0.60): Explicit confirmation of the single highest intent
    if confidence >= 0.28:
        top_result = predictor.predict_top(user_query, top_k=1)
        top_predictions = top_result.get("top_predictions", [])

        if top_predictions:
            top_intent = top_predictions[0]["intent"]
            display_name = HUMAN_INTENT_NAMES.get(top_intent, top_intent.replace("_", " ").title())

            return (
                f"It sounds like you're asking about **{display_name}** ({confidence:.1%}). Is that correct?\n\n"
                f"If so, please let me know or feel free to rephrase your request, and I'll be glad to assist!"
            )

    # 2. Low Confidence (< 0.28 or Out-of-Domain): Explain hotel-specific scope
    return (
        f"I apologize, but I may not be able to assist with general topics or services outside **{HOTEL_NAME}**.\n\n"
        f"As the dedicated virtual concierge for **{HOTEL_NAME}**, I specialize strictly in hotel inquiries and reservations. I can help you with:\n"
        f"- 🏨 **Book, Modify & Cancel Reservations**\n"
        f"- 💰 **Room Rates & Availability Check**\n"
        f"- 🏊 **Resort Facilities & Operating Hours**\n"
        f"- 🍳 **Breakfast, Dining & Pet Policy**\n"
        f"- ⏰ **Check-in / Check-out Times & Free Parking**\n\n"
        f"Please let me know how I can assist with your stay at **{HOTEL_NAME}**!"
    )


def get_response(intent: str, entities: dict | None = None, sentiment: dict | None = None) -> str:
    """
    Return a response based on the predicted intent, dynamically bound extracted entities,
    and optional sentiment empathy awareness.
    """
    response_text = None

    if entities and isinstance(entities, dict):
        room_type = entities.get("room_type")
        guests = entities.get("guests")

        # 1. Specific Room Pricing Card
        if room_type and intent == "check_hotel_prices":
            if room_type in ROOM_DETAILS:
                response_text = get_single_room_card(room_type)

        # 2. Guest Count Recommendation
        elif guests and intent == "check_hotel_prices":
            if guests >= 4:
                response_text = (
                    f"For a group of **{guests} guests**, we highly recommend our **Family Suite** (RM450/night, sleeps up to 5) "
                    f"or **Ocean Villa** (RM780/night with private pool)!\n\n"
                    + get_room_price_table()
                )
            elif guests == 1 or guests == 2:
                response_text = (
                    f"For **{guests} guest{'s' if guests > 1 else ''}**, our **Standard Room** (RM180/night) or **Deluxe Room** (RM280/night) "
                    f"would be a great fit!\n\n"
                    + get_room_price_table()
                )

    if response_text is None:
        response_text = random.choice(
            responses.get(
                intent,
                ["I'm sorry, I didn't understand your question. Could you please rephrase it?"]
            )
        )

    # Prepend empathetic apology if user sentiment is strongly negative/frustrated
    if sentiment and sentiment.get("is_frustrated"):
        empathy_options = [
            (
                "🙏 **We sincerely apologize for your unpleasant experience.** "
                "We deeply understand your frustration and are prioritizing your request.\n\n"
            ),
            (
                "🙏 **We are truly sorry to hear about this issue.** "
                "Your satisfaction is our top priority, and we are here to help make things right.\n\n"
            ),
            (
                "🙏 **We apologize for the inconvenience and frustration caused.** "
                "Please allow us to assist and resolve this matter for you immediately.\n\n"
            )
        ]
        response_text = random.choice(empathy_options) + response_text

    return response_text
