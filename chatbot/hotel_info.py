"""
BookMate Chatbot
Hotel Information

Purpose:
Store all hotel information in one place.
"""

HOTEL_NAME = "Oriented Resort"

PHONE = "+60 4-987 8888"
EMAIL = "booking@orientedresort.com"
ADDRESS = "Pantai Cenang, Langkawi, Kedah, Malaysia"

CHECK_IN = "3:00 PM"
CHECK_OUT = "12:00 PM"

BREAKFAST_TIME = "7:00 AM - 10:00 AM"

PARKING = "Free parking for all hotel guests."

PAYMENT_METHODS = [
    "Visa",
    "Mastercard",
    "DuitNow QR",
    "Online Banking",
    "Cash"
]

FACILITIES = [
    "Swimming Pool",
    "Gym",
    "Spa",
    "Restaurant",
    "Free WiFi",
    "Beach Access",
    "Kids Playground"
]

ROOM_PRICES = {
    "Standard Room": "RM180",
    "Deluxe Room": "RM280",
    "Family Suite": "RM450",
    "Ocean Villa": "RM780"
}

# Official room types
ROOM_TYPES = list(ROOM_PRICES.keys())

# Common user expressions → official room type
ROOM_TYPE_ALIASES = {
    "standard": "Standard Room",
    "standard room": "Standard Room",

    "deluxe": "Deluxe Room",
    "deluxe room": "Deluxe Room",

    "family": "Family Suite",
    "family suite": "Family Suite",

    "suite": "Family Suite",

    "ocean": "Ocean Villa",
    "ocean villa": "Ocean Villa",

    "villa": "Ocean Villa"
}