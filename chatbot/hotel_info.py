"""
BookMate Chatbot
Hotel Information & Knowledge Base

Purpose:
Single Source of Truth (SSOT) storing all hotel constants, rich room metadata,
facility operating hours, and booking policies.
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

# Detailed Facility Operating Hours and Highlights
FACILITY_DETAILS = {
    "Swimming Pool": {
        "hours": "7:00 AM - 9:00 PM",
        "description": "Infinity beachfront swimming pool with sun loungers and towel service."
    },
    "Gym": {
        "hours": "24 Hours (Keycard access)",
        "description": "Modern fitness centre equipped with cardio machines, free weights, and yoga mats."
    },
    "Spa": {
        "hours": "10:00 AM - 8:00 PM",
        "description": "Traditional Malay and aromatherapy wellness massages with ocean views."
    },
    "Restaurant": {
        "hours": "6:30 AM - 10:30 PM",
        "description": "The Orient Bistro serving local Malaysian favorites and international buffets."
    },
    "Free WiFi": {
        "hours": "24 Hours",
        "description": "High-speed complimentary Wi-Fi throughout resort rooms and public areas (SSID: OrientedResort_Guest)."
    },
    "Beach Access": {
        "hours": "All Day",
        "description": "Direct private access to Pantai Cenang beachfront with complimentary beach umbrellas."
    },
    "Kids Playground": {
        "hours": "8:00 AM - 7:00 PM",
        "description": "Safe indoor & outdoor playground area with fun activities for children."
    }
}

# Rich Room Specifications & Metadata
ROOM_DETAILS = {
    "Standard Room": {
        "price": "RM180",
        "price_num": 180,
        "capacity": "2 adults",
        "max_guests": 2,
        "bed": "1 Queen Bed",
        "view": "Lush Garden View",
        "size": "28 sqm",
        "features": ["Free WiFi", "Air Conditioning", "En-suite Bathroom", "Coffee/Tea Maker", "Smart TV"]
    },
    "Deluxe Room": {
        "price": "RM280",
        "price_num": 280,
        "capacity": "2 adults + 1 child",
        "max_guests": 3,
        "bed": "1 King Bed or 2 Single Beds",
        "view": "Panoramic Sea / Pool View",
        "size": "38 sqm",
        "features": ["Free Buffet Breakfast", "Private Balcony", "Bathtub", "Mini Bar", "Work Desk"]
    },
    "Family Suite": {
        "price": "RM450",
        "price_num": 450,
        "capacity": "4-5 guests",
        "max_guests": 5,
        "bed": "2 Queen Beds + 1 Sofa Bed",
        "view": "Stunning Ocean View",
        "size": "65 sqm",
        "features": ["Free Buffet Breakfast", "Spacious Living Room", "Kitchenette", "2 Bathrooms", "Dining Area"]
    },
    "Ocean Villa": {
        "price": "RM780",
        "price_num": 780,
        "capacity": "4 guests",
        "max_guests": 4,
        "bed": "1 King Bed + 2 Twin Beds",
        "view": "Direct Private Beachfront",
        "size": "110 sqm",
        "features": ["Private Plunge Pool", "Free Breakfast & High Tea", "Dedicated Butler Service", "Direct Beach Walkway", "Jacuzzi"]
    }
}

ROOM_PRICES = {
    room: details["price"]
    for room, details in ROOM_DETAILS.items()
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

# Hotel Business Policies
HOTEL_POLICIES = {
    "cancellation_window_hours": 48,
    "cancellation_policy": "Free cancellation up to 48 hours before check-in date.",
    "late_cancellation_fee": "1-night room charge for cancellations within 48 hours of arrival.",
    "refund_timeframe": "5-7 business days to the original payment method.",
    "sst_tax_rate": 0.06,
    "pet_policy": "Pets welcome in designated Ocean Villas (under 10kg, pre-registration required).",
    "shuttle_service": "Complimentary hourly shuttle buses to Langkawi Airport (LGK) and Cenang Mall.",
    "luggage_storage": "Free 24/7 luggage storage at the front desk concierge.",
    "early_checkin": "Subject to room availability upon arrival; early check-in before 12:00 PM may incur RM50 fee."
}