import pandas as pd
import json
import random
import os

print("=== Step 1: Loading Kaggle Dataset (hotel_booking.csv) ===")
kaggle_path = "dataset/hotel_booking.csv"
if not os.path.exists(kaggle_path):
    print(f"Error: Kaggle dataset not found at {kaggle_path}")
    exit(1)

df = pd.read_csv(kaggle_path)
print(f"Loaded Kaggle dataset with {len(df)} rows and {len(df.columns)} columns.")

intents_data = []

# 1. Greetings
greetings = [
    "Hello BookMate", "Hi there", "Good morning", "Good afternoon", "Good evening",
    "Hey BookMate", "Greetings", "Is anyone online?", "Hi, I have a question",
    "Hello, can I ask something?", "Hey, need help with hotel", "Hi BookMate team",
    "Good day BookMate", "Hello, is this resort support?", "Hi, are you active?",
    "Hey there chatbot", "Hello, I want to inquire about resort"
]
for g in greetings:
    intents_data.append({"text": g, "intent": "greeting"})

# 2. Help & Capability Queries
help_phrases = [
    "what can you do?", "what can u do?", "what can you do for me?", "what services do you offer?",
    "what features do you have?", "how can you help me?", "help me", "what are your capabilities?",
    "what options do you have?", "show me what you can do", "how to use this chatbot?", "help",
    "what can i ask you?", "what services are supported?", "how can you assist me?",
    "what can bookmate do?", "list your services", "what can you help me with?"
]
for hp in help_phrases:
    intents_data.append({"text": hp, "intent": "help"})

# 2. Book Room Phrases
book_room_phrases = [
    "i want booking", "i want to book", "i want to book a room", "book a room",
    "book room", "how to book?", "how to book a room?", "make a booking",
    "make a reservation", "reserve a room", "can I book a room?", "I'd like to book a room",
    "want to reserve a room", "I need to book a room", "room reservation", "book now",
    "I want a room", "book a stay", "reserve room", "new booking", "start booking",
    "I would like to make a booking", "how do I make a reservation?"
]
for b in book_room_phrases:
    intents_data.append({"text": b, "intent": "book_room"})

# 3. Hotel Facilities Phrases
facility_phrases = [
    "what is your resort facilities", "what are your resort facilities", "resort facilities",
    "hotel facilities", "what facilities do you have?", "what facilities are available?",
    "facilities", "what amenities do you offer?", "do you have a swimming pool?",
    "do you have a gym?", "do you have spa?", "what amenities do you have?",
    "resort amenities", "hotel amenities", "tell me about your facilities",
    "facilities available", "resort features",
    "need to book for the facilities?", "do I need to book for facilities?",
    "do I need to book for the facilities?", "need to book for facilities?",
    "is booking required for facilities?", "do I need reservation to use facilities?",
    "how to use resort facilities?", "do I need booking to use pool or gym?"
]
for f in facility_phrases:
    intents_data.append({"text": f, "intent": "hotel_facilities"})

# 4. Room Price & Recommendation Phrases
price_phrases = [
    "room price", "room rates", "how much per night?", "what are the prices?",
    "price list", "cost of room", "how much does it cost?", "room pricing",
    "what is the room rate?", "hotel rates", "daily rate",
    "if i family 4 members suitable book which room?", "which room suitable for family of 4?",
    "family 4 members book which room?", "what room for 4 people?", "room recommendation for 4 guests",
    "maximum guests per room", "how many people can stay in standard room?", "how many people per room?",
    "room capacity", "max capacity", "which room is recommended for 4 people?", "room size and capacity",
    "which room should I book for 4 guests?", "suitable room for 4 guests"
]
for p in price_phrases:
    intents_data.append({"text": p, "intent": "room_price"})

# 5. Availability Phrases
avail_phrases = [
    "room availability", "check availability", "are there available rooms?",
    "do you have open rooms?", "is there any room available?", "check room availability",
    "available rooms"
]
for a in avail_phrases:
    intents_data.append({"text": a, "intent": "availability"})

# 6. Breakfast Phrases
breakfast_phrases = [
    "breakfast included?", "is breakfast available?", "food options", "meal options",
    "does it include breakfast?", "breakfast package", "dining options",
    "lunch provided?", "dinner provided?", "is lunch available?", "is dinner available?",
    "do you serve lunch?", "do you serve dinner?", "is lunch included?", "is dinner included?",
    "lunch", "dinner", "lunch and dinner", "meals provided?", "do you serve food?",
    "dinner got?", "got dinner?", "lunch got?", "got lunch?", "do you provide lunch?",
    "do you provide dinner?", "why not provide lunch?", "why no lunch?", "why no dinner?",
    "why not serve lunch?", "is there lunch?", "is there dinner?", "lunch service", "dinner service"
]
for bf in breakfast_phrases:
    intents_data.append({"text": bf, "intent": "breakfast"})

# 7. Parking Phrases
parking_phrases = [
    "parking available?", "is parking free?", "car parking", "do you provide parking?",
    "parking space", "parking lot", "parking can how many cars?", "how many cars can park?",
    "how many parking spaces?", "how many vehicles can park?", "parking capacity",
    "can I park 2 cars?", "parking space for cars", "how many cars allowed?"
]
for pk in parking_phrases:
    intents_data.append({"text": pk, "intent": "parking"})

# 8. Checkin / Checkout Phrases
checkin_phrases = [
    "check in time", "check out time", "when is checkin?", "checkin and checkout times",
    "what is the check in time?", "what time is check out?", "early check in",
    "how to check in?", "how do I check in?", "how to check in", "check in process",
    "check in procedure", "what is the check in process?", "how to check out?",
    "how do I check out?", "check out procedure", "check-in instructions", "where to check in?"
]
for ci in checkin_phrases:
    intents_data.append({"text": ci, "intent": "checkin_checkout"})

# 9. Payment Phrases
payment_phrases = [
    "how to make payment", "how to make a payment", "make payment", "payment",
    "payment methods", "how to pay?", "do you accept credit card?", "payment options",
    "how can I pay?", "deposit requirement", "payment process", "how do I make a payment?",
    "how can I make payment?", "how to pay for booking", "what are the payment methods?",
    "accepted payment methods"
]
for pay in payment_phrases:
    intents_data.append({"text": pay, "intent": "payment"})

# 10. Contact Phrases
contact_phrases = [
    "how to contact?", "contact number", "phone number", "email address",
    "how to contact you?", "contact details", "customer service number",
    "call resort", "hotel contact", "phone contact", "email contact",
    "how to reach you?", "how can I contact the hotel?", "contact info",
    "contact us"
]
for c in contact_phrases:
    intents_data.append({"text": c, "intent": "contact"})

# 12. Location Phrases
location_phrases = [
    "where are you located?", "hotel address", "resort location", "where is the hotel?",
    "location of resort", "address of hotel", "how to get there?", "where is oriented resort?",
    "location", "address"
]
for l in location_phrases:
    intents_data.append({"text": l, "intent": "location"})

# 13. Modify Booking Phrases
modify_booking_phrases = [
    "can i change the room type for alice?", "can i change room type?", "can i change the room type?",
    "change room type", "modify room", "modify booking", "change my room", "change my booking",
    "update booking", "change room type for booking", "change room category", "can I modify my booking?",
    "how to change my reservation?", "edit booking details", "change reservation dates",
    "change guest name", "modify reservation"
]
for mb in modify_booking_phrases:
    intents_data.append({"text": mb, "intent": "modify_booking"})

# 14. Booking Status & Forgot ID/Number Phrases
status_phrases = [
    "i forgot my booking id", "forgot booking id", "don't know booking id", "no booking id",
    "i forgot my reservation id", "forgot reservation id", "don't have reservation id",
    "i forgot my booking number", "forgot my booking number", "don't have booking number",
    "don't know my booking number", "no booking number", "how to check without booking number?",
    "check booking by name and check in date", "find reservation for alice", "alice check in date 25/08/2026",
    "check booking status for guest alice", "look up booking for alice", "find my booking by name",
    "check reservation by name", "search reservation for alice", "help me check my booking",
    "help check booking", "can you check my booking?", "check my reservation status"
]
for sp in status_phrases:
    intents_data.append({"text": sp, "intent": "booking_status"})

# 15. Cancel Booking Phrases
cancel_phrases = [
    "i want cancel my booking", "i want to cancel my booking", "i want cancel booking",
    "cancel my booking", "i want to cancel", "cancel booking", "how to cancel booking",
    "cancel reservation", "i need to cancel my booking", "i would like to cancel",
    "can i cancel booking?", "how do i cancel my booking?", "cancel room"
]
for c in cancel_phrases:
    intents_data.append({"text": c, "intent": "cancel_booking"})

# Sample rows to inject Kaggle attributes into intent questions
sample_rows = df.sample(n=300, random_state=42)

for idx, r in sample_rows.iterrows():
    hotel_name = r['hotel']
    room = r['reserved_room_type']
    adults = int(r['adults'])
    children = int(r['children']) if pd.notnull(r['children']) else 0
    month = r['arrival_date_month']
    year = r['arrival_date_year']
    day = r['arrival_date_day_of_month']
    nights = int(r['stays_in_week_nights']) + int(r['stays_in_weekend_nights'])
    meal_code = r['meal']
    cust_name = r['name']
    deposit = r['deposit_type']
    status = r['reservation_status']
    parking = int(r['required_car_parking_spaces'])
    cust_type = r['customer_type']
    adr = r['adr']
    phone = r['phone-number']

    # 1. Availability
    intents_data.append({"text": f"Is Room Type {room} available at {hotel_name} for {cust_name}?", "intent": "availability"})
    intents_data.append({"text": f"Do you have room availability for {adults} adults in {month} {year}?", "intent": "availability"})

    # 2. Book Room (Kaggle parameters)
    intents_data.append({"text": f"I want to book Room Type {room} for {adults} guests at {hotel_name}", "intent": "book_room"})
    intents_data.append({"text": f"Can I reserve a room for {cust_name} for {nights} nights?", "intent": "book_room"})

    # 3. Room Price
    intents_data.append({"text": f"How much is Room Type {room} at rate {adr} per night?", "intent": "room_price"})
    intents_data.append({"text": f"What is the average daily rate for {hotel_name} in {month}?", "intent": "room_price"})

    # 4. Breakfast
    intents_data.append({"text": f"Is {meal_code} meal package included for {cust_name}'s room?", "intent": "breakfast"})
    intents_data.append({"text": f"What breakfast package is available for Room Type {room}?", "intent": "breakfast"})

    # 5. Parking
    intents_data.append({"text": f"Do you provide {parking} parking space for reservation under {cust_name}?", "intent": "parking"})
    intents_data.append({"text": f"Is parking space available at {hotel_name} for guest {phone}?", "intent": "parking"})

    # 6. Cancellation
    intents_data.append({"text": f"Can I cancel my booking under {cust_name}?", "intent": "cancel_booking"})
    intents_data.append({"text": f"What is the cancellation policy for {deposit} reservation at {hotel_name}?", "intent": "cancel_booking"})

    # 7. Booking Status
    intents_data.append({"text": f"Check reservation status for {cust_name} with status {status}", "intent": "booking_status"})
    intents_data.append({"text": f"Is booking status for {cust_name} confirmed at {hotel_name}?", "intent": "booking_status"})

    # 8. Payment
    intents_data.append({"text": f"What payment methods do you accept for {deposit} booking under {cust_name}?", "intent": "payment"})
    intents_data.append({"text": f"Do I need to pay deposit {deposit} for Room Type {room}?", "intent": "payment"})

    # 9. Hotel Facilities
    intents_data.append({"text": f"What facilities are available at {hotel_name} for {cust_type} guests?", "intent": "hotel_facilities"})
    intents_data.append({"text": f"Does {hotel_name} feature a swimming pool, spa and gym for {cust_name}?", "intent": "hotel_facilities"})

    # 10. Check-in / Check-out
    intents_data.append({"text": f"What are the check-in times at {hotel_name} for arrival on {day} {month}?", "intent": "checkin_checkout"})
    intents_data.append({"text": f"Can guest {cust_name} request early check-in or late check-out at {hotel_name}?", "intent": "checkin_checkout"})

    # 11. Modify Booking
    intents_data.append({"text": f"I want to modify booking dates for guest {cust_name} at {hotel_name}", "intent": "modify_booking"})
    intents_data.append({"text": f"Can I change my reservation from {nights} nights for {cust_name}?", "intent": "modify_booking"})

# Convert to DataFrame & deduplicate
intents_df = pd.DataFrame(intents_data).drop_duplicates(subset=["text"])

print(f"\n=== Step 2: Generated {len(intents_df)} Kaggle-derived intent samples ===")
print("Intents distribution:")
print(intents_df['intent'].value_counts())

# Overwrite dataset/intents.csv completely
intents_df.to_csv("dataset/intents.csv", index=False)
print("\n✅ Successfully replaced dataset/intents.csv with Kaggle-derived dataset!")

# Step 3: Populate data/booking.json with Kaggle booking records
booking_records = [
    {
        "booking_id": "BK1005",
        "name": "Alice",
        "email": "alice@gmail.com",
        "room": "Family Suite",
        "check_in": "24/08/2026",
        "check_out": "28/08/2026",
        "guests": "4",
        "status": "Confirmed"
    }
]
for idx, r in df.head(500).iterrows():
    b_id = f"BK{1000 + idx}"
    check_in = f"{r['arrival_date_day_of_month']}/{r['arrival_date_month']}/{r['arrival_date_year']}"
    guests = str(int(r['adults'] + (r['children'] if pd.notnull(r['children']) else 0)))
    
    booking_records.append({
        "booking_id": b_id,
        "name": str(r['name']),
        "email": str(r['email']),
        "room": f"Room Type {r['reserved_room_type']}",
        "check_in": check_in,
        "check_out": f"{r['arrival_date_month']} {r['arrival_date_year']}",
        "guests": guests,
        "status": str(r['reservation_status'])
    })

os.makedirs("data", exist_ok=True)
with open("data/booking.json", "w", encoding="utf-8") as f:
    json.dump(booking_records, f, indent=4)

print("✅ Successfully updated data/booking.json with 500 real Kaggle reservation records!")
