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

# Diverse conversational greetings & bot capability queries
greetings = [
    "Hello BookMate", "Hi there", "Good morning", "Good afternoon", "Good evening",
    "Hey BookMate", "Greetings", "Is anyone online?", "Can someone help me?", "I need assistance",
    "Nice to meet you", "Hello there", "Hi BookMate chatbot", "Good day", "Morning BookMate",
    "Evening BookMate", "Hi, I have a question", "Hello, can I ask something?", "Hey, need help with hotel",
    "Hi BookMate team", "Good day BookMate", "Hello, is this resort support?", "Hi, are you active?",
    "Hey there chatbot", "Hello, I want to inquire about resort",
    # Capability & Help Queries
    "what can you do?", "what can u do?", "what can you do for me?", "what services do you offer?",
    "what features do you have?", "how can you help me?", "help me", "what are your capabilities?",
    "what options do you have?", "show me what you can do", "how to use this chatbot?", "help",
    "what can i ask you?", "what services are supported?"
]
for g in greetings:
    intents_data.append({"text": g, "intent": "greeting"})

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

    # 2. Book Room
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
booking_records = []
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
