"""
BookMate Chatbot
Dialogue Manager

Purpose:
Control multi-turn conversation flow between:
- Intent Prediction
- Entity Extraction
- Validation
- Booking Management
- Hotel Information

The Dialogue Manager decides what the chatbot should do next.
"""

import json
import os
import random
import re
import sys

# Ensure PROJECT_ROOT is in sys.path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from chatbot.hotel_info import HOTEL_NAME, PHONE, ROOM_PRICES
from chatbot.entity_extractor import extract_entities
from chatbot.validation import (
    validate_name,
    validate_checkin_date,
    validate_checkout_date,
    validate_guests,
    validate_room_type
)

BOOKING_FILE = os.path.join(PROJECT_ROOT, "data", "booking.json")



class DialogueManager:
    """
    Rule-based Dialogue Manager for BookMate hotel chatbot.

    Responsibilities:
    - Manage conversation state
    - Handle new booking flow
    - Handle booking status
    - Handle cancellation
    - Handle modification request
    - Validate booking information
    - Save and retrieve bookings
    """

    def __init__(self):
        self.state = self._create_initial_state()

    # STATE MANAGEMENT
    def _create_initial_state(self):
        """Create a fresh conversation state."""
        return {
            "active": False,
            "action": None,
            "step": 0,

            # Booking information
            "booking_id": None,
            "name": None,

            "room": None,
            "rooms": None,
            "guests": None,

            "checkin": None,
            "checkout": None,
            "nights": None,

            # Contact information
            "email": None,
            "phone": None,

            # Pending context tracking
            "pending_intent": None,
            "awaiting_slot": None
        }

    def reset(self):
        """Reset the current conversation state."""
        self.state = self._create_initial_state()

    # BOOKING STORAGE
    def _load_bookings(self):
        """Load bookings from booking.json."""
        if not os.path.exists(BOOKING_FILE):
            return []

        try:
            with open(BOOKING_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            return data if isinstance(data, list) else []

        except (json.JSONDecodeError, OSError):
            return []

    def _save_bookings(self, bookings):
        """Save bookings to booking.json."""
        directory = os.path.dirname(BOOKING_FILE)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(BOOKING_FILE, "w", encoding="utf-8") as f:
            json.dump(bookings, f, indent=4, ensure_ascii=False)

    def _save_booking(self, booking):
        """Save a new booking."""
        bookings = self._load_bookings()
        bookings.append(booking)
        self._save_bookings(bookings)

    def _get_booking(self, booking_id):
        """Find booking by booking ID."""
        if not booking_id:
            return None

        booking_id = booking_id.upper()

        for booking in self._load_bookings():
            if booking.get("booking_id", "").upper() == booking_id:
                return booking

        return None

    # BOOKING ID
    def _generate_booking_id(self):
        """Generate a unique booking ID."""
        bookings = self._load_bookings()
        existing_ids = {
            b.get("booking_id", "").upper()
            for b in bookings
        }

        while True:
            booking_id = f"BK{random.randint(1000, 9999)}"

            if booking_id not in existing_ids:
                return booking_id

    # CANCEL BOOKING
    def _cancel_booking(self, booking_id):
        """Cancel an existing booking."""
        booking_id = booking_id.upper()
        bookings = self._load_bookings()

        for booking in bookings:
            if booking.get("booking_id", "").upper() == booking_id:

                if booking.get("status") == "Cancelled":
                    return (
                        False,
                        f"Booking **{booking_id}** is already cancelled."
                    )

                booking["status"] = "Cancelled"
                self._save_bookings(bookings)

                return (
                    True,
                    f"❌ Booking **{booking_id}** has been successfully cancelled."
                )

        return (
            False,
            f"Could not find any reservation with Booking ID "
            f"**{booking_id}**."
        )

    # BOOKING STATUS
    def _format_booking_status(self, booking):
        """Format booking information for chatbot response."""

        status = booking.get("status", "Confirmed")

        emoji = "✅" if status == "Confirmed" else "❌"

        return (
            f"📋 **Booking Details ({booking.get('booking_id')})**\n\n"
            f"- **Guest Name** : {booking.get('name', 'N/A')}\n"
            f"- **Room Type**  : {booking.get('room', 'N/A')}\n"
            f"- **Check-in**   : {booking.get('check_in', 'N/A')}\n"
            f"- **Check-out**  : {booking.get('check_out', 'N/A')}\n"
            f"- **Guests**     : {booking.get('guests', 'N/A')}\n"
            f"- **Status**     : {emoji} {status}"
        )

    def _format_invoice_details(self, booking):
        """Format official tax invoice for chatbot response."""
        b_id = booking.get("booking_id", "N/A")
        name = booking.get("name", "N/A")
        room = booking.get("room", "Standard Room")
        check_in = booking.get("check_in", "N/A")
        check_out = booking.get("check_out", "N/A")
        guests = booking.get("guests", 1)
        status = booking.get("status", "Confirmed")

        rate_str = ROOM_PRICES.get(room, "RM 180")
        digits = re.findall(r"\d+", rate_str)
        rate_num = int(digits[0]) if digits else 180
        total_price = rate_num * 2
        tax = int(total_price * 0.06)
        grand_total = total_price + tax

        return (
            f"🧾 **Official Tax Invoice ({b_id})**\n\n"
            f"- **Guest Name**            : {name}\n"
            f"- **Room Reserved**         : {room}\n"
            f"- **Check-in / Check-out**  : {check_in} - {check_out}\n"
            f"- **Guests**                : {guests}\n"
            f"- **Room Rate**             : {rate_str}/night\n"
            f"- **Subtotal**              : RM{total_price}\n"
            f"- **SST Tax (6%)**          : RM{tax}\n"
            f"- **Grand Total Paid**      : **RM{grand_total}**\n"
            f"- **Payment Status**        : ✅ Paid ({status})"
        )

    # BOOKING VALIDATION
    def _apply_entity(self, entity_name, value):
        """
        Validate and save one extracted entity into conversation state.

        Returns:
            (success, error_message)
        """

        if value is None:
            return True, None

        # Name
        if entity_name == "name":
            valid, cleaned, error = validate_name(value)

            if valid:
                self.state["name"] = cleaned
                return True, None

            return False, error

        # Check-in
        if entity_name == "check_in":
            valid, cleaned, error = validate_checkin_date(value)

            if valid:
                self.state["checkin"] = cleaned
                return True, None

            return False, error

        # Check-out
        if entity_name == "check_out":

            if not self.state.get("checkin"):
                return (
                    False,
                    "⚠️ Please provide your check-in date first."
                )

            valid, cleaned, error = validate_checkout_date(
                self.state["checkin"],
                value
            )

            if valid:
                self.state["checkout"] = cleaned
                return True, None

            return False, error

        # Guests
        if entity_name == "guests":
            valid, cleaned, error = validate_guests(value)

            if valid:
                self.state["guests"] = cleaned
                return True, None

            return False, error

        # Room
        if entity_name == "room_type":
            valid, cleaned, error = validate_room_type(value)

            if valid:
                self.state["room"] = cleaned
                return True, None

            return False, error

        if entity_name == "rooms":
            self.state["rooms"] = value
            return True, None

        if entity_name == "nights":
            self.state["nights"] = value
            return True, None

        if entity_name == "email":
            self.state["email"] = value
            return True, None

        if entity_name == "phone":
            self.state["phone"] = value
            return True, None

        if entity_name == "booking_id":
            self.state["booking_id"] = value
            return True, None

        return True, None

    # APPLY ALL EXTRACTED BOOKING ENTITIES
    def _update_booking_state(self, entities):
        """
        Apply all extracted entities to current booking state.

        Important:
        Invalid entities are not silently stored.
        The first validation error is returned.
        """

        # Order matters:
        # check-in should be processed before check-out.
        ordered_entities = [
            "booking_id",
            "name",
            "check_in",
            "check_out",
            "nights",
            "guests",
            "rooms",
            "room_type",
            "email",
            "phone"
        ]

        for key in ordered_entities:

            if key not in entities:
                continue

            value = entities[key]

            if value is None:
                continue

            success, error = self._apply_entity(key, value)

            if not success:
                return error

        return None

    # NEXT BOOKING QUESTION
    def _get_missing_booking_field(self):
        """Return the first missing booking field."""
        if not self.state.get("name"):
            return "name"

        if not self.state.get("checkin"):
            return "checkin"

        if not self.state.get("checkout"):
            return "checkout"

        if not self.state.get("guests"):
            return "guests"

        if not self.state.get("room"):
            return "room"

        return None

    def _ask_next_booking_question(self):
        """Ask user for the next missing booking field."""

        field = self._get_missing_booking_field()

        if field == "name":
            self.state["step"] = 1
            return "Sure! Let me help you book a room.\n\nMay I have your name?"

        if field == "checkin":
            self.state["step"] = 2
            return "What is your check-in date?"

        if field == "checkout":
            self.state["step"] = 3
            return "What is your check-out date?"

        if field == "guests":
            self.state["step"] = 4
            return "How many guests will be staying?"

        if field == "room":
            self.state["step"] = 5
            return (
                "Which room type would you like?\n\n"
                "- Standard Room\n"
                "- Deluxe Room\n"
                "- Family Suite\n"
                "- Ocean Villa"
            )

        return self._create_booking()

    # CREATE BOOKING
    def _create_booking(self):
        """Create and save confirmed booking."""

        booking_id = self._generate_booking_id()

        booking = {
            "booking_id": booking_id,
            "name": self.state["name"],
            "room": self.state["room"],
            "check_in": self.state["checkin"],
            "check_out": self.state["checkout"],
            "guests": self.state["guests"],
            "status": "Confirmed"
        }

        self._save_booking(booking)

        reply = (
            f"✅ **Booking Confirmed!**\n\n"
            f"- **Booking ID** : {booking_id}\n"
            f"- **Name**       : {booking['name']}\n"
            f"- **Room**       : {booking['room']}\n"
            f"- **Check-in**   : {booking['check_in']}\n"
            f"- **Check-out**  : {booking['check_out']}\n"
            f"- **Guests**     : {booking['guests']}\n\n"
            f"Thank you for choosing **{HOTEL_NAME}**!"
        )

        self.reset()

        return reply

    # BOOKING FLOW
    def _handle_new_booking(self, entities):
        """
        Handle new booking conversation.

        The user can provide:
        - One field at a time
        - Several fields at once
        - All booking information in one sentence
        """

        error = self._update_booking_state(entities)

        if error:
            return error

        return self._ask_next_booking_question()

    # ACTIVE BOOKING FLOW
    def _handle_active_booking(self, user_input, entities):
        """
        Continue an existing booking conversation.

        Important:
        During an active booking wizard, the current step has priority
        over the generic entity extraction result.

        This prevents a single date such as "2026-10-05" from being
        incorrectly interpreted as check-in when the chatbot is actually
        asking for check-out.
        """

        step = self.state.get("step")

        # STEP 1: NAME
        if step == 1 and not self.state.get("name"):

            value = entities.get("name")

            if value is not None:
                valid, cleaned, error = validate_name(value)
            else:
                valid, cleaned, error = validate_name(user_input)

            if not valid:
                return error

            self.state["name"] = cleaned

            return self._ask_next_booking_question()

        # STEP 2: CHECK-IN
        if step == 2 and not self.state.get("checkin"):

            value = entities.get("check_in")

            if value is not None:
                valid, cleaned, error = validate_checkin_date(value)
            else:
                valid, cleaned, error = validate_checkin_date(user_input)

            if not valid:
                return error

            self.state["checkin"] = cleaned

            return self._ask_next_booking_question()

        # STEP 3: CHECK-OUT
        if step == 3 and not self.state.get("checkout"):

            # IMPORTANT:
            # Do NOT use entities["check_in"] here.
            #
            # EntityExtractor sees a standalone date as check_in.
            # But in this dialogue step, that date is actually check-out.

            value = entities.get("check_out")

            if value is not None:
                checkout_value = value
            else:
                # If user supplied a standalone date, validate the
                # original user input as the checkout date.
                checkout_value = user_input.strip()

            valid, cleaned, error = validate_checkout_date(
                self.state["checkin"],
                checkout_value
            )

            if not valid:
                return error

            self.state["checkout"] = cleaned

            return self._ask_next_booking_question()

        # STEP 4: GUESTS
        if step == 4 and not self.state.get("guests"):

            value = entities.get("guests")

            if value is not None:
                valid, cleaned, error = validate_guests(value)
            else:
                valid, cleaned, error = validate_guests(user_input)

            if not valid:
                return error

            self.state["guests"] = cleaned

            return self._ask_next_booking_question()

        # STEP 5: ROOM
        if step == 5 and not self.state.get("room"):

            value = entities.get("room_type")

            if value is not None:
                valid, cleaned, error = validate_room_type(value)
            else:
                valid, cleaned, error = validate_room_type(user_input)

            if not valid:
                return error

            self.state["room"] = cleaned

            return self._ask_next_booking_question()

        # FALLBACK
        error = self._update_booking_state(entities)

        if error:
            return error

        return self._ask_next_booking_question()

    # BOOKING ID ACTIONS
    def _handle_booking_id_action(self, action, booking_id):
        """Handle cancel/status/modify actions requiring booking ID."""

        booking_id = booking_id.upper()

        if action == "cancel":
            _, reply = self._cancel_booking(booking_id)
            self.reset()
            return reply

        booking = self._get_booking(booking_id)

        if action == "status":

            if not booking:
                return (
                    f"No reservation found with Booking ID "
                    f"**{booking_id}**. Please verify your booking ID."
                )

            self.reset()
            return self._format_booking_status(booking)

        if action == "modify":

            if not booking:
                return (
                    f"Could not find Booking ID **{booking_id}**. "
                    f"Please check your reference number."
                )

            self.reset()

            return (
                f"Found your booking **{booking_id}**.\n\n"
                f"To modify your reservation, please contact our "
                f"front desk at **{PHONE}**."
            )

        if action == "invoices":
            if not booking:
                return (
                    f"Could not find Booking ID **{booking_id}**. "
                    f"Please check your reference number."
                )
            self.reset()
            return self._format_invoice_details(booking)

        if action == "add_night":
            if not booking:
                return (
                    f"Could not find Booking ID **{booking_id}**. "
                    f"Please check your reference number."
                )
            self.reset()
            return (
                f"Found your booking **{booking_id}** ({booking.get('room', 'N/A')}).\n\n"
                f"To extend your stay, please contact our front desk directly at **{PHONE}** "
                f"or let us know your preferred extension dates!"
            )

        if action == "get_refund":
            if not booking:
                return (
                    f"Could not find Booking ID **{booking_id}**. "
                    f"Please check your reference number."
                )
            b_status = booking.get("status", "Confirmed")
            self.reset()
            if b_status == "Cancelled":
                return (
                    f"💵 **Refund Status for Booking {booking_id}**\n\n"
                    f"Your cancellation has been verified. A full refund is being processed "
                    f"to your original payment method (5-7 business days)."
                )
            else:
                return (
                    f"Reservation **{booking_id}** is currently **{b_status}**.\n\n"
                    f"To request a refund, please process cancellation first or call **{PHONE}**."
                )

        return None

    # MAIN MESSAGE HANDLER
    def handle_message(
        self,
        user_input,
        intent=None,
        extracted_entities=None
    ):
        """
        Main entry point for the Dialogue Manager.

        Args:
            user_input:
                Original user message.

            intent:
                Predicted intent from IntentPredictor.

            extracted_entities:
                Output from EntityExtractor.

        Returns:
            Bot response string.
        """

        if not user_input:
            return "Sorry, I didn't catch that. Could you please try again?"

        # Extract entities if not already provided
        if extracted_entities is None:
            extracted_entities = extract_entities(
                user_input
            ).get("entities_found", {})

        entities = extracted_entities

        booking_id = entities.get("booking_id")

        text_lower = user_input.strip().lower()

        # Cancel current booking wizard
        cancel_keywords = [
            "cancel",
            "stop",
            "exit",
            "quit",
            "never mind",
            "nevermind",
            "abort",
            "dont book",
            "don't book",
            "no thanks"
        ]

        if (
            self.state["active"]
            and self.state["action"] == "book"
            and not booking_id
            and any(keyword in text_lower for keyword in cancel_keywords)
        ):
            self.reset()

            return (
                "No problem! I've cancelled the booking process. "
                "Is there anything else I can help you with?"
            )

        # ACTIVE BOOKING-ID ACTION
        if (
            self.state["active"]
            and self.state["action"] in [
                "cancel",
                "status",
                "modify",
                "invoices",
                "add_night",
                "get_refund"
            ]
        ):

            if booking_id:
                return self._handle_booking_id_action(
                    self.state["action"],
                    booking_id
                )

            # User might simply type:
            # BK1234
            possible_id = user_input.strip().upper()

            if possible_id.startswith("BK"):
                return self._handle_booking_id_action(
                    self.state["action"],
                    possible_id
                )

            return "Please provide a valid Booking ID, for example **BK1234**."

        # CANCEL BOOKING
        if intent == "cancel_hotel_reservation":

            if booking_id:
                return self._handle_booking_id_action(
                    "cancel",
                    booking_id
                )

            self.state["active"] = True
            self.state["action"] = "cancel"
            self.state["pending_intent"] = "cancel_hotel_reservation"
            self.state["awaiting_slot"] = "booking_id"

            return (
                "Sure! Please provide your **Booking ID** "
                "(e.g. BK1234) so I can cancel your reservation."
            )

        # BOOKING STATUS
        if intent == "check_hotel_reservation":

            if booking_id:
                return self._handle_booking_id_action(
                    "status",
                    booking_id
                )

            self.state["active"] = True
            self.state["action"] = "status"
            self.state["pending_intent"] = "check_hotel_reservation"
            self.state["awaiting_slot"] = "booking_id"

            return (
                "I'd be happy to check your reservation. "
                "Please enter your **Booking ID** (e.g. BK1234)."
            )

        # MODIFY BOOKING
        if intent == "change_hotel_reservation":

            if booking_id:
                return self._handle_booking_id_action(
                    "modify",
                    booking_id
                )

            self.state["active"] = True
            self.state["action"] = "modify"
            self.state["pending_intent"] = "change_hotel_reservation"
            self.state["awaiting_slot"] = "booking_id"

            return (
                "Sure! Please provide your **Booking ID** "
                "(e.g. BK1234) to update your reservation."
            )

        # INVOICES
        if intent == "invoices":
            if booking_id:
                return self._handle_booking_id_action("invoices", booking_id)

            self.state["active"] = True
            self.state["action"] = "invoices"
            self.state["pending_intent"] = "invoices"
            self.state["awaiting_slot"] = "booking_id"

            return (
                "To retrieve your official tax invoice, please provide your **Booking ID** "
                "(e.g. BK7496)."
            )

        # EXTEND STAY (ADD NIGHT)
        if intent == "add_night":
            if booking_id:
                return self._handle_booking_id_action("add_night", booking_id)

            self.state["active"] = True
            self.state["action"] = "add_night"
            self.state["pending_intent"] = "add_night"
            self.state["awaiting_slot"] = "booking_id"

            return (
                "I'd be happy to help extend your stay! Please provide your **Booking ID** "
                "(e.g. BK7496)."
            )

        # GET REFUND
        if intent == "get_refund":
            if booking_id:
                return self._handle_booking_id_action("get_refund", booking_id)

            self.state["active"] = True
            self.state["action"] = "get_refund"
            self.state["pending_intent"] = "get_refund"
            self.state["awaiting_slot"] = "booking_id"

            return (
                "To process your refund request, please share your **Booking ID** "
                "(e.g. BK7496)."
            )

        # AVAILABILITY INQUIRY CONTEXT TRACKING
        if intent == "availability":
            checkin = entities.get("check_in")
            if not checkin:
                self.state["pending_intent"] = "availability"
                self.state["awaiting_slot"] = "check_in"
                return "I'd be happy to check room availability for you. What is your check-in date?"
            else:
                self.state["pending_intent"] = None
                self.state["awaiting_slot"] = None

        # NEW BOOKING
        if intent == "book_hotel":

            if not self.state["active"]:

                self.state["active"] = True
                self.state["action"] = "book"
                self.state["step"] = 0

            return self._handle_active_booking(
                user_input,
                entities
            )

        # CONTINUE ACTIVE BOOKING
        if (
            self.state["active"]
            and self.state["action"] == "book"
        ):
            return self._handle_active_booking(
                user_input,
                entities
            )

        # PRICE INQUIRY CONTEXT TRACKING
        if intent in ["check_hotel_prices", "room_price"]:
            if not entities.get("room_type"):
                self.state["pending_intent"] = "check_hotel_prices"
                self.state["awaiting_slot"] = "room_type"
            else:
                self.state["pending_intent"] = None
                self.state["awaiting_slot"] = None

        # STANDALONE BOOKING ID EXPRESS RULE
        if booking_id and not self.state["active"]:
            return self._handle_booking_id_action("status", booking_id)

        return None

# DEFAULT GLOBAL INSTANCE
_default_dialogue_manager = DialogueManager()


def handle_message(
    user_input,
    intent=None,
    extracted_entities=None
):
    """
    Convenience function for chatbot application.
    """

    return _default_dialogue_manager.handle_message(
        user_input,
        intent,
        extracted_entities
    )


def reset_conversation():
    """Reset current chatbot conversation."""
    _default_dialogue_manager.reset()


# TEST
if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    manager = DialogueManager()


    conversation = [
        ("I want to book a deluxe room for 2 guests", "book_hotel"),
        ("My name is John Doe", None),
        ("2026-10-01", None),
        ("2026-10-05", None),
        ("2 guests", None),
        ("Deluxe Room", None),
    ]

    print("=== Testing Dialogue Manager ===")

    for message, intent in conversation:

        entities = extract_entities(
            message
        ).get("entities_found", {})

        response = manager.handle_message(
            message,
            intent=intent,
            extracted_entities=entities
        )

        print(f"\nUser: {message}")
        print(f"Bot: {response}")