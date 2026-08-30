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

from chatbot.hotel_info import HOTEL_NAME, PHONE, EMAIL, HOTEL_POLICIES, ROOM_PRICES
from chatbot.entity_extractor import extract_entities
from chatbot.response import get_response
from chatbot.validation import (
    validate_name,
    validate_phone,
    clean_phone_digits,
    validate_checkin_date,
    validate_checkout_date,
    validate_guests,
    validate_room_type
)

BOOKING_FILE = os.path.join(PROJECT_ROOT, "data", "booking.json")

# FAQ & Informational Intents that can interrupt an active booking workflow
FAQ_INTENTS = {
    "check_hotel_facilities",
    "check_hotel_prices",
    "check_hotel_offers",
    "check_in",
    "check_out",
    "change_hotel_reservation",
    "book_parking_space",
    "bring_pets",
    "check_menu",
    "cancellation_fees",
    "customer_service",
    "human_agent",
    "host_event",
    "store_luggage",
    "shuttle_service",
    "search_hotel",
    "leave_review",
    "redeem_points",
    "invoices",
    "file_complaint"
}



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
            "awaiting_slot": None,
            "confirmed": False,

            # Cancellation verification state
            "target_booking_id": None,
            "target_guest_name": None,

            # Modification state
            "modify_target": None,
            "modify_substep": None,
            "new_checkin": None,
            "new_checkout": None
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

    def _update_booking(self, booking_id, updates):
        """Update fields of an existing booking in booking.json."""
        booking_id = booking_id.upper()
        bookings = self._load_bookings()
        for booking in bookings:
            if booking.get("booking_id", "").upper() == booking_id:
                booking.update(updates)
                self._save_bookings(bookings)
                return True, booking
        return False, None

    def _get_booking(self, identifier):
        """
        Find booking by Booking ID (e.g. BK1021) or Contact Phone Number (e.g. 012-583 2147).
        """
        if not identifier:
            return None

        clean_id = str(identifier).strip()
        clean_id_upper = clean_id.upper().replace("#", "").replace("-", "").replace(" ", "")

        # 1. Search by Booking ID
        for booking in self._load_bookings():
            b_id = booking.get("booking_id", "").upper().replace("-", "").replace(" ", "")
            if clean_id_upper == b_id:
                return booking

        # 2. Search by Contact Phone Number (digit-based matching)
        query_digits = clean_phone_digits(clean_id)
        if query_digits and len(query_digits) >= 8:
            for booking in self._load_bookings():
                b_phone = booking.get("phone", "")
                b_digits = clean_phone_digits(b_phone)
                if b_digits and (
                    query_digits == b_digits
                    or query_digits.endswith(b_digits[-8:])
                    or b_digits.endswith(query_digits[-8:])
                ):
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
        booking = self._get_booking(booking_id)
        if not booking:
            return (
                False,
                f"Could not find any reservation matching **{booking_id}**."
            )

        b_id = booking.get("booking_id", booking_id)
        if booking.get("status") == "Cancelled":
            return (
                False,
                f"Booking **{b_id}** is already cancelled."
            )

        bookings = self._load_bookings()
        for b in bookings:
            if b.get("booking_id", "").upper() == b_id.upper():
                b["status"] = "Cancelled"
                self._save_bookings(bookings)
                return (
                    True,
                    f"❌ Booking **{b_id}** has been successfully cancelled."
                )

        return (
            False,
            f"Could not find any reservation matching **{booking_id}**."
        )

    # BOOKING STATUS
    def _format_booking_status(self, booking):
        """Format booking information for chatbot response."""
        status = booking.get("status", "Confirmed")
        emoji = "✅" if status == "Confirmed" else "❌"

        deposit_paid = booking.get("deposit_paid", "N/A")
        remaining_balance = booking.get("remaining_balance", "N/A")

        return (
            f"📋 **Booking Details ({booking.get('booking_id')})**\n\n"
            f"- **Guest Name**       : {booking.get('name', 'N/A')}\n"
            f"- **Contact Phone**    : `{booking.get('phone', 'N/A')}`\n"
            f"- **Room Type**        : {booking.get('room', 'N/A')}\n"
            f"- **Check-in / Out**   : {booking.get('check_in', 'N/A')} to {booking.get('check_out', 'N/A')} ({booking.get('nights', 1)} nights)\n"
            f"- **Guests**           : {booking.get('guests', 'N/A')} pax\n"
            f"- **Deposit Paid (1N)**: {deposit_paid}\n"
            f"- **Check-in Balance** : {remaining_balance}\n"
            f"- **Booking Status**   : {emoji} **{status}**"
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

    # PRICING CALCULATION
    def _calculate_booking_price(self):
        """Calculate nights, room rate, tax, and total estimated price."""
        room = self.state.get("room", "Standard Room")
        rate_str = ROOM_PRICES.get(room, "RM 180")
        digits = re.findall(r"\d+", rate_str)
        rate_num = int(digits[0]) if digits else 180

        checkin_str = self.state.get("checkin", "")
        checkout_str = self.state.get("checkout", "")

        nights = 1
        try:
            from datetime import datetime
            cin = datetime.strptime(checkin_str, "%Y-%m-%d")
            cout = datetime.strptime(checkout_str, "%Y-%m-%d")
            delta = (cout - cin).days
            if delta > 0:
                nights = delta
        except Exception:
            nights = 1

        subtotal = rate_num * nights
        tax = int(round(subtotal * 0.06))
        grand_total = subtotal + tax

        # 1-Night Deposit Guarantee calculation
        deposit_subtotal = rate_num * 1
        deposit_tax = int(round(deposit_subtotal * 0.06))
        deposit_total = deposit_subtotal + deposit_tax
        remaining_balance = grand_total - deposit_total

        return {
            "rate_str": rate_str,
            "rate_num": rate_num,
            "nights": nights,
            "subtotal": subtotal,
            "tax": tax,
            "grand_total": grand_total,
            "deposit_subtotal": deposit_subtotal,
            "deposit_tax": deposit_tax,
            "deposit_total": deposit_total,
            "remaining_balance": max(0, remaining_balance)
        }

    # NEXT BOOKING QUESTION
    def _get_missing_booking_field(self):
        """Return the first missing booking field."""
        if not self.state.get("name"):
            return "name"

        if not self.state.get("phone"):
            return "phone"

        if not self.state.get("checkin"):
            return "checkin"

        if not self.state.get("checkout"):
            return "checkout"

        if not self.state.get("guests"):
            return "guests"

        if not self.state.get("room"):
            return "room"

        if not self.state.get("deposit_paid"):
            return "deposit"

        return None

    def _ask_next_booking_question(self, has_greeting=False):
        """Ask user for the next missing booking field."""

        field = self._get_missing_booking_field()

        if field == "name":
            self.state["step"] = 1
            greeting_prefix = f"👋 **Hello! Welcome to {HOTEL_NAME}.**\n\n" if has_greeting else ""
            return f"{greeting_prefix}Sure! Let me help you book a room.\n\nMay I have your name?"

        if field == "phone":
            self.state["step"] = 2
            return "May I have your contact phone number (e.g. **0123456789**)?"

        if field == "checkin":
            self.state["step"] = 3
            return "What is your check-in date?"

        if field == "checkout":
            self.state["step"] = 4
            return "What is your check-out date?"

        if field == "guests":
            self.state["step"] = 5
            return "How many guests will be staying?"

        if field == "room":
            self.state["step"] = 6
            return (
                "Which room type would you like?\n\n"
                "- Standard Room\n"
                "- Deluxe Room\n"
                "- Family Suite\n"
                "- Ocean Villa"
            )

        if field == "deposit":
            self.state["step"] = "awaiting_deposit"
            pricing = self._calculate_booking_price()
            return (
                f"📋 **Please Review Your Reservation & Deposit Details:**\n\n"
                f"- **Guest Name**        : {self.state.get('name')}\n"
                f"- **Room Type**         : {self.state.get('room')} ({pricing['rate_str']}/night)\n"
                f"- **Dates**             : {self.state.get('checkin')} to {self.state.get('checkout')} ({pricing['nights']} night{'s' if pricing['nights'] > 1 else ''})\n"
                f"- **Number of Guests**  : {self.state.get('guests')}\n"
                f"- **1-Night Deposit**   : **RM{pricing['deposit_total']}** *(RM{pricing['deposit_subtotal']} + 6% SST)*\n"
                f"- **Remaining Balance** : **RM{pricing['remaining_balance']}** *(Payable upon front desk check-in)*\n"
                f"- **Total Booking**     : **RM{pricing['grand_total']}** (incl. 6% SST)\n\n"
                f"📱 **Please tap [💳 Pay Deposit] on your simulated mobile screen in the sidebar to secure your room!**\n"
                f"*(Or type **Cancel** to abort this reservation)*"
            )

        return self._create_booking()

    # CREATE BOOKING (1-NIGHT DEPOSIT GUARANTEE)
    def _create_booking(self):
        """Create and save confirmed booking with 1-night deposit and check-in balance breakdown."""

        booking_id = self._generate_booking_id()
        pricing = self._calculate_booking_price()

        booking = {
            "booking_id": booking_id,
            "name": self.state["name"],
            "phone": self.state.get("phone", "+60 12-345 6789"),
            "room": self.state["room"],
            "check_in": self.state["checkin"],
            "check_out": self.state["checkout"],
            "guests": self.state["guests"],
            "nights": pricing["nights"],
            "deposit_paid": f"RM{pricing['deposit_total']}",
            "remaining_balance": f"RM{pricing['remaining_balance']}",
            "total_amount": f"RM{pricing['grand_total']}",
            "status": "Confirmed"
        }

        self._save_booking(booking)

        if pricing["nights"] > 1:
            payment_breakdown = (
                f"- **1-Night Deposit**   : **RM{pricing['deposit_total']}** *(RM{pricing['deposit_subtotal']} + 6% SST)*\n"
                f"- **Remaining Balance** : **RM{pricing['remaining_balance']}** *(Payable upon front desk check-in)*\n"
                f"- **Total Booking**     : **RM{pricing['grand_total']}** (incl. 6% SST)\n"
            )
        else:
            payment_breakdown = (
                f"- **Deposit Paid (1N)** : **RM{pricing['deposit_total']}** (incl. 6% SST)\n"
                f"- **Remaining Balance** : **RM0** *(Fully covered)*\n"
            )

        reply = (
            f"🎉 **Reservation Confirmed (1-Night Deposit Guarantee)**\n\n"
            f"- **Booking ID**        : **{booking_id}**\n"
            f"- **Guest Name**        : {booking['name']}\n"
            f"- **Room Type**         : {booking['room']} ({pricing['rate_str']}/night)\n"
            f"- **Dates**             : {booking['check_in']} to {booking['check_out']} ({pricing['nights']} night{'s' if pricing['nights'] > 1 else ''})\n"
            f"- **Guests**            : {booking['guests']} pax\n"
            f"{payment_breakdown}"
            f"- **Status**            : ✅ **Confirmed**\n\n"
            f"Thank you for choosing **{HOTEL_NAME}**! We look forward to hosting you.\n"
            f"💡 *Free cancellation is available up to {HOTEL_POLICIES['cancellation_window_hours']} hours before check-in with 100% deposit refund. You can quote **{booking_id}** anytime to check status or invoices.*"
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

    # FAQ INTERRUPTION & CONTEXTUAL RESUME
    def _handle_faq_interruption(self, intent, entities=None):
        """
        Answer the FAQ/informational inquiry and provide a contextual prompt
        to resume the active booking wizard.
        """
        faq_response = get_response(intent, entities=entities)

        # Build progress breadcrumbs
        progress = []
        if self.state.get("name"):
            progress.append(f"Guest: **{self.state['name']}**")
        if self.state.get("checkin"):
            progress.append(f"Check-in: **{self.state['checkin']}**")
        if self.state.get("checkout"):
            progress.append(f"Check-out: **{self.state['checkout']}**")
        if self.state.get("guests"):
            progress.append(f"Guests: **{self.state['guests']}**")
        if self.state.get("room"):
            progress.append(f"Room: **{self.state['room']}**")

        progress_str = f" ({', '.join(progress)})" if progress else ""

        field = self._get_missing_booking_field()
        if field == "name":
            resume_question = "May I have your name to proceed with the reservation?"
        elif field == "phone":
            resume_question = "May I have your contact phone number to proceed with the reservation?"
        elif field == "checkin":
            resume_question = "What is your check-in date?"
        elif field == "checkout":
            resume_question = "What is your check-out date?"
        elif field == "guests":
            resume_question = "How many guests will be staying?"
        elif field == "room":
            resume_question = (
                "Which room type would you like to reserve?\n\n"
                "- Standard Room\n"
                "- Deluxe Room\n"
                "- Family Suite\n"
                "- Ocean Villa"
            )
        elif field == "deposit":
            resume_question = "Please tap **[💳 Pay Deposit]** on your simulated mobile screen in the sidebar to secure your reservation (or type **Cancel** to abort)."
        else:
            resume_question = "Which room type would you like to reserve?"

        return (
            f"{faq_response}\n\n"
            f"---\n"
            f"💬 *Would you like to continue with your room reservation{progress_str}?*\n\n"
            f"{resume_question}\n"
            f"*(Or type **Cancel** if you wish to stop)*"
        )

    # ACTIVE BOOKING FLOW
    def _handle_active_booking(self, user_input, entities, intent=None):
        """
        Continue an existing booking conversation with FAQ interruption support.

        Important:
        During an active booking wizard, the current step has priority
        over generic entity extraction, and relevant FAQ questions will be answered
        without losing the booking context.
        """

        cleaned_input = user_input.strip().lower()
        cancel_keywords = ["cancel", "stop", "abort", "never mind", "nevermind", "quit", "exit", "don't book", "dont book"]
        if any(re.search(rf"\b{kw}\b", cleaned_input) for kw in cancel_keywords):
            self.reset()
            return "No problem! Your reservation has been cancelled. Feel free to let me know if you would like to book again anytime!"

        step = self.state.get("step")

        confirm_continue_keywords = ["yes", "y", "sure", "continue", "proceed", "ok", "okay", "yeah", "yep"]
        if cleaned_input in confirm_continue_keywords and step not in [1, "awaiting_deposit"]:
            return self._ask_next_booking_question()

        # STEP 1: NAME
        if step == 1 and not self.state.get("name"):

            value = entities.get("name")

            if value is not None:
                valid, cleaned, error = validate_name(value)
            else:
                if intent and intent in FAQ_INTENTS:
                    return self._handle_faq_interruption(intent, entities)
                valid, cleaned, error = validate_name(user_input)

            if not valid:
                if intent and intent in FAQ_INTENTS:
                    return self._handle_faq_interruption(intent, entities)
                return error

            self.state["name"] = cleaned

            return self._ask_next_booking_question()

        # STEP 2: PHONE NUMBER
        if step == 2 and not self.state.get("phone"):

            value = entities.get("phone")

            if value is not None:
                valid, cleaned, error = validate_phone(value)
            else:
                if intent and intent in FAQ_INTENTS:
                    return self._handle_faq_interruption(intent, entities)
                valid, cleaned, error = validate_phone(user_input)

            if not valid:
                if intent and intent in FAQ_INTENTS:
                    return self._handle_faq_interruption(intent, entities)
                return error

            self.state["phone"] = cleaned

            return self._ask_next_booking_question()

        # STEP 3: CHECK-IN
        if step == 3 and not self.state.get("checkin"):

            value = entities.get("check_in")

            if value is not None:
                valid, cleaned, error = validate_checkin_date(value)
            else:
                if intent and intent in FAQ_INTENTS:
                    return self._handle_faq_interruption(intent, entities)
                valid, cleaned, error = validate_checkin_date(user_input)

            if not valid:
                if intent and intent in FAQ_INTENTS:
                    return self._handle_faq_interruption(intent, entities)
                return error

            self.state["checkin"] = cleaned

            return self._ask_next_booking_question()

        # STEP 4: CHECK-OUT
        if step == 4 and not self.state.get("checkout"):

            value = entities.get("check_out")

            if value is not None:
                checkout_value = value
            else:
                if intent and intent in FAQ_INTENTS:
                    return self._handle_faq_interruption(intent, entities)
                checkout_value = user_input.strip()

            valid, cleaned, error = validate_checkout_date(
                self.state["checkin"],
                checkout_value
            )

            if not valid:
                if intent and intent in FAQ_INTENTS:
                    return self._handle_faq_interruption(intent, entities)
                return error

            self.state["checkout"] = cleaned

            return self._ask_next_booking_question()

        # STEP 5: GUESTS
        if step == 5 and not self.state.get("guests"):

            value = entities.get("guests")

            if value is not None:
                valid, cleaned, error = validate_guests(value)
            else:
                if intent and intent in FAQ_INTENTS:
                    return self._handle_faq_interruption(intent, entities)
                valid, cleaned, error = validate_guests(user_input)

            if not valid:
                if intent and intent in FAQ_INTENTS:
                    return self._handle_faq_interruption(intent, entities)
                return error

            self.state["guests"] = cleaned

            return self._ask_next_booking_question()

        # STEP 6: ROOM
        if step == 6 and not self.state.get("room"):

            value = entities.get("room_type")

            if value is not None:
                valid, cleaned, error = validate_room_type(value)
            else:
                if intent and intent in FAQ_INTENTS:
                    return self._handle_faq_interruption(intent, entities)
                valid, cleaned, error = validate_room_type(user_input)

            if not valid:
                if intent and intent in FAQ_INTENTS:
                    return self._handle_faq_interruption(intent, entities)
                return error

            self.state["room"] = cleaned

            return self._ask_next_booking_question()

        # STEP AWAITING DEPOSIT PAYMENT
        if step == "awaiting_deposit" or (self.state.get("room") and not self.state.get("deposit_paid")):
            cleaned_input = user_input.strip().lower()

            cancel_keywords = ["cancel", "no", "n", "stop", "abort", "never mind", "nevermind", "nv"]

            if any(re.search(rf"\b{kw}\b", cleaned_input) for kw in cancel_keywords):
                self.reset()
                return (
                    "No problem! The reservation has been cancelled. No deposit was charged. "
                    "Feel free to let me know if you would like to book again anytime!"
                )

            if intent and intent in FAQ_INTENTS:
                return self._handle_faq_interruption(intent, entities)

            return (
                "📱 **Payment Required**: For secure checkout, please tap **[💳 Pay Deposit]** on your simulated mobile screen in the sidebar to confirm your reservation.\n\n"
                "*(Or type **Cancel** if you wish to abort)*"
            )

        # FALLBACK
        error = self._update_booking_state(entities)

        if error:
            if intent and intent in FAQ_INTENTS:
                return self._handle_faq_interruption(intent, entities)
            return error

        has_greeting = bool(re.search(r"^(?:hi|hello|hey|helo|hiii|hiya|howdy|greetings|good\s+(?:morning|afternoon|evening))\b", user_input.strip().lower()))
        return self._ask_next_booking_question(has_greeting=has_greeting)

    # BOOKING ACTIONS (LOOKUP BY BOOKING ID OR PHONE NUMBER)
    def _handle_booking_id_action(self, action, identifier, entities=None):
        """Handle cancel/status/modify/invoice actions matching by booking ID or contact phone."""
        booking = self._get_booking(identifier)
        if not booking:
            return (
                f"Could not find any reservation matching **{identifier}**. "
                f"Please verify your **Booking ID** (e.g. BK1021) or registered **Phone Number** (e.g. 012-583 2147)."
            )

        b_id = booking.get("booking_id", "N/A")

        if action == "cancel":
            if booking.get("status") == "Cancelled":
                self.reset()
                return f"Booking **{b_id}** is already cancelled."

            target_phone = booking.get("phone", "+60 12-345 6789")

            # Check if user already provided phone in entities or identifier was phone
            provided_phone = (entities.get("phone") if entities else None)
            is_verified = False
            if provided_phone:
                if (clean_phone_digits(provided_phone) == clean_phone_digits(target_phone)
                    or clean_phone_digits(provided_phone).endswith(clean_phone_digits(target_phone)[-8:])
                    or clean_phone_digits(target_phone).endswith(clean_phone_digits(provided_phone)[-8:])):
                    is_verified = True
            elif clean_phone_digits(str(identifier)) == clean_phone_digits(target_phone):
                is_verified = True

            self.state["active"] = True
            self.state["action"] = "confirm_cancel"
            self.state["target_booking_id"] = b_id
            self.state["target_guest_name"] = booking.get("name", "Guest")
            self.state["target_phone"] = target_phone

            if is_verified:
                self.state["mobile_2fa_active"] = True
                return (
                    f"⚠️ **Cancellation Verification for Booking {b_id}**\n\n"
                    f"- **Guest Name**    : {booking.get('name', 'Guest')}\n"
                    f"- **Contact Phone** : `{target_phone}` (✅ Verified)\n"
                    f"- **Room Reserved** : {booking.get('room', 'N/A')}\n"
                    f"- **Stay Dates**    : {booking.get('check_in')} to {booking.get('check_out')}\n"
                    f"- **Current Status**: {booking.get('status', 'Confirmed')}\n\n"
                    f"📱 **Please tap [✅ Confirm Cancel] on your mobile screen in the sidebar to finalize the cancellation** "
                    f"(or type **Keep** to retain your booking)."
                )

            self.state["mobile_2fa_active"] = False
            return (
                f"⚠️ **Cancellation Verification for Booking {b_id}**\n\n"
                f"- **Guest Name**    : {booking.get('name', 'Guest')}\n"
                f"- **Room Reserved** : {booking.get('room', 'N/A')}\n"
                f"- **Stay Dates**    : {booking.get('check_in')} to {booking.get('check_out')}\n"
                f"- **Current Status**: {booking.get('status', 'Confirmed')}\n\n"
                f"🔒 **For security verification, please enter the Contact Phone Number registered under this booking to proceed:**\n"
                f"*(Or type **Keep** / **Never mind** to keep your reservation active)*"
            )

        if action == "status":
            self.reset()
            return self._format_booking_status(booking)

        if action == "modify":
            self.reset()
            return (
                f"✏️ **Modify Reservation ({b_id} - {booking.get('name', 'Guest')}):**\n\n"
                f"Due to room availability adjustments and daily rate differences, stay modifications for **{b_id}** are handled directly by our reservations desk:\n"
                f"- 📞 **Reservations Hotline**: **{PHONE}**\n"
                f"- ✉️ **Email Support**: **{EMAIL}**\n\n"
                f"💡 *Tip: Because **{HOTEL_NAME}** offers free cancellation up to {HOTEL_POLICIES['cancellation_window_hours']} hours before check-in, you can also cancel Booking **{b_id}** here anytime and make a fresh booking with your updated dates!*"
            )

        if action == "invoices":
            self.reset()
            return self._format_invoice_details(booking)

        if action == "add_night":
            self.reset()
            return (
                f"Found your booking **{b_id}** ({booking.get('room', 'N/A')}).\n\n"
                f"To extend your stay, please contact our front desk directly at **{PHONE}** "
                f"or let us know your preferred extension dates!"
            )

        if action == "get_refund":
            b_status = booking.get("status", "Confirmed")
            self.reset()
            if b_status == "Cancelled":
                return (
                    f"💵 **Refund Status for Booking {b_id}**\n\n"
                    f"Your cancellation has been verified. A full refund is being processed "
                    f"to your original payment method (5-7 business days)."
                )
            else:
                return (
                    f"Reservation **{b_id}** is currently **{b_status}**.\n\n"
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
            "nvm",
            "abort",
            "dont book",
            "don't book",
            "no thanks",
            "back",
            "no",
            "keep"
        ]

        if (
            self.state["active"]
            and not booking_id
            and not entities.get("phone")
            and any(re.search(rf"\b{re.escape(kw)}\b", text_lower) for kw in cancel_keywords)
        ):
            action = self.state.get("action")
            target_id = self.state.get("target_booking_id")
            self.reset()

            if action == "book":
                return (
                    "No problem! I've cancelled the booking process. "
                    "Is there anything else I can help you with?"
                )
            elif action == "confirm_cancel" and target_id:
                return f"Cancellation aborted. Your reservation **{target_id}** remains active and confirmed!"
            else:
                return "No problem! Process cancelled. Is there anything else I can help you with?"

        # ACTIVE BOOKING-ID OR PHONE ACTION
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
            # If user switched to an FAQ intent instead of providing an ID
            if intent and intent in FAQ_INTENTS:
                self.reset()
                return get_response(intent, entities=entities)

            if booking_id:
                return self._handle_booking_id_action(
                    self.state["action"],
                    booking_id,
                    entities=entities
                )

            extracted_phone = entities.get("phone")
            if extracted_phone:
                return self._handle_booking_id_action(
                    self.state["action"],
                    extracted_phone,
                    entities=entities
                )

            # User might type: BK1234 or a phone number
            possible_input = user_input.strip()
            if possible_input.upper().startswith("BK"):
                return self._handle_booking_id_action(
                    self.state["action"],
                    possible_input.upper(),
                    entities=entities
                )

            clean_digits = clean_phone_digits(possible_input)
            if len(clean_digits) >= 8:
                return self._handle_booking_id_action(
                    self.state["action"],
                    possible_input,
                    entities=entities
                )

            return "Please provide a valid **Booking ID** (e.g. BK1021) or registered **Phone Number** (e.g. 012-583 2147)."

        # ACTIVE CANCEL CONFIRMATION (PHONE NUMBER VERIFICATION)
        if self.state["active"] and self.state["action"] == "confirm_cancel":
            target_id = self.state.get("target_booking_id")
            target_name = self.state.get("target_guest_name", "Guest")
            target_phone = self.state.get("target_phone", "")

            # 1. Abort / Keep keywords
            abort_keywords = ["keep", "no", "dont", "don't", "stop", "abort", "never mind", "nevermind", "nvm", "exit", "quit"]
            if any(re.search(rf"\b{kw}\b", text_lower) for kw in abort_keywords):
                self.reset()
                return f"Cancellation aborted. Your reservation **{target_id}** remains active and confirmed!"

            # 2. Phone digit matching
            input_digits = clean_phone_digits(user_input)
            target_digits = clean_phone_digits(target_phone)

            extracted_phone = entities.get("phone", "")
            extracted_digits = clean_phone_digits(extracted_phone) if extracted_phone else ""

            is_match = False
            if input_digits and target_digits:
                if input_digits == target_digits or input_digits.endswith(target_digits[-8:]) or target_digits.endswith(input_digits[-8:]):
                    is_match = True
            elif extracted_digits and target_digits:
                if extracted_digits == target_digits or extracted_digits.endswith(target_digits[-8:]) or target_digits.endswith(extracted_digits[-8:]):
                    is_match = True

            if is_match:
                # Activate Mobile 2FA pass in the sidebar
                self.state["mobile_2fa_active"] = True
                return (
                    f"✅ **Identity Verified for Booking {target_id}**\n\n"
                    f"A 2FA cancellation request has been dispatched to your simulated mobile screen in the sidebar.\n\n"
                    f"📱 **Please tap [✅ Confirm Cancel] on your mobile screen in the sidebar to finalize the cancellation** "
                    f"(or type **Keep** to retain your booking)."
                )
            else:
                return (
                    f"⚠️ **Identity Verification Failed**\n\n"
                    f"The contact phone number provided (`{user_input.strip()}`) does not match our records for Booking ID **{target_id}**.\n\n"
                    f"Please enter the correct registered **Contact Phone Number** to proceed, or type **Keep** to retain your booking."
                )

        # CANCEL BOOKING
        if intent == "cancel_hotel_reservation":
            target_identifier = booking_id or entities.get("phone")
            if target_identifier:
                return self._handle_booking_id_action(
                    "cancel",
                    target_identifier,
                    entities=entities
                )

            self.state["active"] = True
            self.state["action"] = "cancel"
            self.state["pending_intent"] = "cancel_hotel_reservation"
            self.state["awaiting_slot"] = "booking_id"

            return (
                "Sure! Please provide your **Booking ID** (e.g. BK1021) or registered **Phone Number** (e.g. 012-583 2147) "
                "so I can cancel your reservation."
            )

        # BOOKING STATUS
        if intent == "check_hotel_reservation":
            target_identifier = booking_id or entities.get("phone")
            if target_identifier:
                return self._handle_booking_id_action(
                    "status",
                    target_identifier,
                    entities=entities
                )

            self.state["active"] = True
            self.state["action"] = "status"
            self.state["pending_intent"] = "check_hotel_reservation"
            self.state["awaiting_slot"] = "booking_id"

            return (
                "I'd be happy to check your reservation. "
                "Please enter your **Booking ID** (e.g. BK1021) or registered **Phone Number** (e.g. 012-583 2147)."
            )

        # MODIFY BOOKING (Front Desk Hotline & Free Re-booking Guidance)
        if intent == "change_hotel_reservation":
            target_identifier = booking_id or entities.get("phone")
            if target_identifier:
                return self._handle_booking_id_action(
                    "modify",
                    target_identifier,
                    entities=entities
                )
            return get_response("change_hotel_reservation", entities=entities)

        # INVOICES
        if intent == "invoices":
            target_identifier = booking_id or entities.get("phone")
            if target_identifier:
                return self._handle_booking_id_action(
                    "invoices",
                    target_identifier,
                    entities=entities
                )

            self.state["active"] = True
            self.state["action"] = "invoices"
            self.state["pending_intent"] = "invoices"
            self.state["awaiting_slot"] = "booking_id"

            return (
                "To retrieve your official tax invoice, please provide your **Booking ID** (e.g. BK1021) "
                "or registered **Phone Number** (e.g. 012-583 2147)."
            )

        # EXTEND STAY (ADD NIGHT)
        if intent == "add_night":
            target_identifier = booking_id or entities.get("phone")
            if target_identifier:
                return self._handle_booking_id_action(
                    "add_night",
                    target_identifier,
                    entities=entities
                )

            self.state["active"] = True
            self.state["action"] = "add_night"
            self.state["pending_intent"] = "add_night"
            self.state["awaiting_slot"] = "booking_id"

            return (
                "I'd be happy to help extend your stay! Please provide your **Booking ID** (e.g. BK1021) "
                "or registered **Phone Number** (e.g. 012-583 2147)."
            )

        # GET REFUND
        if intent == "get_refund":
            target_identifier = booking_id or entities.get("phone")
            if target_identifier:
                return self._handle_booking_id_action(
                    "get_refund",
                    target_identifier,
                    entities=entities
                )

            self.state["active"] = True
            self.state["action"] = "get_refund"
            self.state["pending_intent"] = "get_refund"
            self.state["awaiting_slot"] = "booking_id"

            return (
                "To process your refund request, please share your **Booking ID** (e.g. BK1021) "
                "or registered **Phone Number** (e.g. 012-583 2147)."
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
                entities,
                intent=intent
            )

        # CONTINUE ACTIVE BOOKING
        if (
            self.state["active"]
            and self.state["action"] == "book"
        ):
            return self._handle_active_booking(
                user_input,
                entities,
                intent=intent
            )

        # PRICE INQUIRY CONTEXT TRACKING
        if intent == "check_hotel_prices":
            if not entities.get("room_type"):
                self.state["pending_intent"] = "check_hotel_prices"
                self.state["awaiting_slot"] = "room_type"
            else:
                self.state["pending_intent"] = None
                self.state["awaiting_slot"] = None

        # STANDALONE BOOKING ID OR PHONE EXPRESS RULE
        express_target = booking_id or entities.get("phone")
        if express_target and not self.state["active"]:
            return self._handle_booking_id_action("status", express_target, entities=entities)

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
