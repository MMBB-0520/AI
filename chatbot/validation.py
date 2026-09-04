"""
BookMate Chatbot
Booking Entity Validators

Purpose:
Validate and normalize extracted booking entities.

Validations:
- Check-in date
- Check-out date
- Guest count
- Room type
- Guest name
- Complete booking entity set
"""

import re
import sys
import os
from datetime import datetime, date, timedelta

# Ensure parent directory is available for imports
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

try:
    from chatbot.hotel_info import (
        ROOM_TYPES,
        ROOM_TYPE_ALIASES
    )
except ImportError:
    ROOM_TYPES = ["Standard Room", "Deluxe Room", "Family Suite", "Ocean Villa"]
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
        "villa": "Ocean Villa",
    }



# DATE PARSING
def parse_date(
    date_str: str,
    reference_date: date | None = None
) -> date | None:
    """
    Parse a date string into datetime.date supporting absolute and rich relative formats.

    Supports:
        - Absolute dates: YYYY-MM-DD, DD-MM-YYYY, YYYY/MM/DD, DD/MM/YYYY, etc.
        - Month names: 15 Oct 2026, 15 October 2026, Oct 15 2026, October 15, etc.
        - Relative dates: today, tomorrow, tmrw, day after tomorrow, next week, next month
        - Duration expressions: 1 night, 2 nights, 3 days, for 4 nights, in 2 days, etc.

    Returns:
        datetime.date or None
    """

    if not date_str or not isinstance(date_str, str):
        return None

    clean_str = date_str.strip().lower()

    if reference_date is None:
        reference_date = datetime.now().date()

    # Relative words
    if clean_str in ["today", "tonight", "tdy"]:
        return reference_date

    if clean_str in ["tomorrow", "tmrw", "tomorow", "tomo"]:
        return reference_date + timedelta(days=1)

    if clean_str in ["day after tomorrow", "the day after tomorrow", "overmorrow"]:
        return reference_date + timedelta(days=2)

    if clean_str in ["next week", "in a week", "1 week", "after a week"]:
        return reference_date + timedelta(days=7)

    if clean_str in ["next month", "in a month", "1 month"]:
        return reference_date + timedelta(days=30)

    # Relative night/day duration patterns (e.g., "1 night", "3 nights", "for 2 days", "after 4 nights", "2d", "3n")
    dur_match = re.search(r"(?:for|stay|in|after)?\s*(\d+)\s*(?:night|nights|night's|day|days|d|n)\b", clean_str)
    if dur_match:
        try:
            num_days = int(dur_match.group(1))
            if 1 <= num_days <= 90:
                return reference_date + timedelta(days=num_days)
        except Exception:
            pass

    formats = [
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y.%m.%d",

        "%d-%m-%Y",
        "%d/%m/%Y",
        "%d.%m.%Y",

        "%d-%m-%y",
        "%d/%m/%y",

        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%d %b",
        "%d %B",
        "%b %d",
        "%B %d"
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(
                clean_str,
                fmt
            ).date()

            # If no year was provided in format, assume current reference year
            if dt.year == 1900:
                dt = dt.replace(year=reference_date.year)
                if dt < reference_date:
                    dt = dt.replace(year=reference_date.year + 1)
            return dt

        except ValueError:
            continue

    return None


# CHECK-IN VALIDATION
def validate_checkin_date(
    checkin_str: str,
    reference_date: date | None = None
) -> tuple[bool, str | None, str | None]:
    """
    Validate check-in date.

    Rules:
        - Must be a valid date
        - Cannot be in the past

    Returns:
        (is_valid, normalized_date, error_message)
    """

    if reference_date is None:
        reference_date = datetime.now().date()

    parsed = parse_date(
        checkin_str,
        reference_date
    )

    if not parsed:
        return (
            False,
            None,
            "⚠️ Invalid check-in date. "
            "Please provide a valid date such as **2026-10-01**, **today**, "
            "or **tomorrow**."
        )

    if parsed < reference_date:
        return (
            False,
            None,
            f"⚠️ Check-in date cannot be in the past. "
            f"Today is **{reference_date.strftime('%Y-%m-%d')}**."
        )

    return (
        True,
        parsed.strftime("%Y-%m-%d"),
        None
    )


# CHECK-OUT VALIDATION
def validate_checkout_date(
    checkin_str: str,
    checkout_str: str,
    reference_date: date | None = None
) -> tuple[bool, str | None, str | None]:
    """
    Validate check-out date with support for relative offsets (e.g. 'tomorrow', '3 nights').

    Rules:
        - Must be a valid date
        - Check-out must be after check-in
    """

    if reference_date is None:
        reference_date = datetime.now().date()

    parsed_in = parse_date(
        checkin_str,
        reference_date
    )

    if not parsed_in:
        return (
            False,
            None,
            "⚠️ Please provide a valid check-in date first."
        )

    # Note: When validating checkout, relative terms like "tomorrow", "3 nights", "day after tomorrow"
    # are calculated relative to the check-in date!
    parsed_out = parse_date(
        checkout_str,
        reference_date=parsed_in
    )

    if not parsed_out:
        return (
            False,
            None,
            "⚠️ Invalid check-out date. "
            "Please provide a valid date such as **2026-10-05**, **tomorrow**, or **3 nights**."
        )

    if parsed_in < reference_date:
        return (
            False,
            None,
            "⚠️ Check-in date cannot be in the past."
        )

    if parsed_out <= parsed_in:
        return (
            False,
            None,
            f"⚠️ Check-out date "
            f"({parsed_out.strftime('%Y-%m-%d')}) "
            f"must be after check-in date "
            f"({parsed_in.strftime('%Y-%m-%d')})."
        )

    return (
        True,
        parsed_out.strftime("%Y-%m-%d"),
        None
    )


# CALCULATE NIGHTS
def calculate_nights(
    checkin_str: str,
    checkout_str: str
) -> int | None:
    """
    Calculate number of nights from check-in/check-out.
    """

    checkin = parse_date(checkin_str)
    checkout = parse_date(checkout_str)

    if not checkin or not checkout:
        return None

    nights = (checkout - checkin).days

    if nights <= 0:
        return None

    return nights


# GUEST VALIDATION
def validate_guests(
    guests_input: str | int
) -> tuple[bool, int | None, str | None]:
    """
    Validate number of guests.

    Rules:
        - Positive integer
        - Maximum 20 guests
    """

    if guests_input is None:
        return (
            False,
            None,
            "⚠️ Please specify the number of guests."
        )

    # Integer input
    if isinstance(guests_input, int):
        num = guests_input

    else:
        value = str(guests_input).strip()

        # Only accept a complete integer
        if not re.fullmatch(r"\d+", value):
            return (
                False,
                None,
                "⚠️ Please provide a valid number of guests, "
                "such as **2** or **4**."
            )

        num = int(value)

    if num <= 0:
        return (
            False,
            None,
            "⚠️ Number of guests must be at least **1**."
        )

    if num > 20:
        return (
            False,
            None,
            "⚠️ Maximum capacity per reservation is "
            "**20 guests**. Please contact the front desk "
            "for larger group bookings."
        )

    return (
        True,
        num,
        None
    )


# ROOM TYPE VALIDATION
def validate_room_type(
    room_input: str
) -> tuple[bool, str | None, str | None]:
    """
    Validate room type against official hotel room list.
    """

    if not room_input or not isinstance(room_input, str):
        return (
            False,
            None,
            "⚠️ Please specify a room type."
        )

    clean_room = room_input.strip().lower()

    # Exact official room name
    for room_type in ROOM_TYPES:

        if clean_room == room_type.lower():
            return (
                True,
                room_type,
                None
            )

    # Alias matching
    for alias, official_name in sorted(
        ROOM_TYPE_ALIASES.items(),
        key=lambda item: len(item[0]),
        reverse=True
    ):

        if clean_room == alias:
            return (
                True,
                official_name,
                None
            )

    return (
        False,
        None,
        "⚠️ We couldn't recognize that room type. "
        "Please choose from:\n"
        "- **Standard Room**\n"
        "- **Deluxe Room**\n"
        "- **Family Suite**\n"
        "- **Ocean Villa**"
    )


# ROOM COUNT VALIDATION
def validate_rooms(
    rooms_input: str | int
) -> tuple[bool, int | None, str | None]:
    """
    Validate number of rooms.

    Current rule:
        1-10 rooms per reservation.
    """

    if rooms_input is None:
        return (
            False,
            None,
            "⚠️ Please specify the number of rooms."
        )

    if isinstance(rooms_input, int):
        num = rooms_input

    else:
        value = str(rooms_input).strip()

        if not re.fullmatch(r"\d+", value):
            return (
                False,
                None,
                "⚠️ Please provide a valid number of rooms."
            )

        num = int(value)

    if num <= 0:
        return (
            False,
            None,
            "⚠️ Number of rooms must be at least **1**."
        )

    if num > 10:
        return (
            False,
            None,
            "⚠️ Please contact the front desk "
            "for reservations of more than 10 rooms."
        )

    return (
        True,
        num,
        None
    )


# NAME VALIDATION
def validate_name(
    name_input: str
) -> tuple[bool, str | None, str | None]:
    """
    Validate guest name.

    Allows:
        John Doe
        Nur Aisyah
        O'Connor
        Muhammad Ali

    Rejects:
        empty strings
        numbers
        obviously invalid characters
    """

    if not name_input or not isinstance(name_input, str):
        return (
            False,
            None,
            "⚠️ Please enter your full name."
        )

    clean_name = re.sub(
        r"\s+",
        " ",
        name_input.strip()
    )

    if len(clean_name) < 2:
        return (
            False,
            None,
            "⚠️ Guest name must contain at least "
            "**2 characters**."
        )

    if len(clean_name) > 80:
        return (
            False,
            None,
            "⚠️ Guest name is too long."
        )

    if clean_name.isdigit():
        return (
            False,
            None,
            "⚠️ Please enter a valid guest name."
        )

    name_pattern = (
        r"^[A-Za-zÀ-ÿ]"
        r"[A-Za-zÀ-ÿ' -]{1,79}$"
    )

    if not re.fullmatch(
        name_pattern,
        clean_name
    ):
        return (
            False,
            None,
            "⚠️ Please enter a valid guest name "
            "(letters, spaces, or apostrophes only)."
        )

    return (
        True,
        clean_name.title(),
        None
    )


# PHONE VALIDATION & DIGIT MATCHING
def clean_phone_digits(phone_str: str) -> str:
    """Extract clean numeric digits from phone string for robust matching."""
    if not phone_str:
        return ""
    digits = re.sub(r"\D", "", str(phone_str))
    # Normalize Malaysian leading 60 (e.g. 60123456789 -> 0123456789)
    if digits.startswith("60") and len(digits) >= 10:
        digits = "0" + digits[2:]
    return digits


def validate_phone(
    phone_input: str
) -> tuple[bool, str | None, str | None]:
    """
    Validate contact phone number.

    Allows:
        +60 12-345 6789
        0123456789
        011-2345678
        +1 555-123-4567

    Rejects:
        numbers under 8 digits or over 16 digits
        letters without numbers
    """
    if not phone_input or not isinstance(phone_input, str):
        return (
            False,
            None,
            "⚠️ Please enter a valid contact phone number."
        )

    clean_digits = re.sub(r"\D", "", phone_input)

    if len(clean_digits) < 8 or len(clean_digits) > 16:
        return (
            False,
            None,
            "⚠️ Invalid phone number length. "
            "Please provide a valid phone number (e.g. **+60 12-345 6789**)."
        )

    return (
        True,
        phone_input.strip(),
        None
    )


# COMPLETE BOOKING VALIDATION
def validate_booking_entities(
    entities: dict,
    reference_date: date | None = None
) -> dict:
    """
    Validate extracted booking entities.

    Returns:
        {
            "valid": bool,
            "validated": {...},
            "errors": {...}
        }
    """

    if reference_date is None:
        reference_date = datetime.now().date()

    validated = {}
    errors = {}

    # Room type
    if entities.get("room_type") is not None:

        valid, value, error = validate_room_type(
            entities["room_type"]
        )

        if valid:
            validated["room_type"] = value
        else:
            errors["room_type"] = error

    # Rooms
    if entities.get("rooms") is not None:

        valid, value, error = validate_rooms(
            entities["rooms"]
        )

        if valid:
            validated["rooms"] = value
        else:
            errors["rooms"] = error

    # Guests
    if entities.get("guests") is not None:

        valid, value, error = validate_guests(
            entities["guests"]
        )

        if valid:
            validated["guests"] = value
        else:
            errors["guests"] = error

    # Check-in
    if entities.get("check_in") is not None:

        valid, value, error = validate_checkin_date(
            entities["check_in"],
            reference_date
        )

        if valid:
            validated["check_in"] = value
        else:
            errors["check_in"] = error

    # Check-out
    if entities.get("check_out") is not None:

        checkin_value = validated.get(
            "check_in",
            entities.get("check_in")
        )

        valid, value, error = validate_checkout_date(
            checkin_value,
            entities["check_out"],
            reference_date
        )

        if valid:
            validated["check_out"] = value
        else:
            errors["check_out"] = error

    # Nights
    if (
        validated.get("check_in")
        and validated.get("check_out")
    ):
        nights = calculate_nights(
            validated["check_in"],
            validated["check_out"]
        )

        if nights is not None:
            validated["nights"] = nights

    elif entities.get("nights") is not None:

        try:
            nights = int(entities["nights"])

            if nights > 0:
                validated["nights"] = nights
            else:
                errors["nights"] = (
                    "⚠️ Number of nights must be at least **1**."
                )

        except (ValueError, TypeError):

            errors["nights"] = (
                "⚠️ Please provide a valid number of nights."
            )

    # Name
    if entities.get("name") is not None:

        valid, value, error = validate_name(
            entities["name"]
        )

        if valid:
            validated["name"] = value
        else:
            errors["name"] = error

    # Contact information
    if entities.get("email") is not None:
        validated["email"] = entities["email"]

    if entities.get("phone") is not None:
        validated["phone"] = entities["phone"]

    # Booking ID
    if entities.get("booking_id") is not None:
        validated["booking_id"] = entities["booking_id"]

    return {
        "valid": len(errors) == 0,
        "validated": validated,
        "errors": errors
    }
