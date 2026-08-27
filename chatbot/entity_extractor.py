"""
BookMate Chatbot
Entity Extraction

Purpose:
Extract key domain entities/slots from user input
for hotel reservation and inquiry handling.

Entities:
- booking_id
- room_type
- rooms
- guests
- check_in
- check_out
- nights
- name
- email
- phone
"""

import re
import sys
import os
from datetime import datetime, timedelta

# Ensure parent directory is available for imports
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

try:
    from chatbot.hotel_info import ROOM_TYPES, ROOM_TYPE_ALIASES
except ImportError:
    # Fallback definitions if hotel_info does not export these names
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



class EntityExtractor:
    """
    Rule-based and regex-based entity extractor
    for BookMate hotel chatbot.
    """

    def __init__(self):
        self.room_types_map = ROOM_TYPE_ALIASES

        self.word_to_num = {
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10
        }

    # ROOM TYPE
    def extract_room_type(self, text: str) -> str | None:
        """
        Extract official hotel room type.

        Example:
            "I want a deluxe room"
            -> "Deluxe Room"
        """

        if not text:
            return None

        text_lower = text.lower()

        # Longer phrases first
        for alias in sorted(
            self.room_types_map.keys(),
            key=len,
            reverse=True
        ):
            pattern = r"\b" + re.escape(alias) + r"\b"

            if re.search(pattern, text_lower):
                return self.room_types_map[alias]

        return None

    # NUMBER HELPER
    def _extract_number(self, text: str) -> int | None:
        """
        Extract a numeric value from text.

        Supports:
            2
            two
            5
            five
        """

        if not text:
            return None

        match = re.search(r"\b(\d+)\b", text)

        if match:
            return int(match.group(1))

        text_lower = text.lower()

        for word, number in self.word_to_num.items():
            if re.search(r"\b" + word + r"\b", text_lower):
                return number

        return None

    # ROOMS
    def extract_rooms(self, text: str) -> int | None:
        """
        Extract number of rooms.

        Examples:
            "2 rooms" -> 2
            "two rooms" -> 2
            "I need 3 room" -> 3
        """

        if not text:
            return None

        text_lower = text.lower()

        match = re.search(
            r"\b(\d+)\s*(?:room|rooms)\b",
            text_lower
        )

        if match:
            return int(match.group(1))

        for word, number in self.word_to_num.items():
            pattern = (
                r"\b"
                + word
                + r"\s+(?:room|rooms)\b"
            )

            if re.search(pattern, text_lower):
                return number

        return None

    # GUESTS
    def extract_guests(self, text: str) -> int | None:
        """
        Extract number of guests.

        Supported examples:
            "2 guests" -> 2
            "3 people" -> 3
            "4 pax" -> 4
            "2 adults" -> 2
            "for 5 people" -> 5
            "for 2 adults" -> 2

        Avoids interpreting:
            "for 3 nights"
            "for 2 rooms"
        as guest count.
        """

        if not text:
            return None

        text_lower = text.lower()

        # 1. Explicit guest expressions
        numeric_match = re.search(
            r"\b(\d+)\s*"
            r"(?:people|person|guests?|pax|adults?)\b",
            text_lower
        )

        if numeric_match:
            return int(numeric_match.group(1))

        # 2. Word-number guest expressions
        for word, number in self.word_to_num.items():

            pattern = (
                r"\b"
                + word
                + r"\s+"
                r"(?:people|person|guests?|pax|adults?)\b"
            )

            if re.search(pattern, text_lower):
                return number

        # 3. "for N" booking context

        # Only accept if N is NOT followed by:
        # nights / rooms / days
        numeric_for_match = re.search(
            r"\b(?:for|with)\s+(\d+)"
            r"(?!\s*(?:night|nights|room|rooms|day|days))"
            r"(?:\s*(?:people|person|guests?|pax|adults?))?\b",
            text_lower
        )

        if numeric_for_match:
            return int(numeric_for_match.group(1))

        return None
        
    # NIGHTS
    def extract_nights(self, text: str) -> int | None:
        """
        Extract explicit number of nights.

        Examples:
            "3 nights" -> 3
            "two nights" -> 2

        Note:
            "3 days" is NOT treated as 3 nights.
        """

        if not text:
            return None

        text_lower = text.lower()

        match = re.search(
            r"\b(\d+)\s*(?:night|nights)\b",
            text_lower
        )

        if match:
            return int(match.group(1))

        for word, number in self.word_to_num.items():
            pattern = (
                r"\b"
                + word
                + r"\s+(?:night|nights)\b"
            )

            if re.search(pattern, text_lower):
                return number

        return None

    # DATE EXTRACTION
    def _normalize_date(
        self,
        date_string: str,
        reference_date: datetime | None = None
    ) -> str | None:
        """
        Convert recognized date expressions into YYYY-MM-DD.

        Supports:
            YYYY-MM-DD
            YYYY/MM/DD
            DD-MM-YYYY
            DD/MM/YYYY
            DD.MM.YYYY
            today
            tomorrow
        """

        if not date_string:
            return None

        if reference_date is None:
            reference_date = datetime.now()

        value = date_string.strip().lower()

        # Relative dates
        if value == "today":
            return reference_date.strftime("%Y-%m-%d")

        if value == "tomorrow":
            return (
                reference_date + timedelta(days=1)
            ).strftime("%Y-%m-%d")

        # Explicit date formats
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y.%m.%d",

            "%d-%m-%Y",
            "%d/%m/%Y",
            "%d.%m.%Y",

            "%d-%m-%y",
            "%d/%m/%y"
        ]

        for fmt in formats:
            try:
                parsed = datetime.strptime(value, fmt)
                return parsed.strftime("%Y-%m-%d")
            except ValueError:
                continue

        return None

    def extract_dates(
        self,
        text: str,
        reference_date: datetime | None = None
    ) -> dict:
        """
        Extract check-in and check-out dates.

        Returns:
            {
                "check_in": str | None,
                "check_out": str | None
            }
        """

        results = {
            "check_in": None,
            "check_out": None
        }

        if not text:
            return results

        if reference_date is None:
            reference_date = datetime.now()

        text_lower = text.lower()

        # 1. Date regex (ISO YYYY-MM-DD / YYYY/MM/DD and UK/EU DD/MM/YYYY / DD-MM-YYYY)
        dates_found = re.findall(
            r"\b(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})\b",
            text_lower
        )

        if len(dates_found) >= 2:
            results["check_in"] = self._normalize_date(
                dates_found[0],
                reference_date
            )

            results["check_out"] = self._normalize_date(
                dates_found[1],
                reference_date
            )

            return results

        if len(dates_found) == 1:
            results["check_in"] = self._normalize_date(
                dates_found[0],
                reference_date
            )

        # 2. from X to Y
        from_to_match = re.search(
            r"\bfrom\s+(.+?)\s+to\s+(.+?)(?:\.|,|$)",
            text_lower
        )

        if from_to_match:
            checkin_raw = from_to_match.group(1).strip()
            checkout_raw = from_to_match.group(2).strip()

            checkin = self._normalize_date(
                checkin_raw,
                reference_date
            )

            checkout = self._normalize_date(
                checkout_raw,
                reference_date
            )

            if checkin:
                results["check_in"] = checkin

            if checkout:
                results["check_out"] = checkout

            return results

        # 3. tomorrow / today
        if re.search(r"\btomorrow\b", text_lower):
            results["check_in"] = (
                reference_date + timedelta(days=1)
            ).strftime("%Y-%m-%d")

        elif re.search(r"\btoday\b", text_lower):
            results["check_in"] = (
                reference_date
            ).strftime("%Y-%m-%d")

        return results

    # NAME
    def extract_name(self, text: str) -> str | None:
        """
        Extract guest name only from explicit name patterns.

        Examples:
            "My name is John Doe"
            "Name: John Doe"
            "Please book under John Doe"
        """

        if not text:
            return None

        patterns = [
            r"\bmy\s+name\s+is\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ' -]{1,79})",
            r"\bmy\s+name's\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ' -]{1,79})",
            r"\bname\s*:\s*([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ' -]{1,79})",
            r"\bname\s+is\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ' -]{1,79})",
            r"\bbook\s+(?:the\s+)?reservation\s+under\s+([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ' -]{1,79})"
        ]

        for pattern in patterns:
            match = re.search(
                pattern,
                text,
                re.IGNORECASE
            )

            if match:
                name = match.group(1).strip()

                # Remove trailing booking-related words
                name = re.split(
                    r"\s+(?:and|with|for|email|phone)\b",
                    name,
                    flags=re.IGNORECASE
                )[0].strip()

                if len(name) >= 2 and not name.isdigit():
                    return name.title()

        return None

    # CONTACT INFORMATION
    def extract_contact_info(self, text: str) -> dict:
        """
        Extract email and phone number.
        """

        if not text:
            return {
                "email": None,
                "phone": None
            }

        email_pattern = (
            r"[a-zA-Z0-9._%+-]+"
            r"@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        )

        phone_pattern = (
            r"(?<!\d)"
            r"(?:\+?60|0)"
            r"[\s.-]?"
            r"\d{1,3}"
            r"[\s.-]?"
            r"\d{3,4}"
            r"[\s.-]?"
            r"\d{3,4}"
            r"(?!\d)"
        )

        emails = re.findall(
            email_pattern,
            text
        )

        phones = re.findall(
            phone_pattern,
            text
        )

        valid_phones = []

        for phone in phones:
            digits = re.sub(r"\D", "", phone)

            if 9 <= len(digits) <= 12:
                valid_phones.append(phone.strip())

        return {
            "email": emails[0] if emails else None,
            "phone": valid_phones[0] if valid_phones else None
        }

    # BOOKING ID
    def extract_booking_id(self, text: str) -> str | None:
        """
        Extract booking ID.

        Examples:
            BK1234
            BK-1234
            BK_1234
        """

        if not text:
            return None

        match = re.search(
            r"\b(BK[-_]?\d{4,6})\b",
            text,
            re.IGNORECASE
        )

        if match:
            return (
                match.group(1)
                .upper()
                .replace("-", "")
                .replace("_", "")
            )

        return None

    # CALCULATE NIGHTS
    def calculate_nights(
        self,
        check_in: str | None,
        check_out: str | None
    ) -> int | None:
        """
        Calculate number of nights from check-in/check-out dates.
        """

        if not check_in or not check_out:
            return None

        try:
            checkin_date = datetime.strptime(
                check_in,
                "%Y-%m-%d"
            ).date()

            checkout_date = datetime.strptime(
                check_out,
                "%Y-%m-%d"
            ).date()

            nights = (checkout_date - checkin_date).days

            if nights > 0:
                return nights

        except ValueError:
            pass

        return None

    # EXTRACT ALL
    def extract_all(
        self,
        text: str,
        reference_date: datetime | None = None
    ) -> dict:
        """
        Perform complete entity extraction.
        """

        dates = self.extract_dates(
            text,
            reference_date
        )

        contact = self.extract_contact_info(text)

        calculated_nights = self.calculate_nights(
            dates["check_in"],
            dates["check_out"]
        )

        explicit_nights = self.extract_nights(text)

        # Prefer calculated nights when both dates exist
        nights = (
            calculated_nights
            if calculated_nights is not None
            else explicit_nights
        )

        entities = {
            "booking_id": self.extract_booking_id(text),

            "room_type": self.extract_room_type(text),
            "rooms": self.extract_rooms(text),
            "guests": self.extract_guests(text),

            "check_in": dates["check_in"],
            "check_out": dates["check_out"],
            "nights": nights,

            "name": self.extract_name(text),

            "email": contact["email"],
            "phone": contact["phone"]
        }

        entities_found = {
            key: value
            for key, value in entities.items()
            if value is not None
        }

        return {
            "all_entities": entities,
            "entities_found": entities_found
        }

# Default global instance
_default_extractor = EntityExtractor()

def extract_entities(
    text: str,
    reference_date: datetime | None = None
) -> dict:
    """
    Convenience function returning extracted entities.
    """

    return _default_extractor.extract_all(
        text,
        reference_date
    )

# Testing
if __name__ == "__main__":
    test_reference_date = datetime(2026, 8, 12)

    test_queries = [
        "I want to book a Deluxe room for 2 guests from 2026-10-01 to 2026-10-05.",

        "I need 2 standard rooms for 4 people.",

        "How much is the ocean villa for 3 nights?",

        "Can I reserve a family suite for 4 people starting tomorrow?",

        "My name is John Doe, email john@example.com, phone +60123456789.",

        "Please check my booking BK12345."
    ]

    print("=== Testing EntityExtractor ===")

    for query in test_queries:

        result = extract_entities(
            query,
            reference_date=test_reference_date
        )

        print(f"\nQuery: {query}")
        print(
            "Extracted Entities:",
            result["entities_found"]
        )