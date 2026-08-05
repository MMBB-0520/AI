"""
BookMate Chatbot - Entity Extraction Module
===========================================

Purpose:
Extract key domain entities/slots from user input text for hotel reservation and inquiry handling.

Entities Extracted:
- room_type (Standard Room, Deluxe Room, Family Suite, Ocean Villa)
- guests (number of guests/people)
- check_in / check_out (dates)
- nights (number of stay nights)
- name (guest name)
- phone / email (contact information)
"""

import re
import sys
import os
from datetime import datetime, timedelta

# Ensure parent directory is in sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chatbot.hotel_info import ROOM_PRICES


class EntityExtractor:
    """
    Rule-based & Regex Domain Entity Extractor for BookMate Chatbot.
    """

    def __init__(self):
        # Room Type mapping (normalized search term -> official room name)
        self.room_types_map = {
            "standard": "Standard Room",
            "standard room": "Standard Room",
            "deluxe": "Deluxe Room",
            "deluxe room": "Deluxe Room",
            "family": "Family Suite",
            "family suite": "Family Suite",
            "suite": "Family Suite",
            "ocean villa": "Ocean Villa",
            "ocean": "Ocean Villa",
            "villa": "Ocean Villa"
        }

    def extract_room_type(self, text: str) -> str | None:
        """Extract room type from user message."""
        text_lower = text.lower()
        # Sort keys by length descending to prioritize longer phrases like 'deluxe room' over 'deluxe'
        for key in sorted(self.room_types_map.keys(), key=len, reverse=True):
            pattern = r'\b' + re.escape(key) + r'\b'
            if re.search(pattern, text_lower):
                return self.room_types_map[key]
        return None

    def extract_guests(self, text: str) -> int | None:
        """Extract guest count from user message."""
        text_lower = text.lower()

        # Match phrases: "2 guests", "3 people", "1 person", "for 4 pax", "2 adults"
        match = re.search(r'\b(\d+)\s*(?:people|person|guest|guests|pax|adult|adults)\b', text_lower)
        if match:
            return int(match.group(1))

        # Match "for N" / "with N" in booking context
        match = re.search(r'\b(?:for|with)\s+(\d+)\b', text_lower)
        if match:
            return int(match.group(1))

        # Word numbers: "two guests", "three people"
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        for word, num in word_to_num.items():
            pattern = r'\b' + word + r'\s+(?:people|person|guest|guests|pax|adult|adults)\b'
            if re.search(pattern, text_lower):
                return num

        return None

    def extract_nights(self, text: str) -> int | None:
        """Extract stay duration in nights."""
        text_lower = text.lower()
        match = re.search(r'\b(\d+)\s*(?:night|nights|day|days)\b', text_lower)
        if match:
            return int(match.group(1))

        word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
        for word, num in word_to_num.items():
            pattern = r'\b' + word + r'\s+(?:night|nights)\b'
            if re.search(pattern, text_lower):
                return num
        return None

    def extract_dates(self, text: str) -> dict:
        """
        Extract check-in and check-out dates.
        Returns dict: {'check_in': str|None, 'check_out': str|None}
        """
        text_lower = text.lower()
        results = {"check_in": None, "check_out": None}

        # 1. Match ISO formatted dates (e.g., 2026-10-01 or 2026/10/01)
        iso_dates = re.findall(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', text)
        if len(iso_dates) >= 2:
            results["check_in"] = iso_dates[0]
            results["check_out"] = iso_dates[1]
            return results
        elif len(iso_dates) == 1:
            results["check_in"] = iso_dates[0]

        # 2. Match "from [DATE] to [DATE]" pattern
        from_to_match = re.search(r'from\s+([a-zA-Z0-9/\-\s,]+?)\s+to\s+([a-zA-Z0-9/\-\s,]+)', text_lower)
        if from_to_match:
            cin_str = from_to_match.group(1).strip()
            cout_str = from_to_match.group(2).strip()
            results["check_in"] = cin_str
            results["check_out"] = cout_str
            return results

        # 3. Relative date terms
        today = datetime.now()
        if "tomorrow" in text_lower:
            results["check_in"] = (today + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "today" in text_lower:
            results["check_in"] = today.strftime("%Y-%m-%d")

        return results

    def extract_name(self, text: str) -> str | None:
        """Extract guest name if explicitly introduced."""
        patterns = [
            r"\bmy name is ([a-zA-Z\s]+?)(?:\.|$|,| and)",
            r"\bi am ([a-zA-Z\s]+?)(?:\.|$|,| and)",
            r"\bcall me ([a-zA-Z\s]+?)(?:\.|$|,| and)",
            r"\bname[:\s]+([a-zA-Z\s]+?)(?:\.|$|,)"
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                name = match.group(1).strip()
                # Filter out non-name expressions
                if name.lower() not in ["looking", "trying", "booking", "here", "interested"]:
                    return name.title()
        return None

    def extract_contact_info(self, text: str) -> dict:
        """Extract email and phone number."""
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        phone_pattern = r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'

        emails = re.findall(email_pattern, text)
        phones = re.findall(phone_pattern, text)

        valid_phones = [p for p in phones if len(re.sub(r'\D', '', p)) >= 7]

        return {
            "email": emails[0] if emails else None,
            "phone": valid_phones[0] if valid_phones else None
        }

    def extract_booking_id(self, text: str) -> str | None:
        """Extract booking ID (e.g., BK1234)."""
        match = re.search(r'\b(BK[-_]?\d{4,6})\b', text, re.IGNORECASE)
        if match:
            return match.group(1).upper().replace("-", "").replace("_", "")
        return None

    def extract_all(self, text: str) -> dict:
        """
        Perform complete entity extraction on user text.

        Returns:
            dict containing:
            - all_entities: dict with all entity keys (value is None if not found)
            - entities_found: dict containing only successfully extracted entities
        """
        dates = self.extract_dates(text)
        contact = self.extract_contact_info(text)

        entities = {
            "booking_id": self.extract_booking_id(text),
            "room_type": self.extract_room_type(text),
            "guests": self.extract_guests(text),
            "nights": self.extract_nights(text),
            "check_in": dates["check_in"],
            "check_out": dates["check_out"],
            "name": self.extract_name(text),
            "email": contact["email"],
            "phone": contact["phone"]
        }

        entities_found = {k: v for k, v in entities.items() if v is not None}

        return {
            "all_entities": entities,
            "entities_found": entities_found
        }

# Default global instance
_default_extractor = EntityExtractor()


def extract_entities(text: str) -> dict:
    """
    Convenience function returning dictionary of extracted entities from user text.
    """
    return _default_extractor.extract_all(text)


if __name__ == "__main__":
    test_queries = [
        "I want to book a Deluxe room for 2 guests from 2026-10-01 to 2026-10-05.",
        "My name is John Doe, email john@example.com, phone +60123456789.",
        "How much is the ocean villa for 3 nights?",
        "Can I reserve a family suite for 4 people starting tomorrow?"
    ]

    print("=== Testing EntityExtractor ===")
    for q in test_queries:
        res = extract_entities(q)
        print(f"\nQuery: '{q}'")
        print(f"Extracted Entities: {res['entities_found']}")
