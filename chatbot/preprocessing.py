"""
BookMate Chatbot - Preprocessing & NLP Pipeline
===============================================

Pipeline Steps:
1. Text Normalization: Lowercasing, Domain & Common-Typo Spelling Correction, Basic Cleaning
2. PII Detection & Masking: Detects & masks Email, Credit Card, IC/ID, and Phone numbers
3. Tokenization: Word tokenization
4. Lemmatization: Context-aware lemmatization using POS-tagged WordNetLemmatizer
"""

import os
import re
import string
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.metrics.distance import edit_distance

# Auto-download required NLTK resources silently
REQUIRED_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger")
]

for res_path, res_name in REQUIRED_NLTK_RESOURCES:
    try:
        nltk.data.find(res_path)
    except LookupError:
        try:
            nltk.download(res_name, quiet=True)
        except Exception:
            pass


def _is_luhn_valid(number_str: str) -> bool:
    """Validate numeric string using Luhn algorithm (mod 10)."""
    digits = [int(ch) for ch in number_str if ch.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    reverse_digits = digits[::-1]
    for i, d in enumerate(reverse_digits):
        if i % 2 == 1:
            d = d * 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


class TextPreprocessor:
    """
    NLP Preprocessing Pipeline:
    Text Normalization -> PII Masking -> Tokenization -> Lemmatization
    """

    def __init__(self, enable_spell_check: bool = True):
        self.lemmatizer = WordNetLemmatizer()
        self.enable_spell_check = enable_spell_check

        # Explicit common typos mapping for chatbot domains
        self.common_typos = {
            "helo": "hello",
            "hllo": "hello",
            "hallo": "hello",
            "helooo": "hello",
            "hiii": "hi",
            "heyya": "hey",
            "bookin": "booking",
            "bok": "book",
            "prce": "price",
            "pric": "price",
            "chkin": "checkin",
            "chkout": "checkout",
            "cancle": "cancel",
            "reserv": "reserve",
            "resrvation": "reservation",
            "delux": "deluxe",
            "availble": "available",
            "avialable": "available",
            "paymnt": "payment",
            "locaton": "location"
        }

        # Structured Domain Vocabularies for Proper Nouns and Hotel Concepts
        self.domain_vocab = {
            "hello", "hi", "hey", "booking", "book", "room", "price", "cost",
            "checkin", "checkout", "wifi", "parking", "breakfast", "contact",
            "cancel", "deluxe", "suite", "location", "status", "payment",
            "king", "queen", "minibar", "jacuzzi", "lunch", "dinner",
            "restaurant", "food", "reservation", "amenities", "pool", "view",
            "guests", "person", "people", "night", "nights", "rate", "rates",
            "modify", "change", "update", "deposit", "card", "cash", "reception",
            "frontdesk", "service", "check_in", "check_out", "room_service", "free_wifi"
        }

        self.location_vocab = {
            "bukit", "bintang", "kuala", "lumpur", "klcc", "petaling", "jaya",
            "twin", "tower", "towers", "penang", "langkawi", "malaysia",
            "selangor", "georgetown", "subang", "ttdi", "bangsar", "mont", "kiara",
            "bukit_bintang", "kuala_lumpur", "petaling_jaya", "twin_towers"
        }

        self.hotel_vocab = {
            "bookmate", "oriented", "resort", "standard", "deluxe", "family",
            "ocean", "villa", "executive", "presidential"
        }

        self.brand_vocab = {
            "agoda", "booking", "expedia", "trip", "hilton", "marriott",
            "airbnb", "trivago", "klook", "grab", "traveloka"
        }

        # Build domain vocabulary from dataset for precision spell checking
        self.vocab = set()
        self.vocab.update(self.domain_vocab)
        self.vocab.update(self.location_vocab)
        self.vocab.update(self.hotel_vocab)
        self.vocab.update(self.brand_vocab)

        dataset_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "intents.csv"))
        if os.path.exists(dataset_path):
            try:
                df = pd.read_csv(dataset_path)
                for t in df['text'].dropna():
                    for w in str(t).lower().split():
                        clean_w = re.sub(f"[{re.escape(string.punctuation)}]", "", w)
                        if clean_w:
                            self.vocab.add(clean_w)
            except Exception:
                pass

        # Compound domain phrases (prevent splitting by punctuation/tokenization)
        self.compound_phrases = {
            r"\bcheck[ -]in\b": "check_in",
            r"\bcheck[ -]out\b": "check_out",
            r"\broom[ -]service\b": "room_service",
            r"\bfree[ -]wifi\b": "free_wifi",
            r"\bwi[ -]fi\b": "wifi",
            r"\bair[ -]conditioning\b|\bair[ -]con\b": "air_conditioning",
            r"\bswimming[ -]pool\b": "swimming_pool",
            r"\bocean[ -]view\b": "ocean_view",
            r"\btwin[ -]bed\b": "twin_bed",
            r"\bdouble[ -]bed\b": "double_bed",
            r"\bking[ -]bed\b": "king_bed",
            r"\bqueen[ -]bed\b": "queen_bed",
            r"\bfront[ -]desk\b": "front_desk",
            r"\bsea[ -]view\b": "sea_view",
            r"\bbukit[ -]bintang\b": "bukit_bintang",
            r"\bkuala[ -]lumpur\b": "kuala_lumpur",
            r"\bpetaling[ -]jaya\b": "petaling_jaya",
            r"\btwin[ -]tower\b|\btwin[ -]towers\b": "twin_towers"
        }

        # Spoken number word normalization (e.g. "two guests" -> "2 guests")
        self.number_words = {
            r"\bone\b": "1",
            r"\btwo\b": "2",
            r"\bthree\b": "3",
            r"\bfour\b": "4",
            r"\bfive\b": "5",
            r"\bsix\b": "6",
            r"\bseven\b": "7",
            r"\beight\b": "8",
            r"\bnine\b": "9",
            r"\bten\b": "10"
        }

        # PII Regex Patterns (ordered by specificity)
        self.pii_patterns = [
            ("CREDIT_CARD", r"\b(?:\d[ -]*?){13,19}\b"),
            ("EMAIL", r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
            ("BOOKING_REFERENCE", r"\b(?:BK|BM|RES)[ -]?\d{4,8}\b"),
            ("PASSPORT", r"\b[A-PR-WYa-pr-wy][0-9]{7,8}\b"),
            ("IC_ID", r"\b\d{6}-\d{2}-\d{4}\b"),
            ("PHONE", r"\b(?:\+?60|0)1[0-9][-.\s]?\d{3,4}[-.\s]?\d{3,4}\b|\b(?:\+?60|0)[3-9][-.\s]?\d{3,4}[-.\s]?\d{3,4}\b|\b(?:\+\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b"),
            ("ADDRESS", r"\b(?:no\.?\s*\d+\s*,?\s*jalan\s+[a-zA-Z0-9\s]+|jalan\s+[a-zA-Z0-9\s]+|street|avenue|road|st\.?|rd\.?|lorong\s+[a-zA-Z0-9\s]+)\b")
        ]

    def _get_wordnet_pos(self, tag: str):
        """Map NLTK POS tag to WordNet POS tag for accurate lemmatization."""
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN

    def _correct_word_spelling(self, word: str) -> str:
        """
        Domain-aware & Dictionary-based spell checking:
        1. Explicit common typo dictionary match (`self.common_typos`).
        2. Keep word if in domain vocabulary (`self.vocab`), short (len <= 2),
           or valid English word in WordNet dictionary.
        3. Strict edit-distance match (dist == 1) for unknown non-dictionary words against domain vocabulary.
        4. No TextBlob fallback (prevents distorting out-of-vocabulary terms like 'lunch').
        """
        clean_word = word.strip(string.punctuation).lower()

        if not clean_word or clean_word.isdigit() or "_" in clean_word:
            return word

        # 1. Check explicit typo dictionary
        if clean_word in self.common_typos:
            corrected = self.common_typos[clean_word]
            return word.lower().replace(clean_word, corrected)

        # 2. Keep if already in domain vocabulary or short word
        if clean_word in self.vocab or len(clean_word) <= 2:
            return word

        # 3. Keep if valid English word in WordNet dictionary
        try:
            if wordnet.synsets(clean_word):
                return word
        except Exception:
            pass

        # 4. Strict edit distance match (distance == 1 only) against domain vocabulary
        if len(clean_word) >= 4:
            candidates = []
            for target in self.vocab:
                if abs(len(clean_word) - len(target)) <= 1:
                    dist = edit_distance(clean_word, target)
                    if dist == 1:
                        candidates.append(target)

            if len(candidates) == 1:
                return word.lower().replace(clean_word, candidates[0])

        # Return original word if no match (No TextBlob fallback)
        return word

    def detect_and_mask_pii(self, text: str) -> tuple[str, dict]:
        """
        Detect and mask Personally Identifiable Information (PII).

        Returns:
            masked_text (str): Text with PII replaced by tokens (e.g., [EMAIL]).
            detected_pii (dict): Dictionary mapping PII types to matched values.
        """
        masked_text = text
        detected_pii = {}

        for pii_type, pattern in self.pii_patterns:
            matches = re.findall(pattern, masked_text, re.IGNORECASE)
            if matches:
                filtered_matches = []
                for m in matches:
                    m_str = str(m)
                    if pii_type == "CREDIT_CARD":
                        digits_only = re.sub(r'\D', '', m_str)
                        if _is_luhn_valid(digits_only):
                            filtered_matches.append(m_str)
                    elif pii_type == "PHONE":
                        digits_only = re.sub(r'\D', '', m_str)
                        if len(digits_only) >= 8:
                            filtered_matches.append(m_str)
                    else:
                        filtered_matches.append(m_str)

                if filtered_matches:
                    detected_pii[pii_type] = filtered_matches
                    for m in filtered_matches:
                        masked_text = masked_text.replace(str(m), f"[{pii_type}]")

        return masked_text, detected_pii

    def normalize_text(self, text: str) -> str:
        """
        Text Normalization:
        - Lowercasing
        - Compound Phrase & Spoken Number Normalization
        - Spelling Correction (Domain & Typo Aware)
        - Basic Cleaning (removing extra punctuation/spaces while preserving compounds & PII)
        """
        # 1. Lowercasing
        text = text.lower()

        # 2. Compound Phrase & Spoken Number Normalization
        for pattern, replacement in self.compound_phrases.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        for pattern, replacement in self.number_words.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 3. Spelling Correction
        if self.enable_spell_check:
            words = text.split()
            corrected_words = []
            for w in words:
                if (w.startswith("[") and w.endswith("]")) or "_" in w:
                    corrected_words.append(w)
                else:
                    corrected_words.append(self._correct_word_spelling(w))
            text = " ".join(corrected_words)

        # 4. Basic Cleaning (keep PII bracket tags intact)
        pii_tokens = re.findall(r"\[[A-Z_]+\]", text)
        for i, tag in enumerate(pii_tokens):
            text = text.replace(tag, f" PII_TOKEN_{i} ")

        # Remove punctuation except underscores (preserves check_in, room_service, etc.)
        text = re.sub(r"[^\w\s_]", " ", text)

        # Restore PII tokens
        for i, tag in enumerate(pii_tokens):
            clean_tag_name = tag.strip("[]").lower()
            text = text.replace(f" PII_TOKEN_{i} ", f" {clean_tag_name} ")

        # Clean whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text


    def tokenize(self, text: str) -> list[str]:
        """Tokenize normalized text into a list of word tokens."""
        try:
            return word_tokenize(text)
        except Exception:
            return text.split()

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Lemmatize word tokens using context POS tagging and WordNetLemmatizer."""
        try:
            pos_tags = nltk.pos_tag(tokens)
            lemmatized = [
                self.lemmatizer.lemmatize(token, self._get_wordnet_pos(tag))
                for token, tag in pos_tags
            ]
        except Exception:
            lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]

        return lemmatized

    def process(self, text: str) -> dict:
        """
        Execute full NLP Preprocessing Pipeline.

        Returns dict:
        - original_text
        - pii_masked_text
        - detected_pii
        - normalized_text
        - tokens
        - lemmatized_tokens
        - preprocessed_text (string ready for ML intent classification)
        """
        masked_text, detected_pii = self.detect_and_mask_pii(text)
        normalized_text = self.normalize_text(masked_text)
        tokens = self.tokenize(normalized_text)
        lemmatized_tokens = self.lemmatize(tokens)
        preprocessed_text = " ".join(lemmatized_tokens)

        return {
            "original_text": text,
            "pii_masked_text": masked_text,
            "detected_pii": detected_pii,
            "normalized_text": normalized_text,
            "tokens": tokens,
            "lemmatized_tokens": lemmatized_tokens,
            "preprocessed_text": preprocessed_text
        }


# Default global instance (Spell checking enabled by default)
_default_preprocessor = TextPreprocessor(enable_spell_check=True)


def preprocess_text(text: str) -> str:
    """
    Convenience function returning preprocessed string for model training & prediction.
    """
    return _default_preprocessor.process(text)["preprocessed_text"]


def process_input(text: str) -> dict:
    """
    Convenience function returning detailed result dict of full pipeline.
    """
    return _default_preprocessor.process(text)


if __name__ == "__main__":
    test_cases = [
        "helo, I would like to book a room at BookMate near Bukit Bintang",
        "I need a room with free wifi and room service for two guests",
        "What is the check-in time and check out policy?",
        "My email is john.doe@gmail.com, booking ID BK1234, and phone is +6012-3456789.",
        "My card is 4532015112830366 (valid Luhn) and fake is 1234567890123456 (invalid Luhn)."
    ]

    print("=== Testing Domain-Aware Spelling & Enhanced NLP Preprocessing Pipeline ===")
    for text in test_cases:
        res = process_input(text)
        print(f"\n[Original]   : {res['original_text']}")
        print(f"[PII Masked] : {res['pii_masked_text']}")
        print(f"[Detected PII]: {res['detected_pii']}")
        print(f"[Normalized] : {res['normalized_text']}")
        print(f"[Lemmatized] : {res['preprocessed_text']}")

