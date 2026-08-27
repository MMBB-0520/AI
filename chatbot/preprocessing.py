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

        # Informal expression normalization
        self.informal_expressions = {
            r"\bwanna\b": "want to",
            r"\bgonna\b": "going to",
            r"\bgotta\b": "got to",
            r"\blemme\b": "let me",
            r"\bgimme\b": "give me",
            r"\bkinda\b": "kind of",
            r"\bsorta\b": "sort of"
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

        for pattern, replacement in self.informal_expressions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        for pattern, replacement in self.number_words.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 3. Spelling Correction
        if self.enable_spell_check:
            words = text.split()
            corrected_words = []

            for w in words:
                clean_word = w.strip(string.punctuation).lower()

                if clean_word in self.common_typos:
                    corrected_words.append(
                        self.common_typos[clean_word]
                    )
                else:
                    corrected_words.append(w)

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
        "I want to book a hotel for two nights",
        "wanna check my invocies where could i do it",
        "Can I cancel my hotel reservation?",
        "What time is check in?",
        "Can I bring my pet?",
        "Do you have free parking?",
        "I need to change my reservation",
        "My email is john.doe@gmail.com and my phone is +6012-3456789"
    ]

    print("=== BookMate Preprocessing Test ===")

    for text in test_cases:
        result = process_input(text)

        print(f"\nOriginal: {result['original_text']}")
        print(f"Masked: {result['pii_masked_text']}")
        print(f"Normalized: {result['normalized_text']}")
        print(f"Tokens: {result['tokens']}")
        print(f"Lemmatized: {result['preprocessed_text']}")