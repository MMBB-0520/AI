"""
BookMate Chatbot - Text Preprocessing Pipeline
==============================================

Pipeline Steps:
1. PII Detection & Masking: Detects & masks Email, Credit Card, IC/ID, and Phone numbers
2. Text Normalization:
   - Repeated Character Reduction (e.g. 'helloooo' -> 'hello')
   - Contraction Expansion (e.g. "can't" -> "cannot", "I'd" -> "I would")
   - Chat Slang & Abbreviation Expansion (e.g. 'pls' -> 'please', 'thx' -> 'thank you')
   - Domain-Specific Spelling Correction (Hotel, Room, Facilities, Cancellation)
   - Compound Phrase & Number Word Binding
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

# Auto-download required NLTK resources silently
REQUIRED_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("sentiment/vader_lexicon", "vader_lexicon")
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

        # 1. Contraction Expansion Patterns
        self.contractions = {
            r"\bcan'?t\b": "cannot",
            r"\bwon'?t\b": "will not",
            r"\bshan'?t\b": "shall not",
            r"\bshouldn'?t\b": "should not",
            r"\bcouldn'?t\b": "could not",
            r"\bwouldn'?t\b": "would not",
            r"\bdon'?t\b": "do not",
            r"\bdoesn'?t\b": "does not",
            r"\bdidn'?t\b": "did not",
            r"\bisn'?t\b": "is not",
            r"\baren'?t\b": "are not",
            r"\bwasn'?t\b": "was not",
            r"\bweren'?t\b": "were not",
            r"\bhaven'?t\b": "have not",
            r"\bhasn'?t\b": "has not",
            r"\bhadn'?t\b": "had not",
            r"\bi'?m\b": "i am",
            r"\byou'?re\b": "you are",
            r"\bwe'?re\b": "we are",
            r"\bthey'?re\b": "they are",
            r"\bit'?s\b": "it is",
            r"\bthat'?s\b": "that is",
            r"\bwhat'?s\b": "what is",
            r"\bthere'?s\b": "there is",
            r"\bwho'?s\b": "who is",
            r"\bhow'?s\b": "how is",
            r"\bi'?d\b": "i would",
            r"\byou'?d\b": "you would",
            r"\bhe'?d\b": "he would",
            r"\bshe'?d\b": "she would",
            r"\bwe'?d\b": "we would",
            r"\bthey'?d\b": "they would",
            r"\bi'?ll\b": "i will",
            r"\byou'?ll\b": "you will",
            r"\bhe'?ll\b": "he will",
            r"\bshe'?ll\b": "she will",
            r"\bwe'?ll\b": "we will",
            r"\bthey'?ll\b": "they will",
            r"\bi'?ve\b": "i have",
            r"\byou'?ve\b": "you have",
            r"\bwe'?ve\b": "we have",
            r"\bthey'?ve\b": "they have",
            r"\blet'?s\b": "let us"
        }

        # 2. Chat Slang & Informal Expressions
        self.chat_slang = {
            r"\bpls\b|\bplz\b": "please",
            r"\bthx\b|\bty\b|\btq\b|\bthanx\b": "thank you",
            r"\basap\b": "as soon as possible",
            r"\bpromo\b": "promotion",
            r"\binfo\b": "information",
            r"\bpic\b|\bpics\b": "picture",
            r"\bappt\b": "appointment",
            r"\bidk\b": "i do not know",
            r"\brsv\b": "reserve",
            r"\brsvp\b": "reserve",
            r"\bwanna\b": "want to",
            r"\bgonna\b": "going to",
            r"\bgotta\b": "got to",
            r"\blemme\b": "let me",
            r"\bgimme\b": "give me",
            r"\bkinda\b": "kind of",
            r"\bsorta\b": "sort of"
        }

        # 3. Expanded Hotel Domain Typos
        self.common_typos = {
            # Greetings
            "helo": "hello", "hllo": "hello", "hallo": "hello", "helooo": "hello", "hiii": "hi", "heyya": "hey",
            # Booking & Reserving
            "bookin": "booking", "bok": "book", "boking": "booking", "reserv": "reserve", "resrv": "reserve",
            "resrvation": "reservation", "reseveration": "reservation", "reseravtion": "reservation",
            # Pricing & Rooms
            "prce": "price", "pric": "price", "prces": "prices", "delux": "deluxe", "dlx": "deluxe",
            "suit": "suite", "vila": "villa", "stadard": "standard", "standerd": "standard",
            "accomodation": "accommodation", "acommodation": "accommodation", "availble": "available",
            "avialable": "available", "availibility": "availability", "availablity": "availability",
            # Dates & Times
            "chkin": "checkin", "chkout": "checkout", "checkn": "checkin", "chekin": "checkin",
            "chekout": "checkout", "tomorow": "tomorrow", "tmrw": "tomorrow", "tonite": "tonight",
            # Facilities & Services
            "swiming": "swimming", "swimingpool": "swimming pool", "facilites": "facilities",
            "facilties": "facilities", "fcilities": "facilities", "restaraunt": "restaurant",
            "resturant": "restaurant", "restraunt": "restaurant", "breakfst": "breakfast",
            "brekfast": "breakfast", "brakefast": "breakfast", "parkng": "parking",
            "shutle": "shuttle", "shytle": "shuttle", "lugage": "luggage", "bagage": "baggage",
            # Cancellation & Finance
            "cancle": "cancel", "cnacel": "cancel", "cancelation": "cancellation", "cancelling": "canceling",
            "invoce": "invoice", "invocies": "invoices", "reciept": "receipt", "recipt": "receipt",
            "paymnt": "payment", "paymet": "payment", "refnd": "refund",
            # Communication & Local
            "complante": "complaint", "complain": "complaint", "locaton": "location", "adress": "address",
            "servise": "service", "custumer": "customer"
        }

        # 4. Compound Domain Phrases (Preserves key multi-word phrases)
        self.compound_phrases = {
            r"\boriented[ -]resort\b": "oriented_resort",
            r"\bpantai[ -]cenang\b": "pantai_cenang",
            r"\blangkawi\b": "langkawi",
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
            r"\bduitnow[ -]?qr\b": "duitnow_qr"
        }

        # 5. Spoken Number Words
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

        # 6. PII Regex Patterns
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

    def _normalize_elongations(self, text: str) -> str:
        """Reduce 3+ repeated characters (e.g. 'helloooo' -> 'hello', 'plssss' -> 'pls')."""
        return re.sub(r'([a-zA-Z])\1{2,}', r'\1\1', text)

    def detect_and_mask_pii(self, text: str) -> tuple[str, dict]:
        """
        Detect and mask Personally Identifiable Information (PII).
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
        - Repeated Character Reduction
        - Contraction Expansion
        - Slang / Abbreviation Normalization
        - Compound Phrase & Number Word Binding
        - Spelling Correction (Domain & Typo Aware)
        - Basic Cleaning
        """
        # 1. Lowercasing
        text = text.lower()

        # 2. Repeated character reduction (e.g. 'helooo' -> 'heloo')
        text = self._normalize_elongations(text)

        # 3. Contraction Expansion (e.g. "can't" -> "cannot", "i'm" -> "i am")
        for pattern, replacement in self.contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 4. Chat Slang & Informal Expression Normalization
        for pattern, replacement in self.chat_slang.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 5. Compound Domain Phrases & Numbers
        for pattern, replacement in self.compound_phrases.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        for pattern, replacement in self.number_words.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # 6. Spelling Correction
        if self.enable_spell_check:
            words = text.split()
            corrected_words = []
            for w in words:
                clean_word = w.strip(string.punctuation).lower()
                if clean_word in self.common_typos:
                    corrected_words.append(self.common_typos[clean_word])
                else:
                    corrected_words.append(w)
            text = " ".join(corrected_words)

        # 7. Basic Cleaning (preserving PII tokens and underscores)
        pii_tokens = re.findall(r"\[[A-Z_]+\]", text)
        for i, tag in enumerate(pii_tokens):
            text = text.replace(tag, f" PII_TOKEN_{i} ")

        text = re.sub(r"[^\w\s_]", " ", text)

        for i, tag in enumerate(pii_tokens):
            clean_tag_name = tag.strip("[]").lower()
            text = text.replace(f" PII_TOKEN_{i} ", f" {clean_tag_name} ")

        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> list[str]:
        """Tokenize normalized text into word tokens."""
        try:
            return word_tokenize(text)
        except Exception:
            return text.split()

    def lemmatize(self, tokens: list[str]) -> list[str]:
        """Lemmatize tokens using context POS tagging and WordNet."""
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
        Execute full Text Preprocessing Pipeline.
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


# Default global instance
_default_preprocessor = TextPreprocessor(enable_spell_check=True)


def preprocess_text(text: str) -> str:
    """
    Convenience function returning preprocessed string for ML training and inference.
    """
    return _default_preprocessor.process(text)["preprocessed_text"]


def process_input(text: str) -> dict:
    """
    Convenience function returning detailed result dict of full preprocessing pipeline.
    """
    return _default_preprocessor.process(text)
