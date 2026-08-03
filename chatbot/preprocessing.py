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
from textblob import TextBlob

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
            "cancle": "cancel"
        }

        # Build domain vocabulary from dataset for precision spell checking
        self.vocab = set()
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

        # Fallback core domain words
        self.vocab.update({
            "hello", "hi", "hey", "booking", "book", "room", "price", "cost",
            "checkin", "checkout", "wifi", "parking", "breakfast", "contact",
            "cancel", "deluxe", "suite", "location", "status", "payment",
            "bookmate", "king", "queen", "minibar", "jacuzzi"
        })

        # PII Regex Patterns (ordered by specificity)
        self.pii_patterns = [
            ("CREDIT_CARD", r"\b(?:\d{4}[ -]?){3}\d{4}\b|\b(?:\d[ -]*?){13,19}\b"),
            ("EMAIL", r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
            ("IC_ID", r"\b\d{6}-\d{2}-\d{4}\b|\b[A-Za-z]\d{7,8}\b"),
            ("PHONE", r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b")
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
        Domain-aware spell checking:
        1. Explicit common typo dictionary match.
        2. Exact match in domain vocabulary -> keep word.
        3. Short words (len <= 2) or numbers -> keep word.
        4. Match nearest word in domain vocabulary by edit distance.
        5. Fall back to TextBlob correction.
        """
        clean_word = word.strip(string.punctuation).lower()

        if not clean_word or clean_word.isdigit():
            return word

        # 1. Check explicit typo dictionary
        if clean_word in self.common_typos:
            corrected = self.common_typos[clean_word]
            return word.lower().replace(clean_word, corrected)

        # 2. Keep if already in domain vocabulary
        if clean_word in self.vocab or len(clean_word) <= 2:
            return word

        # 3. Edit distance match against domain vocabulary
        candidates = []
        for target in self.vocab:
            if abs(len(clean_word) - len(target)) <= 2:
                dist = edit_distance(clean_word, target)
                if dist <= 2:
                    candidates.append((dist, target))

        if candidates:
            candidates.sort(key=lambda x: (x[0], len(x[1])))
            best_dist, best_match = candidates[0]
            return word.lower().replace(clean_word, best_match)

        # 4. TextBlob fallback
        try:
            corrected = str(TextBlob(clean_word).correct())
            return word.lower().replace(clean_word, corrected)
        except Exception:
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
            matches = re.findall(pattern, masked_text)
            if matches:
                filtered_matches = [m for m in matches if len(re.sub(r'\D', '', str(m))) >= 7 or pii_type != "PHONE"]
                if filtered_matches:
                    detected_pii[pii_type] = filtered_matches
                    for m in filtered_matches:
                        masked_text = masked_text.replace(str(m), f"[{pii_type}]")

        return masked_text, detected_pii

    def normalize_text(self, text: str) -> str:
        """
        Text Normalization:
        - Lowercasing
        - Spelling Correction (Domain & Typo Aware)
        - Basic Cleaning (removing extra punctuation/spaces)
        """
        # 1. Lowercasing
        text = text.lower()

        # 2. Spelling Correction
        if self.enable_spell_check:
            words = text.split()
            corrected_words = []
            for w in words:
                if w.startswith("[") and w.endswith("]"):
                    corrected_words.append(w)
                else:
                    corrected_words.append(self._correct_word_spelling(w))
            text = " ".join(corrected_words)

        # 3. Basic Cleaning (keep PII bracket tags intact)
        pii_tokens = re.findall(r"\[[A-Z_]+\]", text)
        for i, tag in enumerate(pii_tokens):
            text = text.replace(tag, f" PII_TOKEN_{i} ")

        # Remove punctuation
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

        # Restore PII tokens
        for i, tag in enumerate(pii_tokens):
            clean_tag_name = tag.strip("[]").lower()
            text = text.replace(f" pii_token_{i} ", f" {clean_tag_name} ")

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
        "helo, I would like to book a room",
        "bookin a room for 2 people",
        "What is the prce of deluxe room?",
        "My email is john.doe@gmail.com and phone is +60123456789."
    ]

    print("=== Testing Domain-Aware Spelling & NLP Preprocessing Pipeline ===")
    for text in test_cases:
        res = process_input(text)
        print(f"\n[Original]   : {res['original_text']}")
        print(f"[Normalized] : {res['normalized_text']}")
        print(f"[Lemmatized] : {res['preprocessed_text']}")
