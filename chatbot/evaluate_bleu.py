"""
BookMate Hotel Chatbot / Customer Support Chatbot
=================================================
Response Generation Evaluation (BLEU Score)

Purpose:
    Evaluate the quality of chatbot responses against ground truth
    dataset responses using BLEU (Bilingual Evaluation Understudy) metric.
"""

import os
import sys
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# PROJECT PATH SETUP
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from chatbot.intent_classifier import IntentPredictor

# Check if ResponseGenerator class or get_response function is used
try:
    from chatbot.response import get_response
    def generate_reply(intent):
        return get_response(intent)
except ImportError:
    from chatbot.response import ResponseGenerator
    _responder = ResponseGenerator()
    def generate_reply(intent):
        return _responder.get_response(intent)


def evaluate_bleu(sample_size=500, random_seed=42):
    print("=" * 65)
    print("      EVALUATING CHATBOT RESPONSE GENERATION (BLEU SCORE)       ")
    print("=" * 65)

    # 1. Locate Dataset
    dataset_candidates = [
        os.path.join(PROJECT_ROOT, "dataset", "bitext-hospitality-llm-chatbot-training-dataset.csv"),
        os.path.join(PROJECT_ROOT, "dataset", "Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv")
    ]
    
    dataset_path = None
    for path in dataset_candidates:
        if os.path.exists(path):
            dataset_path = path
            break

    if not dataset_path:
        print(f"Error: Dataset not found in dataset directory.")
        return

    print(f"[*] Dataset: {os.path.basename(dataset_path)}")
    df = pd.read_csv(dataset_path).dropna(subset=["instruction", "intent", "response"])
    
    actual_sample_size = min(sample_size, len(df))
    df_test = df.sample(actual_sample_size, random_state=random_seed).copy()
    print(f"[*] Total Test Samples: {actual_sample_size}")

    # 2. Load Predictor
    print("[*] Loading Intent Classifier...")
    try:
        classifier = IntentPredictor(model_name="Support Vector Machine")
    except Exception:
        classifier = IntentPredictor()

    smoothie = SmoothingFunction().method1
    bleu_1_scores = []
    bleu_2_scores = []
    relevance_count = 0

    print("[*] Calculating BLEU scores...")
    for _, row in df_test.iterrows():
        user_input = str(row["instruction"])
        ground_truth_intent = str(row["intent"])
        reference_response = str(row["response"]).lower().split()

        # Predict intent and generate response
        pred = classifier.predict(user_input)
        pred_intent = pred["intent"]
        bot_reply = generate_reply(pred_intent).lower().split()

        # Calculate BLEU-1 (unigram) and BLEU-2 (bigram)
        b1 = sentence_bleu([reference_response], bot_reply, weights=(1.0, 0, 0, 0), smoothing_function=smoothie)
        b2 = sentence_bleu([reference_response], bot_reply, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)

        bleu_1_scores.append(b1)
        bleu_2_scores.append(b2)

        if pred_intent == ground_truth_intent:
            relevance_count += 1

    avg_bleu_1 = np.mean(bleu_1_scores)
    avg_bleu_2 = np.mean(bleu_2_scores)
    relevancy_rate = relevance_count / actual_sample_size

    # 3. Print Results Summary
    print("\n" + "=" * 65)
    print("                     EVALUATION RESULTS                         ")
    print("=" * 65)
    print(f"  Total Test Samples       : {actual_sample_size}")
    print(f"  Response Relevancy Rate  : {relevancy_rate:.4f} ({relevancy_rate * 100:.2f}%)")
    print(f"  Average BLEU-1 Score     : {avg_bleu_1:.4f} ({avg_bleu_1 * 100:.2f}%)")
    print(f"  Average BLEU-2 Score     : {avg_bleu_2:.4f} ({avg_bleu_2 * 100:.2f}%)")
    print("=" * 65)

    # 4. Show Case Study
    sample_row = df_test.iloc[0]
    sample_query = str(sample_row["instruction"])
    sample_ref = str(sample_row["response"])
    sample_pred_intent = classifier.predict(sample_query)["intent"]
    sample_bot_reply = generate_reply(sample_pred_intent)

    print("\n[Sample Case Comparison]")
    print(f"  User Query       : \"{sample_query}\"")
    print(f"  Ground Truth Ref : \"{sample_ref}\"")
    print(f"  Chatbot Response : \"{sample_bot_reply}\"")
    print("=" * 65)
    print("\n[Note on BLEU Score Interpretation]")
    print("A lower BLEU score is expected and normal for template-driven chatbots,")
    print("as the bot uses professionally polished, brand-specific templates rather")
    print("than verbatim copies of the raw dataset text.")
    print("=" * 65)
