"""
Financial Lexicon Module
========================
Provides domain-specific keyword dictionaries and a hybrid sentiment
adjustment function for financial news headlines.

The `adjust_sentiment()` function layers financial keyword awareness
on top of a base VADER compound score to correct common misclassifications
in financial text.

Author: FinSight Engine
"""

import re
import logging

logger = logging.getLogger(__name__)

# ============================================================
# FINANCIAL KEYWORD DICTIONARIES
# Each keyword maps to a sentiment adjustment weight.
# Positive weights boost sentiment; negative weights reduce it.
# ============================================================

FINANCIAL_POSITIVE_KEYWORDS = {
    # Growth & Performance
    "surge": 0.40,
    "surges": 0.40,
    "surged": 0.40,
    "rally": 0.35,
    "rallies": 0.35,
    "rallied": 0.35,
    "soar": 0.35,
    "soars": 0.35,
    "soared": 0.35,
    "boom": 0.30,
    "booming": 0.30,
    "bull run": 0.35,
    "bullish": 0.30,
    "uptick": 0.25,
    "upbeat": 0.25,
    "outperform": 0.30,
    "outperforms": 0.30,
    "outperformed": 0.30,
    "breakout": 0.25,

    # Profit & Revenue
    "profit growth": 0.30,
    "record profit": 0.35,
    "record revenue": 0.35,
    "strong earnings": 0.30,
    "beats estimates": 0.30,
    "beat estimates": 0.30,
    "earnings beat": 0.30,
    "revenue growth": 0.25,
    "revenue surge": 0.35,
    "profit surge": 0.35,

    # Market & Stock
    "all-time high": 0.35,
    "new high": 0.25,
    "52-week high": 0.30,
    "market rally": 0.35,
    "stock rally": 0.35,
    "shares jump": 0.30,
    "shares surge": 0.35,
    "gains momentum": 0.25,

    # Corporate Actions (Positive)
    "upgrade": 0.30,
    "upgraded": 0.30,
    "expansion": 0.20,
    "acquisition": 0.15,
    "merger": 0.15,
    "ipo success": 0.25,
    "dividend increase": 0.25,
    "dividend hike": 0.25,
    "buyback": 0.20,
    "share buyback": 0.20,
    "fda approval": 0.30,
    "breakthrough": 0.25,

    # Economic (Positive for markets)
    "rate cut": 0.25,
    "interest rate cut": 0.30,
    "cuts interest rate": 0.30,
    "stimulus": 0.20,
    "economic recovery": 0.25,
    "gdp growth": 0.25,
    "job growth": 0.20,
    "consumer confidence": 0.20,
}

FINANCIAL_NEGATIVE_KEYWORDS = {
    # Decline & Loss
    "crash": -0.45,
    "crashes": -0.45,
    "crashed": -0.45,
    "plunge": -0.40,
    "plunges": -0.40,
    "plunged": -0.40,
    "plummet": -0.40,
    "plummets": -0.40,
    "plummeted": -0.40,
    "slump": -0.35,
    "slumps": -0.35,
    "slumped": -0.35,
    "tumble": -0.35,
    "tumbles": -0.35,
    "tumbled": -0.35,
    "tank": -0.35,
    "tanks": -0.35,
    "tanked": -0.35,
    "selloff": -0.35,
    "sell-off": -0.35,
    "bloodbath": -0.45,
    "rout": -0.35,
    "bearish": -0.30,
    "bear market": -0.35,

    # Loss & Failure
    "loss": -0.20,
    "losses": -0.25,
    "massive loss": -0.40,
    "net loss": -0.30,
    "quarterly loss": -0.30,
    "revenue decline": -0.30,
    "profit decline": -0.30,
    "earnings miss": -0.30,
    "misses estimates": -0.30,
    "missed estimates": -0.30,
    "disappointing earnings": -0.35,
    "disappointing results": -0.35,
    "weak earnings": -0.30,
    "poor results": -0.30,

    # Market & Stock
    "all-time low": -0.35,
    "52-week low": -0.30,
    "market crash": -0.45,
    "stock crash": -0.45,
    "shares plunge": -0.40,
    "shares fall": -0.30,
    "shares drop": -0.30,
    "shares sink": -0.35,
    "shares tumble": -0.35,

    # Corporate Actions (Negative)
    "downgrade": -0.30,
    "downgraded": -0.30,
    "layoffs": -0.30,
    "layoff": -0.30,
    "job cuts": -0.30,
    "bankruptcy": -0.50,
    "bankrupt": -0.50,
    "default": -0.40,
    "defaults": -0.40,
    "defaulted": -0.40,
    "delisted": -0.35,
    "fraud": -0.45,
    "scam": -0.45,
    "scandal": -0.40,
    "investigation": -0.20,
    "probe": -0.20,
    "penalty": -0.25,
    "fine": -0.20,
    "fined": -0.25,
    "lawsuit": -0.25,
    "recall": -0.20,

    # Economic (Negative)
    "recession": -0.40,
    "inflation rises": -0.30,
    "inflation surges": -0.35,
    "rate hike": -0.20,
    "interest rate hike": -0.25,
    "raises interest rate": -0.25,
    "debt crisis": -0.40,
    "trade war": -0.30,
    "tariff": -0.20,
    "tariffs": -0.20,
    "sanctions": -0.25,
    "geopolitical risk": -0.25,
    "market correction": -0.25,
    "correction": -0.20,
    "bubble": -0.25,
    "volatility": -0.15,
    "uncertainty": -0.15,

    # Context-sensitive overrides (commonly misclassified)
    "deep discount": -0.20,
    "discount offering": -0.15,
    "price cut": -0.15,
}

FINANCIAL_NEUTRAL_KEYWORDS = {
    "unchanged",
    "steady",
    "flat",
    "sideways",
    "range-bound",
    "consolidation",
    "consolidating",
    "mixed signals",
    "wait and watch",
    "hold rating",
    "holds steady",
    "status quo",
    "in line with expectations",
    "meets estimates",
    "met estimates",
    "stable",
    "on par",
}

# ============================================================
# NEGATION WORDS
# If a negation word appears within 3 words before a keyword,
# the keyword's weight is flipped.
# ============================================================
NEGATION_WORDS = {"not", "no", "never", "neither", "nor", "hardly", "barely", "doesn't", "didn't", "won't", "isn't", "aren't", "wasn't", "weren't", "couldn't", "wouldn't", "shouldn't", "despite"}


def _find_keyword_matches(text_lower, keyword_dict):
    """
    Find all matching keywords in text with negation awareness.
    Returns list of (keyword, weight, negated) tuples.
    Longest-match-first to handle multi-word phrases correctly.
    """
    matches = []
    words = text_lower.split()

    # Sort keywords by length (longest first) to prioritize multi-word phrases
    sorted_keywords = sorted(keyword_dict.keys(), key=len, reverse=True)

    # Track which character positions have been matched (to avoid double-counting)
    matched_positions = set()

    for keyword in sorted_keywords:
        # Find all occurrences of this keyword in text
        start = 0
        while True:
            idx = text_lower.find(keyword, start)
            if idx == -1:
                break

            # Check if this position is already matched by a longer keyword
            positions = set(range(idx, idx + len(keyword)))
            if positions & matched_positions:
                start = idx + 1
                continue

            # Check for negation: look at the 3 words preceding the keyword
            prefix = text_lower[:idx].strip()
            prefix_words = prefix.split()[-3:]  # last 3 words before keyword
            negated = any(neg in prefix_words for neg in NEGATION_WORDS)

            weight = keyword_dict[keyword]
            if negated:
                weight = -weight  # flip the weight

            matches.append((keyword, weight, negated))
            matched_positions.update(positions)

            start = idx + len(keyword)

    return matches


def adjust_sentiment(text, base_compound_score):
    """
    Adjust a VADER compound score using financial domain keywords.

    Parameters
    ----------
    text : str
        The original news headline or text.
    base_compound_score : float
        The VADER compound score (-1.0 to +1.0).

    Returns
    -------
    tuple : (adjusted_score, triggered_keywords_list, adjustment_delta)
        - adjusted_score: float, the financially-adjusted compound score
        - triggered_keywords_list: list of dicts with keyword details
        - adjustment_delta: float, the total adjustment applied
    """
    if not isinstance(text, str) or not text.strip():
        return base_compound_score, [], 0.0

    text_lower = text.lower().strip()
    triggered = []
    total_adjustment = 0.0

    # --- Step 1: Check for neutral keywords (highest priority) ---
    for neutral_kw in FINANCIAL_NEUTRAL_KEYWORDS:
        if neutral_kw in text_lower:
            # Push the score toward 0 (neutral)
            neutralizing_factor = 0.6  # dampen the score by 60%
            adjusted = base_compound_score * (1 - neutralizing_factor)
            triggered.append({
                "keyword": neutral_kw,
                "type": "neutral_override",
                "effect": f"Dampened score by {neutralizing_factor:.0%}"
            })
            logger.debug(
                f"NEUTRAL OVERRIDE | keyword='{neutral_kw}' | "
                f"original={base_compound_score:.4f} → adjusted={adjusted:.4f}"
            )
            return (
                max(-1.0, min(1.0, adjusted)),
                triggered,
                adjusted - base_compound_score
            )

    # --- Step 2: Find positive keyword matches ---
    pos_matches = _find_keyword_matches(text_lower, FINANCIAL_POSITIVE_KEYWORDS)
    for kw, weight, negated in pos_matches:
        total_adjustment += weight
        triggered.append({
            "keyword": kw,
            "type": "positive" if not negated else "positive_negated",
            "weight": weight,
            "negated": negated
        })

    # --- Step 3: Find negative keyword matches ---
    neg_matches = _find_keyword_matches(text_lower, FINANCIAL_NEGATIVE_KEYWORDS)
    for kw, weight, negated in neg_matches:
        total_adjustment += weight  # weight is already negative
        triggered.append({
            "keyword": kw,
            "type": "negative" if not negated else "negative_negated",
            "weight": weight,
            "negated": negated
        })

    # --- Step 4: Cap the total adjustment to prevent keyword-only dominance ---
    total_adjustment = max(-0.6, min(0.6, total_adjustment))

    # --- Step 5: Blend VADER score with financial adjustment ---
    adjusted_score = base_compound_score + total_adjustment

    # Clamp to valid range
    adjusted_score = max(-1.0, min(1.0, adjusted_score))

    # --- Step 6: Logging ---
    if triggered:
        kw_names = [t["keyword"] for t in triggered]
        logger.info(
            f"SENTIMENT ADJUSTMENT | text='{text[:80]}...' | "
            f"vader={base_compound_score:.4f} | adjustment={total_adjustment:+.4f} | "
            f"final={adjusted_score:.4f} | keywords={kw_names}"
        )
    else:
        logger.debug(
            f"NO ADJUSTMENT | text='{text[:80]}...' | "
            f"vader={base_compound_score:.4f} (no financial keywords detected)"
        )

    return adjusted_score, triggered, total_adjustment


def get_sentiment_label(score):
    """Classify a compound score into Positive / Negative / Neutral."""
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"


# ============================================================
# OPTIONAL: FinBERT Integration (Advanced Enhancement)
# ============================================================
# To use a transformer-based financial sentiment model, uncomment
# the following and install: pip install transformers torch
#
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
#
# class FinBERTAnalyzer:
#     """
#     Uses the FinBERT model (yiyanghkust/finbert-tone) for
#     financial sentiment analysis. Much more accurate than VADER
#     for domain-specific text, but requires ~500MB model download.
#     """
#     def __init__(self):
#         self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
#         self.model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
#         self.labels = ["Positive", "Negative", "Neutral"]
#
#     def predict(self, text):
#         inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
#         pred_idx = probs.argmax().item()
#         return self.labels[pred_idx], probs[0][pred_idx].item()
# ============================================================
