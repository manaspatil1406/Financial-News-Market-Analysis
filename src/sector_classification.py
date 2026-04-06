import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
import re
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.naive_bayes import MultinomialNB  # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # type: ignore
from sklearn.utils import resample  # type: ignore
import joblib  # type: ignore
import os
import sys

# ============================================================
# FIX: Use dynamic path resolution instead of a hardcoded path.
# This file lives in src/, so the project root is one level up.
# ============================================================
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add src/ to path so config can be imported
sys.path.append(os.path.join(PROJECT_DIR, "src"))
from config import PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR, CHARTS_DIR

DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "preprocessed_finance_news.csv")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# ============================================================
# KEYWORD-BASED SECTOR RULES (for boosting noisy predictions)
# ============================================================
SECTOR_KEYWORDS = {
    "IT": [
        "infosys", "tcs", "wipro", "hcl", "tech mahindra", "cognizant",
        "software", "it services", "cloud", "saas", "artificial intelligence",
        "ai platform", "machine learning", "data center", "cybersecurity",
        "digital transformation", "automation", "computing", "semiconductor",
        "microchip", "oracle", "sap", "microsoft", "google", "amazon web",
    ],
    "Banking": [
        "rbi", "reserve bank", "interest rate", "repo rate", "hdfc bank",
        "sbi", "icici bank", "axis bank", "kotak", "banking", "loan",
        "credit", "deposit", "npa", "non performing", "monetary policy",
        "fiscal policy", "central bank", "federal reserve", "inflation",
        "gdp", "economic growth", "treasury", "bond yield", "forex",
    ],
    "Energy": [
        "reliance", "ongc", "oil", "petroleum", "gas", "crude",
        "opec", "energy", "power grid", "solar", "wind energy",
        "renewable", "coal", "mining", "fuel", "petrol", "diesel",
        "natural gas", "refinery", "adani green", "ntpc", "power plant",
    ],
    "Pharma": [
        "pharma", "fda", "drug", "medicine", "clinical trial", "vaccine",
        "sun pharma", "cipla", "dr reddy", "lupin", "biocon",
        "pharmaceutical", "biotech", "healthcare", "hospital",
        "medical device", "therapy", "patent", "generic drug",
    ],
    "Automobile": [
        "tata motors", "maruti", "hyundai", "mahindra", "bajaj auto",
        "hero motocorp", "tvs motor", "electric vehicle", "ev",
        "automobile", "car sales", "vehicle", "automotive", "tesla",
        "ford", "toyota", "bmw", "suv", "sedan", "motorcycle",
    ],
}


def preprocess_text(text):
    """
    Clean text for inference — matches the preprocessing used during training.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]
    return " ".join(tokens)


def keyword_sector_match(text):
    """
    Check if the input text contains strong keyword signals for a sector.
    Returns (sector_name, confidence_count) or (None, 0).
    """
    text_lower = text.lower()
    scores = {}
    for sector, keywords in SECTOR_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[sector] = count

    if scores:
        best_sector = max(scores, key=scores.get)
        return best_sector, scores[best_sector]
    return None, 0


def predict_sector(text, model, vectorizer, debug=False):
    """
    Robust sector prediction pipeline:
      1. Preprocess the input text
      2. Transform using trained TF-IDF vectorizer
      3. Get model prediction
      4. Apply keyword boosting if model confidence is low
    """
    cleaned = preprocess_text(text)
    if debug:
        print(f"  [DEBUG] Input text   : {text[:80]}...")
        print(f"  [DEBUG] Cleaned text : {cleaned[:80]}...")

    text_tfidf = vectorizer.transform([cleaned])

    model_pred = model.predict(text_tfidf)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(text_tfidf)[0]
        max_confidence = proba.max()
        if debug:
            for cls, p in zip(model.classes_, proba):
                print(f"  [DEBUG] P({cls:12s}) = {p:.4f}")
    else:
        max_confidence = 1.0

    kw_sector, kw_score = keyword_sector_match(text)
    if debug:
        print(f"  [DEBUG] Model pred   : {model_pred} (conf={max_confidence:.3f})")
        print(f"  [DEBUG] Keyword match: {kw_sector} (score={kw_score})")

    if kw_sector and (max_confidence < 0.45 or kw_score >= 2):
        final_pred = kw_sector
    elif kw_sector and model_pred != kw_sector and kw_score >= 1:
        final_pred = kw_sector
    else:
        final_pred = model_pred

    if debug:
        print(f"  [DEBUG] Final pred   : {final_pred}")

    return final_pred


def main():
    print("=" * 60)
    print("SECTOR CLASSIFICATION — Training Pipeline")
    print("=" * 60)
    print(f"Project root : {PROJECT_DIR}")
    print(f"Data path    : {DATA_PATH}")

    # ── Load Data ───────────────────────────────────────────────
    print("\nLoading data...")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: Data file not found at: {DATA_PATH}")
        print("Please run the data collection and preprocessing steps first.")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    df.dropna(subset=["clean_text", "sector"], inplace=True)

    print(f"Original shape: {df.shape}")
    df = df[df["sector"] != "Others"]
    print(f"Shape after removing 'Others': {df.shape}")

    # ── Handle Class Imbalance via Oversampling ──────────────────
    print("\nClass distribution BEFORE balancing:")
    print(df["sector"].value_counts())

    max_size = df["sector"].value_counts().max()
    balanced_dfs = []
    for sector in df["sector"].unique():
        sector_df = df[df["sector"] == sector]
        if len(sector_df) < max_size:
            sector_df = resample(
                sector_df, replace=True, n_samples=max_size, random_state=42
            )
        balanced_dfs.append(sector_df)
    df = pd.concat(balanced_dfs)

    print("\nClass distribution AFTER balancing:")
    print(df["sector"].value_counts())

    X = df["clean_text"]
    y = df["sector"]

    # ── TF-IDF ──────────────────────────────────────────────────
    print("\nExtracting TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", ngram_range=(1, 2)
    )
    X_tfidf = vectorizer.fit_transform(X)

    # ── Train/Test Split ────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Logistic Regression ─────────────────────────────────────
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(
        class_weight="balanced", max_iter=1000, solver="lbfgs"
    )
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)

    # ── Naive Bayes ─────────────────────────────────────────────
    print("Training Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)

    # ── Evaluate ────────────────────────────────────────────────
    print("\n--- Model Evaluation ---")
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    print(classification_report(y_test, lr_pred))

    print(f"Naive Bayes Accuracy: {nb_acc:.4f}")
    print(classification_report(y_test, nb_pred))

    # ── Select Best ─────────────────────────────────────────────
    print("\n--- Model Comparison ---")
    print(f"| {'Model':<22} | Accuracy |")
    print(f"|{'-'*24}|----------|")
    print(f"| {'Naive Bayes':<22} | {nb_acc*100:.2f}%   |")
    print(f"| {'Logistic Regression':<22} | {lr_acc*100:.2f}%   |")

    if lr_acc > nb_acc:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_pred = lr_pred
    else:
        best_model = nb_model
        best_name = "Naive Bayes"
        best_pred = nb_pred

    print(f"\n-> Best Model: {best_name} — Saving...")

    # ── Confusion Matrix ────────────────────────────────────────
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=best_model.classes_,
        yticklabels=best_model.classes_,
    )
    plt.title(f"Confusion Matrix ({best_name})")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    cm_path = os.path.join(CHARTS_DIR, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    print(f"Saved confusion matrix: {cm_path}")

    # ── Save Models ─────────────────────────────────────────────
    model_out = os.path.join(MODELS_DIR, "final_sector_model.pkl")
    vec_out   = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    joblib.dump(best_model, model_out)
    joblib.dump(vectorizer, vec_out)
    print(f"Saved model      : {model_out}")
    print(f"Saved vectorizer : {vec_out}")

    # ── Test Custom Headlines ───────────────────────────────────
    print("\n--- Testing Custom Headlines (with keyword boosting) ---")
    custom_headlines = [
        ("Infosys launches new AI platform", "IT"),
        ("RBI cuts interest rates to boost economy", "Banking"),
        ("Oil prices crash causing losses for ONGC", "Energy"),
        ("Tesla reports strong sales growth", "Automobile"),
        ("Sun Pharma gets FDA approval", "Pharma"),
        ("Reliance Industries reports strong quarterly profits", "Energy"),
        ("TCS announces expansion in AI services and global hiring", "IT"),
        ("HDFC Bank reports record quarterly profits", "Banking"),
    ]

    correct = 0
    for text, expected in custom_headlines:
        pred = predict_sector(text, best_model, vectorizer, debug=False)
        match = "✓" if pred == expected else "✗"
        if pred == expected:
            correct += 1
        print(f"  [{match}] \"{text}\" → {pred} (expected: {expected})")

    print(f"\nCustom headline accuracy: {correct}/{len(custom_headlines)}")


if __name__ == "__main__":
    main()