import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import PROCESSED_DATA_DIR, OUTPUTS_DIR, PROJECT_ROOT
from financial_lexicon import adjust_sentiment, get_sentiment_label

# Set paths using config
INPUT_PATH = os.path.join(PROCESSED_DATA_DIR, "preprocessed_finance_news.csv")
OUTPUT_PATH = os.path.join(OUTPUTS_DIR, "financial_news_with_sentiment.csv")

# Force UTF-8 encoding for stdout to avoid Windows charmap errors
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def analyze_sentiment_hybrid(text, analyzer):
    """
    Perform hybrid sentiment analysis:
    1. Get base VADER compound score
    2. Adjust with financial domain keywords
    3. Return both original and adjusted scores + triggered keywords
    """
    if not isinstance(text, str) or not text.strip():
        return 0.0, 0.0, "Neutral", []

    # Step 1: VADER base score
    vader_scores = analyzer.polarity_scores(text)
    vader_compound = vader_scores['compound']

    # Step 2: Financial keyword adjustment
    adjusted_score, triggered_keywords, delta = adjust_sentiment(text, vader_compound)

    # Step 3: Label from adjusted score
    sentiment_label = get_sentiment_label(adjusted_score)

    return vader_compound, adjusted_score, sentiment_label, triggered_keywords


def main():
    # Step 1: Initialize VADER
    analyzer = SentimentIntensityAnalyzer()

    # Step 2: Load the Dataset
    print(f"Loading dataset from: {INPUT_PATH}")
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found!")
        return

    df = pd.read_csv(INPUT_PATH)
    print(f"Dataset shape: {df.shape}")

    # Step 3: Apply Hybrid Sentiment Analysis
    print("\n📊 Applying HYBRID Sentiment Analysis (VADER + Financial Lexicon)...")
    print("   This combines general VADER scores with domain-specific financial keyword adjustments.\n")

    # Apply hybrid analysis to each headline
    results = df['headline'].astype(str).apply(
        lambda x: analyze_sentiment_hybrid(x, analyzer)
    )

    # Unpack results into separate columns
    df['vader_compound'] = results.apply(lambda x: x[0])
    df['adjusted_compound'] = results.apply(lambda x: x[1])
    df['sentiment'] = results.apply(lambda x: x[2])
    df['triggered_keywords'] = results.apply(
        lambda x: ", ".join([kw['keyword'] for kw in x[3]]) if x[3] else ""
    )

    # Step 4: Show adjustment statistics
    adjustments_made = (df['vader_compound'] != df['adjusted_compound']).sum()
    total = len(df)
    print(f"   ✅ Analyzed {total} headlines")
    print(f"   🔧 Financial adjustments applied: {adjustments_made} ({adjustments_made/total*100:.1f}%)")
    print(f"   📝 Headlines unchanged: {total - adjustments_made} ({(total-adjustments_made)/total*100:.1f}%)\n")

    # Step 5: Check Sentiment Distribution
    print("--- Sentiment Distribution (After Financial Adjustment) ---")
    dist = df['sentiment'].value_counts()
    print(dist)

    # Save pie chart
    plt.figure(figsize=(8, 8))
    colors = ['#4CAF50', '#F44336', '#FFC107']
    plt.pie(dist, labels=dist.index.astype(str), autopct='%1.1f%%',
            colors=colors, startangle=140)
    plt.title('Overall Sentiment Distribution (Hybrid Analysis)')
    chart_path = os.path.join(PROJECT_ROOT, 'phase6_sentiment_distribution.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved: {chart_path}")

    # Step 6: Sector-wise Sentiment Breakdown
    print("\n--- Sector-wise Sentiment Breakdown ---")
    sector_sentiment = df.groupby(['sector', 'sentiment']).size().unstack(fill_value=0)
    sector_sentiment_pct = sector_sentiment.div(sector_sentiment.sum(axis=1), axis=0) * 100
    print("Percentage Breakdown per Sector:")
    print(sector_sentiment_pct.round(2))

    # Save sector-wise bar chart
    plt.figure(figsize=(12, 6))
    sector_sentiment_pct.plot(kind='bar', stacked=True,
                              color=['#F44336', '#FFC107', '#4CAF50'], ax=plt.gca())
    plt.title('Sentiment Distribution by Sector (Hybrid Analysis)')
    plt.ylabel('Percentage (%)')
    plt.xlabel('Sector')
    plt.legend(title='Sentiment')
    plt.xticks(rotation=45)
    plt.tight_layout()
    chart_path = os.path.join(PROJECT_ROOT, 'phase6_sector_sentiment.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"Saved: {chart_path}")

    # Step 7: Test with Custom Headlines — Including recent financial news
    print("\n" + "=" * 70)
    print("  🧪 TESTING: Custom Headlines — VADER vs Hybrid Comparison")
    print("=" * 70)

    test_headlines = [
        # Classic misclassification cases
        "Stock market sees major correction after rally",
        "Company offers deep discount on shares",
        "RBI cuts interest rates boosting market confidence",
        "Oil prices crash causing massive losses",

        # Recent financial news patterns (2025-2026)
        "Tesla reports disappointing quarterly earnings amid EV slowdown",
        "Sun Pharma gets breakthrough FDA approval for cancer drug",
        "Infosys launches new AI platform with record profits",
        "HDFC Bank shares surge after strong Q4 results",
        "Sensex crashes 1000 points on global selloff fears",
        "Adani stocks plunge amid fraud allegations",
        "Reliance Jio reports massive revenue surge in Q3",
        "IT sector faces layoffs amid recession fears",
        "Gold prices hit all-time high amid geopolitical uncertainty",
        "Fed raises interest rate by 50 basis points",
        "Tata Motors EV sales soar as demand surges",
        "Rupee falls to all-time low against dollar",
        "Zomato shares rally after surprise profit announcement",
        "Nvidia stock hits new high on AI chip demand",
        "Paytm shares tank after RBI regulatory action",
        "Indian GDP growth beats estimates at 7.2 percent",
    ]

    print(f"\n{'Headline':<55} {'VADER':>7} {'Hybrid':>7} {'Label':>10}  {'Keywords'}")
    print("-" * 120)

    for h in test_headlines:
        vader_score, adjusted_score, label, keywords = analyze_sentiment_hybrid(h, analyzer)
        kw_str = ", ".join([kw['keyword'] for kw in keywords]) if keywords else "—"

        # Mark rows where adjustment changed the outcome
        vader_label = get_sentiment_label(vader_score)
        marker = " ✨" if vader_label != label else ""

        print(f"{h[:54]:<55} {vader_score:>+7.3f} {adjusted_score:>+7.3f} {label:>10}{marker}  {kw_str}")

    print("-" * 120)
    print("  ✨ = Sentiment was corrected by financial keyword adjustment\n")

    # Step 8: Save the Updated Dataset
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved: {OUTPUT_PATH}")
    print(f"   Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
