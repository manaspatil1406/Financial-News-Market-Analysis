import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import joblib  # type: ignore
import os
import sys
import logging
import subprocess
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
from PIL import Image  # type: ignore

# Add src to path for config
CHDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(CHDIR, "src"))
from config import (
    MODEL_PATH, VECTORIZER_PATH, SUMMARY_CSV, TRENDS_CSV, 
    CHART_FILES, MAIN_DATA_CSV, PROJECT_ROOT
)
from financial_lexicon import adjust_sentiment, get_sentiment_label as fin_get_sentiment_label

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Live analysis imports
try:
    from live_analyzer import smart_fetch, get_live_market_summary, analyze_live_news
except ImportError:
    # If not in path, try adding src
    sys.path.append(os.path.join(CHDIR, "src"))
    from live_analyzer import smart_fetch, get_live_market_summary, analyze_live_news
from datetime import datetime

# ============================================================
# PIPELINE RUNNER
# ============================================================
def run_pipeline():
    """Triggers the aggregation and visualization scripts."""
    with st.spinner("🚀 Running Pipeline... This may take a moment."):
        try:
            # Run Aggregation
            logger.info("Executing Aggregation Pipeline...")
            agg_script = os.path.join(PROJECT_ROOT, "src", "aggregation.py")
            subprocess.run([sys.executable, agg_script], check=True, capture_output=True)
            
            # Run Visualization
            logger.info("Executing Visualization Pipeline...")
            viz_script = os.path.join(PROJECT_ROOT, "src", "visualisation", "dashboard.py")
            subprocess.run([sys.executable, viz_script], check=True, capture_output=True)
            
            st.success("✅ Pipeline executed successfully! Reloading data...")
            st.rerun()
        except subprocess.CalledProcessError as e:
            logger.error(f"Pipeline failed: {e.stderr.decode()}")
            st.error(f"❌ Pipeline failed: {e.stderr.decode()}")
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            st.error(f"❌ An error occurred: {e}")

def check_files():
    """Checks if required data files exist."""
    missing = []
    if not os.path.exists(SUMMARY_CSV): missing.append("Sector Summary CSV")
    if not os.path.exists(TRENDS_CSV): missing.append("Market Trends CSV")
    if not os.path.exists(CHART_FILES["dashboard"]): missing.append("Master Dashboard Image")
    return missing

# ============================================================
# CACHED LOADERS
# ============================================================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_vectorizer():
    return joblib.load(VECTORIZER_PATH)

@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()

@st.cache_data
def load_summary():
    return pd.read_csv(SUMMARY_CSV, index_col=0)

@st.cache_data
def load_trends():
    return pd.read_csv(TRENDS_CSV)

def get_sentiment(score):
    """Legacy sentiment label function. Used for non-adjusted scores."""
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def safe_image(path):
    if os.path.exists(path):
        return Image.open(path)
    return None

# ============================================================
# SECTOR PREDICTION — Robust pipeline with keyword boosting
# ============================================================
import re

SECTOR_KEYWORDS = {
    "IT": [
        "infosys", "tcs", "wipro", "hcl", "tech mahindra", "cognizant",
        "software", "it services", "cloud", "saas", "artificial intelligence",
        "ai platform", "machine learning", "data center", "cybersecurity",
        "digital transformation", "automation", "computing", "semiconductor",
    ],
    "Banking": [
        "rbi", "reserve bank", "interest rate", "repo rate", "hdfc bank",
        "sbi", "icici bank", "axis bank", "kotak", "banking", "loan",
        "credit", "deposit", "npa", "monetary policy", "central bank",
        "federal reserve", "inflation", "gdp", "treasury", "bond yield",
    ],
    "Energy": [
        "reliance", "ongc", "oil", "petroleum", "gas", "crude",
        "opec", "energy", "solar", "wind energy", "renewable", "coal",
        "mining", "fuel", "petrol", "diesel", "natural gas", "refinery",
    ],
    "Pharma": [
        "pharma", "fda", "drug", "medicine", "clinical trial", "vaccine",
        "sun pharma", "cipla", "dr reddy", "lupin", "biocon",
        "pharmaceutical", "biotech", "healthcare", "hospital",
    ],
    "Automobile": [
        "tata motors", "maruti", "hyundai", "mahindra", "bajaj auto",
        "hero motocorp", "tvs motor", "electric vehicle", "ev",
        "automobile", "car sales", "vehicle", "automotive", "tesla",
    ],
}

def preprocess_text(text):
    """Clean text for inference — matches training preprocessing."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [t for t in tokens if len(t) > 2]
    return " ".join(tokens)

def keyword_sector_match(text):
    """Check for strong keyword signals in input text."""
    text_lower = text.lower()
    scores = {}
    for sector, keywords in SECTOR_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > 0:
            scores[sector] = count
    if scores:
        return max(scores, key=scores.get), scores[max(scores, key=scores.get)]
    return None, 0

def predict_sector(text, model, vectorizer):
    """
    Robust sector prediction:
    1. Preprocess text (same as training)
    2. TF-IDF transform
    3. Model prediction with confidence check
    4. Keyword boosting for domain-specific terms
    """
    cleaned = preprocess_text(text)
    text_tfidf = vectorizer.transform([cleaned])
    model_pred = model.predict(text_tfidf)[0]

    # Get confidence
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(text_tfidf)[0]
        max_confidence = proba.max()
    else:
        max_confidence = 1.0

    # Keyword boosting
    kw_sector, kw_score = keyword_sector_match(text)

    if kw_sector and (max_confidence < 0.45 or kw_score >= 2):
        return kw_sector
    elif kw_sector and model_pred != kw_sector and kw_score >= 1:
        return kw_sector
    return model_pred

# ============================================================
# APP CONFIG
# ============================================================
st.set_page_config(
    page_title="FinSight Engine",
    page_icon="📊",
    layout="wide",
)

# Initialize Session State
if 'live_df' not in st.session_state:
    st.session_state.live_df = None
if 'last_fetch_time' not in st.session_state:
    st.session_state.last_fetch_time = None
if 'fetch_count' not in st.session_state:
    st.session_state.fetch_count = 0

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    /* Global */
    .block-container { padding-top: 1rem; }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 2.2rem; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 1.05rem; }

    /* Metric cards */
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 2rem; color: #0f3460; }
    .metric-card p  { margin: 0.2rem 0 0 0; color: #666; font-size: 0.9rem; }

    /* Result cards */
    .result-card {
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .result-positive { background: #e8f5e9; border-left: 5px solid #4CAF50; }
    .result-negative { background: #ffebee; border-left: 5px solid #F44336; }
    .result-neutral  { background: #fff3e0; border-left: 5px solid #FF9800; }
    .result-sector   { background: #e3f2fd; border-left: 5px solid #2196F3; }

    /* Sector trend cards */
    .trend-card {
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        color: white;
    }
    .trend-bullish { background: linear-gradient(135deg, #2e7d32, #4CAF50); }
    .trend-bearish { background: linear-gradient(135deg, #c62828, #F44336); }
    .trend-stable  { background: linear-gradient(135deg, #e65100, #FF9800); }
    .trend-card h4 { margin: 0 0 0.4rem 0; font-size: 1.15rem; }
    .trend-card p  { margin: 0; font-size: 0.9rem; opacity: 0.9; }

    /* About badges */
    .tech-badge {
        display: inline-block;
        background: #e3f2fd;
        color: #1565c0;
        padding: 0.35rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.85rem;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 📊 Navigation")
    page = st.radio(
        "Go to",
        ["🏠 Home", "🔍 Live News Analyzer", "📊 Market Dashboard", "📈 Sector Trends", "🔴 Live Feed"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    
    # Missing Files Warning & Pipeline Trigger in Sidebar
    missing_files = check_files()
    if missing_files:
        st.warning(f"⚠️ Missing: {', '.join(missing_files)}")
        if st.button("🔄 Run Full Pipeline", use_container_width=True):
            run_pipeline()
    else:
        if st.button("🔄 Update Aggregates", use_container_width=True):
            run_pipeline()
            
    st.markdown("---")
    st.caption("FinSight Engine v1.0")

# ============================================================
# PAGE 1 — HOME
# ============================================================
if page == "🏠 Home":
    st.markdown("""
    <div class="main-header">
        <h1>📊 FinSight Engine</h1>
        <p>AI-powered sector classification and sentiment analysis for financial news</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### System Architecture")
    st.markdown("""
    ```
    Input News  →  Preprocessing  →  TF-IDF Vectorization  →  Sector Classification  →  Sentiment Analysis  →  Market Trends
    ```
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="metric-card"><h3>48,742</h3><p>Total Articles Analyzed</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>6</h3><p>Sectors Covered</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>~85%</h3><p>Model Accuracy</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌟 Master Dashboard")
    img = safe_image(CHART_FILES["dashboard"])
    if img:
        st.image(img, use_container_width=True)
    else:
        st.warning("dashboard_master.png not found. Run visualization_dashboard.py first.")

# ============================================================
# PAGE 2 — LIVE NEWS ANALYZER
# ============================================================
elif page == "🔍 Live News Analyzer":
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Live News Analyzer</h1>
        <p>Enter any financial headline to instantly predict the sector and sentiment</p>
    </div>
    """, unsafe_allow_html=True)

    # Example buttons
    st.markdown("##### 💡 Try an example:")
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    with ex_col1:
        if st.button("Infosys launches new AI platform"):
            st.session_state["news_input"] = "Infosys launches new AI platform"
    with ex_col2:
        if st.button("RBI cuts interest rates"):
            st.session_state["news_input"] = "RBI cuts interest rates"
    with ex_col3:
        if st.button("Oil prices crash causing losses"):
            st.session_state["news_input"] = "Oil prices crash causing losses"

    user_input = st.text_area(
        "Enter a financial news headline or paragraph:",
        value=st.session_state.get("news_input", ""),
        height=120,
        placeholder="e.g. Tesla reports disappointing quarterly earnings...",
    )

    if st.button("🔍 Analyze", type="primary", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                model = load_model()
                vectorizer = load_vectorizer()
                analyzer = load_vader()

                # Sector prediction (robust pipeline with preprocessing + keyword boosting)
                predicted_sector = predict_sector(user_input, model, vectorizer)

                # Hybrid Sentiment: VADER + Financial Lexicon
                vader_compound = analyzer.polarity_scores(user_input)["compound"]
                adjusted_score, triggered_keywords, adjustment_delta = adjust_sentiment(user_input, vader_compound)
                sentiment = fin_get_sentiment_label(adjusted_score)
                vader_sentiment = get_sentiment(vader_compound)

            st.markdown("---")
            st.markdown("### 📋 Analysis Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"""
                <div class="result-card" style="background:#f5f5f5; border-left:5px solid #333;">
                    <p style="color:#666; margin:0;">📰 Input Text</p>
                    <h4 style="margin:0.3rem 0;">{user_input[:120]}{'...' if len(user_input) > 120 else ''}</h4>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="result-card result-sector">
                    <p style="color:#666; margin:0;">🏭 Predicted Sector</p>
                    <h3 style="margin:0.3rem 0; color:#1565c0;">{predicted_sector}</h3>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                css_class = "result-positive" if sentiment == "Positive" else ("result-negative" if sentiment == "Negative" else "result-neutral")
                emoji = "😊" if sentiment == "Positive" else ("😟" if sentiment == "Negative" else "😐")
                st.markdown(f"""
                <div class="result-card {css_class}">
                    <p style="color:#666; margin:0;">{emoji} Sentiment</p>
                    <h3 style="margin:0.3rem 0;">{sentiment}</h3>
                </div>
                """, unsafe_allow_html=True)

            # Compound score visualization — using ADJUSTED score
            st.markdown("#### 📊 Adjusted Compound Score")
            normalized = (adjusted_score + 1) / 2  # map -1..1 to 0..1
            st.progress(normalized)
            st.caption(f"Adjusted compound score: **{adjusted_score:.4f}** (range: -1 to +1)")

            # Confidence interpretation — using adjusted score
            if adjusted_score > 0.5:
                st.success("🟢 Strong Positive Signal — Very favorable news")
            elif adjusted_score > 0.05:
                st.info("🟡 Mild Positive Signal — Slightly favorable")
            elif adjusted_score > -0.05:
                st.warning("⚪ Neutral Signal — No strong sentiment detected")
            elif adjusted_score > -0.5:
                st.warning("🟠 Mild Negative Signal — Slightly unfavorable")
            else:
                st.error("🔴 Strong Negative Signal — Very unfavorable news")



# ============================================================
# PAGE 3 — MARKET DASHBOARD
# ============================================================
elif page == "📊 Market Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Market Dashboard</h1>
        <p>Professional visualizations generated from the analysis pipeline</p>
    </div>
    """, unsafe_allow_html=True)

    # Row 1: Pie + Grouped Bar
    col1, col2 = st.columns(2)
    with col1:
        img = safe_image(CHART_FILES["chart1"])
        if img:
            st.image(img, caption="Overall Sentiment Distribution (Pie Chart)", use_container_width=True)
    with col2:
        img = safe_image(CHART_FILES["chart2"])
        if img:
            st.image(img, caption="Sector-wise Sentiment Distribution (Grouped Bar)", use_container_width=True)

    # Row 2: Market Trends + Heatmap
    col1, col2 = st.columns(2)
    with col1:
        img = safe_image(CHART_FILES["chart3"])
        if img:
            st.image(img, caption="Market Trend by Sector (Horizontal Bar)", use_container_width=True)
    with col2:
        img = safe_image(CHART_FILES["chart7"])
        if img:
            st.image(img, caption="Sector vs Sentiment Heatmap", use_container_width=True)

    # Row 3: Box Plot (full width)
    st.markdown("---")
    img = safe_image(CHART_FILES["chart4"])
    if img:
        st.image(img, caption="Sentiment Compound Score Distribution by Sector (Box Plot)", use_container_width=True)

    # Row 4: Year-wise Line (full width)
    if os.path.exists(CHART_FILES["chart5"]):
        st.markdown("---")
        img = safe_image(CHART_FILES["chart5"])
        if img:
            st.image(img, caption="Year-wise Sentiment Trend (Line Chart)", use_container_width=True)

    # Row 5: Stacked Bar (full width)
    st.markdown("---")
    img = safe_image(CHART_FILES["chart6"])
    if img:
        st.image(img, caption="Article Count by Sector and Sentiment (Stacked Bar)", use_container_width=True)

    # Row 6: Headlines table
    st.markdown("---")
    img = safe_image(CHART_FILES["chart8"])
    if img:
        st.image(img, caption="Top Positive & Negative Financial Headlines", use_container_width=True)

# ============================================================
# PAGE 4 — SECTOR TRENDS
# ============================================================
elif page == "📈 Sector Trends":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Sector Trends</h1>
        <p>Interactive breakdown of sentiment across market sectors</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        summary_df = load_summary()
        trends_df = load_trends()
    except Exception as e:
        st.error(f"Could not load data files: {e}")
        st.stop()

    # Merge for display
    merged = trends_df.copy()

    # Sector filter
    all_sectors = merged["Sector"].tolist()
    selected = st.selectbox("Filter by Sector:", ["All Sectors"] + all_sectors)

    if selected != "All Sectors":
        merged = merged[merged["Sector"] == selected]

    # Trend cards in 3-col grid
    st.markdown("### Market Trend Cards")
    cols = st.columns(3)
    for i, (_, row) in enumerate(merged.iterrows()):
        sector = row["Sector"]
        trend = row["Trend"]

        # Get percentages from summary
        if sector in summary_df.index:
            pos = summary_df.loc[sector, "Positive"] if "Positive" in summary_df.columns else 0
            neg = summary_df.loc[sector, "Negative"] if "Negative" in summary_df.columns else 0
            neu = summary_df.loc[sector, "Neutral"] if "Neutral" in summary_df.columns else 0
        else:
            pos, neg, neu = 0, 0, 0

        css = "trend-bullish" if "Bullish" in trend else ("trend-bearish" if "Bearish" in trend else "trend-stable")

        with cols[i % 3]:
            st.markdown(f"""
            <div class="trend-card {css}">
                <h4>{sector} — {trend}</h4>
                <p>Positive: {pos:.1f}% &nbsp;|&nbsp; Negative: {neg:.1f}% &nbsp;|&nbsp; Neutral: {neu:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(pos / 100 if pos else 0)
            st.caption(f"Positive sentiment ratio for {sector}")

    # Full table
    st.markdown("---")
    st.markdown("### 📊 Detailed Sentiment Table")

    display_df = summary_df.copy()
    display_df = display_df.round(2)
    st.dataframe(display_df, use_container_width=True)

    # Trend labels table
    st.markdown("### 🏷️ Market Trend Labels")
    trends_display = trends_df.copy()

    def color_trend(val):
        if "Bullish" in str(val):
            return "background-color: #e8f5e9; color: #2e7d32; font-weight: bold;"
        elif "Bearish" in str(val):
            return "background-color: #ffebee; color: #c62828; font-weight: bold;"
        elif "Stable" in str(val):
            return "background-color: #fff3e0; color: #e65100; font-weight: bold;"
        return ""

    styled = trends_display.style.map(color_trend, subset=["Trend"])
    st.dataframe(styled, use_container_width=True)

# ============================================================
# PAGE 5 — LIVE FEED
# ============================================================
elif page == "🔴 Live Feed":
    st.markdown("""
    <div class="main-header">
        <h1>🔴 Live Financial News Feed</h1>
        <p>Real-time news processing from NewsAPI & Yahoo RSS</p>
    </div>
    """, unsafe_allow_html=True)

    # Show last updated time
    if st.session_state.last_fetch_time:
        st.caption(f"🕐 Last updated: {st.session_state.last_fetch_time}")
    else:
        st.caption("🕐 Not fetched yet. Click 'Fetch Latest Now' below.")

    # Sidebar parameters for auto-refresh
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🔴 Live Settings")
        auto_refresh = st.toggle("🔄 Auto Refresh (5 mins)", value=False)
        if auto_refresh:
            st.caption("Page will reload periodically.")

    # Main action bar
    col_btn, col_spacer = st.columns([1, 2])
    with col_btn:
        if st.button("🔄 Fetch Latest Now", use_container_width=True, type="primary"):
            with st.spinner("🔍 Fetching fresh financial news..."):
                # Force fresh fetch — bypass ALL cache
                df_fresh = smart_fetch(force_refresh=True)
                
                # Update session state
                st.session_state.live_df = df_fresh
                st.session_state.last_fetch_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                st.session_state.fetch_count += 1
                
                if df_fresh is not None and not df_fresh.empty:
                    st.success(f"✅ Fetched {len(df_fresh)} articles (including history)!")
                else:
                    st.warning("⚠️ No new articles found. Try again in a few minutes.")

    # Load data if not already loaded (initial load)
    if st.session_state.live_df is None:
        with st.spinner("Loading cached news..."):
            st.session_state.live_df = smart_fetch()

    df_to_show = st.session_state.live_df

    if df_to_show is not None and not df_to_show.empty:
        # Visual metrics for proof of freshness
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("📰 Total Articles", len(df_to_show))
        with m_col2:
            st.metric("🔄 Fetch Session Count", st.session_state.fetch_count)
        with m_col3:
            if 'fetched_at' in df_to_show.columns:
                latest_fetch = df_to_show['fetched_at'].max()
                st.metric("🕐 Source Fetch Time", latest_fetch.split()[-1])
        
        if 'publishedAt' in df_to_show.columns:
            newest_pub = df_to_show['publishedAt'].max()
            st.caption(f"📅 Most recent article published: **{newest_pub}**")

        st.markdown("---")

        # Sector Navigation Tabs
        sectors = ['IT', 'Banking', 'Pharma', 'Energy', 'Automobile']
        tabs = st.tabs(['🖥️ IT', '🏦 Banking', '💊 Pharma', '⚡ Energy', '🚗 Auto'])
        
        for tab, sector in zip(tabs, sectors):
            with tab:
                # Filter and sort by newest first
                sector_news = df_to_show[df_to_show['predicted_sector'] == sector]
                if sector_news.empty:
                    st.info(f"No recent news found for {sector}.")
                else:
                    sector_news = sector_news.sort_values('publishedAt', ascending=False)
                    for _, row in sector_news.iterrows():
                        sentiment = row['sentiment']
                        score = row['vader_compound']
                        
                        # Use card highlighting from Fix 4
                        if sentiment == 'Positive':
                            color = "#e8f5e9" # light green
                            border = "#4CAF50"
                            tag = "🟢 Positive"
                        elif sentiment == 'Negative':
                            color = "#ffebee" # light red
                            border = "#F44336"
                            tag = "🔴 Negative"
                        else:
                            color = "#f5f5f5" # light gray
                            border = "#9e9e9e"
                            tag = "⚪ Neutral"
                        
                        st.markdown(f"""
                        <div style="background:{color}; border-left:8px solid {border}; padding:1.2rem; border-radius:8px; margin-bottom:1rem; box-shadow:0 1px 3px rgba(0,0,0,0.1);">
                            <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.5rem;">
                                <span style="font-weight:bold;">{tag} | Score: {score:+.2f}</span>
                                <span style="font-size:0.8rem; color:#666;">{row['source']} | {row['publishedAt']}</span>
                            </div>
                            <h4 style="margin:0.2rem 0; color:#1a1a2e; font-size:1.1rem;">{row['title']}</h4>
                            <p style="margin:0.5rem 0; font-size:0.9rem; color:#444; line-height:1.4;">{row.get('description', '')[:250]}{'...' if len(str(row.get('description', ''))) > 250 else ''}</p>
                            <div style="margin-top:0.8rem;">
                                <a href="{row['url']}" target="_blank" style="text-decoration:none; color:#1565c0; font-size:0.85rem; font-weight:600;">🔗 Read Full Article →</a>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

        # Market Summary Comparison
        st.markdown("---")
        st.markdown("### 📊 Live Sentiment Summary by Sector")
        summary_df, _, _ = get_live_market_summary(df_to_show)
        
        st.dataframe(summary_df.set_index('Sector').style.format({
            'Positive': '{:.1f}%',
            'Negative': '{:.1f}%',
            'Neutral': '{:.1f}%'
        }), use_container_width=True)

    else:
        st.warning("No live news data available. Click 'Fetch Latest Now' to start.")

    # Auto refresh logic
    if auto_refresh:
        import time
        time.sleep(300) # Wait 5 minutes
        st.rerun()

# ============================================================
# To run: streamlit run app.py
# ============================================================
