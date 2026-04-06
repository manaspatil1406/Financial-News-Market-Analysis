# Financial News Analysis System (FNAS)

# 📊 Financial News Analysis System (FNAS)

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An end-to-end **Machine Learning + NLP pipeline** for classifying financial news into industry sectors and analyzing market sentiment in real time.

---

## 🚀 Project Highlights

- Multi-sector classification: Banking, IT, Pharma, Energy, Automobile, Others  
- Hybrid sentiment analysis using **VADER + financial lexicon**  
- Market trend detection: **Bullish, Bearish, Stable**  
- Interactive **Streamlit dashboard**  
- Live news integration via **NewsAPI + Yahoo Finance RSS fallback**

---

## 🧠 System Workflow

```text
Raw Financial News Data
        ↓
Data Cleaning & Preprocessing
        ↓
TF-IDF Feature Extraction
        ↓
Model Training (Logistic Regression / Naive Bayes)
        ↓
Sector Prediction
        ↓
Sentiment Analysis (VADER + Financial Lexicon)
        ↓
Aggregation & Trend Detection
        ↓
Visualization (Dashboard)

Financial-News-Market-Analysis/
├── app/
├── data/
├── models/
├── notebooks/
├── reports/
├── src/
├── README.md
└── requirements.txt

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/manaspatil1406/Financial-News-Market-Analysis.git
cd Financial-News-Market-Analysis
```

### 2. Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK and spaCy resources

```bash
python -m nltk.downloader stopwords punkt punkt_tab
python -m spacy download en_core_web_sm
```

### 5. Optional: set a NewsAPI key

Without a key, the app falls back to Yahoo Finance RSS feeds.

```bash
# Windows PowerShell
$env:NEWS_API_KEY="your_actual_key_here"

# Windows Command Prompt
set NEWS_API_KEY=your_actual_key_here

# macOS / Linux
export NEWS_API_KEY=your_actual_key_here
```

---

## Running the Pipeline

Run these scripts in order from the project root:

```bash
python src/sector_classification.py
python src/sentiment_analysis.py
python src/aggregation.py
python src/visualisation/dashboard.py
```

Then launch the app:

```bash
streamlit run app/app.py
```

---

## Technologies Used

- Python, Pandas, NumPy
- NLTK, spaCy, VADER Sentiment
- scikit-learn, joblib
- Matplotlib, Seaborn
- Streamlit, Pillow
- NewsAPI, feedparser, requests

---

## Notes

- The historical dataset used in the project is a filtered financial subset of the larger raw news corpus.
- The sector classifier is trained on the main sectors after excluding the `Others` class from training.
- Live analysis uses the saved model and vectorizer for real-time inference.

---

## About the Project

This project was developed as a practical financial text analytics system to help users understand sector-wise news flow, market sentiment, and trend signals through machine learning, NLP, and interactive dashboards.