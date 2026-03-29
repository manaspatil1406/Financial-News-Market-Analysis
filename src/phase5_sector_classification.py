import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.naive_bayes import MultinomialNB  # type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # type: ignore
import joblib  # type: ignore
import os

# Set paths
DATA_PATH = r"c:\Users\Rajesh Patil\Financial-News-Market-Analysis\data\processed\preprocessed_finance_news.csv"
MODELS_DIR = r"c:\Users\Rajesh Patil\Financial-News-Market-Analysis\models"
os.makedirs(MODELS_DIR, exist_ok=True)
SRC_DIR = r"c:\Users\Rajesh Patil\Financial-News-Market-Analysis\src"
os.makedirs(SRC_DIR, exist_ok=True)
PROJECT_DIR = r"c:\Users\Rajesh Patil\Financial-News-Market-Analysis"

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # Handle NaN values
    df.dropna(subset=['clean_text', 'sector'], inplace=True)
    
    # Step 1: Remove the Others Category
    print(f"Original shape: {df.shape}")
    df = df[df['sector'] != 'Others']
    print(f"Shape after removing 'Others': {df.shape}")
    
    # Step 2: Handle Class Imbalance
    print("\nClass distribution after removal:")
    print(df['sector'].value_counts())
    
    X = df['clean_text']
    y = df['sector']
    
    # Retrain TF-IDF
    print("\nExtracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(X)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    # Step 3: Train Logistic Regression Model
    print("\nTraining Logistic Regression...")
    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs')
    lr_model.fit(X_train, y_train)
    lr_pred = lr_model.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    
    # Step 4: Train Naive Bayes
    print("\nTraining Naive Bayes...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_acc = accuracy_score(y_test, nb_pred)
    
    # Step 5: Evaluate Both Models
    print("\n--- Model Evaluation ---")
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    print("Logistic Regression Classification Report:")
    print(classification_report(y_test, lr_pred))
    
    print(f"Naive Bayes Accuracy: {nb_acc:.4f}")
    print("Naive Bayes Classification Report:")
    print(classification_report(y_test, nb_pred))
    
    # Step 6: Compare & Select Best Model
    print("\n--- Model Comparison ---")
    print(f"| Model               | Accuracy |")
    print(f"|---------------------|----------|")
    print(f"| Naive Bayes         | {nb_acc*100:.2f}%   |")
    print(f"| Logistic Regression | {lr_acc*100:.2f}%   |")
    
    if lr_acc > nb_acc:
        best_model = lr_model
        best_name = "Logistic Regression"
        best_pred = lr_pred
    else:
        best_model = nb_model
        best_name = "Naive Bayes"
        best_pred = nb_pred
        
    print(f"-> Select the better model: {best_name}")
    print(f"[OK] Best Model: {best_name} -> Saved!")

    # Confusion matrix for best model
    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=best_model.classes_, 
                yticklabels=best_model.classes_)
    plt.title(f'Confusion Matrix ({best_name})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_DIR, 'phase5_results.png'))
    print("Saved confusion matrix to phase5_results.png")
    
    # Step 7: Save the Final Model
    joblib.dump(best_model, os.path.join(MODELS_DIR, 'final_sector_model.pkl'))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))
    print("Saved models to models/ folder.")
    
    # Step 8: Test with Custom News Headlines
    custom_headlines = [
        "Infosys launches new AI platform",
        "RBI cuts interest rates",
        "Oil prices crash causing losses",
        "Tesla reports strong sales growth",
        "Sun Pharma gets FDA approval"
    ]
    
    print("\n--- Testing Custom Headlines ---")
    custom_tfidf = vectorizer.transform(custom_headlines)
    custom_preds = best_model.predict(custom_tfidf)
    for text, pred in zip(custom_headlines, custom_preds):
        print(f'"{text}" \t-> Predicted: {pred}')

if __name__ == "__main__":
    main()
