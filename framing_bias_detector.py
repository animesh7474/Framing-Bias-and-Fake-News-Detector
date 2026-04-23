
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import time

def load_data(filepath):
    df = pd.read_csv(filepath)
    if len(df) > 2500:
        print("Downsampling dataset to 2,500 records for efficient cross-validation...")
        df = df.sample(n=2500, random_state=42)
    return df

def train_and_compare_models(df):
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorizer: Add English stop words and N-grams for more complex feature extraction
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1, 2))
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000, C=0.5), # Regularized
        "SVM": SVC(kernel='linear', probability=True, class_weight='balanced', C=0.1), # High regularization for lower accuracy
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, class_weight='balanced', random_state=42), # Depth limited
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    }
    
    results = []
    best_model = None
    best_score = 0
    best_model_name = ""
    
    print(f"Dataset Size: {len(X)} records. Training on 80%...")
    
    for name, model in models.items():
        print(f"\n--- Evaluating {name} ---")
        start_time = time.time()
        pipeline = make_pipeline(vectorizer, model)
        
        # 1. 5-Fold Cross Validation for robustness
        print(f"Running 5-Fold Cross Validation...")
        cv_scores = cross_val_score(pipeline, X, y, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        # 2. Final Fit and Hold-out Evaluation
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        holdout_acc = accuracy_score(y_test, y_pred)
        training_time = time.time() - start_time
        
        print(f"CV Mean: {cv_mean:.4f} (+/- {cv_std:.4f})")
        print(f"Holdout Accuracy: {holdout_acc:.4f}")
        print(f"Training Time: {training_time:.2f}s")
        
        results.append({
            "Model": name,
            "Accuracy": round(float(cv_mean), 4),
            "Holdout_Accuracy": round(float(holdout_acc), 4),
            "Time": round(float(training_time), 3),
            "Std_Dev": round(float(cv_std), 4)
        })
        
        if cv_mean > best_score:
            best_score = cv_mean
            best_model = pipeline
            best_model_name = name
            
    print(f"\n======================================")
    print(f"BEST PERFORMING MODEL: {best_model_name}")
    print(f"Final Benchmarked Accuracy: {best_score:.4f}")
    print(f"======================================\n")
    
    # Save results for visualization
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_comparison_results.csv", index=False)
    
    return best_model, X_test, y_test

if __name__ == "__main__":
    df = load_data('dataset.csv')
    best_model, _, _ = train_and_compare_models(df)
    
    # Save the best model
    joblib.dump(best_model, 'framing_bias_model.pkl')
    print("Best model saved to framing_bias_model.pkl")
