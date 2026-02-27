
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
    return df

def train_and_compare_models(df):
    X = df['text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Vectorizer
    vectorizer = TfidfVectorizer(max_features=5000)
    
    models = {
        "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
        "SVM": SVC(kernel='linear', probability=True, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = []
    best_model = None
    best_score = 0
    best_model_name = ""
    
    print("Starting model comparison...")
    
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        pipeline = make_pipeline(vectorizer, model)
        
        # Train
        pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        training_time = time.time() - start_time
        
        print(f"{name} Accuracy: {accuracy:.4f} (Time: {training_time:.2f}s)")
        
        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Time": training_time
        })
        
        if accuracy > best_score:
            best_score = accuracy
            best_model = pipeline
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with Accuracy: {best_score:.4f}")
    
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
