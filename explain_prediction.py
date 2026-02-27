
import joblib
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt

def explain_prediction(text, model_path='framing_bias_model.pkl'):
    model = joblib.load(model_path)
    explainer = LimeTextExplainer(class_names=model.classes_)
    
    exp = explainer.explain_instance(text, model.predict_proba, num_features=6)
    
    print(f"Text: {text}")
    print(f"Prediction: {model.predict([text])[0]}")
    print("Explanation:")
    
    # Show explanation in console
    for feature, weight in exp.as_list():
        print(f"{feature}: {weight}")
        
    # Ideally, we would visualize this, but for a script, we print the weights.
    # To visualize: exp.show_in_notebook(text=True) if in Jupyter, or save as html
    # exp.save_to_file('explanation.html')
    # print("Explanation saved to explanation.html")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = sys.argv[1]
        explain_prediction(text)
    else:
        print("Please provide a text string to explain.")
