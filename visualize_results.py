
import pandas as pd
import matplotlib.pyplot as plt

def visualize_results():
    try:
        df = pd.read_csv("model_comparison_results.csv")
    except FileNotFoundError:
        print("Results file not found. Run framing_bias_detector.py first.")
        return

    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df['Accuracy'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')
        
    plt.savefig('model_accuracy_comparison.png')
    print("Saved accuracy chart to model_accuracy_comparison.png")
    
    # Plot Training Time
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df['Model'], df['Time'], color=['#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
    plt.title('Model Training Time Comparison')
    plt.ylabel('Time (seconds)')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.2f}s', ha='center', va='bottom')

    plt.savefig('model_time_comparison.png')
    print("Saved time chart to model_time_comparison.png")

if __name__ == "__main__":
    visualize_results()
