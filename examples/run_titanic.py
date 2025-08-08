import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from casuallite import RCSE_IBFS, plot_graph
import seaborn as sns
import numpy as np


if __name__ == '__main__':
    # Load and preprocess the Titanic dataset
    df = sns.load_dataset("titanic")
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    df['embarked'] = df['embarked'].astype('category').cat.codes  
    df['class'] = df['class'].astype('category').cat.codes
    df['who'] = df['who'].astype('category').cat.codes
    df['deck'] = df['deck'].astype('category').cat.codes
    df['alive'] = df['alive'].map({'no': 0, 'yes': 1})
    df['adult_male'] = df['adult_male'].astype(int)
    df['alone'] = df['alone'].astype(int)
    df = df.select_dtypes(include=np.number).dropna()

    X_data = df.to_numpy()
    feature_names = df.columns.tolist()

    print("\n🚀 Running RCSE-IBFS...")
    # Correctly unpack the two return values
    G, importances = RCSE_IBFS(X_data, beta=0.9, alpha_init=0.15, max_cond_set_size=2, feature_names=feature_names)

    print("\n📌 Recovered Causal Skeleton:")
    if G and G.nodes():
        for node in G.nodes():
            print(f"{feature_names[node]}: {[feature_names[n] for n in G.neighbors(node)]}")
        # Pass the importances dictionary to the plotting function
        plot_graph(G, labels=feature_names, title="Titanic Causal Skeleton (RCSE-IBFS)", node_importance=importances)
    else:
        print("No causal edges discovered.")
