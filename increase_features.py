import sys

import pandas as pd
import numpy as np

from pathlib import Path
from IPython import embed
from sklearn.neural_network import MLPClassifier
from utils import normalize_dataset, stratify, clean_dataset

def increase_features(df: pd.DataFrame, output_path: Path):
    # coefs_ -> weights
    # intercepts_ -> biases

    # X * weights + biases -> feedforward

    # np.maximum -> ReLU
    map_features = lambda X, clf: np.maximum(np.matmul(X, clf.coefs_[0]) + clf.intercepts_[0], 0)
    mlp = MLPClassifier(
        hidden_layer_sizes=(4096 // 2,), 
        verbose=True, 
        max_iter=250, 
        early_stopping=True, 
        learning_rate='adaptive', 
        random_state=1701,
    )

    X = df.drop('class', axis=1)
    y = df['class']

    mlp.fit(X, y)

    mapped_df = pd.DataFrame(data=map_features(X, mlp))
    mapped_df['class'] = y

    mapped_df.to_csv(output_path, index=False)    

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: ./{sys.argv[0]} <csv_path> <output_path>")
        exit(1)
    
    CSV_PATH, OUTPUT_PATH = sys.argv[1:]

    CSV_PATH = Path(CSV_PATH)
    OUTPUT_PATH = Path(OUTPUT_PATH)

    df = pd.read_csv(CSV_PATH)
    df = clean_dataset(df)
    df = normalize_dataset(df, view=CSV_PATH.stem.split("_")[0])
    df = stratify(df, 0.2, 1701)

    increase_features(df, OUTPUT_PATH)
