#!/usr/bin/env python3

import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.neural_network import MLPClassifier
from utils import (
    normalize_dataset, 
    stratify, 
    clean_dataset
)

from pickle import (
    dumps as pdumps,
    loads as ploads
)

from datetime import datetime as dt

def _print(*args):
    print(f"[{dt.now().strftime('%Y-%m-%d %H:%M:%S')}]", *args, flush=True)

def save_model(model: MLPClassifier, output_path: Path) -> None:
    with open(output_path, 'wb') as fp:
        fp.write(pdumps(model))

def load_model(model_path: Path) -> MLPClassifier:
    with open(model_path, 'rb') as fp:
        return ploads(fp.read())

def increase_features(df: pd.DataFrame, output_path: Path, model_output_path: Path, train=True):
    # coefs_ -> weights
    # intercepts_ -> biases

    # X * weights + biases -> feedforward

    # np.maximum -> ReLU
    map_features = lambda X, clf: np.maximum(np.matmul(X, clf.coefs_[0]) + clf.intercepts_[0], 0)
    mlp = MLPClassifier(
        hidden_layer_sizes=(2048,), 
        verbose=True, 
        max_iter=250, 
        early_stopping=True, 
        learning_rate='adaptive', 
        random_state=1701,
    )

    X = df.drop('class', axis=1)
    y = df['class']

    if train == True:
        _print('Training MLP')
        mlp.fit(X, y)
    else:
        _print("Loading model from", model_output_path)
        mlp = load_model(model_output_path)

    _print('Increasing features')
    X_mapped = map_features(X, mlp)
    X_mapped[X_mapped > 1.] = 1
    mapped_df = pd.DataFrame(data=X_mapped)
    mapped_df['class'] = y
    mapped_df['class'] = mapped_df['class'].map(lambda x: 1 if x == 'attack' else 0)

    _print('Saving')
    mapped_df.to_csv(output_path, index=False)

    if train:
        _print("Saving model at", model_output_path)
        save_model(mlp, model_output_path)


if __name__ == "__main__":

    months = [str(i).zfill(2) for i in range(11, 13)]
    for month in months:

        CSV_PATH = Path(f"data/csv/NIGEL/NIGEL_2014_{month}.csv")
        OUTPUT_PATH = Path(f"/data/img_nids/mlp/NIGEL_2014_{month}.csv")
        MODEL_OUTPUT_PATH = Path(f"data/models/mlp.model")

        _print("Saving data at", OUTPUT_PATH)

        df = pd.read_csv(CSV_PATH)
        _print(f'Dataset {CSV_PATH} loaded')

        df = clean_dataset(df)
        _print('Dataset cleaned')

        df = normalize_dataset(df, view=CSV_PATH.stem.split("_")[0])
        _print('Dataset normalized')

        df = stratify(df, 0.3, 1701)
        _print('Dataset stratified')

        train = False
        if month == '01': 
            _print("Training model at", month)
            train = True

        increase_features(df, OUTPUT_PATH, MODEL_OUTPUT_PATH, train=train)
        _print("Done month", month)
