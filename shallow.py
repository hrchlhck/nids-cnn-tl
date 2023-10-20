from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from pathlib import Path
from utils import normalize_dataset, clean_dataset
from datetime import datetime as dt

import numpy as np
import sys
import pandas as pd
import argparse

RANDOM_STATE = 1701
PARSER = argparse.ArgumentParser(prog=sys.argv[0])
PARSER.add_argument("-i", "--input", help="Input data directory", required=True, type=str)
PARSER.add_argument("-o", "--output", help="Output file path", required=True, type=str)
PARSER.add_argument("-u", "--update", help="Perform monthly model updates", action="store_true")

def get_years(dataset_path: Path) -> list:
    ret = set()

    for file in dataset_path.iterdir():
        fname = file.stem

        year = fname.split("_")[1]
        ret.add(year)

        if year in ret:
            continue
    
    return sorted(list(ret))

def get_files_year(dataset_path: Path, year: str) -> list:
    return sorted([f for f in dataset_path.iterdir() if year in f.name])

def get_view(file: Path) -> str:
    return file.stem.split("_")[0]

if __name__ == '__main__':
    args = PARSER.parse_args()

    dataset_path = Path(args.input)
    output_name = Path(args.output)
    update = args.update
    update_delay = 1

    classifiers = {
        'rf': RandomForestClassifier(n_jobs=-1, random_state=RANDOM_STATE),
        'dt': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'gbt': GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = pd.DataFrame()
    for key, classifier in classifiers.items():
        X_train_last, y_train_last = None, None 
        print(f"[ {dt.now()} ] Started classifier {key.upper()}")

        for year in get_years(dataset_path):

            # December does not have events on the dataset
            # Therefore, skipping the whole year
            if year == '2017' or year == 2017:
                print("Skipping year", year)
                continue
        
            files = get_files_year(dataset_path, year)

            for i, file in enumerate(files):
                view = get_view(file)
                df = pd.read_csv(file)

                df = clean_dataset(df)
                df = normalize_dataset(df, view=view)

                X = df.drop('class', axis=1).to_numpy()
                y = df['class'].to_numpy()

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=RANDOM_STATE)

                # Train in january
                if i == 0:
                    print(f"[ {dt.now()} ] Training classifier {key.upper()} at january ({year})")
                    classifier.fit(X_train, y_train)
                    X_train_last, y_train_last = np.copy(X_train), np.copy(y_train)
                
                if update == True and i >= update_delay:
                    print(f"[ {dt.now()} ] Updating classifier {key.upper()} with month {i} ({year})")
                    classifier.fit(X_train_last, y_train_last)
                
                print(f"[ {dt.now()} ] Evaluating classifier {key.upper()}")
                # Evaluation
                y_pred = classifier.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

                temp = dict(
                    tp=tp,
                    tn=tn,
                    fp=fp,
                    fn=fn,
                    fnr=fn/(fn+tp),
                    fpr=fp/(fp+tn),
                    month=str(i+1).zfill(2),
                    year=year,
                    classifier=key
                )

                results = pd.concat([results, pd.DataFrame([temp])], ignore_index=True)
                results.to_csv(output_name, index=False)  
                print(f"[ {dt.now()} ] Saved {key.upper()} data of month {i+1} at {output_name}")
                X_train_last, y_train_last = np.copy(X_train), np.copy(y_train)