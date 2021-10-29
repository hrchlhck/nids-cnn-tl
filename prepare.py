import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from pyDeepInsight import ImageTransformer

from multiprocessing import Process

DATA = Path('./data')

def stratify_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.groupby('class').apply(lambda x: x.sample(n=n))
    return df.reset_index(drop=True)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    ss = MinMaxScaler(feature_range=(0, 1))
    X, y = df.drop('class', axis=1), df['class']
    tmp = pd.DataFrame(ss.fit_transform(X), columns=X.columns.values)
    tmp['class'] = y
    return tmp

def pick_instances(year: str, n: int) -> None:
    file = DATA / 'csv' / year 
    months = list(file.iterdir())
    days_per_month = [list(days.iterdir()) for days in months if days.is_dir()]

    for days in tqdm(days_per_month):
        df = pd.DataFrame() 

        for d in tqdm(days):
            df = df.append(pd.read_csv(d), ignore_index=True)
    
        df = df.sample(frac=1).reset_index(drop=True)

        # Removing unwanted columns
        unwanted_columns = [
            'MAWILAB_taxonomy', 'MAWILAB_label', 
            'MAWILAB_nbDetectors', 'MAWILAB_distance',
        ]
        df.drop(unwanted_columns, axis=1, inplace=True)

        df.drop_duplicates(inplace=True)

        if len(df) < n:
            n = len(df)

        df = stratify_df(df=df, n=n)

        df = normalize(df=df)

        print("Writing csv for year", year, 'month', days[0].parent.name)
        df.to_csv(DATA / 'csv' / days[0].parent.parent.name / days[0].parent.name / 'all.csv', index=False)    

def pick_all_years() -> None:
    def func(fr, to) -> None:
        for year in range(fr, to):
            print("Started", year)
            pick_instances(str(year), 25_000)
    
    Process(target=func, args=(2010, 2013)).start()
    Process(target=func, args=(2014, 2017)).start()
    Process(target=func, args=(2018, 2019)).start()

def to_image() -> None:
    out_file = DATA / 'image'

    files = [sorted([Path(f'data/csv/{f}/{m.name}/augmented_features.csv') for m in Path(f'data/csv/{f}').iterdir()]) for f in range(2010, 2020)]
    
    for f in files:
        it = ImageTransformer(feature_extractor='tsne', 
                    pixels=100, random_state=1701, 
                    n_jobs=-1)

        df = pd.read_csv(f[0])

        # Sempre treinar com janeiro
        X = df.drop('class', axis=1).to_numpy()
        y = df['class'].to_numpy()

        feat = it.fit_transform(X, format='scalar')

        for month in f:
            for i in tqdm(range(feat.shape[0])):
                fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
                
                sns.heatmap(feat[i], cbar=False, xticklabels=False, yticklabels=False, ax=ax)

                atk = y[i]
                if atk == 1:
                    fig.savefig(out_file / month.parent.parent.name / month.parent.name / 'attack' / f'instance_{i}.png')
                else:
                    fig.savefig(out_file / month.parent.parent.name / month.parent.name / 'normal' / f'instance_{i}.png')
                break

            fig.tight_layout()
    
def test_classifiers(clf, pos) -> None:
    files = [sorted([Path(f'data/csv/{f}/{m.name}/all.csv') for m in Path(f'data/csv/{f}').iterdir()]) for f in range(2010, 2020)]
    
    ret = pd.DataFrame(columns=['year', 'month', 'fp', 'fn', 'tp', 'tn', 'accuracy', 'recall', 'precision'])

    pbar_files = tqdm(files, position=pos)
    for f in pbar_files:
        pbar_files.set_description("Classifier: %s" % clf.__class__.__name__)
        
        # Attempt to fix "FloatingPointError: invalid value encountered in double_scalars"
        df = pd.read_csv(f[0])
        num_cols = df.select_dtypes('number').columns.values
        df[num_cols] = df[num_cols].round(5)

        X_train, y_train = df.drop('class', axis=1), df['class']

        clf.fit(X_train, y_train)

        pbar_months = tqdm(f, position=pos)
        for month in pbar_months:

            df = pd.read_csv(month)
            X_test, y_test = df.drop('class', axis=1), df['class']

            y_pred = clf.predict(X_test)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

            tmp = 1 if tp + fn == 0 else tp + fn
            recall = tp / (tmp)

            tmp = 1 if tp + fp == 0 else tp + fp
            precision = tp / (tmp)
            data = {
                'year': month.parent.parent.name,
                'month': month.parent.name,
                'fp': fp,
                'fn': fn,
                'tp': tp,
                'tn': tn,
                'accuracy': (tp + tn) / len(df),
                'recall': recall,
                'precision': precision,
                # 'f1score': 2 * ((recall * precision) / (recall + precision)),
            }
            ret = ret.append(pd.Series(data), ignore_index=True)
            ret.to_csv(DATA / 'ml_tests' / f'{clf.__class__.__name__}.csv', index=False)

def run_test_classifiers():
    Process(target=test_classifiers, args=(RandomForestClassifier(n_jobs=-1), 0)).start()
    Process(target=test_classifiers, args=(GradientBoostingClassifier(), 1)).start()
    p = Process(target=test_classifiers, args=(MLPClassifier(4096), 2))
    p.start()
    p.join()

def plot_metrics():
    result_path = DATA / 'ml_tests'
    output_path = DATA / 'plots' / 'batch'

    years = list(range(2010, 2020))
    
    for year in years:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
        for file, ax in zip(result_path.iterdir(), axes):
            df = pd.read_csv(file)

            df['fnr'] = df['fn'] / (df['fn'] + df['tp'])
            df['fpr'] = df['fp'] / (df['fp'] + df['tn'])

            X, Y1, Y2 = list(range(1, 13)), df[df['year'] == year]['fn'], df[df['year'] == year]['fp']

            ax.plot(X, Y1, label='FN', marker='s', ms=9, linestyle='dotted', fillstyle='none', color='red')
            ax.plot(X, Y2, label='FP', marker='s', ms=9, linestyle='dotted', fillstyle='none', color='black')
            ax.set(xticks=X, xlim=(0, 13), xlabel='Month')
            ax.set(ylim=(-400, 26_000), ylabel='Average Error Rate (%)')
            ax.set_title(file.stem)
            ax.legend()

        fig.suptitle(year)
        fig.savefig(output_path / f'metrics_{year}.png', dpi=210)

def feature_extractor(_from: int, to: int) -> None:
    print(f"Started feature extractor from {_from} to {to}")
    files = [sorted([Path(f'data/csv/{f}/{m.name}/all.csv') for m in Path(f'data/csv/{f}').iterdir()]) for f in range(_from, to)]

    # np.maximum as 'relu'
    map_features = lambda X, clf: np.maximum(np.matmul(X, clf.coefs_[0]) + clf.intercepts_[0], 0)

    for f in tqdm(files):
        mlp = MLPClassifier(hidden_layer_sizes=(2048,), verbose=True, max_iter=50)

        # Getting january data 
        df = pd.read_csv(f[0])
        X = df.drop('class', axis=1).to_numpy()
        y = df['class'].to_numpy()

        mlp.fit(X, y)

        for month in tqdm(f):    
            df = pd.read_csv(month)
            X = df.drop('class', axis=1).to_numpy()
            y = df['class'].to_numpy()

            mapped_df = pd.DataFrame(data=map_features(X, mlp), columns=[f'neuron_{i}' for i in range(mlp.hidden_layer_sizes[0])])
            mapped_df['class'] = y

            mapped_df.to_csv(DATA / 'csv' / month.parent.parent.name / month.parent.name / 'augmented_features.csv', index=False)    

def run_feature_extractor() -> None:
    Process(target=feature_extractor, args=(2010, 2013)).start()
    Process(target=feature_extractor, args=(2014, 2017)).start()
    p = Process(target=feature_extractor, args=(2018, 2020))
    p.start()
    p.join()
    
def main() -> None:
    # pick_all_years()
    # run_test_classifiers()
    # plot_metrics()
    # to_image()
    # run_feature_extractor()

    it = ImageTransformer(feature_extractor='tsne', 
                    pixels=75, random_state=1701, 
                    n_jobs=-1)

    df = pd.read_csv('data/csv/2010/01/augmented_features.csv')

    X = df.drop('class', axis=1).to_numpy()

    feat = it.fit_transform(X, format='scalar')

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    
    sns.heatmap(feat[0], cbar=False, xticklabels=False, yticklabels=False, ax=ax)

    fig.savefig('teste.png')
    fig.tight_layout()


if __name__ == '__main__':
    main()
