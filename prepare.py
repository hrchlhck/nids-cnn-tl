import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from pyDeepInsight import ImageTransformer

from multiprocessing import Process

from random import Random 

DATA = Path('./data')
R = Random(1)

def stratify_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.groupby('class').apply(lambda x: x.sample(n=n))
    return df.reset_index(drop=True)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    ss = MinMaxScaler(feature_range=(-1, 1))
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

    files = [sorted([Path(f'data/csv/{f}/{m.name}/all.csv') for m in Path(f'data/csv/{f}').iterdir()]) for f in range(2010, 2020)]

    for f in files:
        it = ImageTransformer(feature_extractor='tsne', 
                    pixels=50, random_state=1701, 
                    n_jobs=-1)

        # Fitting with January data
        df_train = pd.read_csv(f[0])
        X_train = df_train.drop('class', axis=1).to_numpy()
        y_train = df_train['class'].to_numpy()

        it.fit(X_train, y_train)

        for month in f:
            df = pd.read_csv(month)

            mat = it.transform(df.drop('class', axis=1).to_numpy(), format='scalar')
            print(mat.shape)

            for i in tqdm(range(mat.shape[0])):
                fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)

                cax = sns.heatmap(mat[i], cmap='hot',
                      linewidth=0.01, linecolor='dimgrey',
                      square=True, ax=ax, cbar=False)
                cax.axis('off')
                ax.grid(False)

                fig.savefig(out_file / month.parent.parent.name / month.parent.name / f'instance_{i}.png')
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
    Process(target=test_classifiers, args=(RandomForestClassifier(n_jobs=-1),0)).start()
    Process(target=test_classifiers, args=(GradientBoostingClassifier(),1)).start()
    p = Process(target=test_classifiers, args=(GaussianNB(),2))
    p.start()
    p.join()

def plot_metrics():
    result_path = DATA / 'ml_tests'
    output_path = DATA / 'plots' / 'batch'

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    axes = (ax0, ax1, ax2)

    for file, ax in zip(result_path.iterdir(), axes):
        df = pd.read_csv(file)

        X, Y1, Y2 = list(range(2010, 2020)), df.groupby('year').mean()['fn'], df.groupby('year').mean()['fp']

        ax.plot(X, Y1, label='FN', marker='s', ms=9, linestyle='dotted', fillstyle='none', color='red')
        ax.plot(X, Y2, label='FP', marker='s', ms=9, linestyle='dotted', fillstyle='none', color='black')
        ax.set(xticks=X, xlim=(2009, 2020), xlabel='Year')
        ax.set(ylim=(-300, 25_300), ylabel='Average number of instances')
        ax.set_title(file.stem)
        ax.legend()
    
    fig.savefig(output_path / 'metrics.png', dpi=210)
    plt.show()

def main() -> None:
    # run_test_classifiers()
    # plot_metrics()
    to_image()

if __name__ == '__main__':
    main()
