import pandas as pd
import numpy as np
import seaborn as sns
import warnings
import pickle

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager

from pathlib import Path
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

from pyDeepInsight import ImageTransformer

from multiprocessing import Process

DATA = Path('./data')
MONTHS_NAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS = [(str(i)) if i >= 10 else f"0{i}" for i in range(1, 13)]
MONTHS = {month: name for month, name in zip(MONTHS, MONTHS_NAME)}

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

def pick_allget_years() -> None:
    def func(fr, to) -> None:
        for year in range(fr, to):
            print("Started", year)
            pick_instances(str(year), 25_000)
    
    Process(target=func, args=(2010, 2013)).start()
    Process(target=func, args=(2014, 2017)).start()
    Process(target=func, args=(2018, 2019)).start()

def save_fig(features, y, start_idx, out, _id):
    prog = tqdm(range(features.shape[0]), position=0)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)

    for i in prog:
        sns.heatmap(features[i], cbar=False, xticklabels=False, yticklabels=False, ax=ax)

        if y[i] == 1:
            fig.savefig(out / 'attack' / f'instance_{i+start_idx}.jpg', format='jpg')
        else:
            fig.savefig(out / 'normal' / f'instance_{i+start_idx}.jpg', format='jpg')
        plt.cla()
        prog.update(1)

def to_image(max_processes=4) -> None:
    global DATA
    out_file = DATA / 'image'

    files = [sorted([Path(f'data/csv/{f}/{m.name}/augmented_features.csv') for m in Path(f'data/csv/{f}').iterdir()]) for f in range(2012, 2013)]

    for f in files:
        if not Path('image_transformer.obj') in list(Path('.').glob('*.obj')):
            print("Creating Image Transformer")
            it = ImageTransformer(feature_extractor='tsne', 
                    pixels=64, random_state=1701, 
                    n_jobs=4)

            df = pd.read_csv(f[0], low_memory=True)

            # Always train with january data
            X = df.drop('class', axis=1).to_numpy()
            y = df['class'].to_numpy()

            print("Fitting Image Transformer")
            feat = it.fit(X)

            with open('image_transformer.obj', mode='wb+') as fp:
                fp.write(pickle.dumps(it))
            
            del df, X, y
        else:
            print("Loading image transformer")
            with open('image_transformer.obj', mode='rb') as fp:
                it = pickle.loads(fp.read())

        for month in f:
            print('Generating images for month', month.parent.name)
            df = pd.read_csv(month, low_memory=True)

            # Always train with january data
            y = df['class'].to_numpy()
            print(len(y))

            print("Transforming data")
            feat = it.transform(df.drop('class', axis=1).to_numpy(), format='scalar')

            out = out_file / month.parent.parent.name / month.parent.name
            actual_boundary = 0
            for i in range(max_processes):
                slice_ = slice(actual_boundary, len(y) // max_processes + actual_boundary)
                p = Process(target=save_fig, args=(feat[slice_], y[slice_], actual_boundary, out, i))
                p.start()
                actual_boundary += len(y) // max_processes 
            p.join()

            del df, feat, y                
    
def test_classifiers(clf, pos, update=False) -> None:
    files = [sorted([Path(f'data/csv/{f}/{m.name}/all.csv') for m in Path(f'data/csv/{f}').iterdir()]) for f in range(2012, 2013)]
    output_prefix = DATA / 'ml_tests'
    clf_name = clf.__class__.__name__
    out_fname = output_prefix / f'{clf_name}.csv'

    ret = pd.DataFrame(columns=['year', 'month', 'fp', 'fn', 'tp', 'tn', 'accuracy', 'recall', 'precision'])

    pbar_files = tqdm(files, position=pos)
    for f in pbar_files:
        last_month = '01'
        
        pbar_files.set_description(f"Classifier: {clf_name}")
        
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

            if update == True and last_month < str(month.parent.name):
                out_fname = output_prefix / f'{clf_name}_update.csv'
                df_last = pd.read_csv(f'data/csv/2012/{last_month}/all.csv')

                print(last_month, month.parent.name)
                X_train, y_train = df_last.drop('class', axis=1), df['class']
                clf.fit(X_train, y_train)
                last_month = month.parent.name

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
            ret.to_csv(out_fname, index=False)

def run_test_classifiers():
    print("Starting tests without monthly updates")
    Process(target=test_classifiers, args=(RandomForestClassifier(n_jobs=-1), 0)).start()
    Process(target=test_classifiers, args=(GradientBoostingClassifier(), 1)).start()
    p = Process(target=test_classifiers, args=(MLPClassifier(), 2))
    p.start()
    p.join()

    # print("Starting tests with monthly updates")
    # Process(target=test_classifiers, args=(RandomForestClassifier(n_jobs=-1), 0, True)).start()
    # Process(target=test_classifiers, args=(GradientBoostingClassifier(), 1, True)).start()
    # p = Process(target=test_classifiers, args=(MLPClassifier(), 2, True))
    # p.start()
    # p.join()

def plot_metrics():
    """ The function intent was to pick all years and select the best of them in terms of accuracy """
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

def plot_metrics2(year: int) -> None:
    result_path = DATA / 'ml_tests'
    output_path = DATA / 'plots' / 'batch'
    
    for file in result_path.iterdir():
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        df = pd.read_csv(file)

        df['fnr'] = df['fn'] / (df['fn'] + df['tp']) * 100
        df['fpr'] = df['fp'] / (df['fp'] + df['tn']) * 100

        df_year = df[df['year'] == year]
        X, Y1, Y2 = list(MONTHS.values()), df_year['fnr'], df_year['fpr']
        print(file.stem)

        ax.plot(X, Y1, label='FN', marker='s', ms=9, linestyle='dotted', fillstyle='none', color='red')
        ax.plot(X, Y2, label='FP', marker='o', ms=9, linestyle='dotted', fillstyle='none', color='black')
        ax.set(xticks=X, xlim=(-1, 12), xlabel='Month')
        ax.tick_params(axis='x', rotation=60)
        ax.set(ylim=(-10, 100), ylabel='Average Error Rate (%)')
        ax.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.2), ncol=2)
        
        fig.savefig(output_path / f'{year}_{file.stem}.png', dpi=210, transparent=True)

def feature_extractor(_from: int, to: int) -> None:
    print(f"Started feature extractor from {_from} to {to}")
    files = [sorted([Path(f'data/csv/{f}/{m.name}/all.csv') for m in Path(f'data/csv/{f}').iterdir()]) for f in range(_from, to)]

    # np.maximum as 'relu'
    map_features = lambda X, clf: np.maximum(np.matmul(X, clf.coefs_[0]) + clf.intercepts_[0], 0)

    for f in tqdm(files):
        mlp = MLPClassifier(hidden_layer_sizes=(4096 // 2,), verbose=True, max_iter=250, learning_rate='')

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
    # Process(target=feature_extractor, args=(2010, 2013)).start()
    # Process(target=feature_extractor, args=(2014, 2017)).start()
    p = Process(target=feature_extractor, args=(2012, 2013))
    p.start()
    p.join()
    
def main() -> None:
    # pick_allget_years()
    # run_test_classifiers()
    # plot_metrics()
    # plot_metrics2(2012)
    to_image()
    # run_feature_extractor()

if __name__ == '__main__':
    warnings.simplefilter('ignore')

    # Font setup
    fonts_path = ['/usr/local/share/fonts/p/']
    fonts = mpl.font_manager.findSystemFonts(fontpaths=fonts_path, fontext='ttf')

    for font in fonts:
        font_manager.fontManager.addfont(font)

    plt.rc('font', family='Palatino', size=20)
    # End font setup

    # Entrypoint
    main()
