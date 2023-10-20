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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from imblearn.under_sampling import RandomUnderSampler

from pyDeepInsight import ImageTransformer

from multiprocessing import Process
from threading import Thread, Semaphore

from lib import *

from IPython import embed

MULTIPLEXER = Semaphore(3)

def stratify_df(df: pd.DataFrame, n: int) -> pd.DataFrame:
    df = df.groupby('class').apply(lambda x: x.sample(n=n, replace=True))
    return df.reset_index(drop=True)

def normalize(df: pd.DataFrame) -> pd.DataFrame:
    ss = MinMaxScaler(feature_range=(0, 1))
    X, y = df.drop('class', axis=1), df['class']
    tmp = pd.DataFrame(ss.fit_transform(X), columns=X.columns.values)
    tmp['class'] = y
    return tmp

def pick_instances(year: str, view='VIEGAS') -> None:
    file = DATA / 'csv' / view / year 
    months = list(sorted(file.iterdir()))
    days_per_month = [list(days.iterdir()) for days in months if days.is_dir()]

    print(year)
    for days in tqdm(days_per_month):
        df = pd.DataFrame() 

        for d in tqdm(days):
            df = df.append(pd.read_csv(d), ignore_index=True)

        # Removing unwanted columns
        unwanted_columns = [
            'MAWILAB_taxonomy', 'MAWILAB_label', 
            'MAWILAB_nbDetectors', 'MAWILAB_distance',
        ]

        if view == 'VIEGAS':
            unwanted_columns += ['VIEGAS_numberOfDifferentServices_A', 'VIEGAS_numberOfDifferentDestinations_A']

        df.drop(unwanted_columns, axis=1, inplace=True)

        ss = MinMaxScaler(feature_range=(0, 1))
        X, y = df.drop('class', axis=1), df['class']
        df[X.columns.values] = ss.fit_transform(X)
        df['class'] = y

        rus = RandomUnderSampler(random_state=0)

        X_sampled, y_sampled = rus.fit_resample(df.drop('class', axis=1), df['class'])

        df_bal = pd.DataFrame(columns=X_sampled.columns)
        df_bal[X_sampled.columns] = X_sampled
        df_bal['class'] = y_sampled

        print(df_bal['class'].value_counts() / len(df_bal) * 100)

        print("Writing csv for year", year, 'month', days[0].parent.name, 'at view', view)
        df_bal.to_csv(DATA / 'csv' / view /days[0].parent.parent.name / days[0].parent.name / 'all.csv', index=False)    

def pick_allget_years() -> None:
    Process(target=pick_instances, args=('2016',)).start()
    # Process(target=pick_instances, args=('2017',)).start()
    # Process(target=pick_instances, args=('2018',)).start()
    # Process(target=pick_instances, args=('2019',)).start()

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
                    n_jobs=-1)

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
    
def test_classifiers(clf, pos, semaphore, update=False, view='VIEGAS') -> None:
    files = [sorted([Path(f'data/csv/{view}/{f}/{m.name}/all.csv') for m in Path(f'data/csv/{view}/{f}').iterdir()]) for f in range(2016, 2020)]
    output_prefix = DATA / 'ml_tests'
    clf_name = clf.__class__.__name__

    ret = pd.DataFrame(columns=['year', 'month', 'fp', 'fn', 'tp', 'tn', 'fnr','fpr', 'accuracy', 'recall', 'precision'])

    msg = f"Starting tests with {clf_name} without monthly updates"

    if update:
        msg = msg.replace('without', 'with')
    with semaphore:
        print(msg)
        pbar_files = tqdm(files, position=pos)
        
        last_month = '01'
        last_year = files[0][0].parent.parent.name

        for f in pbar_files:    
            pbar_files.set_description(f"Classifier: {clf_name}_update_{update}")
            
            current_year = f[0].parent.parent.name

            pbar_months = tqdm(f, position=pos)
            for month in pbar_months:
                current_month = month.parent.name

                if current_month == '01':
                    df = pd.read_csv(f[0])
                    X_train, y_train = df.drop('class', axis=1), df['class']
                    clf.fit(X_train, y_train)

                if update == True and last_month != current_month:                    
                    df_last = pd.read_csv(f'data/csv/{view}/{last_year}/{last_month}/all.csv')

                    print('Updating %s with %s/%s (current: %s/%s)' % (clf_name, last_year, last_month, current_year, current_month))

                    X, y = df_last.drop('class', axis=1), df_last['class']
                    clf.fit(X, y)
                    last_month = current_month
                    last_year = current_year

                df = pd.read_csv(month)
                X_test, y_test = df.drop('class', axis=1), df['class']
                y_pred = clf.predict(X_test)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

                tmp = 1 if tp + fn == 0 else tp + fn
                recall = tp / (tmp)

                tmp = 1 if tp + fp == 0 else tp + fp
                precision = tp / (tmp)
                
                fnr = fn / (fn + tp)
                fpr = fp / (fp + tn)

                data = {
                    'year': current_year,
                    'month': current_month,
                    'fp': fp,
                    'fn': fn,
                    'tp': tp,
                    'tn': tn,
                    'fnr': fnr,
                    'fpr': fpr,
                    'accuracy': (tp + tn) / len(df),
                    'recall': recall,
                    'precision': precision,
                    'f1score': 2 * ((recall * precision) / (recall + precision)),
                }

                out_fname = output_prefix / f'{clf_name}_{view}.csv'
                if update == True:
                    out_fname = output_prefix / f'{clf_name}_{view}_update.csv'

                ret = ret.append(pd.Series(data), ignore_index=True)
                ret.to_csv(out_fname, index=False)

def run_test_classifiers():
    rs = 3
    classif = [
        RandomForestClassifier(n_jobs=-1, random_state=rs),
        DecisionTreeClassifier(random_state=rs),
        GradientBoostingClassifier(random_state=rs),
        RandomForestClassifier(n_jobs=-1, random_state=rs),
        DecisionTreeClassifier(random_state=rs),
        GradientBoostingClassifier(random_state=rs),
    ]

    for i in range(len(classif)):
        params = (classif[i], 0, MULTIPLEXER, True)

        if i < len(classif) // 2:
            params = (classif[i], 0, MULTIPLEXER, False)
    
        t = Thread(target=test_classifiers, args=params).start()


def plot_metrics2(year: int, how='monthly') -> None:
    """
        how: groupby month, bimester, trimester, quarter, semester
    """

    result_path = DATA / 'ml_tests'
    output_path = DATA / 'plots' / 'batch'
    
    for file in result_path.glob('*.csv'):
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), constrained_layout=True)
        df = pd.read_csv(file)

        if not 'fnr' in df.columns.values or not 'fnr' in df.columns.values:
            df['fnr'] = df['fn'] / (df['fn'] + df['tp']) * 100
            df['fpr'] = df['fp'] / (df['fp'] + df['tn']) * 100
        else:
            df['fnr'] = df['fnr'] * 100
            df['fpr'] = df['fpr'] * 100
        
        print(file.stem, how)

        if how == 'monthly':
            outname = plot_monthly()

        if how == 'quarter':
            outname = plot_quarter(year, file, output_path, df, ax)
        
        if how == 'semester':
            outname = plot_semester(year, file, output_path, df, ax)
        
        fig.savefig(outname, dpi=210, transparent=True)

def feature_extractor(_from: int, to: int, view='VIEGAS') -> None:
    print(f"Started feature extractor from {_from} to {to}")
    files = [sorted([Path(f'data/csv/{view}/{f}/{m.name}/all.csv') for m in Path(f'data/csv/{view}/{f}').iterdir()]) for f in range(_from, to)]

    # np.maximum as 'relu'
    # coefs_ -> weights
    # intercepts_ -> biases

    # X * weights + biases -> feedforward
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
    p = Process(target=feature_extractor, args=(2016, 2017))
    p.start()
    p.join()
    
def main() -> None:
    # pick_allget_years()
    # run_test_classifiers()
    # plot_metrics()
    plot_metrics2(2016, 'semester')
    # to_image()
    # run_feature_extractor()

if __name__ == '__main__':
    warnings.simplefilter('ignore')
    
    # plt.rc('pgf', texsystem='pdflatex') 

    # Font setup
    fonts_path = ['/usr/local/share/fonts/']
    fonts = mpl.font_manager.findSystemFonts(fontpaths=fonts_path, fontext='ttf')

    for font in fonts:
        font_manager.fontManager.addfont(font)

    plt.rc('font', family='Palatino', size=20)
    # End font setup

    # Entrypoint
    main()
