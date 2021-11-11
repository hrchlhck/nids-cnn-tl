from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any

import pandas as pd

from .constants import DATA, MONTHS

__all__ = ['plot_semester', 'plot_monthly', 'plot_quarter']

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

def plot_semester(year: int, file: Path, output_path: Path, df: pd.DataFrame, ax: plt.Axes) -> Path:
    def get_semester(x: Any) -> str:
        s = 'S2'
        if x >= 1 and x <= 6:
            s = 'S1'
        return s
    
    df = df.copy()
    df['semester'] = df['month'].apply(get_semester)
    df['year_by_semester'] = df['year'].astype(str) + '-' + df['semester']

    grouped = df.groupby('year_by_semester').mean()[['fnr', 'fpr']]
    X = df['year_by_semester'].unique()
    Y1, Y2 = grouped['fnr'], grouped['fpr']

    ax.plot(X, Y1, label='FN', marker='s', ms=9, linestyle='dotted', fillstyle='none', color='red')
    ax.plot(X, Y2, label='FP', marker='o', ms=9, linestyle='dotted', fillstyle='none', color='black')
    ax.set(xticks=X, xlim=(-1, 8), xlabel='Month')
    ax.tick_params(axis='x', rotation=90)
    ax.set(ylim=(-10, 100), ylabel='Error Rate (%)')
    ax.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.25), ncol=2)

    return output_path / f'{year}_{file.stem}_semester.png'

def plot_monthly(year: int, file: Path, output_path: Path, df: pd.DataFrame, ax: plt.Axes) -> Path:
    df = df.copy()
    df_year = df[df['year'] == year]

    X, Y1, Y2 = list(MONTHS.values()), df_year['fnr'], df_year['fpr']

    ax.plot(X, Y1, label='FN', marker='s', ms=9, linestyle='dotted', fillstyle='none', color='red')
    ax.plot(X, Y2, label='FP', marker='o', ms=9, linestyle='dotted', fillstyle='none', color='black')
    ax.set(xticks=X, xlim=(-1, 12), xlabel='Month')
    ax.tick_params(axis='x', rotation=60)
    ax.set(ylim=(-10, 100), ylabel='Error Rate (%)')
    ax.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.2), ncol=2)

    return output_path / f'{year}_{file.stem}.png'

def plot_quarter(year: int, file: Path, output_path: Path, df: pd.DataFrame, ax: plt.Axes) -> Path:
    def get_quarter(x: Any) -> str:
        q = 'Q4'

        if x >= 1 and x <= 3:
            q = 'Q1'
        elif x >= 4 and x <= 6:
            q = 'Q2'
        elif x >= 7 and x <= 9:
            q = 'Q3'
        
        return q

    df = df.copy()
    df['quarter'] = df['month'].apply(get_quarter)
    df['year_by_quarter'] = df['year'].astype(str) + '-' + df['quarter']

    grouped = df.groupby('year_by_quarter').mean()[['fnr', 'fpr']]
    X = df['year_by_quarter'].unique()
    Y1, Y2 = grouped['fnr'], grouped['fpr']

    ax.plot(X, Y1, label='FN', marker='s', ms=9, linestyle='dotted', fillstyle='none', color='red')
    ax.plot(X, Y2, label='FP', marker='o', ms=9, linestyle='dotted', fillstyle='none', color='black')
    ax.set(xticks=X, xlim=(-1, 16), xlabel='Month')
    ax.tick_params(axis='x', rotation=90)
    ax.set(ylim=(-10, 100), ylabel='Error Rate (%)')
    ax.legend(loc='upper center', frameon=False, bbox_to_anchor=(0.5, 1.25), ncol=2)

    return output_path / f'{year}_{file.stem}_quartile.png'