import sys

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

MONTHS_NAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS = [str(i).zfill(2) for i in range(1, 13)]
MONTHS = {month: name for month, name in zip(MONTHS, MONTHS_NAME)}

if __name__ == '__main__':
    plt.rc('font', family='serif', size=16)

    if len(sys.argv) < 2:
        print(f"Usage: ./{sys.argv[0]} <results_file>")
        exit(1)

    results_file = Path(sys.argv[1])

    names = {'dt': 'Decision Tree', 'rf': 'Random Forest', 'gbt': 'Gradient Boosting'}

    df = pd.read_csv(results_file)

    for year in df.year.unique(): 
        for name in names.keys():
            fig, ax = plt.subplots(constrained_layout=True, figsize=(7, 5))
            ax.set(ylim=(-2.5, 100), xlabel='Month', ylabel='Error Rate (%)')
            ax.tick_params(axis='x', rotation=45)
            temp = df[(df['classifier'] == name) & (df['year'] == year)]

            fpr = temp["fpr"] * 100
            fnr = temp["fnr"] * 100

            ax.plot(MONTHS_NAME, fpr, label='FPR', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='black')
            ax.plot(MONTHS_NAME, fnr, label='FNR', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='red')

            ax.legend(loc='upper left', frameon=False)

            fig.savefig(f'data/results/image/{results_file.stem}_{year}_{name}.pdf')
