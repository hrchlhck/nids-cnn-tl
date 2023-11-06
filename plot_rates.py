#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime as dt

MONTHS_NAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS = [str(i).zfill(2) for i in range(1, 13)]
MONTHS = {month: name for month, name in zip(MONTHS, MONTHS_NAME)}

def _print(*args):
    print(f"[{dt.now().strftime('%Y-%m-%d %H:%M:%S')}]", *args, flush=True)

if __name__ == '__main__':
    plt.rc('font', family='serif', size=16)

    if len(sys.argv) < 2:
        print(f"Usage: ./{sys.argv[0]} <results_file>")
        exit(1)

    results_file = Path(sys.argv[1])

    names = {'dt': 'Decision Tree', 'rf': 'Random Forest', 'gbt': 'Gradient Boosting', 'alexnet': 'alexnet', 'nb': 'Naive Bayes'}

    df = pd.read_csv(results_file)

    
    df["fpr"] = df["fp"] / (df["fp"] + df["tn"])
    df["fnr"] = df["fn"] / (df["fn"] + df["tp"])

    for year in df.year.unique(): 
        for name in names.keys():   
            if name == 'dt' or name == 'alexnet':
                continue
            
            fig, ax = plt.subplots(constrained_layout=True, figsize=(4, 4))
            ax.set(ylim=(-2.5, 50), xlabel='Month', ylabel='Error Rate (%)')
            ax.tick_params(axis='x', rotation=45)
            temp = df[(df['classifier'] == name) & (df['year'] == year)]
            fpr = temp["fpr"] * 100
            fnr = temp["fnr"] * 100
            
            if len(MONTHS_NAME) != len(fpr):
                print(f"{results_file.stem}({year}, {name}) different sizes ({len(MONTHS_NAME)} != {len(fpr)})")
                break

            ax.plot(MONTHS_NAME, fpr, label='FPR', marker='s', ms=12, linestyle='dotted', fillstyle='none', color='black')
            ax.plot(MONTHS_NAME, fnr, label='FNR', marker='^', ms=12, linestyle='dotted', fillstyle='none', color='red')

            ax.legend(loc='upper left', frameon=False)

            out = Path(f'data/results/image/{results_file.stem}_{year}_{name}.pdf')
            fig.savefig(out)
            _print(out, "OK")
