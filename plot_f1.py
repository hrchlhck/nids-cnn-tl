#!/usr/bin/env python3

from pathlib import Path
from plot_rates import _print

import matplotlib.pyplot as plt
import pandas as pd

MONTHS_NAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS = [str(i).zfill(2) for i in range(1, 13)]
MONTHS = {month: name for month, name in zip(MONTHS, MONTHS_NAME)}

MARKERS = {
    'vgg16': 's',
    'alexnet': 'o',
    'googlenet': '^'
}

LINESTYLES = {
    'vgg16': "dotted",
    'alexnet': "-",
    'googlenet': "dashdot"
}

COLORS = {
    'vgg16': "red",
    'alexnet': "blue",
    'googlenet': "black"
}

if __name__ == "__main__":
    plt.rc('font', family='serif', size=16)
    df = pd.DataFrame()
    for file in Path("data/results/csv/pytorch").iterdir():
        _print(file)

        temp = pd.read_csv(file)
        temp["precision"] = temp["tp"] / (temp["tp"] + temp["fp"])
        temp["recall"] = temp["tp"] / (temp["tp"] + temp["fn"])
        temp["f1_score"] = (2 * temp["precision"] * temp["recall"]) / (temp["precision"] + temp["recall"])

        temp["classifier"] = file.stem.split("_")[-1]

        df = pd.concat([df, temp])

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.set(xlabel="Month", ylabel="F1-Score", ylim=(.8, 1.0))
    ax.tick_params(axis='x', rotation=60)

    for cnn in df.classifier.unique():
        temp = df[df["classifier"] == cnn]
        ax.plot(MONTHS_NAME, temp["f1_score"], label=cnn.upper(), marker=MARKERS[cnn], ms=12, linestyle=LINESTYLES[cnn], fillstyle='none', color=COLORS[cnn])
    ax.legend(loc='lower left', frameon=False, prop={'size': 12})
    
    fig.savefig("f1_scores.pdf")