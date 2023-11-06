#!/usr/bin/env python3

import sys

import matplotlib.pyplot as plt
from IPython import embed
import torch
from sklearn.metrics import auc as auc_score, roc_curve
from pathlib import Path
from datetime import datetime as dt
from typing import Tuple
from tqdm import tqdm
import random

from tf import VGG16, ALEXNET, GOOGLENET, CustomDataset, BATCH_SIZE, DataLoader, transforms, ToImage, Resize,MinMaxScaler, now, SEED

from numpy import loadtxt, float32 as f32, array, ndarray, random as np_random

MONTHS_NAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS = [str(i).zfill(2) for i in range(1, 13)]
MONTHS = {month: name for month, name in zip(MONTHS, MONTHS_NAME)}
MODELS = {
    'vgg16': VGG16,
    'alexnet': ALEXNET,
    'googlenet': GOOGLENET
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

def _print(*args):
    print(f"[{dt.now().strftime('%Y-%m-%d %H:%M:%S')}]", *args, flush=True)

def load_model(models: dict, model_path: Path) -> torch.nn.Module:
    model: torch.nn.Module = models.get(model_path.stem, VGG16)(2)
    model_params = torch.load(model_path, map_location=torch.device("cpu"))

    model.load_state_dict(model_params)
    _print("Loaded model", model_path.stem)

    return model

def load_dataset(data_path: Path) -> DataLoader:
    data = loadtxt(data_path, delimiter=",", skiprows=1, dtype=f32)
    _print("Loaded data", data_path)

    X, y = data[:, :-1], data[:, -1]

    X_test = torch.Tensor(X)
    y_test = torch.LongTensor(y)

    transform = transforms.Compose([
        ToImage(),
        Resize((224, 224)),
        MinMaxScaler(feature_range=(0.0, 1.0)),
    ])

    del X, y, data

    cd_test  = CustomDataset(subset=(X_test, y_test), transform=transform)

    return DataLoader(cd_test, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)

def get_predict_proba(model: torch.nn.Module, dataset: DataLoader, device: str) -> Tuple[ndarray, ndarray]:
    it_eval = tqdm(enumerate(dataset), total=len(dataset))
    y_true = list()
    y_pred_proba = list()

    x: torch.Tensor
    y: torch.Tensor
    output: torch.Tensor

    with torch.no_grad():
        model = model.eval()
        for i, (x, y) in it_eval:
            x = x.to(device)
            y = y.to(device)

            output = model(x)

            y_true.extend(y.cpu().numpy().ravel())
            y_pred_proba.extend(output.cpu().numpy())

            desc = f"[{now()}] {i+1}/{len(dataset)}"
            it_eval.set_description(desc)

    return array(y_true).reshape(-1, 1), array(y_pred_proba)

if __name__ == '__main__':
    torch.manual_seed(SEED)
    np_random.seed(SEED)
    random.seed(SEED)

    plt.rc('font', family='serif', size=16)

    if len(sys.argv) < 2:
        print(f"Usage: ./{sys.argv[0]} <models_path>")
        exit(1)

    device = 'cpu'
    if torch.cuda.is_available():
        _print("Using CUDA")
        device = 'cuda'

    model_path = Path(sys.argv[1])

    fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", xlim=(-0.05, 1.05), ylim=(-0.05, 1.05))

    for model_file in model_path.iterdir():
        if model_file.stem == 'mlp':
            _print("Skipping MLP")
            continue

        model = load_model(MODELS, model_file)
        model = model.to(device)

        dataset = load_dataset(Path("/data/img_nids/mlp/NIGEL_2014_02.csv"))

        y, proba = get_predict_proba(model, dataset, device)
        
        fpr, tpr, _ = roc_curve(y, proba[:, 1])
        auc = auc_score(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{model_file.stem.upper()} (AUC={auc:.2f})", alpha=.8, color=COLORS[model_file.stem], linestyle=LINESTYLES[model_file.stem])
        _print("Plotted", model_file.stem.upper())

    ax.legend(loc="lower right", frameon=False, prop={"size": 12})
    fig.savefig(f"data/results/image/roc_no_update.pdf")