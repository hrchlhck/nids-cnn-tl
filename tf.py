#!/usr/bin/env python3

from torch import nn
from torchvision import transforms
from torchvision.models.alexnet import AlexNet_Weights
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from IPython import embed
from datetime import datetime as dt

import random
import pandas as pd
import torch.optim as optim
import torch
import numpy as np

SEED = 1701
EPOCHS = 50
MODEL_REPO = "pytorch/vision:v0.10.0"
BATCH_SIZE = 512

def __freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def ALEXNET():
    alexnet = __freeze(torch.hub.load(MODEL_REPO, "alexnet", weights=AlexNet_Weights.DEFAULT).eval())
    alexnet._modules["classifier"][-1] = nn.Linear(4096, 2, bias=True)

    alexnet._modules["classifier"][-1].weight.data.uniform_(0.0, 1.0)
    alexnet._modules["classifier"][-1].bias.data.fill_(0) 
    
    alexnet._modules["classifier"].append(nn.Softmax(dim=1))
    return alexnet.eval()

class CustomModel(nn.Module):
    def __init__(self, output_size: int, tf_model, *args, **kwargs):
        super(CustomModel, self).__init__(*args, **kwargs)
        self._output_size = output_size
        self._tf_model = tf_model
    
    def forward(self, x):
        # Add 1 dimention to input (3, 224, 224) -> (1, 3, 224, 224)
        # x = x.unsqueeze(0)
        return self._tf_model(x)
    

def validate(epoch: int, model: CustomModel, dataset: DataLoader):
    # Validation
    model.eval()
    it_eval = tqdm(enumerate(dataset), total=len(dataset))
    running_loss = 0.
    correct = 0
    qt = 1
    metrics = dict(tp=0, tn=0, fp=0, fn=0)
    y_pred = list()
    y_true = list()
    with torch.no_grad():
        for _, (x, y) in it_eval:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            running_loss += loss_fn(output, y).item()
            y_pred.extend((torch.max(output, 1)[1]).data.cpu().numpy())
            y_true.extend(y.data.cpu().numpy())
            correct += torch.sum(output.data.max(1)[1].eq(y)).item()
            qt += len(x)
            desc = f"[{now()}] Epoch {str(epoch).zfill(3)} Val. Acc: {correct/qt:.4f} Val. Loss: {running_loss / len(dataset):.8f}"
            it_eval.set_description(desc)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["tn"] = tn
    metrics["fn"] = fn
    return running_loss / len(dataset), correct/qt, metrics

def train(epoch: int, model: CustomModel, dataset: DataLoader):
    model.train()
    running_loss = 0.
    qt = 1
    correct = 0
    it = tqdm(enumerate(dataset), total=len(dataset))
    model.train()
    for _, (x, y) in it:
        x = x.to(device)
        y = y.to(device)
        
        # Make predictions for this batch
        outputs = model(x)

        # Zero your gradients for every batch!
        optimizer.zero_grad()
        loss = loss_fn(outputs, y)
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        correct += torch.sum(outputs.data.max(1)[1].eq(y)).item()
        qt += len(x)
    
        # Gather data and report
        running_loss += loss.item()

        desc = f"[{now()}] Epoch {str(epoch).zfill(3)} Acc: {correct/qt:.4f} Loss: {running_loss / len(dataset):.8f}"
        it.set_description(desc)
    return running_loss / len(dataset), correct/qt

def now():
    return dt.now().strftime("%d-%m-%Y %H-%M-%S")

if __name__ == "__main__":
    # Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = 'cpu'
    if torch.cuda.is_available():
        print("Using device CUDA")
        device = 'cuda'

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        # Normalize for Alexnet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    df_metrics = pd.DataFrame()
    for month in range(1, 13):
        month_str = str(month).zfill(2)
        model = CustomModel(2, ALEXNET())
        model.to(device)
        
        dataset = ImageFolder(f'/data/img_nids/image/{month_str}', transform=preprocess)
        targets = np.array(dataset.targets)
        train_idx, test_idx = train_test_split(np.arange(len(dataset.targets)), test_size=.3, random_state=SEED, stratify=targets)
        test_idx, val_idx = train_test_split(test_idx, test_size=.3, random_state=SEED, stratify=targets[test_idx])

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        data_train = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8, sampler=train_sampler)
        data_test = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8, sampler=test_sampler)
        data_val = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=8, sampler=val_sampler)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        losses = pd.DataFrame()

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train(epoch, model, data_train)
            val_loss, val_acc, _ = validate(epoch, model, data_val)

            losses = pd.concat(
                [losses, pd.DataFrame(
                    [
                        {
                            'train_loss': train_loss, 'train_acc': train_acc, 
                            'val_loss': val_loss, 'val_acc': val_acc, 'month': month_str
                        }
                    ])
                ]
            )

            losses.to_csv("data/results/csv/transfer_learning_train.csv", index=False)
        
        _, _, metrics = validate(0, model, data_test)
        print(metrics)
        df_metrics = pd.concat([df_metrics, pd.DataFrame([{**metrics, 'month': month_str}])])
        

        