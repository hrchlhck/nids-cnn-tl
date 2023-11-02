#!/usr/bin/env python3

from torch import nn
from torchvision import transforms
from torchvision.models.alexnet import AlexNet_Weights
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Tuple
from datetime import datetime as dt
from PIL.Image import Image
import cv2

import random
from pandas import DataFrame, concat
import torch.optim as optim
import torch
import numpy as np

SEED = 1701
EPOCHS = 50
MODEL_REPO = "pytorch/vision:v0.10.0"
BATCH_SIZE = 512
LEARNING_RATE = 1e-3

def __freeze(model):
    for param in model.parameters():
        param.requires_grad = False
    return model

def ALEXNET(num_classes: int):
    alexnet = __freeze(torch.hub.load(MODEL_REPO, "alexnet", weights=AlexNet_Weights.DEFAULT))

    alexnet._modules["classifier"][-1] = nn.Linear(4096, num_classes, bias=True)

    nn.init.xavier_uniform_(alexnet._modules["classifier"][-1].weight)
    
    alexnet._modules["classifier"].append(nn.Softmax(dim=1))
    return alexnet

class MinMaxScaler:
    def __init__(self, feature_range: tuple = (0.0, 1.0)) -> None:
        self._feature_range = feature_range
        
    def __call__(self, array: torch.Tensor) -> torch.Tensor:
        # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
        _min, _max = self._feature_range
        X_std = (array - array.min()) / (array.max() - array.min())
        return X_std * (_max - _min) + _min

# Custom transform
class ToImage:
    def __call__(self, array: torch.Tensor, keep_normalization=True):
        """ The idea is to convert a 1d array to a 2d array by resizing (with padding) to the square root of the 1d shape

        Ex: 
            - shape: 2048  
            - sqrt(shape) = 45.25 -> round to ceil (46)
            - resize the feature vector to 46x46 
            - return the new feature vector as a RGB PIL Image for torchvision transforms
         """
        feat = array.shape[0]
        n = int(np.ceil(feat ** 0.5))

        array = array.cpu().numpy().copy()
        
        # Squared size with padding
        array.resize((n, n))
        if not keep_normalization:
            return (array * 255).astype(np.uint8)

        return torch.Tensor(array.astype(np.float32)).unsqueeze(0)
    
class Resize:
    def __init__(self, shape):
        self._shape = shape

    def __call__(self, X, rgb=True):
        device = 'cpu'
        
        if isinstance(X, Image):
            X = np.array(X)
        
        if isinstance(X, torch.Tensor):
            device = X.device.type
            X = X.squeeze(0).cpu().numpy()

        ret = cv2.resize(X, dsize=self._shape, interpolation=cv2.INTER_CUBIC)
        
        if rgb:
            ret = cv2.cvtColor(ret, cv2.COLOR_GRAY2BGR)

        ret = torch.Tensor(ret)
        if rgb:
            ret = ret.view(3, *self._shape)
        
        if device == 'cuda':
            ret = ret.to('cuda')
            
        return ret

class CustomDataset(Dataset):
    def __init__(self, subset: Tuple[torch.Tensor, torch.Tensor], transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[0][index, :], self.subset[1][index]
    
        if self.transform:
            x = self.transform(x)

        return x, y
        
    def __len__(self):
        return self.subset[0].size(0)

# https://stackoverflow.com/a/73704579
class EarlyStopper:
    def __init__(self, patience: int, min_delta: float = 0.0):
        self._patience = patience
        self._min_delta = min_delta
        self._counter = 0
        self._min_validation_loss = float('inf')

    def early_stop(self, epoch, val_loss):
        if val_loss < self._min_validation_loss:
            self._min_validation_loss = val_loss
            self._counter = 0
        elif val_loss > (self._min_validation_loss + self._min_delta):
            self._counter += 1
            if self._counter >= self._patience:
                print(f"[{now()}] Early stopped at epoch {epoch}")
                return True
        return False

    def reset(self):
        self._min_validation_loss = float('inf')
        self._counter = 0
    

def validate(device: str, epoch: int, optimizer, loss_fn, model, dataset: DataLoader):
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
            y_pred.extend(torch.argmax(output, 1).cpu().numpy())
            y_true.extend(y.data.cpu().numpy())
            correct += torch.sum(torch.argmax(output, 1).eq(y)).item()
            qt += len(x)
            desc = f"[{now()}] Epoch {str(epoch).zfill(3)} Val. Acc: {correct/qt:.4f} Val. Loss: {running_loss / len(dataset):.8f}"
            it_eval.set_description(desc)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics["tp"] = tp
    metrics["fp"] = fp
    metrics["tn"] = tn
    metrics["fn"] = fn
    return running_loss / len(dataset), correct/qt, metrics

def train(device: str, epoch: int, optimizer, loss_fn, model, dataset: DataLoader):
    model.train()
    running_loss = 0.
    qt = 1
    correct = 0
    it = tqdm(enumerate(dataset), total=len(dataset))

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
        correct += torch.sum(torch.argmax(outputs, 1).eq(y)).item()
        qt += len(x)
    
        # Gather data and report
        running_loss += loss.item()

        desc = f"[{now()}] Epoch {str(epoch).zfill(3)} Acc: {correct/qt:.4f} Loss: {running_loss / len(dataset):.8f}"
        it.set_description(desc)
    return running_loss / len(dataset), correct/qt


def now():
    return dt.now().strftime("%d-%m-%Y %H-%M-%S")

def update():
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

    df_metrics = DataFrame()
    for month in range(1, 13):
        month_str = str(month).zfill(2)
        model = CustomModel(2, ALEXNET())
        model.to(device)
        
        # dataset = ImageFolder(f'/data/img_nids/image/{month_str}', transform=preprocess)
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

        losses = DataFrame()

        raise Exception("FALTA IMPLEMENTAR ATUALIZAÇÃO COM 1 MES DE ATRASO")

        for epoch in range(1, EPOCHS + 1):
            train_loss, train_acc = train(device, epoch, optimizer, loss_fn, model, data_train)
            val_loss, val_acc, _ = validate(device, epoch, optimizer, loss_fn, model, data_val)

            losses = pd.concat(
                [losses, pd.DataFrame(
                    [
                        {
                            'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 
                            'val_loss': val_loss, 'val_acc': val_acc, 'month': month_str
                        }
                    ])
                ]
            )

            losses.to_csv("data/results/csv/update_transfer_learning_train.csv", index=False)
        
        _, _, metrics = validate(device, 0, optimizer, loss_fn, model, data_test)
        print(metrics)
        df_metrics = pd.concat([df_metrics, pd.DataFrame([{**metrics, 'month': month_str}])])
        df_metrics.to_csv("update_alexnet.csv", index=False)

def no_update():
    # Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = 'cpu'
    if torch.cuda.is_available():
        print("Using CUDA")
        device = 'cuda'

    df_metrics = DataFrame()
    es = EarlyStopper(5, 1e-2)
    model = ALEXNET(num_classes=2)
    model = model.to(device)

    transform = transforms.Compose([
        ToImage(),
        Resize((224, 224)),
        MinMaxScaler(feature_range=(0.0, 1.0)),
    ])

    months = [str(i).zfill(2) for i in range(1, 13)]
    for month in range(1, 13):
        month_str = str(month).zfill(2)
        
        data = np.loadtxt(f'/data/img_nids/mlp/NIGEL_2014_{month_str}.csv', skiprows=1, dtype=np.float32, delimiter=',')
        X, y = data[:, :-1], data[:, -1]
        
        del data

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=SEED)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=.3, random_state=SEED)

        del X, y

        X_train = torch.Tensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_test  = torch.Tensor(X_test)
        y_test  = torch.LongTensor(y_test)
        X_val   = torch.Tensor(X_val)
        y_val   = torch.LongTensor(y_val)

        cd_train = CustomDataset(subset=(X_train, y_train), transform=transform)
        cd_test  = CustomDataset(subset=(X_test, y_test), transform=transform)
        cd_val   = CustomDataset(subset=(X_val, y_val), transform=transform)

        data_train = DataLoader(cd_train, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)
        data_test  = DataLoader(cd_test, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)
        data_val   = DataLoader(cd_val, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)

        loss_fn   = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        losses = DataFrame()
            
        if month_str == '01':
            for epoch in range(1, EPOCHS + 1):
                train_loss, train_acc = train(device, epoch, optimizer, loss_fn, model, data_train)
                val_loss, val_acc, _ = validate(device, epoch, optimizer, loss_fn, model, data_val)

                losses = concat([
                    losses, 
                    DataFrame([{
                            'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc, 
                            'val_loss': val_loss, 'val_acc': val_acc, 'month': month_str
                    }])
                ])

                if es.early_stop(epoch, val_loss):
                    break

                losses.to_csv("data/results/csv/no_update_transfer_learning_train.csv", index=False)
        else:
            data = np.loadtxt(f'/data/img_nids/mlp/NIGEL_2014_{month_str}.csv', skiprows=1, dtype=np.float32, delimiter=',')
            X, y = data[:, :-1], data[:, -1].astype(np.uint8)
            X_test = torch.Tensor(X)
            y_test = torch.LongTensor(y)
            del X, y
            cd_test  = CustomDataset(subset=(X_test, y_test), transform=transform)
            data_test  = DataLoader(cd_test, shuffle=True, batch_size=BATCH_SIZE, num_workers=8)

        _, _, metrics = validate(device, 0, optimizer, loss_fn, model, data_test)
        print(metrics, month_str)
        df_metrics = concat([df_metrics, DataFrame([{**metrics, 'month': month_str}])])
        df_metrics.to_csv("no_update_alexnet.csv", index=False)
        es.reset()

if __name__ == "__main__":
    no_update()
