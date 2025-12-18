import numpy as np
import torch

# --- Transform class ---
class CyclicShift:
    def __init__(self, shift_range=(0, 0)):
        """
        shift_range: (low, high) shift amount in indices.
                     Positive shifts right, negative shifts left.
        """
        self.shift_range = shift_range

    def __call__(self, x):
        """x: torch.Tensor shape (1, T) or (N, 1, T)"""
        amt = int(np.random.uniform(*self.shift_range))
        if amt != 0:
            x = torch.roll(x, shifts=amt, dims=-1)
        return x
        
class GlucoseTransform:
    def __init__(self, add_range=(0,0), alpha_range=(1.0,1.0), noise_range=(0,0)):
        """
        add_range: tuple (low, high) range of constant to add
        alpha_range: tuple (low, high) range for random scaling multiplier
        noise_range: tuple (low, high) for random uniform noise
        """
        self.add_range = add_range
        self.alpha_range = alpha_range
        self.noise_range = noise_range

    def __call__(self, x):
        """
        x: torch.Tensor of shape (1, T) or (N, 1, T)
        returns transformed tensor with same shape
        """
        # pick a random constant to add
        c = np.random.uniform(*self.add_range)
        # random scale factor
        a = np.random.uniform(*self.alpha_range)
        # noise sampled per element
        noise = np.random.uniform(self.noise_range[0], self.noise_range[1], size=x.shape)
        
        # Apply transforms
        x = x + c
        x = x * a
        x = x + torch.from_numpy(noise).type_as(x)
        return x
        
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x

import pandas as pd
import numpy as np 
import torch 
from torch.utils.data import Dataset, DataLoader 

class CGMDataset(Dataset):
    def __init__(self, df, seq_col='sequence', label_col='label',
                 offset=0, mode=0, transform=None):
        sequences = np.stack(df[seq_col].values).astype(np.float32)
        labels    = df[label_col].values.astype(np.float32)
        self.mode = mode
        self.mean = 140
        self.std  = 120
        self.offset = offset
        self.transform = transform

        sequences = (sequences - self.mean) / self.std

        self.X = torch.from_numpy(sequences).unsqueeze(1)
        self.y = torch.from_numpy(labels).unsqueeze(1)

    def __len__(self):
        return len(self.X)*7 if self.mode == 0 else len(self.X)

    def __getitem__(self, idx):
        if self.mode == 0:
            patient_id = idx // 7
            day = idx - 7*patient_id
            x = self.X[patient_id,:,day*288:day*288+288]
            y = self.y[patient_id]
        elif self.mode == 1:
            x = self.X[idx]
            y = self.y[idx]
        elif self.mode == 2:
            x = self.X[idx,:,self.offset*288:self.offset*288+288]
            y = self.y[idx]

        # apply transform if provided
        if self.transform is not None:
            x = self.transform(x)

        return x, y



import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    classification_report, roc_auc_score, roc_curve, f1_score,
    precision_recall_curve, average_precision_score
)

def print_results(y_test, y_prob):
    # Predicted labels (threshold = 0.5)
    y_pred = (y_prob > 0.5).astype(int)

    # ----- Classification Metrics -----
    acc = accuracy_score(y_test, y_pred)
    ba = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("Accuracy:",acc )
    print("Balanced Accuracy:",ba )
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("F1 score",f1)

    # ----- ROC AUC -----
    auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC Score: {auc:.4f}")

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # ----- AUPRC -----
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    auprc = average_precision_score(y_test, y_prob)
    print(f"AUPRC Score: {auprc:.4f}")

    results = {'AUC':auc, 'AUPRC':auprc, 'Accuracy':acc, 'BA': ba, 'F1':f1}
    return results
    '''
    # ----- Plot ROC Curve -----
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='darkorange', linewidth=2,
             label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    #plt.show()

    # ----- Plot Precision–Recall Curve -----
    plt.figure(figsize=(6,5))
    plt.plot(recall, precision, color='purple', linewidth=2,
             label=f'PR Curve (AUPRC = {auprc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision–Recall Curve')
    plt.legend()
    plt.grid(True)
    #plt.show()
    '''
