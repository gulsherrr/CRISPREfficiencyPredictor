# CRISPREfficiencyPredictor

# Project Overview

This project chronicles a multi-day, step-by-step development of a machine-learning pipeline for predicting CRISPR-Cas9 guide RNA editing efficiency. It begins with synthetic data and progresses toward real experimental CRISPR screening data from the Azimuth repository. The workflow includes dataset construction, one-hot DNA encoding, PyTorch model building, CNN training, evaluation, visualization, debugging NaNs, and transitioning toward a richer, biologically informed model.

Up through Day 10, the model operates on raw 20-mer guide sequences alone and inevitably collapses toward predicting the mean—an expected limitation given the biological complexity of CRISPR editing.
Starting Day 11, the plan is to expand the model’s biological context using additional real features: percent peptide, construct barcode, extended spacer, explicit PAM encoding, gene symbol embeddings, and amino acid cut position.
The 20-mer CNN branch will stay intact, but the architecture will become multimodal.

Day 11 onwards details will be added later.

# Day-by-Day Progress

## Day 0 — Environment Setup

I installed and verified Anaconda, created a clean project folder structure, and set up a dedicated conda environment named crispr-ai.
I installed all required packages and verified them via:

```python
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import sklearn 

print("NumPy:", np.__version__) 
print("Pandas:", pd.__version__) 
print("Matplotlib:", plt.__version__) 
print("Scikit-learn:", sklearn.__version__)
```

## Day 1 — Synthetic CRISPR Dataset Generation

I created a synthetic CRISPR guide RNA dataset, assigned each guide a fake efficiency score, and saved it to the data/ folder.
I plotted the synthetic efficiency distribution using matplotlib.

## Day 2 — One-Hot Encoding DNA

I loaded the Day 1 dataset and defined a one-hot encoding scheme:

```python
BASE_TO_VEC = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1]
}

def one_hot_encode(seq):
    return np.array([BASE_TO_VEC[base] for base in seq])
```

I encoded the whole dataset:
```python
encoded_sequences = np.array([one_hot_encode(seq) for seq in df["guide_sequence"]])
encoded_sequences.shape
```

I visualized one encoded guide as a heatmap and saved:
```python
X_day2.npy

y_day2.npy
```

## Day 3 — Train/Validation/Test Split

I reloaded the encoded data and performed a 60/20/20 split:

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print("Training set:", X_train.shape, y_train.shape)
print("Validation set:", X_val.shape, y_val.shape)
print("Test set:", X_test.shape, y_test.shape)
```

I saved all split arrays and visualized the ratios.


## Day 4 — PyTorch Datasets and DataLoaders

After installing PyTorch, I converted all arrays into tensors:

```python
import torch

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_val_t = torch.tensor(X_val, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

print(X_train_t.shape, y_train_t.shape)
```

I created a custom PyTorch Dataset class, then DataLoaders:

```python
from torch.utils.data import DataLoader

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
```

Jupyter Notebook plotting broke, so I switched to saving figures for Colab.


## Day 5 — Building the CRISPR CNN Architecture

I rebuilt the pipeline and defined the first CNN:

```python
import torch.nn as nn
import torch.nn.functional as F

class CrisprCNN(nn.Module):
    def __init__(self):
        super(CrisprCNN, self).__init__()

        # Input: (batch, 20, 4) → transpose for Conv1D
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3)

        self.dropout = nn.Dropout(0.3)

        # Flatten → regress
        self.fc1 = nn.Linear(64 * 16, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

I ran the first forward pass:

```python
predictions[:5].detach().numpy().flatten(), y_batch[:5].numpy()
```

I saved the untrained model.


## Day 6 — Training on Synthetic Data

I configured:

```python
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

I trained for 20 epochs.
Because the labels were random synthetic noise, the model found no real signal. The loss stopped improving early, which is expected.

I saved logs as CSV for Colab plotting.


## Day 7 — Final Evaluation on Synthetic Data

I loaded the test tensors, rebuilt the CNN, loaded weights, made predictions, and computed metrics:

```python
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

mse, mae, r2
```

### Results:

```bash
MSE = 0.116

MAE = 0.291

R² = –0.731
```

### Interpretation:

Moderate errors (expected with random data)

30% average error magnitude

Negative R² means worse than predicting the mean. Exactly what happens when the labels contain no signal.


## Day 8 — Importing Real CRISPR Efficiency Data

I downloaded V2_data.xlsx from the Azimuth repository (GitHub).

I used only:

1. Construct Barcode
2. Percent Peptide

I normalized efficiency:

```python
df["efficiency"] = df["efficiency"] / 100.0
print(df["efficiency"].describe())
```

I saved a cleaned dataset (for reproducibility), reloaded it, and one-hot encoded real guides:

```python
mapping = {"A":0, "C":1, "G":2, "T":3}

def one_hot_encode(seq):
    arr = np.zeros((20,4))
    for i, base in enumerate(seq):
        arr[i, mapping[base]] = 1
    return arr

X_real = np.stack(df["sequence"].apply(one_hot_encode).values)
y_real = df["efficiency"].values

print(X_real.shape, y_real.shape)
```

I performed the 60/20/20 split and saved:

```bash
X_train_real.npy

y_train_real.npy

X_val_real.npy

y_val_real.npy

X_test_real.npy

y_test_real.npy
```

## Day 9 — Training on Real Data (Debugging NaNs)

I loaded all saved tensors and created DataLoaders.
When training, both train and val loss became NaN.

I debugged using:

```python
print("X_train contains NaN:", torch.isnan(X_train_t).any().item())
print("X_train contains Inf:", torch.isinf(X_train_t).any().item())

print("y_train contains NaN:", torch.isnan(y_train_t).any().item())
print("y_train contains Inf:", torch.isinf(y_train_t).any().item())
```

Output:

```bash
X_train contains NaN: False  
X_train contains Inf: False  
y_train contains NaN: True  
y_train contains Inf: False  
```

I filtered NaNs:

```python
valid_mask = ~torch.isnan(y_train_t)
X_train_t = X_train_t[valid_mask]
y_train_t = y_train_t[valid_mask]
print("New training set size:", X_train_t.shape)
```

I rebuilt validation DataLoader:

```python
val_dataset = TensorDataset(X_val_t, y_val_t)
val_loader = DataLoader(val_dataset, batch_size=64)
```

Training resumed successfully.

Training behavior:

```bash
Train loss: 0.0817

Val loss: 0.0825
```

This tight alignment shows:

1. No overfitting
2. Good normalization
3. Real biological signal extracted, though, limited
4. Plateau reflects true biological noise floor


# Day 10 — Testing on Real Data and Visualization

I loaded the final weights and generated predictions:

```python
with torch.no_grad():
    X_test_t = X_test_t.to(device)
    preds = model(X_test_t).cpu().numpy()

preds[:5], y_test[:5]
```

Metrics:

```python
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)
```

Results:

```bash
MSE = 0.08299

MAE = 0.25117

R² = 0.00579
```

### Interpretation:

Stable and consistent with validation

Average prediction error still ~25% of scale

R² ≈ 0 → model barely better than predicting the mean

True vs predicted plot shows a horizontal band → mean collapse

Residual plot shows wide, unbiased scatter → consistently mediocre everywhere

Biological reason:

The model only sees:

1. 20-mer guide sequence

But CRISPR efficiency depends on:

1. PAM structure
2. Gene-specific repair bias
3. Chromatin accessibility
4. Nearby sequence context
5. Cut position
6. Nucleosome occupancy


The model simply doesn’t have enough biological information to learn meaningful variance.

# Plans Starting Day 11

I will expand the model using the following additional real features:

1. Percent peptide
2. Construct Barcode
3. Extended Spacer
4. Gene Symbol
5. Amino Acid cut position


Rules for the new multimodal design:

1. The original 20-mer CNN branch stays unchanged.
2. PAM is encoded explicitly, not merged into the 20-mer.
3. Gene symbols are learned using an embedding layer.
4. Cut position enters as a continuous auxiliary feature.


This transition will transform the model from pure sequence learning into a biologically informed predictor.
