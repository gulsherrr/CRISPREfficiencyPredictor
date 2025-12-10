# CRISPR Efficiency Predictor
A multi-stage deep learning project focused on predicting CRISPR guide efficiency using real experimental data. This repository documents each development step from preprocessing to model evaluation, building toward a cleaner, multi-feature architecture that integrates sequence, PAM, and biological context for improved accuracy.

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

## Day 11 — Data Preprocessing & Feature Engineering

On Day 11, I preprocessed the data. I loaded the results sheet from the V2_data.xlsx file and performed basic cleaning. I retained only the required columns:

```python
required_cols = [

    "Construct Barcode",

    "Extended Spacer(NNNN[20nt]NGGNNN)",

    "Gene Symbol",

    "Amino Acid Cut position",

    "Percent Peptide",

]
```

I then dropped rows with missing values in these columns:

```python
df = df.dropna(subset=required_cols).reset_index(drop=True)

print("Shape after dropping NaNs in required cols:", df.shape)
```

### 20-mer Sequence Encoding

Next, I one-hot encoded the 20-nt guide sequence using the following function:

```python
BASE2IDX = {"A": 0, "C": 1, "G": 2, "T": 3}

def one_hot_20mer(seq, length=20):
    """
    One-hot encode a 20-nt sequence into shape (4, length).
    Assumes characters are only A/C/G/T (uppercased).
    """
    seq = str(seq).upper().strip()
    if len(seq) != length:
        raise ValueError(f"Sequence length {len(seq)} != {length} for: {seq}")

    arr = np.zeros((4, length), dtype=np.float32)
    for i, base in enumerate(seq):
        idx = BASE2IDX.get(base, None)
        if idx is None:
            raise ValueError(f"Invalid base '{base}' in sequence: {seq}")
        arr[idx, i] = 1.0
    return arr
```

This function was applied to all rows.

### PAM Encoding

I extracted PAMs from the extended spacer and encoded them as one-hot vectors:

```python
unique_pams = sorted(df["PAM"].unique())
pam2id = {p: i for i, p in enumerate(unique_pams)}
print("pam2id mapping:", pam2id)

pam_ids = df["PAM"].map(pam2id).values  # shape: (N,)

X_pam = np.zeros((len(df), len(unique_pams)), dtype=np.float32)
X_pam[np.arange(len(df)), pam_ids] = 1.0

print("X_pam shape:", X_pam.shape)
```

### Gene Symbol Encoding

Gene symbols were encoded as integer IDs for use in an embedding layer:

```python
unique_genes = sorted(df["Gene Symbol"].astype(str).unique())
gene2id = {g: i for i, g in enumerate(unique_genes)}
print("Number of unique genes:", len(unique_genes))

gene_ids = df["Gene Symbol"].astype(str).map(gene2id).astype(np.int64).values  # shape: (N,)
print("gene_ids shape:", gene_ids.shape, "min:", gene_ids.min(), "max:", gene_ids.max())
```

### Cut Position & Target Processing

I kept amino-acid cut position as a numeric scalar:

```python
cut_raw = df["Amino Acid Cut position"].astype(np.float32).values  # shape: (N,)
print("Cut position raw stats: min", cut_raw.min(), "max", cut_raw.max())
```

I extracted and inspected the target variable:

```python
y_raw = df["Percent Peptide"].astype(np.float32).values  # shape: (N,)
print("Percent Peptide raw stats: min", y_raw.min(), "max", y_raw.max())
```

I then created a 60/20/20 train/validation/test split, fit scalers on the training set only, and saved all features, targets, and metadata as .npy and .json files.

## Day 12 — Multi-Input Model Design & Diagnostic Training

On Day 12, I loaded the preprocessed data and defined a multi-input dataset:

```python
class CRISPRDataset(Dataset):

    def __init__(self, X_seq, X_pam, X_gene, X_cut, y):

        self.X_seq = torch.tensor(X_seq, dtype=torch.float32)

        self.X_pam = torch.tensor(X_pam, dtype=torch.float32)

        self.X_gene = torch.tensor(X_gene, dtype=torch.long)

        self.X_cut = torch.tensor(X_cut, dtype=torch.float32)

        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        return {

            "seq": self.X_seq[idx],

            "pam": self.X_pam[idx],

            "gene": self.X_gene[idx],

            "cut": self.X_cut[idx],

            "y": self.y[idx]

        }
```

I designed a context-aware architecture with three branches.

### Sequence CNN Branch

```python
class SequenceCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):

        x = F.relu(self.conv1(x))

        x = F.relu(self.conv2(x))

        x = self.pool(x).squeeze(-1)

        return x  # (batch, 64)
```

### Gene Embedding Branch

```python
class GeneEmbedding(nn.Module):

    def __init__(self, num_genes, embed_dim=6):

        super().__init__()

        self.embed = nn.Embedding(num_genes, embed_dim)

    def forward(self, x):

        return self.embed(x)
```

### PAM Projection Branch

```python
class PAMProjector(nn.Module):

    def __init__(self, pam_dim, out_dim=4):

        super().__init__()

        self.fc = nn.Linear(pam_dim, out_dim)

    def forward(self, x):

        return F.relu(self.fc(x))
```

These branches were fused into a single context-aware but sequence-respecting model.

### Training Protocol

I defined the following training loop:

```python
def train_one_epoch(model, loader):

    model.train()

    losses = []

    for batch in loader:

        optimizer.zero_grad()

        y_pred = model(

            batch["seq"].to(device),

            batch["pam"].to(device),

            batch["gene"].to(device),

            batch["cut"].to(device)

        )

        loss = criterion(y_pred, batch["y"].to(device))

        loss.backward()

        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)
```

And the validation loop:

```python
def evaluate(model, loader):

    model.eval()

    ys, preds = [], []

    with torch.no_grad():

        for batch in loader:

            y_pred = model(

                batch["seq"].to(device),

                batch["pam"].to(device),

                batch["gene"].to(device),

                batch["cut"].to(device)

            )

            preds.append(y_pred.cpu().numpy())

            ys.append(batch["y"].cpu().numpy())

    return mean_squared_error(

        np.vstack(ys), np.vstack(preds)

    )
```
I trained the model briefly and evaluated behavior using:

1. training vs validation loss curves
2. prediction distribution
3. true vs predicted plots

This confirmed elimination of mean-collapse. This transition will transform the model from pure sequence learning into a biologically informed predictor.

## Day 13 — Test Evaluation, Interpretation & Ablation

On Day 13, I retrained the model cleanly using the same architecture and then evaluated it on the held-out test set.

Test predictions were generated using:

```python
model.eval()

y_true_scaled = []

y_pred_scaled = []

with torch.no_grad():

    for batch in test_loader:

        preds = model(

            batch["seq"].to(device),

            batch["pam"].to(device),

            batch["gene"].to(device),

            batch["cut"].to(device)

        )

        y_pred_scaled.append(preds.cpu().numpy())

        y_true_scaled.append(batch["y"].cpu().numpy())

y_true_scaled = np.vstack(y_true_scaled).flatten()

y_pred_scaled = np.vstack(y_pred_scaled).flatten()

Test-Set Performance (Scaled)

MSE: 0.0142

MAE: 0.0836

R²: 0.8332
```

This represents a ~5–6× error reduction over the sequence-only model (R² ≈ 0.006).

### Ablation Analysis

I created a reusable evaluation helper:

```python
def evaluate_on_arrays(model, X_seq, X_pam, X_gene, X_cut, y_scaled):

    """Evaluate model on given arrays (scaled y). Returns metrics + predictions."""

    loader = DataLoader(

        CRISPRDataset(X_seq, X_pam, X_gene, X_cut, y_scaled),

        batch_size=128,

        shuffle=False

    )

    model.eval()

    y_true_list, y_pred_list = [], []

    with torch.no_grad():

        for batch in loader:

            preds = model(

                batch["seq"].to(device),

                batch["pam"].to(device),

                batch["gene"].to(device),

                batch["cut"].to(device)

            )

            y_pred_list.append(preds.cpu().numpy())

            y_true_list.append(batch["y"].cpu().numpy())

    y_true = np.vstack(y_true_list).flatten()

    y_pred = np.vstack(y_pred_list).flatten()

    mse = mean_squared_error(y_true, y_pred)

    mae = mean_absolute_error(y_true, y_pred)

    r2  = r2_score(y_true, y_pred)

    return mse, mae, r2, y_true, y_pred
```

### Results Summary

Gene-shuffled: R² = −0.1258 → catastrophic failure

PAM-neutralized: R² = 0.8299 → minimal effect

Cut-neutralized: R² = −0.5077 → catastrophic failure

These results confirm that gene identity and cut position are foundational, while PAM acts as a secondary refinement signal.

# Final Outcome

This project demonstrates that sequence alone is insufficient for predicting CRISPR guide efficiency. Incorporating biological context transforms the model from a mean-collapsing regressor into a high-performing, explainable system. The final model was frozen and saved after test-set evaluation and ablation analysis.
