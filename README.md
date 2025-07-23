

# 🫀 ECG Arrhythmia Classification Using Lead I (Chapman Dataset)

This project focuses on detecting and classifying various arrhythmia conditions from Lead I ECG waveforms using the **Chapman 12-lead ECG dataset**. The pipeline includes data extraction, filtering, label preprocessing, and training a multi-label classification model using PyTorch.

---

## 📁 Dataset Overview

* **Source:** [PhysioNet - Chapman ECG Arrhythmia Dataset](https://physionet.org/content/ecg-arrhythmia/)
* **Total Records:** 45,152 ECGs
* **Sampling Rate:** 500 Hz
* **Duration per ECG:** 10 seconds → 5000 samples
* **Lead Used:** `Lead I` only (out of 12 leads)
* **Label Format:** Multi-label using SNOMED-CT codes (mapped to acronym labels like `AF`, `AFIB`, `SB`, etc.)

---

## 🛠️ Pipeline

### 1. **Extract Labels**

* Parsed `.hea` files to extract `#Dx` line containing SNOMED-CT codes.
* Mapped SNOMED codes to human-readable acronyms using `ConditionNames_SNOMED-CT.csv`.
* Saved all results in a `REFERENCE.csv`.

### 2. **Load and Preprocess Signals**

* Read `.mat` signal files using `wfdb`.
* Extracted only `Lead I` signals → `(5000,)` shape.
* Filtered out ECGs not present in `REFERENCE.csv`.
* Stored ECGs in `X` and their corresponding labels in `Y`.

### 3. **Label Binarization**

* Used `MultiLabelBinarizer` to convert string labels to multi-hot vectors.

### 4. **Class Filtering**

* Focused on most common classes:

  ```
  ['AF', 'AFIB', 'APB', 'SA', 'SB', 'SR', 'ST', 'SVT']
  ```
* Removed rare/underrepresented classes for more balanced learning.

---

## 📊 Final Dataset Summary

* ✅ **Filtered Samples:** 44,111
* ✅ **Selected Labels:** 8
* ✅ **Final Shape:** `X: (44111, 5000, 1)`, `Y: (44111, 8)`

---

## 🤖 Model Training (PyTorch)

### Data Split:

```python
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42
)
```

### Network Architecture:

A simple 1D CNN:

```python
class ECGClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(32 * 2500, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))        # (N, 32, 2500)
        x = x.view(x.size(0), -1)                   # Flatten
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))
```

* **Loss:** `BCELoss` (binary cross-entropy for multi-label)
* **Optimizer:** `Adam`
* **Input Shape:** `(batch_size, 1, 5000)`

---

## 🧪 Evaluation

### Metrics:

* **Micro F1-score:** `0.70` → good overall performance
* **Macro F1-score:** `0.36` → reflects difficulty with rare classes

### Class-wise Performance (excerpt):

| Label                  | F1-score |
| ---------------------- | -------- |
| SB (Sinus Bradycardia) | 0.95     |
| SR (Sinus Rhythm)      | 0.69     |
| ST (Sinus Tachycardia) | 0.73     |
| AFIB, SVT, APB, SA     | \~0.00   |

---

## 🔍 Interpretation

* **Lead I** alone performs well for dominant rhythms like SB, SR, ST.
* **Rare arrhythmias** like `SVT`, `AFIB`, `APB` are underrepresented → poor performance.
* Needs **class imbalance handling** (e.g., weighted loss or sampling).

---

## ✅ To Do / Next Steps

* [ ] Add class-weighted `BCELoss` for better rare label performance.
* [ ] Train deeper CNN or hybrid model (e.g., CNN-LSTM).
* [ ] Data augmentation: jitter, shift, add noise.
* [ ] Tune thresholds for better multi-label precision/recall tradeoff.

---

## 🧠 Author Notes

* All experiments are done using PyTorch and ECGs are downselected to only **Lead I**.
* For full multi-lead input (12 leads), additional dimensional preprocessing is required.

---

Let me know if you want this saved to a `README.md` file!
