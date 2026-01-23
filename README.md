# Crypto Price Direction Prediction with LSTM (RNN)

This project implements a deep learning pipeline using **LSTM-based Recurrent Neural Networks** to predict short-term **price direction (up/down)** for cryptocurrencies.

The goal is not to overfit prices, but to explore **sequence modeling**, **time-series validation**, and **robust experimentation practices** in noisy financial data.

---

## Problem Formulation

- **Task**: Binary classification (price goes up or down)
- **Asset**: LTC-USD (with BTC, ETH, BCH as auxiliary features)
- **Model**: Multi-layer LSTM (RNN)
- **Output**: Probability of upward price movement

---

## Model Architecture

- 3 stacked LSTM layers
- Dropout + Batch Normalization
- Dense projection head
- Softmax output (2 classes)

Loss:
- `sparse_categorical_crossentropy`

Optimizer:
- Adam (with gradient clipping)

---

## Feature Engineering

- Returns
- Moving averages (SMA 20 / 50)
- Volatility (rolling std)
- RSI (14)
- Price-to-SMA ratios
- Volume-based features

All features are **scaled using StandardScaler**, fitted **only on training data** to prevent leakage.

---

## Time-Series Validation

- Walk-forward split (no shuffling across time)
- Separate scaler per split
- Balanced classes via undersampling
- No future leakage

---

## Hyperparameter Search

The pipeline automatically explores combinations of:

- Sequence length (`SEQ_LEN`)
- Prediction horizon (`FUTURE_PERIOD`)
- Learning rate
- Batch size
- Number of LSTM units

Each experiment:
- Saves TensorBoard logs
- Saves best model checkpoint
- Logs results to CSV

---

## Best Result (So Far)

**Best performing configuration:**

- Asset: **LTC-USD**
- Sequence Length: **60**
- Future Horizon: **3**
- Learning Rate: **0.001**
- Batch Size: **64**
- LSTM Units: **64**
- **Validation Accuracy: ~65.7%**
- Best Epoch: **15**

Model name:
LTC-USD_SEQ60_FUT3_LR0_001_BATCH64_UNITS64_1768957697

> Note: Achieving >60% validation accuracy in crypto direction prediction is non-trivial due to market noise.

---

## Logging & Reproducibility

- TensorBoard logs are generated per experiment
- Results saved to `results.csv`
- Best model checkpoint saved automatically

Logs and models are excluded from version control (see `.gitignore`).

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python cryptoRNN.py
