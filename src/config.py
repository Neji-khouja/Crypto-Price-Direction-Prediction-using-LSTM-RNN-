import itertools
import time

SYMBOL = "LTC-USD"
SEQ_LENGTHS = [30, 60]
FUTURE_PERIODS = [1, 3]

FEATURE_COLUMNS = [
    "close", "volume"
]

EPOCHS = 30

LEARNING_RATES = [1e-3, 5e-4]
BATCH_SIZES = [32, 64]
LSTM_UNITS = [64, 128]
DROPOUT_RATES = [0.2, 0.3]


RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
MODEL_SAVE_PATH = "models/saved"
LOGS_PATH = "logs/tensorboard"
RESULTS_PATH = "results/results.csv"


EXPERIMENTS = list(itertools.product(
    SEQ_LENGTHS,
    FUTURE_PERIODS,
    LEARNING_RATES,
    BATCH_SIZES,
    LSTM_UNITS,
    DROPOUT_RATES
))

def experiment_name(seq, fut, lr, batch, units, dropout):
    ts = int(time.time())
    return f"{SYMBOL}_SEQ{seq}_FUT{fut}_LR{lr}_BATCH{batch}_UNITS{units}_DROP{dropout}_{ts}"
