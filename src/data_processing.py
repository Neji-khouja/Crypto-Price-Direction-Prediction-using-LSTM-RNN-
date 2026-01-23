import os
import pandas as pd
import numpy as np
from config import *
from utils import scale_features, create_sequences

os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

def process_data(csv_file):
    df = pd.read_csv(csv_file)
    df = df.sort_values("timestamp")

    df, _ = scale_features(df, FEATURE_COLUMNS)

    for seq in SEQ_LENGTHS:
        for fut in FUTURE_PERIODS:
            X, y = create_sequences(df, seq, fut, FEATURE_COLUMNS)

            split = int(0.8 * len(X))
            X_train, X_val = X[:split], X[split:]
            y_train, y_val = y[:split], y[split:]

            np.save(f"{PROCESSED_DATA_PATH}/X_train_SEQ{seq}_FUT{fut}.npy", X_train)
            np.save(f"{PROCESSED_DATA_PATH}/y_train_SEQ{seq}_FUT{fut}.npy", y_train)
            np.save(f"{PROCESSED_DATA_PATH}/X_val_SEQ{seq}_FUT{fut}.npy", X_val)
            np.save(f"{PROCESSED_DATA_PATH}/y_val_SEQ{seq}_FUT{fut}.npy", y_val)

            print(f"Saved processed data SEQ={seq} FUT={fut}")

if __name__ == "__main__":
    process_data(f"{RAW_DATA_PATH}/LTC-USD.csv")
