import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def scale_features(df, columns):
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    return df_scaled, scaler

def create_sequences(df, seq_len, future_period, feature_cols):
    X, y = [], []

    for i in range(len(df) - seq_len - future_period):
        X.append(df[feature_cols].iloc[i:i+seq_len].values)

        future_price = df["close"].iloc[i + seq_len + future_period]
        current_price = df["close"].iloc[i + seq_len]

        y.append(1 if future_price > current_price else 0)

    return np.array(X), np.array(y)
