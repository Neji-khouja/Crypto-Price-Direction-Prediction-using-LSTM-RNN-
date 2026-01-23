import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.optimizers import Adam

from config import *

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs("results", exist_ok=True)

def build_model(input_shape, units, dropout, lr):
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout),
        LSTM(units),
        Dropout(dropout),
        Dense(2, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=Adam(lr),
        metrics=["accuracy"]
    )
    return model

def train():
    with open(RESULTS_PATH, "a", newline="") as f:
        writer = csv.writer(f)

        for seq, fut, lr, batch, units, dropout in EXPERIMENTS:
            name = experiment_name(seq, fut, lr, batch, units, dropout)
            print(f"\n Training {name}")

            X_train = np.load(f"{PROCESSED_DATA_PATH}/X_train_SEQ{seq}_FUT{fut}.npy")
            y_train = np.load(f"{PROCESSED_DATA_PATH}/y_train_SEQ{seq}_FUT{fut}.npy")
            X_val = np.load(f"{PROCESSED_DATA_PATH}/X_val_SEQ{seq}_FUT{fut}.npy")
            y_val = np.load(f"{PROCESSED_DATA_PATH}/y_val_SEQ{seq}_FUT{fut}.npy")

            model = build_model(
                input_shape=X_train.shape[1:],
                units=units,
                dropout=dropout,
                lr=lr
            )

            checkpoint = ModelCheckpoint(
                filepath=f"{MODEL_SAVE_PATH}/{name}",
                monitor="val_accuracy",
                save_best_only=True,
                mode="max"
            )

            tensorboard = TensorBoard(
                log_dir=f"{LOGS_PATH}/{name}"
            )

            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=EPOCHS,
                batch_size=batch,
                callbacks=[checkpoint, tensorboard],
                verbose=2
            )

            best_val_acc = max(history.history["val_accuracy"])
            writer.writerow([name, seq, fut, lr, batch, units, dropout, best_val_acc])
            f.flush()

if __name__ == "__main__":
    train()
