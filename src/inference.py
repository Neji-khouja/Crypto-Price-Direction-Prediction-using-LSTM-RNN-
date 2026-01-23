import numpy as np
import tensorflow as tf
import sys

MODEL_PATH = sys.argv[1]
SEQUENCE_FILE = sys.argv[2]

model = tf.keras.models.load_model(MODEL_PATH)
X = np.load(SEQUENCE_FILE)

pred = model.predict(X[-1:].reshape(1, *X.shape[1:]))
direction = "UP " if pred[0][1] > pred[0][0] else "DOWN "

print("Prediction:", direction)
print("Confidence:", pred)
