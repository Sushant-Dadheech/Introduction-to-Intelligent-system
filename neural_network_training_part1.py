"""
Neural Network Training and Hyper-parameter Tuning - Part 1
Multi-layer neural network for income prediction using age and experience.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/content/multi.csv")
print("Dataset shape:", df.shape)
print(df.head())

# Features and target
X = df[['age', 'experience']].values
Y = df['income'].values

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Scaling
scaler_X = StandardScaler()
scaler_Y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).flatten()
Y_test_scaled = scaler_Y.transform(Y_test.reshape(-1, 1)).flatten()

# Build model
model = Sequential([
    Input(shape=(2,)),
    Dense(12, activation='relu'),
    Dense(8, activation='relu'),
    Dense(4, activation='relu'),
    Dense(1)
])

model.summary()

# Compile model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Train model
history = model.fit(
    X_train, Y_train_scaled,
    epochs=500,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# Evaluate model
loss, mae = model.evaluate(X_test, Y_test_scaled, verbose=0)
print(f"\nTest MSE: {loss:.4f}")
print(f"Test MAE: {mae:.4f}")

# Predictions
predictions_scaled = model.predict(X_test)
predictions = scaler_Y.inverse_transform(predictions_scaled).flatten()

print("\n--- Sample Predictions ---")
print(f"{'Actual':>12} {'Predicted':>12} {'Error':>10}")
print("-" * 38)
for actual, pred in zip(Y_test, predictions):
    error = actual - pred
    print(f"${actual:>10,.0f} ${pred:>10,.2f} {error:>+10.2f}")

# Plot training history
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('MSE (scaled)')
plt.legend()
plt.tight_layout()
plt.show()
