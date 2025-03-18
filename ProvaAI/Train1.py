import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 1. Carica il Dataset
X = np.load(r"AI\NPYs\X.npy")
Y = np.load(r"AI\NPYs\Y.npy")

# 2. Prepara i Dati
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 3. Normalizzazione dei dati
X_train = X_train.astype('float32') / 1.0
X_test = X_test.astype('float32') / 1.0

# 4. Definisci il Modello (CNN)
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(6, 9, 9), padding='same'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])

# Compila il modello
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 5. Addestra il Modello
history = model.fit(X_train, Y_train, epochs=30, batch_size=32, validation_split=0.1)

# 6. Valuta il Modello
loss, mae = model.evaluate(X_test, Y_test)
print(f"Loss sul set di test: {loss}")
print(f"MAE sul set di test: {mae}")

# 7. Salva il Modello (formato .keras)
model.save("tablut_model.keras")

# 8. Analizza i Risultati (come prima)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.show()

Y_pred = model.predict(X_test)
plt.scatter(Y_test, Y_pred)
plt.xlabel("Valori Reali")
plt.ylabel("Previsioni")
plt.title("Valori Reali vs Previsioni")
plt.show()
