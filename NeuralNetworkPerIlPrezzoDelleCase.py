#In questo esempio, costruiremo una rete neurale che predice il prezzo di una casa in base a variabili come la superficie, 
#il numero di camere da letto, e altre caratteristiche. 
#Utilizzeremo il dataset Boston Housing che è disponibile in Keras e contiene informazioni su varie case (superficie, numero di camere, età della casa, ecc.).
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Carica il dataset Boston Housing
(x, y), (x_test, y_test) = tf.keras.datasets.boston_housing.load_data()

# Suddividi il dataset in training e test
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

# Normalizzare i dati
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(x_train.shape[1],)),  # Layer nascosto
    layers.Dense(32, activation='relu'),  # Secondo layer nascosto
    layers.Dense(1)  # Layer di output (un singolo valore per la previsione)
])
model.compile(optimizer='adam',
              loss='mse',  # Per regressione
              metrics=['mae'])  # Mean Absolute Error per monitorare la performance
test_loss, test_mae = model.evaluate(x_test, y_test)
print(f"Test Loss (MSE): {test_loss}")
print(f"Test MAE: {test_mae}")
predictions = model.predict(x_test)
print(f"Prime 5 previsioni: {predictions[:5]}")
print(f"Valori reali delle prime 5 case: {y_test[:5]}")
