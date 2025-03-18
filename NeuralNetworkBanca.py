#Capisco, ti propongo un esempio pratico che si focalizza su un problema di classificazione di dati tabulari usando una rete neurale. 
#Usiamo un dataset di esempio di clienti bancari, 
#dove l'obiettivo è predire se un cliente sottoscriverà un prodotto bancario, come un deposito a termine, in base a caratteristiche come l'età, il reddito, lo stato civile, ecc.
#pip install tensorflow pandas scikit-learn
import pandas as pd
import numpy as np

# Creiamo un dataset fittizio di clienti bancari
np.random.seed(42)
data = {
    'Age': np.random.randint(18, 70, 1000),  # Età tra 18 e 70 anni
    'Income': np.random.randint(15000, 120000, 1000),  # Reddito annuale tra 15,000 e 120,000
    'MaritalStatus': np.random.choice(['Single', 'Married'], 1000),  # Stato civile
    'Education': np.random.choice(['Highschool', 'Bachelor', 'Master'], 1000),  # Educazione
    'Subscribed': np.random.choice([0, 1], 1000)  # Target: 0 = non sottoscritto, 1 = sottoscritto
}

df = pd.DataFrame(data)
print(df.head())
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Convertiamo le variabili categoriche in numeriche
label_encoder = LabelEncoder()

df['MaritalStatus'] = label_encoder.fit_transform(df['MaritalStatus'])  # Single = 0, Married = 1
df['Education'] = label_encoder.fit_transform(df['Education'])  # Highschool = 0, Bachelor = 1, Master = 2

# Separiamo le feature (X) dal target (y)
X = df[['Age', 'Income', 'MaritalStatus', 'Education']]
y = df['Subscribed']

# Normalizziamo le feature numeriche
scaler = StandardScaler()
X[['Age', 'Income']] = scaler.fit_transform(X[['Age', 'Income']])

# Suddividiamo i dati in training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.head())
import tensorflow as tf
from tensorflow.keras import layers, models

# Creiamo il modello
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Primo layer nascosto
    layers.Dense(32, activation='relu'),  # Secondo layer nascosto
    layers.Dense(1, activation='sigmoid')  # Layer di output (1 = sottoscritto, 0 = non sottoscritto)
])

# Compiliamo il modello
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Per classificazione binaria
              metrics=['accuracy'])

# Visualizziamo la struttura del modello
model.summary()
# Allenamento del modello
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
# Valutiamo il modello sui dati di test
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")
# Fare previsioni sui dati di test
predictions = model.predict(X_test)
predicted_classes = (predictions > 0.5).astype(int)  # Se la probabilità è maggiore di 0.5, predici 1 (sottoscritto)

# Visualizziamo le prime 5 previsioni
print(f"Prime 5 previsioni: {predicted_classes[:5]}")
print(f"Valori reali delle prime 5 osservazioni: {y_test[:5]}")
