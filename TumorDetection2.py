import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import random

# Definizione dei percorsi per le cartelle contenenti le immagini
no_tumor = "/content/drive/MyDrive/Colab Notebooks/notumor"
glioma = "/content/drive/MyDrive/Colab Notebooks/glioma"
meningioma = "/content/drive/MyDrive/Colab Notebooks/meningioma"
pituitary = "/content/drive/MyDrive/Colab Notebooks/pituitary"

# Funzione per caricare e ridimensionare le immagini
def load_image(folder):
    images = []
    for file_name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file_name))
        if img is not None:
            img = cv2.resize(img, (128, 128))
            images.append(img)
    return images

# Caricamento delle immagini e creazione delle etichette
no_tumor_imgs = load_image(no_tumor)
glioma_imgs = load_image(glioma)
meningioma_imgs = load_image(meningioma)
pituitary_imgs = load_image(pituitary)
print("\nDimensione no_tumor_imgs:", len(no_tumor_imgs)) 
print("\nDimensione glioma_imgs:", len(glioma_imgs)) 
print("\nDimensione meningioma_imgs:", len(meningioma_imgs)) 
print("\nDimensione pituitary_imgs:", len(pituitary_imgs)) 
height, width, channels = no_tumor_imgs[0].shape
print(f"Dimensioni dell'immagine: {height}x{width}")
print(f"Numero di canali: {channels}")

no_labels = [0] * len(no_tumor_imgs)
glioma_labels = [1] * len(glioma_imgs)
meningioma_labels = [2] * len(meningioma_imgs)
pituitary_labels = [3] * len(pituitary_imgs)

# Creazione degli array x e y per le immagini e le etichette
x = np.array(no_tumor_imgs + glioma_imgs + meningioma_imgs + pituitary_imgs)
y = np.array(no_labels + glioma_labels + meningioma_labels + pituitary_labels)

# Normalizzazione delle immagini e conversione delle etichette in categorico
x = x.astype("float32") / 255.0
y = to_categorical(y, num_classes=4)

# Definizione del numero di classi
num_classes = 4
# Suddivisione dei dati in addestramento e validazione
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)
# Creazione del modello sequenziale
model = keras.Sequential([
    # Primo blocco
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    # Secondo blocco
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    # Terzo blocco
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    # Quarto blocco
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    # Quinto blocco
    layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    # Flatten e strati finali densi
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])
print(model.summary())
# Caricamento delle immagini
no_tumor_imgs = load_image(no_tumor)
glioma_imgs = load_image(glioma)
meningioma_imgs = load_image(meningioma)
pituitary_imgs = load_image(pituitary)

# Selezione casuale di 2 immagini distinti per ciascun gruppo (senza ripetizioni)
random_no_tumor_indices = random.sample(range(len(no_tumor_imgs)), 2)
random_glioma_indices = random.sample(range(len(glioma_imgs)), 2)
random_meningioma_indices = random.sample(range(len(meningioma_imgs)), 2)
random_pituitary_indices = random.sample(range(len(pituitary_imgs)), 2)

# Creazione del grafico
plt.figure(figsize=(12, 8))

# Plot delle immagini casuali con indice e label
all_random_indices = random_no_tumor_indices + random_glioma_indices + random_meningioma_indices + random_pituitary_indices
all_random_images = (
    [no_tumor_imgs[i] for i in random_no_tumor_indices] +
    [glioma_imgs[i] for i in random_glioma_indices] +
    [meningioma_imgs[i] for i in random_meningioma_indices] +
    [pituitary_imgs[i] for i in random_pituitary_indices]
)

labels = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']
for i, (img, idx) in enumerate(zip(all_random_images, all_random_indices)):
    plt.subplot(2, 4, i + 1)  # 2 righe e 4 colonne
    plt.imshow(img)
    plt.title(f"Index: {all_random_indices[i]}, Label: {labels[i // 2]}")  # Mostra l'indice e la label
    plt.axis('off')

plt.tight_layout()
plt.show()
no_tumorTest = "/content/drive/MyDrive/Colab Notebooks/notumorTest"
gliomaTest = "/content/drive/MyDrive/Colab Notebooks/gliomaTest"
meningiomaTest = "/content/drive/MyDrive/Colab Notebooks/meningiomaTest"
pituitaryTest = "/content/drive/MyDrive/Colab Notebooks/pituitaryTest"
# Caricamento delle immagini di test
no_tumor_imgsTest = load_image(no_tumorTest)
glioma_imgsTest = load_image(gliomaTest)
meningioma_imgsTest = load_image(meningiomaTest)
pituitary_imgsTest = load_image(pituitaryTest)

# Stampa delle dimensioni dei gruppi di immagini di test
print("\nDimensione no_tumor_imgsTest:", len(no_tumor_imgsTest)) 
print("\nDimensione glioma_imgsTest:", len(glioma_imgsTest)) 
print("\nDimensione meningioma_imgsTest:", len(meningioma_imgsTest)) 
print("\nDimensione pituitary_imgsTest:", len(pituitary_imgsTest)) 

# Creazione delle etichette per le immagini di test
no_labelsTest = [0] * len(no_tumor_imgsTest)
glioma_labelsTest = [1] * len(glioma_imgsTest)
meningioma_labelsTest = [2] * len(meningioma_imgsTest)
pituitary_labelsTest = [3] * len(pituitary_imgsTest)

# Creazione dell'array xTest per le immagini di test
xTest = np.array(no_tumor_imgsTest + glioma_imgsTest + meningioma_imgsTest + pituitary_imgsTest)

# Creazione dell'array yTest per le etichette di test
yTest = np.array(no_labelsTest + glioma_labelsTest + meningioma_labelsTest + pituitary_labelsTest)

# Normalizzazione delle immagini di test
xTest = xTest.astype("float32") / 255.0

# Conversione delle etichette in formato categorico (one-hot encoded)
yTest = to_categorical(yTest, num_classes=4)


#x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

# Compilazione del modello
model.compile(loss='categorical_crossentropy', 
              optimizer=keras.optimizers.Adam(0.001), 
              metrics=['accuracy'])

# Addestramento del modello
history = model.fit(
    x_train, y_train,
    epochs=100,  # Numero di epoche (puoi modificarlo a seconda delle necessità)
    batch_size=32,
    validation_data=(x_val, y_val)  # Usa x_val e y_val come dati di validazione
)
# Valutazione del modello sui dati di test
test_loss, test_accuracy = model.evaluate(xTest, yTest)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predizione sui dati di test
y_pred = model.predict(xTest)

# Report di classificazione
y_pred_classes = np.argmax(y_pred, axis=1)
print(classification_report(np.argmax(yTest, axis=1), y_pred_classes, target_names=labels))

# Visualizzazione dei grafici di addestramento
# Grafico della perdita durante l'allenamento
plt.plot(history.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'], label='Loss Validation')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Grafico dell'accuratezza durante l'allenamento
plt.plot(history.history['accuracy'], label='Accuracy Training')
plt.plot(history.history['val_accuracy'], label='Accuracy Validation')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
'''
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Applica l'augmentation durante il training
train_generator = datagen.flow(x_train, y_train, batch_size=32)
history = model.fit(
    train_generator,
    epochs=100,
    validation_data=(x_val, y_val),
    steps_per_epoch=len(x_train) // 32
)
Modifica Dropout e Batch Normalization
Aumenta il dropout a 0.3 o 0.4 nei blocchi convoluzionali per rendere il modello più robusto.
Sposta BatchNormalization prima dell'attivazione relu nei livelli convoluzionali per normalizzare i dati di input nei livelli.
Se hai abbastanza memoria, puoi aumentare la risoluzione delle immagini a 150x150 o 224x224 per fornire al modello più dettagli.
'''
