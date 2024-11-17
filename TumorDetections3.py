import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Definizione dei percorsi per le cartelle contenenti le immagini
base_path = "/content/drive/MyDrive/Colab Notebooks"
no_tumor = os.path.join(base_path, "notumor")
glioma = os.path.join(base_path, "glioma")
meningioma = os.path.join(base_path, "meningioma")
pituitary = os.path.join(base_path, "pituitary")

# Funzione per caricare e ridimensionare le immagini
def load_image(folder):
    images = []
    for file_name in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, file_name))
        if img is not None:
            img = cv2.resize(img, (128, 128))
            images.append(img)
    return images

# Funzione per caricare immagini e creare etichette
def load_images_and_labels(categories):
    images = []
    labels = []
    for label, folder in enumerate(categories.values()):
        for file_name in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, file_name))
            if img is not None:
                img = cv2.resize(img, (128, 128))
                images.append(img)
                labels.append(label)
    return np.array(images), np.array(labels)

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

# Suddivisione dei dati in addestramento e validazione
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=0)

# Creazione del modello sequenziale
num_classes = 4
model = keras.Sequential([
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.1),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(0.001),
              metrics=['accuracy'])

print(model.summary())

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(x_train)

# Aggiunta di EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Addestramento
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_val, y_val),
    epochs=100,
    callbacks=[early_stopping]
)

# Salvataggio del modello
model.save('/content/drive/MyDrive/ProjectCNNsBrainTumor/tumor_classification_model2.keras')

# Visualizzazione dei grafici
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Loss Training')
plt.plot(history.history['val_loss'], label='Loss Validation')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(history.history['accuracy'], label='Accuracy Training')
plt.plot(history.history['val_accuracy'], label='Accuracy Validation')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()

# Caricamento dei dati di test
test_categories = {
    "No Tumor": os.path.join(base_path, "notumorTest"),
    "Glioma": os.path.join(base_path, "gliomaTest"),
    "Meningioma": os.path.join(base_path, "meningiomaTest"),
    "Pituitary": os.path.join(base_path, "pituitaryTest"),
}
x_test, y_test = load_images_and_labels(test_categories)
x_test = x_test.astype("float32") / 255.0
y_test = to_categorical(y_test, num_classes=4)

# Valutazione del modello
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Predizione e report di classificazione
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
labels = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
print(classification_report(np.argmax(y_test, axis=1), y_pred_classes, target_names=labels))
