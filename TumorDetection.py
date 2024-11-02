#lib
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

from keras.models import Sequential
#funzioni di costruzione del modello
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras import layers

no_tumor_path="/content/drive/MyDrive/Colab Notebooks/no"
yes_tumor_path="/content/drive/MyDrive/Colab Notebooks/si"

def load_image(Folder): #tack the folder of TumorImage
    images=[]
    for file_name in os.listdir(Folder):  #List all photo
        img=cv2.imread(os.path.join(Folder,file_name))
        if img is not None:
            img=cv2.resize(img,(128,128))
            images.append(img)
    return images

no_img = load_image(no_tumor_path)
print("\nDimensione no_img:", len(no_img))  # Stampa la lunghezza della lista no_img

yes_img = load_image(yes_tumor_path)
print("\nDimensione yes_img:", len(yes_img))  # Stampa la lunghezza della lista yes_img

#make labels
no_labels=[0]*len(no_img)
yes_labels=[1]*len(yes_img)
x=np.array(no_img + yes_img)
y=np.array(no_labels + yes_labels )
#visualize random img
import random
figure = plt.figure()
plt.figure(figsize=(16,10))
num_of_images = 50
"""
for index in range(1, num_of_images + 1):
    Rand_image=random.randint(0,len(x)-1)
    if y[Rand_image]==0:
           class_names="no"
    else:
         class_names="yes"
    plt.subplot(5, 10, index).set_title(f'{class_names}')
    plt.axis('off')
    plt.imshow(x[Rand_image], cmap='gray_r')
#data processing
    #first layer, we have 32 filter for 32 activation-maps for output
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=[128, 128, 3]),
    layers.MaxPool2D(), #pool of img 128x128 -> 64x6
    #after cut the img size with pool we can increase the number of filter
    #now we go deeper and deeper in img feald
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.MaxPool2D(),
    layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),
    layers.Conv2D(filters=128, kernel_size=3, activation="relu", padding='same'),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax'),

"""
"""
model = models.Sequential([
    # Primo blocco
    layers.Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)),  # Usa la dimensione corretta per i tuoi dati
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

    # Strato Flatten e Fully Connected
    layers.Flatten(),
    layers.Dense(1024, activation='relu'),
    layers.Dropout(0.2),

    # Strato di output
    layers.Dense(num_classes, activation='softmax')
])
"""
x = x.astype("float32")
y = to_categorical(y, num_classes=2)
x /= 255.0
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
num_classes = 2
model = keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=[128, 128, 3]),
    layers.BatchNormalization(),
    layers.MaxPool2D(),
    layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPool2D(),
    layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPool2D(),
    layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPool2D(),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5)
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax'),
])
print(model.summary())
model.compile(loss = 'categorical_crossentropy', optimizer = keras.optimizers.Adam(0.001), metrics = ['accuracy'])
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.990 and logs.get('accuracy') < 1:
            self.model.stop_training = True
            print("\nReached 99% accuracy so cancelling training!")

back = myCallback()
history = model.fit(X_train,
                    y_train,
                    epochs=100,
                    batch_size=32)
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print("Classification Report:\n", classification_report(y_true, y_pred_classes))
fig, ax = plt.subplots(figsize=(12,5))

ax.plot(history.history['loss'],label='train loss')

ax.plot(history.history['accuracy'],label='train accuracy')

ax.legend()

plt.show()
print("\nSalvataggio del modello!")
model.save('/content/drive/MyDrive/ProjectCNNsBrainTumor/tumor_classification_model1.keras')
print("\nModello salvato con successo!")
def predict_image(image_path, esito):
    # Carica l'immagine
    img = cv2.imread(image_path)
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(img_rgb)
   # plt.axis('off')  # Nasconde gli assi
   # plt.title("Immagine da testare")
    #plt.show()
    img = cv2.resize(img, (128, 128))  # Ridimensiona l'immagine
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(img_rgb)
    #plt.axis('off')  # Nasconde gli assi
    #plt.title("Immagine da testare")
    #plt.show()
    img = np.array(img).astype("float32") / 255.0  # Normalizza
    img = np.expand_dims(img, axis=0)  # Aggiungi dimensione batch
    # Fai la previsione
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)
    # Stampa il risultato
    if (predicted_class == 0 and esito.strip().upper() == "NO") or (predicted_class == 1 and esito.strip().upper() == "SI"):
        print(f"Per l'immagine '{image_path.split('/')[-1]}': Indovinato, {esito} tumore.")
    else:
        print(f"Per l'immagine '{image_path.split('/')[-1]}': Tumore previsto, ma atteso '{esito}'.")



# Percorso dell'immagine da testare
test_image_path1 = '/content/Te-me_0109.jpg'
test_image_path2 = '/content/Te-no_0014.jpg'
test_image_path3 = '/content/Te-no_0017.jpg'
test_image_path4 = '/content/Te-no_0023.jpg'
test_image_path5 = '/content/Te-pi_0236.jpg'
# Esegui la previsione

predict_image(test_image_path1, "SI")
predicted_class = predict_image(test_image_path2, "NO")
predicted_class = predict_image(test_image_path3, "NO")
predicted_class = predict_image(test_image_path4, "NO")
predicted_class = predict_image(test_image_path5, "SI")
