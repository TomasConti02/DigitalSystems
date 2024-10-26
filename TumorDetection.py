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

no_img=load_image(no_tumor_path)
yes_img=load_image(yes_tumor_path)
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
"""
x = x.astype("float32")
y = to_categorical(y, num_classes=2)
x /= 255.0
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
num_classes = 2
model = keras.Sequential([
    layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same', input_shape=[128, 128, 3]),
    layers.MaxPool2D(),
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
                    batch_size=32,
                   callbacks=[back])
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)
print("Classification Report:\n", classification_report(y_true, y_pred_classes))
fig, ax = plt.subplots(figsize=(12,5))

ax.plot(history.history['loss'],label='train loss')

ax.plot(history.history['accuracy'],label='train accuracy')

ax.legend()

plt.show()
