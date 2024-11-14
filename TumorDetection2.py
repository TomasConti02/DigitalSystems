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

no_tumor="/content/drive/MyDrive/Colab Notebooks/notumor"
glioma="/content/drive/MyDrive/Colab Notebooks/glioma"
meningioma="/content/drive/MyDrive/Colab Notebooks/meningioma"
pituitary="/content/drive/MyDrive/Colab Notebooks/pituitary"

#no_tumor_path="/content/drive/MyDrive/Colab Notebooks/no"
#yes_tumor_path="/content/drive/MyDrive/Colab Notebooks/si"

def load_image(Folder): #tack the folder of TumorImage
    images=[]
    for file_name in os.listdir(Folder):  #List all photo
        img=cv2.imread(os.path.join(Folder,file_name))
        if img is not None:
            img=cv2.resize(img,(128,128))
            images.append(img)
    return images

no_tumor_Img = load_image(no_tumor)
glioma_Img = load_image(glioma)
meningioma_Img = load_image(meningioma)
pituitary_Img = load_image(pituitary)
print("\nDimensione no_tumor_Img:", len(no_tumor_Img))  
print("\nDimensione no_tumor_Img:", len(glioma_Img))  
print("\nDimensione no_tumor_Img:", len(meningioma_Img))  
print("\nDimensione no_tumor_Img:", len(pituitary_Img)) 
