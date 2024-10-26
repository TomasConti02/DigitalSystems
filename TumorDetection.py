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
no_tumor_path="/content/no"
yes_tumor_path="/content/si"

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
