import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input

class_names = ['cat', 'dog']
path = './images'

def show_prediction(image, y_pred):
    plt.imshow(image.astype("int32"))
    plt.title(class_names[y_pred.numpy()[0][0]])
    plt.axis("off")

# loading model
new_model = keras.models.load_model('./transfer_learning_cats_dogs.h5')

# loading test image
size = (150, 150)
images = []
y_preds = []

plt.figure(figsize=(10, 10))
i = 0
for img in os.listdir('./images'):
    ax = plt.subplot(3, 3, i + 1)
    original = image.load_img(f"{path}/{img}", target_size=size)
    img = image.img_to_array(original)
    img = np.expand_dims(img, axis=0)

    aux = new_model.predict(img, batch_size=10)
    pred = tf.where(aux < 0.5, 0, 1)
    
    y_preds.append(pred)
    
    show_prediction(np.array(original), pred)
    i+=1
plt.show()
