from train import prep_image, image_pair_split
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from tensorflow import keras
from keras.layers import *
import os

model = keras.models.Sequential()
model.add(Conv2D(filters=128, kernel_size=3, activation='swish', strides=2, padding='same'))
model.add(Conv2D(filters=64, kernel_size=3, activation='swish', strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(128, activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='swish'))
model.add(Dropout(0.2))
model.add(Dense(32*32*32, activation='swish'))
model.add(Reshape((32,32,32)))
model.add(Conv2DTranspose(filters=128, kernel_size=3, activation='swish', strides=2, padding='same'))
model.add(Conv2DTranspose(filters=64, kernel_size=3, activation='swish', strides=2, padding='same'))
model.add(Conv2DTranspose(filters=3, kernel_size=3, activation='sigmoid', padding='same'))

    
model.load_weights('./current models/current model')

# CHANGE THIS
input_image = image_pair_split(os.getcwd().replace("\\", "/") + '/cityscapes_data/train/6.jpg')[0]
input_image_arr = np.resize(prep_image(input_image, 128), (1, 128, 128, 3))

predicted_output_image = model.predict(input_image_arr)[0]

fig = plt.figure(figsize=(10,7))

#INPUT IMAGE
fig.add_subplot(1, 2, 1)
plt.imshow(input_image)
plt.axis('off')
plt.title('Input Image')

#OUTPUT IMAGE
fig.add_subplot(1, 2, 2)
#OUTPUT IMAGE LEGEND
road = mpatches.Patch(color='mediumorchid', label='road')
sidewalk = mpatches.Patch(color='magenta', label='sidewalk')
tree = mpatches.Patch(color='yellowgreen', label='tree')
car = mpatches.Patch(color='blue', label='car')
person = mpatches.Patch(color='red', label='person')
building = mpatches.Patch(color='dimgrey', label='building')
sky = mpatches.Patch(color='cornflowerblue', label='sky')
plt.legend(handles=[road, sidewalk, tree, car, person, building, sky], prop={'size': 6})

plt.imshow(predicted_output_image)
plt.axis('off')
plt.title('Predicted Image Mask')

plt.show()

    

