from train import prep_image, image_pair_split
import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.layers import *


model = keras.models.Sequential(
        [
            Conv2D(filters=64, kernel_size=3, activation='swish', strides=2, padding='same'),
            Conv2D(filters=64, kernel_size=3, activation='swish', strides=2, padding='same'),
            Flatten(),
            Dense(64, activation='swish'),
            Dropout(0.2),
            Dense(64, activation='swish'),
            Dropout(0.2),
            Dense(32*32*32, activation='swish'),
            Reshape((32,32,32)),
            Conv2DTranspose(filters=64, kernel_size=3, activation='swish', strides=2, padding='same'),
            Conv2DTranspose(filters=64, kernel_size=3, activation='swish', strides=2, padding='same'),
            Conv2DTranspose(filters=3, kernel_size=3, activation='sigmoid', padding='same'),
        ]
    )

    
model.load_weights('./current models/current model')

test_input = image_pair_split('C:/Users/Nicholas/Documents/Coding/RoadVision/cityscapes_data/train/2208.jpg')[0]
input_image = prep_image(test_input, 128)

input_image = np.resize(input_image, (1, 128, 128, 3))

predicted_image = model.predict(input_image)

plt.imshow(predicted_image[0])
plt.show()
    

