import numpy as np
from PIL import Image
from tensorflow import keras
from keras.layers import *
import matplotlib.pyplot as plt
import os

def prep_image(image, img_size):
    image = image.resize((img_size, img_size))
    image_sequence = image.getdata()
    image_array = np.array(image_sequence)
    image_sequence = image_array
    image_sequence = np.resize(image_sequence, ((img_size, img_size, 3)))
    return (image_sequence/255)


def image_pair_split(filename):
    image_mask = Image.open(filename)
    
    image, mask = image_mask.crop([0, 0, 256, 256]), image_mask.crop([256, 0, 512, 256])

    return image, mask


def load_training_data(data_size:int):
    training_inputs=[]
    training_outputs=[]

    # CHANGE THIS
    training_dir = os.getcwd().replace("\\", "/") + '/cityscapes_data/train/'

    for i in range(1, data_size+1):
        image, mask = image_pair_split(training_dir + str(i) + '.jpg')

        training_inputs.append(prep_image(image, 128))
        training_outputs.append(prep_image(mask, 128))

    print ('Training Data Loaded')

    return training_inputs, training_outputs

def load_testing_data(data_size:int):
    testing_inputs=[]
    testing_outputs=[]

    # CHANGE THIS
    testing_dir = os.getcwd().replace("\\", "/") + '/cityscapes_data/val/'

    for i in range(1, data_size+1):
        image, mask = image_pair_split(testing_dir + str(i) + '.jpg')

        testing_inputs.append(prep_image(image, 128))
        testing_outputs.append(prep_image(mask, 128))

    print ('Testing Data Loaded')

    return testing_inputs, testing_outputs


model = keras.models.Sequential([
    Conv2D(filters=64, kernel_size=(3,3), activation='swish', kernel_initializer='he_normal', padding='same'),    
    Dropout(0.2), 
    Conv2D(filters=64, kernel_size=(3,3), activation='swish', kernel_initializer='he_normal', padding='same'),          
    MaxPool2D(),        

    Conv2D(filters=128, kernel_size=(3,3), activation='swish', kernel_initializer='he_normal', padding='same'),    
    Dropout(0.2), 
    Conv2D(filters=128, kernel_size=(3,3), activation='swish', kernel_initializer='he_normal', padding='same'),          
    MaxPool2D(),    

    Conv2D(filters=256, kernel_size=(3,3), activation='swish', kernel_initializer='he_normal', padding='same'),    
    Dropout(0.2), 
    Conv2D(filters=256, kernel_size=(3,3), activation='swish', kernel_initializer='he_normal', padding='same'),          
    MaxPool2D(),        

    Conv2D(filters=512, kernel_size=(3,3), activation='swish', kernel_initializer='he_normal', padding='same'),    
    Dropout(0.2), 
    Conv2D(filters=512, kernel_size=(3,3), activation='swish', kernel_initializer='he_normal', padding='same'),           

    Conv2DTranspose(filters=256, kernel_size=(2,2), activation='swish', strides=(2,2), padding='same'),  
    Conv2D(filters=256, kernel_size=(3,3), activation='swish', kernel_initializer='he_normal', padding='same'),
    Dropout(0.2),
    Conv2D(filters=256, kernel_size=(3,3), activation='swish', kernel_initializer='he_normal', padding='same'),

    Conv2DTranspose(filters=128, kernel_size=(2,2), activation='swish', strides=(2,2), padding='same'),  
    Conv2D(filters=128, kernel_size=(3,3), activation='swish', kernel_initializer='he_normal', padding='same'),
    Dropout(0.2),
    Conv2D(filters=128, kernel_size=(3,3), activation='swish', kernel_initializer='he_normal', padding='same'),

    Conv2DTranspose(filters=64, kernel_size=(2,2), activation='swish', strides=(2,2), padding='same'),  
    Conv2D(filters=64, kernel_size=(3,3), activation='swish', kernel_initializer='he_normal', padding='same'),
    Dropout(0.2),
    Conv2D(filters=64, kernel_size=(3,3), activation='swish', kernel_initializer='he_normal', padding='same'),

    Conv2D(3, (1,1), activation='sigmoid')
])

if __name__ == '__main__':

    training_inputs, training_outputs = load_training_data(2750)

    testing_inputs, testing_outputs = load_testing_data(500)

    train_x = np.array(training_inputs)
    test_x = np.array(testing_inputs)
    train_y = np.array(training_outputs)
    test_y = np.array(testing_outputs)

    model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath='./current models/current model',
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    history = model.fit(train_x, train_y, validation_data = (test_x, test_y), batch_size=16, epochs=75, callbacks=[model_checkpoint_callback])

    print(model.summary())

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch/Iteration')
    plt.legend(['Training', 'Testing'], loc='upper left')
    plt.show()
    
