from train import prep_image, image_pair_split, model
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from keras.layers import *
import os

test_model = model
test_model.load_weights('./current models/current model')

input_image = image_pair_split(os.getcwd().replace("\\", "/") + '/cityscapes_data/train/1.jpg')[0]
input_image_arr = np.resize(prep_image(input_image, 128), (1, 128, 128, 3))

predicted_output_image = test_model.predict(input_image_arr)[0]

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
plt.legend(handles=[road, sidewalk, tree, car, person, building, sky], prop={'size': 10})

plt.imshow(predicted_output_image)
plt.axis('off')
plt.title('Predicted Image Mask')

plt.show()
    

