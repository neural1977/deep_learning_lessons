from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge
from keras.applications.resnet50 import ResNet50
from sklearn.model_selection import train_test_split
from skimage.transform import resize
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.optimizers import SGD
from keras.preprocessing.image import array_to_img, img_to_array
import matplotlib.pyplot as plt

import os
import pdb
import cv2
import numpy as np

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)

proxy = False
proxy_server = "http://proxy.istat.it:3128"
default_callbacks = []
epochs = 100
batch_size = 8
data_augmentation = False
full_model_trainable = True
depth = 3
show_dataset = False

data_type = 'float32'
#data_type = 'float16'

#neural_model = 'Xception'
#neural_model = 'VGG16'
#neural_model = 'VGG19'
neural_model = 'ResNet50'
#neural_model = 'MobileNet'
#neural_model = 'InceptionResNetV2'
#neural_model = 'NASNetLarge'

if neural_model == 'ResNet50' or neural_model == 'VGG16' or neural_model == 'MobileNet' or neural_model == 'VGG19': 
    height = 224
    width = 224
    inputShape = (height, width, depth)
elif neural_model == 'InceptionResNetV2' or neural_model == 'Xception': 
    height = 299
    width = 299
    inputShape = (height, width, depth)
elif neural_model == 'NASNetLarge':
    height = 331
    width = 331
    inputShape = (height, width, depth)

pre_trained = True

if pre_trained == True:
    weights = 'imagenet'
else:    
    weights = None

class AdvancedCVModel:
    
    @staticmethod
    def build(neural_model, inputShape, classes):
    
        if neural_model == 'Xception':
            model = Xception(weights = weights, include_top=False, input_shape = inputShape)
        elif neural_model == 'VGG16': 
            model = VGG16(weights = weights, include_top=False, input_shape = inputShape)
        elif neural_model == 'VGG19':
            model = VGG19(weights = weights, include_top=False, input_shape = inputShape)
        elif neural_model == 'ResNet50':
            model = ResNet50(weights = weights, include_top=False, input_shape = inputShape)
        elif neural_model == 'MobileNet':
            model = MobileNet(weights = weights, include_top=False, input_shape = inputShape)
        elif neural_model == 'InceptionResNetV2':
            model = InceptionResNetV2(weights = weights, include_top=False, input_shape = inputShape)
        elif neural_model == 'NASNetLarge':
            model = NASNetLarge(weights = weights, include_top=False, input_shape = inputShape)

        # return the constructed network architecture
        return model

# Set proxy
if proxy == True:
    os.environ["https_proxy"] = proxy_server

# img_arr is of shape (n, h, w, c)
def resize_image_arr(img_arr, height, width):
    x_resized_list = []
    for i in range(img_arr.shape[0]):
        img = img_arr[i]
        #resized_img = resize(img, (height, width))
        resized_img = cv2.resize(img, (height, width))
        x_resized_list.append(resized_img)
    return np.stack(x_resized_list)

# Set CPU or GPU type
gpu = True
gpu_id = "0"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  
if gpu == False: 
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
else: 
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

#download mnist data and split into train and test sets
(X, y), (X_test, y_test) = cifar10.load_data()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20)

# Resize image arrays
X_train = resize_image_arr(X_train, height, width)
X_val = resize_image_arr(X_val, height, width)
X_test = resize_image_arr(X_test, height, width)

if show_dataset == True:
    plt.imshow(array_to_img(X_train[0]))
    plt.savefig("first_resized_cifar10_mnist_train_image.jpg")
    print("First cifar10 mnist train image", y_train[0][0])
    plt.show(block = False)
    plt.pause(3)
    plt.close()

    # Show the first image from the test set
    plt.imshow(array_to_img(X_test[0]))
    plt.savefig("first_resized_cifar10_mnist_test_image.jpg")
    print("First fashion mnist test image", y_test[0][0])
    plt.show(block = False)
    plt.pause(3)
    plt.close()

# Normalize the data
X_train = X_train.astype(data_type)
X_val = X_val.astype(data_type)
X_test = X_test.astype(data_type)
X_train /= 255
X_val /= 255
X_test /= 255

# Convert labels to categorical one-hot encoding
y_train = to_categorical(y_train, num_classes=10)
y_val = to_categorical(y_val, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# create the base pre-trained model
base_model = AdvancedCVModel.build(neural_model, inputShape, 10)

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 10 classes
predictions = Dense(10, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
if full_model_trainable == False: 
    for layer in base_model.layers:
        layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics = ['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics = ['accuracy'])

checkPoint=ModelCheckpoint("advanced_cifar10.cnn", save_weights_only=True, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
default_callbacks = default_callbacks+[checkPoint]
pdb.set_trace()
# Train the model, iterating on the data in batches
if data_augmentation == False: 
	history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, validation_data = (X_val, y_val), callbacks = default_callbacks, verbose = 2)
else:
	datagen = ImageDataGenerator(zoom_range = 0.2, horizontal_flip = True, vertical_flip = True, rotation_range = 30)                                   

	history = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size), steps_per_epoch = X_train.shape[0]/batch_size, epochs = epochs, validation_data = (X_val, y_val), callbacks = default_callbacks, verbose = 2)

score = model.evaluate(X_test, y_test, batch_size=16)
print(score)
