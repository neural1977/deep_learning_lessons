from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.nasnet import NASNetLarge
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import backend as K
import numpy as np
import pdb

depth = 3

neural_model = 'Xception'
#neural_model = 'VGG16'
#neural_model = 'VGG19'
#neural_model = 'ResNet50'
#neural_model = 'MobileNet'
#neural_model = 'InceptionResNetV2'
#neural_model = 'NASNetLarge'

'''
# NASNetLarge
height = 331
width = 331
inputShape = (depth, height, width)
'''

if neural_model == 'ResNet50' or neural_model == 'VGG16' or neural_model == 'MobileNet' or neural_model == 'VGG19': 
    height = 224
    width = 224
    inputShape = (depth, height, width)
elif neural_model == 'InceptionResNetV2' or neural_model == 'Xception': 
    height = 299
    width = 299
    inputShape = (height, width, depth)

pre_trained = True

if pre_trained == True:
    weights = 'imagenet'
else:    
    weights = None


class AdvancedCVModel:
    
    @staticmethod
    def build(neural_model, inputShape, classes, summary):
    
        if neural_model == 'Xception':
            model = Xception(weights=weights, input_shape = inputShape)
        elif neural_model == 'VGG16': 
            model = VGG16(weights = None)
        elif neural_model == 'VGG19':
            model = VGG19(weights = None)
        elif neural_model == 'ResNet50':
            model = ResNet50(weights=weights)
        elif neural_model == 'MobileNet':
            model = MobileNet(weights=weights)
        elif neural_model == 'InceptionResNetV2':
            model = InceptionResNetV2(weights=weights)
        elif neural_model == 'NASNetLarge':
            model = NASNetLarge(weights=weights)

        #if summary==True:
        model.summary()
    
        # return the constructed network architecture
        return model
        
img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(width, height))
x = image.img_to_array(img)
if neural_model == 'Xception':
    x = np.moveaxis(x, 0, 2) # For Xception channels last
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

deepnetwork = AdvancedCVModel.build(neural_model, inputShape, 10, True)

preds = deepnetwork.predict(x)

# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=3)[0])     
