from model import UNET, FCN32, FCN8
import matplotlib.image as mpimg
from tensorflow.keras.utils import multi_gpu_model
import numpy


def read_data(num_images=10):
    images = []
    for i in range(num_images):
        image = mpimg.imread('data/test_{}.jpg'.format(i+1))
        images.append(image)

    images = numpy.array(images, dtype=float)
    return images


model_name = 'FCN8'
models = {'FCN8': FCN8, 'FCN32': FCN32, 'UNET': UNET}

model = models[model_name]

model = model.build(height=256, width=256, num_classes=33)
model = multi_gpu_model(model, gpus=2)
model.load_weights('pretrain/{}.h5'.format(model_name))
predictimages = 8
images = read_data(num_images=predictimages)
##### 预测图片 ####
preds = model.predict(images)
annotations = numpy.reshape(preds.argmax(axis=2), (predictimages, 256, 256))
for i in range(predictimages):
    mpimg.imsave('preds/{}/preds_test_{}.png'.format(model_name, i+1),
                 annotations[i])
