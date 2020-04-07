from model import FCN8
import numpy
from tensorflow.keras.utils import to_categorical, multi_gpu_model
from tensorflow.keras.optimizers import Adam
import matplotlib.image as mpimg


def read_data(num_images=10):
    categoricals = []
    images = []
    num_images = 25
    for i in range(num_images):
        mat = numpy.loadtxt('data/label_{}.numpy'.format(i+1))
        categorical = to_categorical(numpy.reshape(mat, (-1,)).astype(int),
                                     num_classes=33)
        categoricals.append(categorical)
        image = mpimg.imread('data/train_{}.jpg'.format(i+1))
        images.append(image)

    images = numpy.array(images, dtype=float)
    categoricals = numpy.array(categoricals)
    print(images.shape, categoricals.shape)
    return images, categoricals


mode = 'train'

if mode == 'train':
    model = FCN8.build(height=256, width=256, num_classes=33)
    model = multi_gpu_model(model, gpus=2)
    model.summary()
    model.compile(optimizer=Adam(0.00001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    num_images = 25
    images, categoricals = read_data(num_images=num_images)
    model.fit(images, categoricals, epochs=200, steps_per_epoch=num_images)
    model.save_weights('pretrain/FCN8.h5')  # save model
    mode = 'test'


if mode == 'test':
    model = FCN8.build(height=256, width=256, num_classes=33)
    model = multi_gpu_model(model, gpus=2)
    model.load_weights('pretrain/FCN8.h5')
    num_images = 25
    images, categoricals = read_data(num_images=num_images)

    ##### 预测图片 ####
    preds = model.predict(images)
    annotations = numpy.reshape(preds.argmax(axis=2), (num_images, 256, 256))
    for i in range(num_images):
        mpimg.imsave('preds/FCN8/preds_{}.png'.format(i+1), annotations[i])
