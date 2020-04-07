# from keras.models import Model
# from keras.layers import Conv2D, Conv2DTranspose, Input, Cropping2D, add, Dropout, Reshape, Activation, MaxPooling2D, Add
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Cropping2D, add, Dropout, Reshape, Activation, MaxPooling2D, Add


def build(height, width, num_classes):
    """build fcn8 model.
    ----------
    Args:
        - height: image height.
        - width: image width.
        - num_classes: segmentation classes.  
    return:
        - model: fcn8 model.

    operation
        - Conv2D:
            filters: 64 ->128 ->256 ->512 ->4096.
            kernel_size: (3,3).
            strides:
            padding: 'same'(keep the shape) or 'valid'=>( o=(i-k+2p)/s+1 ).
            activation: 'relu'.
        - MaxPooling2D:
            pool_size: (2,2).
            strides: (2,2).  
        - Conv2DTranspose: upsample the feature maps.
            filters:
            kernel_size:
            strides: 
            padding: 'same'(keep the shape) or 'valid'=>( o=(i-k+2p)/s+1 ).
            activation: 'relu'. """

    inputs = Input(shape=(height, width, 3))
    # Layer1
    conv1_1 = Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv1_1')(inputs)  # 320 320 64
    conv1_2 = Conv2D(filters=64,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv1_2')(conv1_1)  # 320 320 64
    pool1 = MaxPooling2D(pool_size=(2, 2,),
                         strides=(2, 2),
                         name='pool1')(conv1_2)  # 160 160 64
    # Layer2
    conv2_1 = Conv2D(filters=128,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv2_1')(pool1)  # 160 160 128
    conv2_2 = Conv2D(filters=128,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv2_2')(conv2_1)  # 160 160 128
    pool2 = MaxPooling2D(pool_size=(2, 2,),
                         strides=(2, 2),
                         name='pool2')(conv2_2)  # 80 80 128
    # Layer3
    conv3_1 = Conv2D(filters=256,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv3_1')(pool2)  # 80 80 256
    conv3_2 = Conv2D(filters=256,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv3_2')(conv3_1)  # 80 80 256
    conv3_3 = Conv2D(filters=256,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv3_3')(conv3_2)  # 80 80 256
    pool3 = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         name='pool3')(conv3_3)  # 40 40 256
    # Layer4
    conv4_1 = Conv2D(filters=512,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv4_1')(pool3)  # 40 40 512
    conv4_2 = Conv2D(filters=512,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv4_2')(conv4_1)  # 40 40 512
    conv4_3 = Conv2D(filters=512,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv4_3')(conv4_2)  # 40 40 512
    pool4 = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         name='pool4')(conv4_3)  # 20 20 512
    # Layer5
    conv5_1 = Conv2D(filters=512,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv5_1')(pool4)  # 20 20 512
    conv5_2 = Conv2D(filters=512,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv5_2')(conv5_1)  # 20 20 512
    conv5_3 = Conv2D(filters=512,
                     kernel_size=(3, 3),
                     activation='relu',
                     padding='same',
                     name='conv5_3')(conv5_2)  # 20 20 512
    pool5 = MaxPooling2D(pool_size=(2, 2),
                         strides=(2, 2),
                         name='pool5')(conv5_3)  # 10 10 512

    vgg = Model(inputs, pool5)
    vgg.load_weights(
        'pretrain/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')

    ###########################
    ########### fcn ###########
    ###########################
    fc6 = Conv2D(filters=4096,
                 kernel_size=(7, 7),
                 activation='relu',
                 padding='same',
                 name='fc6')(pool5)  # 10 10 4096
    fc7 = Conv2D(filters=4096,
                 kernel_size=(1, 1),
                 activation='relu',
                 padding='same',
                 name='fc7')(fc6)  # 10 10 4096
    score_pool5 = Conv2D(filters=num_classes,
                         kernel_size=(1, 1),
                         activation='relu',
                         padding='same',
                         name='score_pool5')(fc7)  # 10 10 21
    # 2xupsample pool5(10,10,21) => (320,320,21)
    upsampled_32x = Conv2DTranspose(filters=num_classes,
                                    kernel_size=(64, 64),
                                    strides=(32, 32),
                                    padding='same',
                                    use_bias=False)(score_pool5)  # 320 320 21

    flatt = Reshape((-1, num_classes))(upsampled_32x)  # 320*320 21
    out = Activation('softmax')(flatt)  # 320*320 21
    model = Model(inputs, out)
    return model
