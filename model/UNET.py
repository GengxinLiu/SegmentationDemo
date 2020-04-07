from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Cropping2D, add, Dropout, Reshape, Activation, MaxPooling2D, Add, UpSampling2D, concatenate


def build(height, width, num_classes):
    inputs = Input(shape=(height, width, 3))
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(inputs)  # (256,256,64)
    conv1 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv1)  # (256,256,64)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)  # (128,128,64)

    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)  # (128, 128, 128)
    conv2 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv2)  # (128, 128, 128)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)  # (64, 64, 128)

    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)  # (64, 64, 256)
    conv3 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv3)  # (64, 64, 256)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)  # (32, 32, 256)

    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)  # (32, 32, 512)
    conv4 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv4)  # (32, 32, 512)
    drop4 = Dropout(0.5)(conv4)  # (32, 32, 512)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)  # (16, 16, 512)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)  # (16, 16, 1024)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv5)  # (16, 16, 1024)
    drop5 = Dropout(0.5)(conv5)  # (16, 16, 1024)

    # UpSampling2D(size=(2, 2))(drop5) => (32,32,1024)
    up6 = Conv2D(512, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(drop5))  # (32,32,512)
    # (32,32,512) +  (32,32,512) =>  (32,32,1024)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge6)  # (32,32,512)
    conv6 = Conv2D(512, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv6)  # (32,32,512)

    up7 = Conv2D(256, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv6))  # (64,64,256)
    merge7 = concatenate([conv3, up7], axis=3)  # (64,64,512)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge7)  # (64,64,256)
    conv7 = Conv2D(256, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv7)  # (64,64,256)

    up8 = Conv2D(128, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv7))  # (128,128,128)
    merge8 = concatenate([conv2, up8], axis=3)  # (128,128,256)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge8)  # (128,128,128)
    conv8 = Conv2D(128, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv8)  # (128,128,128)

    up9 = Conv2D(64, 2, activation='relu', padding='same',
                 kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(conv8))  # (256,256,64)
    merge9 = concatenate([conv1, up9], axis=3)  # (256,256,128)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(merge9)  # (256,256,64)
    conv9 = Conv2D(64, 3, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)  # (256,256,64)
    conv9 = Conv2D(num_classes, 1, activation='relu', padding='same',
                   kernel_initializer='he_normal')(conv9)  # (256,256,21)

    flatt = Reshape((-1, num_classes))(conv9)  # 256*256 21
    out = Activation('softmax')(flatt)  # 256*256 21
    model = Model(inputs, out)
    return model
