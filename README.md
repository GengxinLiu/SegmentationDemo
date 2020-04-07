run model:
    - run_fcn32.py: fcn32 model 以vgg16为特征提取模型
    - run_fcn8.py: fcn8 model 以vgg16为特征提取模型
    - run_unet.py: unet model 常用医学图像分割模型

图像增强操作:
    - augment.ipynb: notebook 简单教程.

model:
    - FCN8.py
    - FCN32.py
    - UNET.py

data:
    - label_{}.numpy: label.
    - train_{}.jpg: image.

pretrain: pretrain model
