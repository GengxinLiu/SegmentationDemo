run model:
    - run_random.py: 随便搭的一个小网络, 模型保存为randomNet.h5.
    - run_fcn8.py: fcn8 model, 以vgg16为特征提取模型, 模型较大未训练过.
    - run_unet.py: unet model, 常用医学图像分割模型, 模型较大未训练过.

图像增强操作:
    - augment.ipynb: notebook 简单教程.
    
segmap: 手动分割的图片.

preds: run_random 预测的分割图片, 在PC上跑的, 网络小, 预测比较粗糙, 但有大概的轮廓了.

model:
    - FCN8.py
    - randomNet.py
    - UNET.py

data:
    - label_{}.numpy: 分割label.
    - train_{}.jpg: 输入图像.

pretrain:
    预训练模型.
    - VGG16
    - randomNet.h5