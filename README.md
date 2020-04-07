first download the vgg16 model in pretrain/

run model:

    - run_fcn32.py: train fcn32 model 
    - run_fcn8.py: train fcn8 model  
    - run_unet.py: train unet model  

augment image:

    - augment.ipynb: notebook.

model:

    - FCN8.py
    - FCN32.py
    - UNET.py

data:

    - label_{}.numpy: label.
    - train_{}.jpg: image.

pretrain: pretrain model
