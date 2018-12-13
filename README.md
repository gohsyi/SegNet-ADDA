# Introduction
This is my project for course CS386: Digital Image Processing (Shanghai Jiao Tong University). Me and Yuqiao He finished this together.

Our aim is to detect glaucoma with computer vision. I implemented SegNet along with ADDA (Adversarial Discriminative Domain Adaptation) for transfer learning.


# Implementation

###  SegNet
The network architecture and some codes in this part are based on [Tensorflow-SegNet](https://github.com/tkuanlun350/Tensorflow-SegNet), 
except for the data queue is implemented with `tf.data.dataset` rather than `tf.train.batch`
which is deprecated by tensorflow currently.

The SegNet is implemented as a class. The decoder and encoder network are seperate rather than in one `inference()`.

### ADDA
For the architecture details, you can refer to the paper [here](https://arxiv.org/abs/1702.05464).

In our case, ADDA will learn a target encoder (as Generator) with GAN. 
The Discriminator judges the encode result is from the source encoder or target encoder. 
Note that the target decoder and classifier are not trained and directly use parameters from source SegNet.

# ARGs

- -preprocess
    - `nothing`: do nothing, choose this if you have all txt files and .png files
    - `txt`: only generate train.txt,test.txt,val.txt, ...
    - `all`: generate .txt files and preprocess images into 240x240 .png images
    - You may want to modify main.py if your images is in somewhere else

- will add others

# Requirement
numpy

tensorflow

Pillow

scikit-image

# Usage
#### training

  python main.py -log_dir logs/ -note dice -loss dice -preprocess all

#### finetune

  python main.py -log_dir logs/ -note finetune -finetune logs/dice/model.ckpt-11999 -loss dice

#### test

  python main.py -test logs/dice/model.ckpt-11999


# Dataset
example format:

"path_to_image1" "path_to_corresponding_label_image1",

"path_to_image2" "path_to_corresponding_label_image2",

"path_to_image3" "path_to_corresponding_label_image3",

.......
