# Introduction

This is project for course CS386: Digital Image Processing (Shanghai Jiao Tong University). Me and Yuqiao He finished this together.

In this project, we aim to automatically detect glaucoma via deep learning. To do that, we need to calculate CDR on fine segmented retina images. To get precise segmenta- tion, we implemented SegNet together with ADDA(Adversarial Discriminative Domain Adaptation). We are the first to combine them together to detect glaucoma on test dataset which have different brightness from our training dataset.


# Structure

```
.
└-- evaluation/              # folder contains evalutation code
└-- figures/                 # folder contains figures for demonstration
└-- model/                   # folder contains two models
|   └-- segnet.py            # SegNet model
|   └-- adda.py              # ADDA model
└-- util/                    # folder containing some utils
|   └-- dataset.py           # use tf.data.dataset to construct data pipeline
|   └-- loss_functions.py    # various loss functions
|   └-- output.py            # output results to file
|   └-- util.py              # utils
└-- main.py                  # main function entry
└-- preprocess.ipynb         # preprocess input images
└-- evaluate.ipynb           # evaluate our experiment results
```


# Implementation

###  SegNet

The network architecture and some codes in this part are based on [Tensorflow-SegNet](https://github.com/tkuanlun350/Tensorflow-SegNet), 

The SegNet model consists of two parts, encoder and decoder. The encoder has four batch-normed convolutional layers, each with a max pooling layer. All the max pooling layer has a kernel size of 1,2,2,1 and a stride of 1,2,2,1. Note the data is stored in the form of ’NHWC’. As for the decoder, we have four deconvolutional layers, each with a batch-normed convolutional layer in replace of original upsampling layers.

The architecture:

<img src="https://github.com/gohsyi/SegNet-ADDA/blob/master/figures/architecture.png" width="600" align="middle"/>

Here is the demonstration:
    
<img src="https://github.com/gohsyi/SegNet-ADDA/blob/master/figures/segnet.png" width="600" align="middle" />
    

### ADDA

For the architecture details, you can refer to the paper [Adversarial Discriminative Domain Adaptation
](https://arxiv.org/abs/1702.05464).

The ADDA(Adversarial Discriminative Domain Adaptation) model learns a discriminative mapping of target images to the source feature space. Here we just consider the encoder in SegNet as the mapping we want the model to learn. 

Once we learned a target encoder well, we can detect glaucoma in pictures from other distribution, like brighter or darker retina photos.

The reason why we choose to learn a target encoder of SegNet to do transfer learning is two-fold. On one hand, the SegNet encoder’s function is to extract features from input images, which is the same with ADDA’s target mapping. On the other hand, the output of SegNet encoder is small enough, which makes our discriminator network lighter and easier to train.

The size of discriminator should be considered carefully. If the discriminator network is small, it may not able to represent enough information, and does no good to generator part. But if the discriminator network is too large, its training could be very slowly. And what’s even worse is that the discriminator may become so powerful that our generator cannot fool it at all. So the loss will keep being high and our target encoder doesn’t make any progress at all. We did some experiments and found that the discriminator works best as a three-layer MLP, and the two hidden layers’ sizes are 576 and 128.

And here is the demonstration:

<img src="https://github.com/gohsyi/SegNet-ADDA/blob/master/figures/adda.png" width="1000" align="middle" />


# Requirement

numpy

tensorflow (>=1.11.0 recommend)

Pillow

scikit-image


# Args

- `gpu`: set `CUDA_VISIBLE_DEVICES`, but if you don't want to use gpu device, just set `-gpu -1`
- `note`: temporary files and model checkpoints will be stored in logs/`note`/
- `save_image`: whether to store test results
- `loss`: which loss function you want to use, `dice` or `normal` or `weighted`
- `test`: set `test` as the path to the checkpoint file when you want to test a stored model
- `transfer`: set `transfer` as the folder that contains the stored pretrained SegNet model
- `finetune`: set `finetune` as the path to the checkpoint file if you want to fine tune a stored SegNet model
- `batch_size`: training batch size, default as 5
- `max_steps`: max training steps when training SegNet as well as ADDA, default as `1000000`
- `learning_rate`: learning rate when training SegNet as well as ADDA, default as `0.001`


# Usage

1. Download and unzip our training and test dataset [glaucoma.zip](https://pan.baidu.com/s/1OxEiYgHoYYGjE1UXU1QY0w)(key :cyve) and put it in the project's root folder.

2. Run preprocess.ipynb, which will preprocess images and labels, and generate three files: train.txt, val.txt, test.txt, representing training set, Validation set and test set. They store the path to each image and path to its corresponding label in one line.

3. If you want to train a model from zero. Run this: `$ python main.py`.

4. If you want to test a stored SegNet model. Download the [stored model](https://pan.baidu.com/s/1f9vz3DsAM5U1PUdadizWug)(key: 91hi) and put model.ckpt-11999/ in logs/. Then run this: `$ python main.py -test model.ckpt-11999/model.ckpt-11999`.

5. If you want to do transfer learning. Run this: `$ python main.py -transfer model.ckpt-11999/`. You may find some codes in `evaluate.ipynb` useful if you want to evaluate your results.


# Results

We did experiments on loss functions and find that dice loss is the most efficient.
<img src="https://github.com/gohsyi/SegNet-ADDA/blob/master/figures/iu.png" width="500">

Our transfer learning approach can reduce the MSE between CDR computed with our segmentation and that with our annotations on target dataset.

<img src="https://github.com/gohsyi/SegNet-ADDA/blob/master/figures/mamse.png" width="450">

Demonstration:

<img src="https://github.com/gohsyi/SegNet-ADDA/blob/master/figures/demo.png" width="500">
