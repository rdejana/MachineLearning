# Homework XX

## Overview
This Homework will compare image classificaiton inference performance with Tensorflow 1.15, Tensorflow 2, TFLite, and Jetson Inference.  Note, this is not currently containerized.

## Framework installation

## Classification images
Find at least 3 images on your own to use for classification.  
You will be submitting these images as part of the homework.

### Prerequisites 
1. Install system packages
```
$ sudo apt-get update
$ sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
```
2. Install and upgrade pip3
``
$ sudo apt-get install python3-pip
``
3. Install Virtualenv
```
$ sudo pip3 install -U virtualenv # system-wide install
```
4. Create the virtualenv directory under /data
```
$ cd /data
$ mkdir virtualenvs
```

 If you /data directory is owned by root, create using sudo, `sudo mkdir virtualenvs` and then change ownership to your id, `sudo chown -R $USER:$USER virtualenvs`
### TFLite
1. Create a TFLite python3 virtualenv in your /data directory. 
```
$ cd /data/virtualenvs
$ virtualenv -p python3 ./tflite
```
 2. Activate virtual enviroment.  This is importand as TFLite will only be available when using this virtual environment. 
 ```
 $ source /data/virtualenvs/tflite/bin/activate
 ```
 3. Install TFLite runtime and additional components. 
 Open https://www.tensorflow.org/lite/guide/python and look for the latest version that supports ARM 64 at your version of python.  Currently this is https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_aarch64.whl
 ```
 pip3 install testresources setuptools
 pip3 install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_aarch64.whl
 pip3 install Pillow numpy
 ```
 4. Deactivate virtualenv
 ```
 deactiate
 ```
### Tensorflow 1.15
1. Create a TF 1.15 python3 virtualenv in your /data directory. 
```
$ cd /data/virtualenvs
$ virtualenv -p python3 ./tf115
```
2. . Activate virtual enviroment.  This is importand as TF 1.15 will only be available when using this virtual environment. 
 ```
 $ source /data/virtualenvs/tf115/bin/activate
 ```
3. Install TF 1.15 runtime and additional components.
```
pip3 install  testresources setuptools
pip3 install numpy==1.16.1 future==0.17.1 mock==3.0.5 h5py==2.9.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11
pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v43 'tensorflow<2'
pip3 install --upgrade tensorflow-hub
pip3 install Pillow
```
Full instructions may be found at https://docs.nvidia.com/deeplearning/frameworks/install-tf-jetson-platform/index.html

### Tensorflow 2.x
### Jetson Inference

## Assignment 
Clone this repository to your TX2.  

## Part 1: TFLite
In this part, you'll run a simple image classificaiton example against a sample image and your classification test images.  You'll need to run against each of the supplied models, returning the (up to) top 5 results.
1. Enable your tflite virtualenv.
2. cd to tflite.
3. Run to following 
```
    python3 classify_image.py --model models/<modelName>  --labels models/imagenet_labels.txt   --input <pathToImage> -k 5
``` 
For each of the following models:
- efficientnet-L_quant.tflite
- efficientnet-M_quant.tflite
- efficientnet-S_quant.tflite
- inception_v4_299_quant.tflite
- mobilenet_v1_1.0_224_quant.tflite
- mobilenet_v2_1.0_224_quant.tflite

using the test image `images/parrot.jpg` and your test images.

### Questions
1. What was the average inference time for model and image combination?  What where the returned classes their score?
2. In your optionion, which model is best and why?

## Part 2: TensorFlow 1.15

This classifier downloads models from TensorFlow Hub.  Downloads are not cached and are redowned with each run.


Test with the following, passing the URL as the -m parameter
- mobilenet_v1_100_224 URL: https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/4
- 

- https://tfhub.dev/google/efficientnet/b0/classification/
- https://tfhub.dev/google/efficientnet/b4/classification/
- https://tfhub.dev/google/efficientnet/b7/classification/

For the EfficientNet models, please read the Usage section carefully. 

If you are seeing memory errors when running, run the flush_buffers.sh script located in the tf115 directory. 
```
sudo sh flush_buffers.sh
```
## TBD
For each framework (Ryan note, provide TFLite exmample and a basic TF 1.15 one), you'll need to write a program that performs image classification with an image. The program should take arguments for the model, labels file, image, the number of times to run interence, the max number of classification results, and the classification score threshold. It then prints the model's prediction for what the image is to the terminal screen.
See the example for TFLite.  

what to test...

## Problems
1. What does it mean to quantize a model?
