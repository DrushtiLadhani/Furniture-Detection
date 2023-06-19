# Furniture-Detection


# Furniture Detection on Jetson Nano 2GB Developer Kit using Yolov5.


Furniture detection system which will detect objects based on whether it is Bed, Bench, Bookshelf, Chair, Lamp, Sofa, Swing, Table,
Vase, Wardrobe.

## Aim and Objectives

### Aim

To create a furniture detection system which will detect objects based on whether it isBed, Bench, Bookshelf, Chair, Lamp, Sofa, Swing, Table,
Vase, Wardrobe.

### Objectives

• The main objective of the project is to create a program which can be either run on
Jetson nano or any pc with YOLOv5 installed and start detecting using the camera
module on the device.

• Using appropriate datasets for recognizing and interpreting data using machine
learning.

• To show on the optical viewfinder of the camera module whether objects are Bed, Bench, Bookshelf, Chair, Lamp, Sofa, Swing, Table,
Vase, Wardrobe.

## Abstract

• An object is classified based on whether it is Bed, Bench, Bookshelf, Chair, Lamp, Sofa, Swing, Table,
Vase, Wardrobe, etc and is detected by the live feed from the system’s camera.

•We have completed this project on jetson nano which is a very small computational
device.

• A lot of research is being conducted in the field of Computer Vision and Machine
Learning (ML), where machines are trained to identify various objects from one
another. Machine Learning provides various techniques through which various objects
can be detected.

• One such technique is to use YOLOv5 with Roboflow model, which generates a small
size trained model and makes ML integration easier.


## Introduction


• This project is based on a furniture detection model with modifications. We are going
to implement this project with Machine Learning and this project can be even run on
jetson nano which we have done.

• This project can also be used to gather information about what category of waste does
the object comes in.

• The objects can even be further classified into Bed, Bench, Bookshelf, Chair, Lamp, Sofa, Swing, Table,
Vase, Wardrobe based on the image annotation we give in roboflow.


training in Roboflow has allowed us to crop images and also change the contrast of certain images to match the time of day
for better recognition by the model.

• Neural networks and machine learning have been used for these tasks and have
obtained good results.

• Machine learning algorithms have proven to be very useful in pattern recognition and
classification, and hence can be used for furniture detection as well.


## Jetson Nano Compatibility

• The power of modern AI is now available for makers, learners, and embedded developers
everywhere.

• NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run
multiple neural networks in parallel for applications like image classification, object
detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as
little as 5 watts.

• Hence due to ease of process as well as reduced cost of implementation we have used Jetson
nano for model detection and training.

• NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated
AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

• In our model we have used JetPack version 4.6 which is the latest production release and
supports all Jetson modules.

# Jetson Nano 2GB





https://user-images.githubusercontent.com/89011801/151482942-43bed9a8-abc9-4548-92b2-0c24e7d43d0b.mp4





## Proposed System

1. Study basics of machine learning and image recognition.
    
2. Start with implementation
        
        ➢ Front-end development
        ➢ Back-end development

3. Testing, analysing and improvising the model. An application using python and
Roboflow and its machine learning libraries will be using machine learning to identify
whether objects are Bed, Bench, Bookshelf, Chair, Lamp, Sofa, Swing, Table,
Vase, Wardrobe.


## Methodology

The  furnirure system is a program that focuses on implementing real time
Garbage detection.

It is a prototype of a new product that comprises of the main module:
furniture detection and then showing on viewfinder whether the object is furniture or not.

Furniture Detection Module


#### This Module is divided into two parts:

#### 1] furniture detection

• Ability to detect the location of object in any input image or frame. The output is
the bounding box coordinates on the detected object.

• For this task, initially the Dataset library Kaggle was considered. But integrating
it was a complex task so then we just downloaded the images from gettyimages.ae
and google images and made our own dataset.

• This Datasets identifies object in a Bitmap graphic object and returns the
bounding box image with annotation of object present in a given image.

#### 2] Classification Detection


• Classification of the object based on whether it is furniture or not.

• Hence YOLOv5 which is a model library from roboflow for image classification
and vision was used.

•There are other models as well but YOLOv5 is smaller and generally easier to use
in production. Given it is natively implemented in PyTorch (rather than Darknet),
modifying the architecture and exporting and deployment to many environments
is straightforward.

• YOLOv5 was used to train and test our model for various classes like Bed, Bench, Bookshelf, Chair, Lamp, Sofa, Swing, Table,
Vase, Wardrobe.
. We trained it for 149 epochs and achieved an accuracy of
approximately 91%.
## Installation

### Initial Setup

Remove unwanted Applications.
```bash
sudo apt-get remove --purge libreoffice*
sudo apt-get remove --purge thunderbird*
```
### Create Swap file

```bash
sudo fallocate -l 10.0G /swapfile1
sudo chmod 600 /swapfile1
sudo mkswap /swapfile1
sudo vim /etc/fstab
```
```bash
#################add line###########
/swapfile1 swap swap defaults 0 0
```
### Cuda Configuration

```bash
vim ~/.bashrc
```
```bash
#############add line #############
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export
LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_P
ATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```
```bash
source ~/.bashrc
```
### Udpade a System
```bash
sudo apt-get update && sudo apt-get upgrade
```
################pip-21.3.1 setuptools-59.6.0 wheel-0.37.1#############################

```bash 
sudo apt install curl
```
``` bash 
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
```
``` bash
sudo python3 get-pip.py
```
```bash
sudo apt-get install libopenblas-base libopenmpi-dev
```


```bash
source ~/.bashrc
```
```bash
sudo pip3 install pillow
```
```bash
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
```
```bash
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
```bash
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
```bash
sudo python3 -c "import torch; print(torch.cuda.is_available())"
```
### Installation of torchvision.

```bash
git clone --branch v0.9.1 https://github.com/pytorch/vision torchvision
cd torchvision/
sudo python3 setup.py install
```
### Clone yolov5 Repositories and make it Compatible with Jetson Nano.

```bash
cd
git clone https://github.com/ultralytics/yolov5.git
cd yolov5/
```

``` bash
sudo pip3 install numpy==1.19.4
history
##################### comment torch,PyYAML and torchvision in requirement.txt##################################
sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
sudo python3 detect.py
sudo python3 detect.py --weights yolov5s.pt --source 0
```
## Garbage Dataset Training
### We used Google Colab And Roboflow

train your model on colab and download the weights and past them into yolov5 folder
link of project


## Running Garbage Detection Model
source '0' for webcam

```bash
!python detect.py --weights best.pt --img 416 --conf 0.1 --source 0

```
# Demo


https://youtu.be/s3xwYaeYZoU


## Conclusion

• In this project our model is trying to detect objects and then showing it on viewfinder, live
as what their class is as whether they are Bed, Bench, Bookshelf, Chair, Lamp, Sofa, Swing, Table,
Vase, Wardrobe as we have specified in Roboflow.

## Refrences

#### 1]Roboflow :- https://roboflow.com/

#### 2] Datasets or images used: https://www.gettyimages.ae/search/2/image?phrase=garbage

#### 3] Google images
