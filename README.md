
# Garbage Detection using Yolov5 on Jetson Nano 2gb Developer Kit.

Garbage detection system which will detect objects based on whether it is
plastic waste or plastics or just garbage.
## Aim and Objectives

### Aim

To create a Garbage detection system which will detect objects based on whether it is
plastic waste or plastics or just garbage.

### Objectives

➢ The main objective of the project is to create a program which can be either run on
Jetson nano or any pc with YOLOv5 installed and start detecting using the camera
module on the device.

➢ Using appropriate datasets for recognizing and interpreting data using machine
learning.

➢ To show on the optical viewfinder of the camera module whether objects are plastics
or plastic waste or garbage.
## Abstract

➢ An object is classified based on whether it is plastics, plastic waste, garbage etc and is
detected by the live feed from the system’s camera.

➢ We have completed this project on jetson nano which is a very small computational
device.

➢ A lot of research is being conducted in the field of Computer Vision and Machine
Learning (ML), where machines are trained to identify various objects from one
another. Machine Learning provides various techniques through which various objects
can be detected.

➢ One such technique is to use YOLOv5 with Roboflow model, which generates a small
size trained model and makes ML integration easier.

➢ Garbage has become a major problem for developing countries as their rate of growth
also results in enormous consumption of products and hence more waste.

➢ Garbage segregation based on recyclable waste and unrecyclable waste like certain
plastics helps solve the problem of garbage as well as helps in the further growth of
economy by reusing certain recyclable material.
## Introduction


➢ This project is based on a Garbage detection model with modifications. We are going
to implement this project with Machine Learning and this project can be even run on
jetson nano which we have done.

➢ This project can also be used to gather information about what category of waste does
the object comes in.

➢ The objects can even be further classified into liquid, solid, organic, inorganic waste
based on the image annotation we give in roboflow.

➢ Garbage detection sometimes becomes difficult as certain waste gets mixed together
and gets harder for the model to detect. However, training in Roboflow has allowed us
to crop images and also change the contrast of certain images to match the time of day
for better recognition by the model.

➢ Neural networks and machine learning have been used for these tasks and have
obtained good results.

➢ Machine learning algorithms have proven to be very useful in pattern recognition and
classification, and hence can be used for Garbage detection as well.
## Literature Review

➢ There is no denying that India has improved its sanitation coverage, but the country’s
biggest shortcoming is its poor waste management infrastructure.

➢ Where solid waste is properly managed, after the waste is generated, it is segregated
at source, then properly stored, collected, transported and treated. In an effective solid
waste management model, there should be a goal to reduce, reuse, recover and recycle
waste by using the appropriate technologies and the waste that is disposed of in
landfills should be minimized, most importantly, landfills should be properly managed
so that they don’t become a source of greenhouse gases and toxins.

➢ The waste that is generated is just recklessly dumped in most cases, some is dumped
on the streets, and some is dumped in landfills that are not properly managed and this
ends up polluting the air, soil, and underground water.

➢ There are not enough public bins, and the available bins are not even covered and, in
many cases, waste overflows out of those bins and ends up going all over the streets.

➢ India’s informal recycling sector that consists of waste pickers plays a crucial role in
segregating and recycling waste, but in most cases, they are not formally trained and
at times they burn wastes at landfills to keep themselves warm at night and end up
setting landfill fires that cause air pollution, and because of inadequate gear, they are
also exposed to diseases and injuries.

➢ The sizes of landfills in India are constantly increasing and that is fast becoming a
major concern.
## Jetson Nano Compatibility

➢ The power of modern AI is now available for makers, learners, and embedded developers
everywhere.

➢ NVIDIA® Jetson Nano™ Developer Kit is a small, powerful computer that lets you run
multiple neural networks in parallel for applications like image classification, object
detection, segmentation, and speech processing. All in an easy-to-use platform that runs in as
little as 5 watts.

➢ Hence due to ease of process as well as reduced cost of implementation we have used Jetson
nano for model detection and training.

➢ NVIDIA JetPack SDK is the most comprehensive solution for building end-to-end accelerated
AI applications. All Jetson modules and developer kits are supported by JetPack SDK.

➢ In our model we have used JetPack version 4.6 which is the latest production release and
supports all Jetson modules.
## Proposed System

1. Study basics of machine learning and image recognition.
    
2. Start with implementation
        
        ➢ Front-end development
        ➢ Back-end development

3. Testing, analysing and improvising the model. An application using python and
Roboflow and its machine learning libraries will be using machine learning to identify
whether objects are plastics, plastic waste like bottles or garbage.

4. Use datasets to interpret the object and suggest whether the object is plastic waste, plastics
or garbage.
## Methodology

The Garbage detection system is a program that focuses on implementing real time
Garbage detection.

It is a prototype of a new product that comprises of the main module:
Garbage detection and then showing on viewfinder whether the object is garbage or not.

Garbage Detection Module

```bash
This Module is divided into two parts:
```

    1] Garbage detection

➢ Ability to detect the location of object in any input image or frame. The output is
the bounding box coordinates on the detected object.

➢ For this task, initially the Dataset library Kaggle was considered. But integrating
it was a complex task so then we just downloaded the images from gettyimages.ae
and google images and made our own dataset.

➢ This Datasets identifies object in a Bitmap graphic object and returns the
bounding box image with annotation of object present in a given image.

    2] Classification Detection


➢ Classification of the object based on whether it is garbage or not.

➢ Hence YOLOv5 which is a model library from roboflow for image classification
and vision was used.

➢ There are other models as well but YOLOv5 is smaller and generally easier to use
in production. Given it is natively implemented in PyTorch (rather than Darknet),
modifying the architecture and exporting and deployment to many environments
is straightforward.

➢ YOLOv5 was used to train and test our model for various classes like Plastics,
plastic waste, garbage. We trained it for 149 epochs and achieved an accuracy of
approximately 91%.

## Jetson Nano 2GB Developer Kit.

### Setup


<img src="https://github.com/rishikeshbondade17/Garbage-Detection-using-Yolov5-on-2gb-Jetson-Nano/blob/main/Jetson_nano_setup/jetson_nano.jpg" alt="Demo1" width="400" height="300">
<img src="https://github.com/rishikeshbondade17/Garbage-Detection-using-Yolov5-on-2gb-Jetson-Nano/blob/main/Jetson_nano_setup/jetson_nano0.jpg" alt="Demo2" width="400" height="300">
<img src="https://github.com/rishikeshbondade17/Garbage-Detection-using-Yolov5-on-2gb-Jetson-Nano/blob/main/Jetson_nano_setup/jetson_nano1.jpg" alt="Demo3" width="400" height="300">
<img src="https://github.com/rishikeshbondade17/Garbage-Detection-using-Yolov5-on-2gb-Jetson-Nano/blob/main/Jetson_nano_setup/jetson_nano2.jpg" alt="Demo4" width="400" height="300">

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
###########add line###########
/swapfile1 swap swap defaults 0 0
```
### Cuda Configuration

```bash
vim ~/.bashrc
```
```bash
#############add line #############
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATh=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
```
```bash
source ~/.bashrc
```
### Udpade and Upgrade a System
```bash
sudo apt-get update
sudo apt-get upgrade
```
### Install Some Required Packages 

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
sudo apt-get install python3-dev build-essential autoconf libtool pkg-config python-opengl python-pil python-pyrex python-pyside.qtopengl idle-python2.7 qt4-dev-tools qt4-designer libqtgui4 libqtcore4 libqt4-xml libqt4-test libqt4-script libqt4-network libqt4-dbus python-qt4 python-qt4-gl libgle3 python-dev libssl-dev libpq-dev python-dev libxml2-dev libxslt1-dev libldap2-dev libsasl2-dev libffi-dev libfreetype6-dev python3-dev
```
```bash
vim ~/.bashrc
###### add line ########
export OPENBLAS_CORETYPE=ARMV8
```

```bash
source ~/.bashrc
```
```bash
sudo pip3 install pillow
```

### Install Torch
```bash
curl -LO https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl
```
```bash
mv p57jwntv436lfrd78inwl7iml6p13fzh.whl torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
```bash
sudo pip3 install torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
#Check Torch, output should be "True" 
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

# comment torch,PyYAML and torchvision in requirement.txt

sudo pip3 install --ignore-installed PyYAML>=5.3.1
sudo pip3 install -r requirements.txt
```
### Download weights and Test Yolov5 Installation on USB webcam
```bash
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
## Output Video


https://user-images.githubusercontent.com/80247111/151414233-78e4dc7e-c36d-4ebd-8a60-0ecaff2ba6ed.mp4



## Advantages

➢ The Garbage detection system will be of great help in reducing diseases that occur because
of poor waste management.

➢ The Garbage detection system shows the classification of the object whether they are
plastics, plastic waste like bottles or just plain garbage.

➢ It can then convey to the person who cleans or if it needs to be completely automated then
to the segregating machine to separate the waste according to the classes specified.

➢ When completely automated no user input is required and therefore works with absolute
efficiency and speed.

➢ As it is completely automated the cost of segregation of waste decreases significantly.

➢ It can work around the clock and therefore becomes more cost efficient.
## Application

➢Detects object class like plastic or plastic waste in a given image frame or viewfinder using
a camera module.

➢ Can be used in various garbage segregation plants.

➢ Can be used as a refrence for other ai models based on Garbage detection.
## Future Scope


➢ As we know technology is marching towards automation, so this project is one of the step
towards automation.

➢ Thus, for more accurate results it needs to be trained for more images, and for a greater
number of epochs.

➢ Garbage segregation will become a necessity in the future due to rise in population and
hence our model will be of great help to tackle the situation in an efficient way.

➢ As more products gets released due to globalization and urbanization new waste will be
created and hence our model which can be trained and modified with just the addition of
images can be very useful.
## Conclusion

➢ In this project our model is trying to detect objects and then showing it on viewfinder, live
as what their class is as whether they are plastics, plastic waste or garbage as we have
specified in Roboflow.

➢ The model solves the problem of garbage segregation in modern India and helps
counteract and prevent water borne diseases.

➢ Lower diseases lead to better economy of country as the workforce doesn’t get affected
and hence can go to work with no problem.
## Refrences

1] Roboflow :- https://roboflow.com/

2] Datasets or images used: https://www.gettyimages.ae/search/2/image?phrase=garbage

3] Google images
## Articles



[1] https://www.recycling-magazine.com/2020/05/06/waste-management-crisis-in-india/#:~:text=Urban%20India%20generates%2062%20million,just%2011.9%20million%20is%20treated.

[2] https://www.downtoearth.org.in/blog/waste/india-s-challenges-in-waste-management-56753
