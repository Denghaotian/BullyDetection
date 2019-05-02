# Bully picture object detection
CPSC8810 Deep Learning Term Project Stage2

## Authors
Qingbo Lai   

qingbol@clemson.edu 

Haotian Deng 

hdeng@clemson.edu

## Note
This project construct and implement Convolutional Neural Networks (CNNs)
to classify the bully picture. All codes were implementated and tested on 
Palmetto www.palmetto.clemson.edu

## Prerequisites
Python3.6; TensorFlow framework 1.12

## Network Structure
We have two different networks structure. one is simple three layers CNN 
model with two fully connected layers which written by ourselves, the 
another model is based VGG16 with some changes by ourselves.

## Training Strategy
We used ten categories images to train model. Nine categories of bully 
images which are laughing, pullinghair, quarrel, slapping, punching, 
stabbing, gossiping, strangle and isolation. The rest of images are 
nonbullying category. 

## Usages
Default location of images in dataset : dataset/JPEGImages/
Default location of testing data: dataset/Annotationsa
### Train
python main.py
### test(predict) a single image (default value of test_num=1)
python main.py --action prediction
### test(predict) a group of  images (assign a value to test_num)
python main.py --action prediction --test_num 3

### modify the dataset path
python main.py --img_path your_img_path 
Or
python main.py --action prediction --img_path your_img_path 

### modify training steps
python main.py --iteration_steps 10000

### modify learning rate
python main.py --learning_rate 0.0001


## Reference
