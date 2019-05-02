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
We rewrited and modified Single Shot MultiBox Detector (SSD), so the network is SSD's net structure.

## Training Strategy
We use the dataset provided in the CPSC-8810 as the basis, but for the model training, the dataset in the classroom can not meet the requirements, and further processing, such as labeling the object, we use the "labelImg" tool to mark the file. we can get "xml" files about images which contains information of labels and regions. And then annotations' formula same as the PASCAL VOC data set. Our data mainly contain two objects, one is bully, the other is victim.

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
