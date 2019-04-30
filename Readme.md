# Bully picture classification 
CPSC8810 Deep Learning Term Project

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
Default location of training data : data_bully/training_data

Default location of testing data: data_bully/testing_data
### Train
python bully_train.py --train_path path-to-training-dataset
### predict a single image
python predict.py --img_file path-to-img/xxx.jpg
### test the accuracy for testing dataset 
python test.py --test_path path-to-testing-dataset



## Testing
You can test model use two methods.

Testing one image 

Command line:

Testing 10 groups of classified images by the tagged file directory
like laughing, pullinghair, quarrel, slapping, punching, stabbing, 
gossiping, strangle, isolation and nonbullying. The output will be
the accuracy of testing files.

Command line:

Best Testing accuracy result is 

## Reference
