import os
import sys
import gc
import random
import numpy as np
import tensorflow as tf
import time
import glob
from tensorflow.python.training.moving_averages import assign_moving_average
from .ObjectDetectNet import OdNet
# import ObjectDetecNet 
sys.path.append("..")
from loaddata import get_data

class Model(object):

    def __init__(self, sess, parameters):
        self.sess = sess
        self.parameters = parameters

    def tst(self):
        print("self.parameters.action is ", self.parameters.action)

    '''
    Training method
    '''
    def training(self):
        running_count = 0
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            my_net= OdNet(sess)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(var_list=tf.trainable_variables())
            if os.path.exists('./session_params/session.ckpt.index') :
                print('\nStart Restore')
                saver.restore(sess, './session_params/session.ckpt')
                print('\nEnd Restore')
            
            print('\n============Training start from here============')
            min_loss_location = 100000.
            min_loss_class = 100000.

            while((min_loss_location + min_loss_class) > 0.001 and running_count < self.parameters.iteration_steps):
                running_count += 1
                
                train_data, actual_data,_ = get_data(self.parameters.batch_size,self.parameters.img_path,self.parameters.xml_path)
                if len(train_data) > 0:
                    loss_all,loss_class,loss_location,pred_class,pred_location = my_net.train_run(train_data, actual_data)
                    l = np.sum(loss_location)
                    c = np.sum(loss_class)
                    if min_loss_location > l:
                        min_loss_location = l
                    if min_loss_class > c:
                        min_loss_class = c

                    sum_loss= min_loss_location + min_loss_class
                    msg = ("Step {0}---Loss:[{1:>4.3}|{2:>4.3}]" 
                        " ** Location::{3:>4.3} ** Class::{4:>4.3}"
                        " ** pred_class::[{5:>4.3}|{6:>4.3}|{7:>4.3}]"
                        " ** pred_location::[{8:>4.3f}|{9:>4.3f}|{10:>4.3f}|")
                    print(msg.format(running_count, sum_loss, loss_all, np.sum(loss_location), 
                        np.sum(loss_class),np.sum(pred_class),np.amax(pred_class),
                        np.min(pred_class),np.sum(pred_location),np.amax(pred_location),
                        np.min(pred_location)))

                    # save ckpt 
                    if running_count % 100 == 0:
                        saver.save(sess, './trained_model/bullymodel.ckpt')
                        print('bullymodel.ckpt has been saved.')
                        gc.collect()
                else:
                    print('No image!,pls check')
                    break
                
            saver.save(sess, './trained_model/bullymodel.ckpt')
            sess.close()
            gc.collect()
                
        print('===========Training ended==========')


    '''
    prediction for bully object detection
    '''
    def prediction(self):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            my_net= OdNet(sess)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(var_list=tf.trainable_variables())
            if os.path.exists('./trained_model/bullymodel.ckpt.index') :
                saver.restore(sess, './trained_model/bullymodel.ckpt')
                #load data
                image, actual,file_list= get_data(1,self.parameters.img_path,self.parameters.xml_path)
                #run
                pred_class, pred_class_val, pred_location =  my_net.prediction_run(image,None)
                print('images for prediction:' + str(file_list))
                
                for index, act in zip(range(len(image)), actual):
                    for a in act :
                        print('【img-'+str(index)+' actual】:' + str(a))
                    msg = ("predicted class is: {0}\n "
                          "predicted class value:{1}\n"
                          "predicted location:{2}")
                    print(msg.format(pred_class[index], pred_class_val[index], 
                          pred_location[index]))
                    # print('predicted location:' + str(pred_location[index]))   
            else:
                print('No trained model Exists!')
            sess.close()