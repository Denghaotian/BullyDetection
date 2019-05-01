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
        # batch_size = 1
        # batch_size = 15
        running_count = 0
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            # ssd_model = ssd300.SSD300(sess,True)
            my_net= OdNet(sess,True)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(var_list=tf.trainable_variables())
            if os.path.exists('./session_params/session.ckpt.index') :
                print('\nStart Restore')
                saver.restore(sess, './session_params/session.ckpt')
                print('\nEnd Restore')
            
            print('\n============Training start from here============')
            min_loss_location = 100000.
            min_loss_class = 100000.

            while((min_loss_location + min_loss_class) > 0.001 and running_count < 1000):
                running_count += 1
                
                train_data, actual_data,_ = get_data(self.parameters.batch_size,self.parameters.img_path,self.parameters.xml_path)
                if len(train_data) > 0:
                    # loss_all,loss_class,loss_location,pred_class,pred_location = self.train_run(train_data, actual_data)
                    loss_all,loss_class,loss_location,pred_class,pred_location = my_net.run(train_data, actual_data)
                    l = np.sum(loss_location)
                    c = np.sum(loss_class)
                    if min_loss_location > l:
                        min_loss_location = l
                    if min_loss_class > c:
                        min_loss_class = c

                    # msg = ("Step {0} --- Loss:{1:>6.2%}|{2:>6.2%}" 
                    #     "||Location::{3:>6.2%}  ||Class::{4:.3f}"
                    #     "||pred_class::{3:>6.2%}|  ||pred_location::{4:.3f}")
                    # print(msg.format(epoch + 1, i, train_acc, val_acc, val_loss)) 

                    print('Step:【' + str(running_count) + '】|Loss: 【'+str(min_loss_location + min_loss_class)+'|'+ str(loss_all) + '】 |Location:【'+ str(np.sum(loss_location)) + '】 |Class:【'+ str(np.sum(loss_class)) + '】 |pred_class:【'+ str(np.sum(pred_class))+'|'+str(np.amax(pred_class))+'|'+ str(np.min(pred_class)) + '】 |pred_location:【'+ str(np.sum(pred_location))+'|'+str(np.amax(pred_location))+'|'+ str(np.min(pred_location)) + '】')
                    
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
