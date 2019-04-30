# Qingbo/Haotian Mar 8,2019

# import loaddata
from my_model import laimodel, vgg16model, vgg17model
import sys  
import os
import tensorflow as tf
from tensorflow.python.platform import flags  
# from tensorflow.python.platform import gfile

import time
from datetime import timedelta
import math
import random
import numpy as np
import gc
import xml.etree.ElementTree as etxml
import skimage.io
import skimage.transform
# import ssd300
import time

#Define some global and local variables
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string("img_path", "/Users/tarus/OnlyInMac/bully_data/bully_merge/JPEGImages/", "path for train img")
flags.DEFINE_string("xml_path", "/Users/tarus/OnlyInMac/bully_data/bully_merge/Annotations/", "path for train xml file")
def set_parameter():
    #set some superparameters which can reset befor run
    flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')
    # ?% of the data will be used for validation
    flags.DEFINE_float('validation_size', 0.2, 'validation size.')
    flags.DEFINE_integer('img_size', 128, 'image width=image height.')
    flags.DEFINE_integer('iteration_steps', 20000, 'Number of epochs to run trainer.')
    flags.DEFINE_string("train_path", "data_bully/training_data", "path of training data")
    flags.DEFINE_integer('batch_size', 64, 'Number of batch size.')
    flags.DEFINE_string("output_labels", "trained_model/output_labels.txt", "store the labels")
    flags.DEFINE_string("saved_dir", "trained_model", "save trained model")
    flags.DEFINE_string("saved_file", "trained_model/bully_action", "save trained model")

    
    flags.DEFINE_integer('img_depth', 3, 'Number of channels.')
    flags.DEFINE_integer('filter_size', 3, 'filter size.')
    flags.DEFINE_integer('filter_depth1', 32, 'filter depth for conv1.')
    flags.DEFINE_integer('filter_depth2', 32, 'filter depth for conv2.')
    flags.DEFINE_integer('filter_depth3', 64, 'filter depth for conv3.')
    flags.DEFINE_integer('pooling_num', 3, 'Number of pooling')
    flags.DEFINE_integer('flatten_num', pow(FLAGS.img_size//
        pow(2,FLAGS.pooling_num),2)*FLAGS.filter_depth3, 
        'Number of features after flattern')
    print("flatten_num is", FLAGS.flatten_num)
    flags.DEFINE_integer('fc_depth', 128, 'fully connected layer depth.')

    flags.DEFINE_integer('vgg_filter_size1', 3, 'filter size.')
    flags.DEFINE_integer('vgg_filter_size2', 3, 'filter size.')
    flags.DEFINE_integer('vgg_filter_size3', 3, 'filter size.')
    flags.DEFINE_integer('vgg_filter_size4', 3, 'filter size.')
    flags.DEFINE_integer('vgg_filter_depth1', 64, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg_filter_depth2', 128, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg_filter_depth3', 256, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg_filter_depth4', 512, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg_num_hidden1', 4096, 'filter depth for conv3.')
    flags.DEFINE_integer('vgg_num_hidden2', 4096, 'filter depth for conv3.')

    flags.DEFINE_integer('vgg17_filter_size1', 3, 'filter size.')
    flags.DEFINE_integer('vgg17_filter_size2', 3, 'filter size.')
    flags.DEFINE_integer('vgg17_filter_size3', 3, 'filter size.')
    flags.DEFINE_integer('vgg17_filter_size4', 3, 'filter size.')
    flags.DEFINE_integer('vgg17_filter_depth1', 64, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg17_filter_depth2', 64, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg17_filter_depth3', 128, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg17_filter_depth4', 128, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg17_filter_depth5', 256, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg17_filter_depth6', 256, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg17_filter_depth7', 256, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg17_filter_depth8', 512, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg17_filter_depth9', 512, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg17_filter_depth10', 512, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg17_filter_depth11', 512, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg17_filter_depth12', 512, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg17_filter_depth13', 512, 'filter depth for conv1.')
    flags.DEFINE_integer('vgg17_num_hidden1', 4096, 'filter depth for conv3.')
    flags.DEFINE_integer('vgg17_num_hidden2', 4096, 'filter depth for conv3.')

    #parameters for object detection
    # return FLAGS

def main(_):
    #call set_parameter function to intialize the super parameters 
    set_parameter()
    print("img_path is :", FLAGS.img_path)
    print("validation percent is :", FLAGS.validation_size)

	#establish the session variable
    sess = tf.Session()
	# establish a model object
	model = Model(sess, PARA)
	#start training or testing
	getattr(model,flags.FLAGS.action)()


    #*prepare the training dataset & load data
    classes = os.listdir(FLAGS.train_path)
    # print(classes)
    classes.sort()
    # print(classes)
    flags.DEFINE_integer('class_number', len(classes), 'classes number.')
    # class_number= len(classes)
    print("class number is:", FLAGS.class_number)
    
    #save classes/labels
    file_label =open(FLAGS.output_labels,mode='w')
    for field in classes:
        # print(field)
        file_label.write(field)
        file_label.write('\n')
        file_label.flush()
    file_label.close()

    label_lst=[]
    rs = os.path.exists(FLAGS.output_labels)
    if rs==True:
        file_handler =open(FLAGS.output_labels,mode='r')
        contents = file_handler.readlines()
        for name in contents:
            name = name.strip('\n')
            label_lst.append(name)
        file_handler.close()
    print(label_lst)
 
    input_data = loaddata.read_dataset(FLAGS.train_path, FLAGS.img_size, 
                                      classes, FLAGS.validation_size)
    print("******The traning data have been loaded**********")
    # print(input_data.train.cls)
    print("Number of files in Training-set:  \t{}" 
         .format(len(input_data.train.labels)))
    print("Number of files in Validation-set:\t{}" 
        .format(len(input_data.valid.labels)))

    #create a dataflow graph
    mygraph = tf.Graph()
    with mygraph.as_default():
        #1) Define some data & labbel placeholder.
        ## data
        data_placeholder = tf.placeholder(tf.float32, 
            shape=[None, FLAGS.img_size,FLAGS.img_size,FLAGS.img_depth], 
            name='data_placeholder')
        ## labels
        label_placeholder = tf.placeholder(tf.float32, 
            shape=[None, FLAGS.class_number], name='label_placeholder')
        # label_index = tf.argmax(label_placeholder, dimension=1)
        label_index = tf.argmax(label_placeholder, axis=1)
        # print("label_index is :", label_in)

        #2) initilize the weight matrices and bias vectors 
        lainet_coefficients = laimodel.define_coefficients(filter_size=FLAGS.filter_size, 
            img_depth=FLAGS.img_depth, filter_depth1=FLAGS.filter_depth1,
            filter_depth2=FLAGS.filter_depth2,
            filter_depth3=FLAGS.filter_depth3,
            flatten_num=FLAGS.flatten_num, fc_depth=FLAGS.fc_depth,
            class_number=FLAGS.class_number)

        vgg16_coefficients = vgg16model.vgg16_coefficients(
            vgg_filter_size1=FLAGS.vgg_filter_size1,
            vgg_filter_size2=FLAGS.vgg_filter_size2, 
            vgg_filter_size3=FLAGS.vgg_filter_size3,
            vgg_filter_size4=FLAGS.vgg_filter_size4, 
            vgg_filter_depth1=FLAGS.vgg_filter_depth1, 
            vgg_filter_depth2=FLAGS.vgg_filter_depth2, 
            vgg_filter_depth3=FLAGS.vgg_filter_depth3, 
            vgg_filter_depth4=FLAGS.vgg_filter_depth4, 
            vgg_num_hidden1=FLAGS.vgg_num_hidden1,
            vgg_num_hidden2=FLAGS.vgg_num_hidden2,
            img_size=FLAGS.img_size, 
            img_depth=FLAGS.img_depth , 
            class_number=FLAGS.class_number)

        vgg17_coefficients = vgg17model.vgg17_coefficients(
            vgg_filter_size1=FLAGS.vgg17_filter_size1,
            vgg_filter_size2=FLAGS.vgg17_filter_size2, 
            vgg_filter_size3=FLAGS.vgg17_filter_size3,
            vgg_filter_size4=FLAGS.vgg17_filter_size4, 
            vgg_filter_depth1=FLAGS.vgg17_filter_depth1, 
            vgg_filter_depth2=FLAGS.vgg17_filter_depth2, 
            vgg_filter_depth3=FLAGS.vgg17_filter_depth3, 
            vgg_filter_depth4=FLAGS.vgg17_filter_depth4, 
            vgg_filter_depth5=FLAGS.vgg17_filter_depth5, 
            vgg_filter_depth6=FLAGS.vgg17_filter_depth6, 
            vgg_filter_depth7=FLAGS.vgg17_filter_depth7, 
            vgg_filter_depth8=FLAGS.vgg17_filter_depth8, 
            vgg_filter_depth9=FLAGS.vgg17_filter_depth9, 
            vgg_filter_depth10=FLAGS.vgg17_filter_depth10, 
            vgg_filter_depth11=FLAGS.vgg17_filter_depth11, 
            vgg_filter_depth12=FLAGS.vgg17_filter_depth12, 
            vgg_filter_depth13=FLAGS.vgg17_filter_depth13, 
            vgg_num_hidden1=FLAGS.vgg17_num_hidden1,
            vgg_num_hidden2=FLAGS.vgg17_num_hidden2,
            img_size=FLAGS.img_size, 
            img_depth=FLAGS.img_depth , 
            class_number=FLAGS.class_number)

        # coefficients =lainet_coefficients 
       # coefficients =vgg16_coefficients 
        coefficients =vgg17_coefficients 

        #3. construct the CNN model
        # train_net=laimodel.lainet
        #train_net=vgg16model.vgg16net
        train_net=vgg17model.vgg17net
        logits = train_net(data_placeholder, coefficients)

        #4. calculate the cross entropy between the logits and actual labels
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits= 
                        logits, labels=tf.stop_gradient(label_placeholder))
        #tensorflow1.5 compatible
        # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits= 
        #                 logits, labels=label_placeholder)
        cost = tf.reduce_mean(cross_entropy)

        #5. use optimizer to calculate the gradients of the loss function 
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate) \
                    .minimize(cost)

        # Predictions for the training, validation, and test data.
        labels_pred = tf.nn.softmax(logits,name='labels_pred')
        class_pred= tf.argmax(labels_pred, axis=1)
        #class_pred= tf.argmax(labels_pred, dimension=1)
        correct_pred = tf.equal(class_pred, label_index)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    with tf.Session(graph=mygraph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        for i in range(FLAGS.iteration_steps):
            train_data_batch, train_label_batch, _, train_cls_batch = \
                input_data.train.next_batch(FLAGS.batch_size)
            val_data_batch, val_label_batch, _, val_cls_batch = \
                input_data.valid.next_batch(FLAGS.batch_size)
            
            feed_dict_train = {data_placeholder: train_data_batch,
                            label_placeholder: train_label_batch}
            feed_dict_val = {data_placeholder: val_data_batch,
                                label_placeholder: val_label_batch}
            sess.run(optimizer, feed_dict=feed_dict_train)

            if i % int(input_data.train.num_examples/FLAGS.batch_size) == 0: 
                val_loss = sess.run(cost, feed_dict=feed_dict_val)
                epoch = int(i / int(input_data.train.num_examples/FLAGS.batch_size))    
                train_acc = sess.run(accuracy, feed_dict=feed_dict_train)
                val_acc = sess.run(accuracy, feed_dict=feed_dict_val)
                msg = ("Epoch/Step {0}/{1} --- Train Acc:{2:>6.1%}" 
                    "||Val Acc:{3:>6.1%}  ||Val Loss:{4:.3f}")
                print(msg.format(epoch + 1, i, train_acc, val_acc, val_loss))
                if val_acc>95:
                    break

                #save the result
                if not os.path.exists(FLAGS.saved_dir):
                    os.makedirs(FLAGS.saved_dir)
                saver.save(sess, FLAGS.saved_file) 

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
    tf.app.run()
