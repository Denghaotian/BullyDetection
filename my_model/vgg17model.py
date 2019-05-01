import tensorflow as tf

# def flatten_the_layer(array):
#     shape = array.get_shape().as_list()
#     return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])

def flatten_the_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer


def vgg17_coefficients(vgg_filter_size1=3 , vgg_filter_size2=3 , vgg_filter_size3=3 , vgg_filter_size4=3 , 
                       vgg_filter_depth1=64 , vgg_filter_depth2=64 , 
                       vgg_filter_depth3=128 , vgg_filter_depth4=128 ,
                       vgg_filter_depth5=256 , vgg_filter_depth6=256 , vgg_filter_depth7=256 , 
                       vgg_filter_depth8=512 , vgg_filter_depth9=512 , vgg_filter_depth10=512 , 
                       vgg_filter_depth11=512 , vgg_filter_depth12=512, vgg_filter_depth13=512 , 
                       vgg_num_hidden1=4096 , vgg_num_hidden2=4096 ,
                       image_size=128, img_depth=3 , class_number=10):
    
    w1 = tf.Variable(tf.truncated_normal([vgg_filter_size1, vgg_filter_size1, img_depth, vgg_filter_depth1],  stddev=0.05))
    # b1 = tf.Variable(tf.zeros([vgg_filter_depth1]))
    b1 = tf.Variable(tf.constant(0.05, shape=[vgg_filter_depth1]))
    w2 = tf.Variable(tf.truncated_normal([vgg_filter_size1, vgg_filter_size1, vgg_filter_depth1, vgg_filter_depth2],  stddev=0.05))
    b2 = tf.Variable(tf.constant(0.05, shape=[vgg_filter_depth2]))

    w3 = tf.Variable(tf.truncated_normal([vgg_filter_size2, vgg_filter_size2, vgg_filter_depth2, vgg_filter_depth3],  stddev=0.05))
    b3 = tf.Variable(tf.constant(0.05, shape = [vgg_filter_depth3]))
    w4 = tf.Variable(tf.truncated_normal([vgg_filter_size2, vgg_filter_size2, vgg_filter_depth3, vgg_filter_depth4],  stddev=0.05))
    b4 = tf.Variable(tf.constant(0.05, shape = [vgg_filter_depth4]))
    
    w5 = tf.Variable(tf.truncated_normal([vgg_filter_size3, vgg_filter_size3, vgg_filter_depth4, vgg_filter_depth5],  stddev=0.05))
    b5 = tf.Variable(tf.constant(0.05, shape = [vgg_filter_depth5]))
    w6 = tf.Variable(tf.truncated_normal([vgg_filter_size3, vgg_filter_size3, vgg_filter_depth5, vgg_filter_depth6],  stddev=0.05))
    b6 = tf.Variable(tf.constant(0.05, shape = [vgg_filter_depth6]))
    w7 = tf.Variable(tf.truncated_normal([vgg_filter_size3, vgg_filter_size3, vgg_filter_depth6, vgg_filter_depth7],  stddev=0.05))
    b7 = tf.Variable(tf.constant(0.05, shape=[vgg_filter_depth7]))

    w8 = tf.Variable(tf.truncated_normal([vgg_filter_size4, vgg_filter_size4, vgg_filter_depth7, vgg_filter_depth8],  stddev=0.05))
    b8 = tf.Variable(tf.constant(0.05, shape = [vgg_filter_depth8]))
    w9 = tf.Variable(tf.truncated_normal([vgg_filter_size4, vgg_filter_size4, vgg_filter_depth8, vgg_filter_depth9],  stddev=0.05))
    b9 = tf.Variable(tf.constant(0.05, shape = [vgg_filter_depth9]))
    w10 = tf.Variable(tf.truncated_normal([vgg_filter_size4, vgg_filter_size4, vgg_filter_depth9, vgg_filter_depth10],  stddev=0.05))
    b10 = tf.Variable(tf.constant(0.05, shape = [vgg_filter_depth10]))
    
    w11 = tf.Variable(tf.truncated_normal([vgg_filter_size4, vgg_filter_size4, vgg_filter_depth10, vgg_filter_depth11],  stddev=0.05))
    b11 = tf.Variable(tf.constant(0.05, shape = [vgg_filter_depth11]))
    w12 = tf.Variable(tf.truncated_normal([vgg_filter_size4, vgg_filter_size4, vgg_filter_depth11, vgg_filter_depth12],  stddev=0.05))
    b12 = tf.Variable(tf.constant(0.05, shape=[vgg_filter_depth12]))
    w13 = tf.Variable(tf.truncated_normal([vgg_filter_size4, vgg_filter_size4, vgg_filter_depth12, vgg_filter_depth13],  stddev=0.05))
    b13 = tf.Variable(tf.constant(0.05, shape = [vgg_filter_depth13]))
    
    polling_layer_num = 5
    flatten_num = (image_size // (2**polling_layer_num))*(image_size // (2**polling_layer_num))*vgg_filter_depth13 
    print("flatten numb is ", flatten_num)

    w14 = tf.Variable(tf.truncated_normal([flatten_num, vgg_num_hidden1],  stddev=0.05))
    b14 = tf.Variable(tf.constant(0.05, shape = [vgg_num_hidden1]))
    
    w15 = tf.Variable(tf.truncated_normal([vgg_num_hidden1, vgg_num_hidden2],  stddev=0.05))
    b15 = tf.Variable(tf.constant(0.05, shape = [vgg_num_hidden2]))
   
    w16 = tf.Variable(tf.truncated_normal([vgg_num_hidden2, class_number],  stddev=0.05))
    b16 = tf.Variable(tf.constant(0.05, shape = [class_number]))
    coefficients = {
        'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'w7': w7, 'w8': w8, 'w9': w9, 'w10': w10, 
        'w11': w11, 'w12': w12, 'w13': w13, 'w14': w14, 'w15': w15, 'w16': w16, 
        'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5, 'b6': b6, 'b7': b7, 'b8': b8, 'b9': b9, 'b10': b10, 
        'b11': b11, 'b12': b12, 'b13': b13, 'b14': b14, 'b15': b15, 'b16': b16
    }
    return coefficients
    
def vgg17net(input_data, coefficients):
    layer1_conv = tf.nn.conv2d(input_data, coefficients['w1'], [1, 1, 1, 1], padding='SAME')
    layer1_actv = tf.nn.relu(layer1_conv + coefficients['b1'])

    layer2_conv = tf.nn.conv2d(layer1_actv, coefficients['w2'], [1, 1, 1, 1], padding='SAME')
    layer2_actv = tf.nn.relu(layer2_conv + coefficients['b2'])
    layer2_pool = tf.nn.max_pool(layer2_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer3_conv = tf.nn.conv2d(layer2_pool, coefficients['w3'], [1, 1, 1, 1], padding='SAME')
    layer3_actv = tf.nn.relu(layer3_conv + coefficients['b3'])   

    layer4_conv = tf.nn.conv2d(layer3_actv, coefficients['w4'], [1, 1, 1, 1], padding='SAME')
    layer4_actv = tf.nn.relu(layer4_conv + coefficients['b4'])
    layer4_pool = tf.nn.max_pool(layer4_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer5_conv = tf.nn.conv2d(layer4_pool, coefficients['w5'], [1, 1, 1, 1], padding='SAME')
    layer5_actv = tf.nn.relu(layer5_conv + coefficients['b5'])

    layer6_conv = tf.nn.conv2d(layer5_actv, coefficients['w6'], [1, 1, 1, 1], padding='SAME')
    layer6_actv = tf.nn.relu(layer6_conv + coefficients['b6'])

    layer7_conv = tf.nn.conv2d(layer6_actv, coefficients['w7'], [1, 1, 1, 1], padding='SAME')
    layer7_actv = tf.nn.relu(layer7_conv + coefficients['b7'])
    layer7_pool = tf.nn.max_pool(layer7_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer8_conv = tf.nn.conv2d(layer7_pool, coefficients['w8'], [1, 1, 1, 1], padding='SAME')
    layer8_actv = tf.nn.relu(layer8_conv + coefficients['b8'])

    layer9_conv = tf.nn.conv2d(layer8_actv, coefficients['w9'], [1, 1, 1, 1], padding='SAME')
    layer9_actv = tf.nn.relu(layer9_conv + coefficients['b9'])

    layer10_conv = tf.nn.conv2d(layer9_actv, coefficients['w10'], [1, 1, 1, 1], padding='SAME')
    layer10_actv = tf.nn.relu(layer10_conv + coefficients['b10'])
    layer10_pool = tf.nn.max_pool(layer10_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    layer11_conv = tf.nn.conv2d(layer10_pool, coefficients['w11'], [1, 1, 1, 1], padding='SAME')
    layer11_actv = tf.nn.relu(layer11_conv + coefficients['b11'])

    layer12_conv = tf.nn.conv2d(layer11_actv, coefficients['w12'], [1, 1, 1, 1], padding='SAME')
    layer12_actv = tf.nn.relu(layer12_conv + coefficients['b12'])

    layer13_conv = tf.nn.conv2d(layer12_actv, coefficients['w13'], [1, 1, 1, 1], padding='SAME')
    layer13_actv = tf.nn.relu(layer13_conv + coefficients['b13'])
    layer13_pool = tf.nn.max_pool(layer13_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
    
    flat_layer  = flatten_the_layer(layer13_pool)
    flatten_num2 = flat_layer.get_shape()[1:4].num_elements()
    print("flatten_num2 is: ", flatten_num2)

    layer14_fccd = tf.matmul(flat_layer, coefficients['w14']) + coefficients['b14']
    layer14_actv = tf.nn.relu(layer14_fccd)
    # layer14_drop = tf.nn.dropout(layer14_actv, 0.5)
    
    layer15_fccd = tf.matmul(layer14_actv, coefficients['w15']) + coefficients['b15']
    layer15_actv = tf.nn.relu(layer15_fccd)
    # layer15_drop = tf.nn.dropout(layer15_actv, 0.5)
    
    logits = tf.matmul(layer15_actv, coefficients['w16']) + coefficients['b16']
    return logits
