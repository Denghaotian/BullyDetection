import tensorflow as tf

# def flatten_the_layer(array):
#     shape = array.get_shape().as_list()
#     return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])

def flatten_the_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer

def define_coefficients(filter_size, img_depth, filter_depth1, filter_depth2,
                  filter_depth3, flatten_num, fc_depth, class_number):
    w1 = tf.Variable(tf.truncated_normal([filter_size, filter_size, img_depth, filter_depth1], stddev=0.05))
    b1 = tf.Variable(tf.constant(0.05, shape=[filter_depth1]))
    w2 = tf.Variable(tf.truncated_normal([filter_size, filter_size, filter_depth1, filter_depth2], stddev=0.05))
    b2 = tf.Variable(tf.constant(0.05, shape=[filter_depth2]))
    w3 = tf.Variable(tf.truncated_normal([filter_size, filter_size, filter_depth2, filter_depth3], stddev=0.05))
    b3 = tf.Variable(tf.constant(0.05, shape=[filter_depth3]))
    w4 = tf.Variable(tf.truncated_normal([flatten_num, fc_depth], stddev=0.05))
    b4 = tf.Variable(tf.constant(0.05, shape=[fc_depth]))
    w5 = tf.Variable(tf.truncated_normal([fc_depth,class_number], stddev=0.05))
    b5 = tf.Variable(tf.constant(0.05, shape=[class_number]))
    coefficients = {
        'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5,
        'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5
    }
    return coefficients

def lainet(input_data, coefficients):
    layer1_conv = tf.nn.conv2d(input=input_data, filter=coefficients['w1'], strides=[1, 1, 1, 1], padding='SAME')
    layer1_actv = tf.nn.relu(layer1_conv + coefficients['b1'])
    layer1_pool = tf.nn.max_pool(value=layer1_actv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    layer2_conv = tf.nn.conv2d(input=layer1_pool, filter=coefficients['w2'], strides=[1, 1, 1, 1], padding='SAME')
    layer2_actv = tf.nn.relu(layer2_conv + coefficients['b2'])
    layer2_pool = tf.nn.max_pool(value=layer2_actv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    layer3_conv = tf.nn.conv2d(input=layer2_pool, filter=coefficients['w3'], strides=[1, 1, 1, 1], padding='SAME')
    layer3_actv = tf.nn.relu(layer3_conv + coefficients['b3'])
    layer3_pool = tf.nn.max_pool(value=layer3_actv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    layer_flat = flatten_the_layer(layer3_pool)
    layer4_fccd = tf.matmul(layer_flat, coefficients['w4']) + coefficients['b4']
    layer4_actv = tf.nn.relu(layer4_fccd)
    logits = tf.matmul(layer4_actv, coefficients['w5']) + coefficients['b5']
    
    return logits