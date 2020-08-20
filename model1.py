# -*- coding: utf-8 -*-
"""
Created on Wed Oct  17  2018

This model is built for .

@author: Junchao Zhang
"""
import tensorflow as tf
import numpy as np
import math

Num_Classes = 2


def max_pool(inputs, name):
    with tf.variable_scope(name) as scope:
        value, index = tf.nn.max_pool_with_argmax(tf.to_double(inputs), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                                  padding='SAME', name=scope.name)
    return tf.to_float(value), index, inputs.get_shape().as_list()


def initialization(k, c):
    std = math.sqrt(2. / (k ** 2 * c))
    return tf.truncated_normal_initializer(stddev=std)


def conv_layer(bottom, name, shape, is_training):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('weights', shape, initializer=initialization(shape[0], shape[2]))
        conv = tf.nn.conv2d(bottom, w, [1, 1, 1, 1], padding='SAME')
        b = tf.get_variable('bias', shape[3], initializer=tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, b)
        bias = tf.layers.batch_normalization(bias, training=is_training)
        conv_out = tf.nn.relu(bias)
    return conv_out


def up_sampling(pool, ind, output_shape, batch_size, name=None):
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
           :param batch_size:
    """
    with tf.variable_scope(name):
        pool_ = tf.reshape(pool, [-1])
        batch_range = tf.reshape(tf.range(batch_size, dtype=tf.int64), [tf.shape(pool)[0], 1, 1, 1])
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, [-1, 1])
        ind_ = tf.reshape(ind, [-1, 1])
        ind_ = tf.concat([b, ind_], 1)
        ret = tf.scatter_nd(ind_, pool_, shape=[batch_size, output_shape[1] * output_shape[2] * output_shape[3]])
        # the reason that we use tf.scatter_nd: if we use tf.sparse_tensor_to_dense, then the gradient is None, which will cut off the network.
        # But if we use tf.scatter_nd, the gradients for all the trainable variables will be tensors, instead of None.
        # The usage for tf.scatter_nd is that: create a new tensor by applying sparse UPDATES(which is the pooling value) to individual values of slices within a
        # zero tensor of given shape (FLAT_OUTPUT_SHAPE) according to the indices (ind_). If we ues the orignal code, the only thing we need to change is: changeing
        # from tf.sparse_tensor_to_dense(sparse_tensor) to tf.sparse_add(tf.zeros((output_sahpe)),sparse_tensor) which will give us the gradients!!!
        ret = tf.reshape(ret, [tf.shape(pool)[0], output_shape[1], output_shape[2], output_shape[3]])
        return ret


def forward(inputs, is_training, with_dropout, keep_prob, batch_size, bayes=True):
    with tf.device('gpu'):
        with tf.variable_scope('forward'):
            img1, img2 = tf.split(inputs, 2, 3)
            img1_g = img1[:,:,:,0:1]*0.3 + img1[:,:,:,1:2]*0.59 + img1[:,:,:,2:3]*0.11
            img2_g = img2[:,:,:,0:1]*0.3 + img2[:,:,:,1:2]*0.59 + img2[:,:,:,2:3]*0.11
            inputs = tf.concat([img1_g,img2_g],-1)

            conv1_1 = conv_layer(inputs, 'conv1_1', [3, 3, 2, 64], is_training)
            conv1_2 = conv_layer(conv1_1, 'conv1_2', [3, 3, 64, 64], is_training)
            pool1, pool1_index, shape_1 = max_pool(conv1_2, 'pool1')

            conv2_1 = conv_layer(pool1, 'conv2_1', [3, 3, 64, 128], is_training)
            conv2_2 = conv_layer(conv2_1, 'conv2_2', [3, 3, 128, 128], is_training)
            pool2, pool2_index, shape_2 = max_pool(conv2_2, 'pool2')

            conv3_1 = conv_layer(pool2, 'conv3_1', [3, 3, 128, 256], is_training)
            conv3_2 = conv_layer(conv3_1, 'conv3_2', [3, 3, 256, 256], is_training)
            conv3_3 = conv_layer(conv3_2, 'conv3_3', [3, 3, 256, 256], is_training)
            pool3, pool3_index, shape_3 = max_pool(conv3_3, 'pool3')

            if bayes:
                dropout1 = tf.layers.dropout(pool3, rate=(1 - keep_prob), training=with_dropout, name='dropout1')
                conv4_1 = conv_layer(dropout1, 'conv4_1', [3, 3, 256, 512], is_training)
            else:
                conv4_1 = conv_layer(pool3, 'conv4_1', [3, 3, 256, 512], is_training)
            conv4_2 = conv_layer(conv4_1, 'conv4_2', [3, 3, 512, 512], is_training)
            conv4_3 = conv_layer(conv4_2, 'conv4_3', [3, 3, 512, 512], is_training)
            pool4, pool4_index, shape_4 = max_pool(conv4_3, 'pool4')

            # if bayes:
            #     dropout2 = tf.layers.dropout(pool4, rate=(1 - keep_prob), training=with_dropout, name='dropout2')
            #     conv5_1 = conv_layer(dropout2, 'conv5_1', [3, 3, 512, 512], is_training)
            # else:
            #     conv5_1 = conv_layer(pool4, 'conv5_1', [3, 3, 512, 512], is_training)
            # conv5_2 = conv_layer(conv5_1, 'conv5_2', [3, 3, 512, 512], is_training)
            # conv5_3 = conv_layer(conv5_2, 'conv5_3', [3, 3, 512, 512], is_training)
            # pool5, pool5_index, shape_5 = max_pool(conv5_3, 'pool5')
            #
            # # Decoder Process
            # if bayes:
            #     dropout3 = tf.layers.dropout(pool5, rate=(1 - keep_prob), training=with_dropout, name='dropout3')
            #     deconv5_1 = up_sampling(dropout3, pool5_index, shape_5, batch_size, name='unpool_5')
            # else:
            #     deconv5_1 = up_sampling(pool5, pool5_index, shape_5, batch_size, name='unpool_5')
            # deconv5_2 = conv_layer(deconv5_1, 'deconv5_2', [3, 3, 512, 512], is_training)
            # deconv5_3 = conv_layer(deconv5_2, 'deconv5_3', [3, 3, 512, 512], is_training)
            # deconv5_4 = conv_layer(deconv5_3, 'deconv5_4', [3, 3, 512, 512], is_training)

            if bayes:
                dropout4 = tf.layers.dropout(pool4, rate=(1 - keep_prob), training=with_dropout, name='dropout4')
                deconv4_1 = up_sampling(dropout4, pool4_index, shape_4, batch_size, name='unpool_4')
            else:
                deconv4_1 = up_sampling(pool4, pool4_index, shape_4, batch_size, name='unpool_4')
            deconv4_2 = conv_layer(deconv4_1, 'deconv4_2', [3, 3, 512, 512], is_training)
            deconv4_3 = conv_layer(deconv4_2, 'deconv4_3', [3, 3, 512, 512], is_training)
            deconv4_4 = conv_layer(deconv4_3, 'deconv4_4', [3, 3, 512, 256], is_training)

            if bayes:
                dropout5 = tf.layers.dropout(deconv4_4, rate=(1 - keep_prob), training=with_dropout, name='dropout5')
                deconv3_1 = up_sampling(dropout5, pool3_index, shape_3, batch_size, name='unpool_3')
            else:
                deconv3_1 = up_sampling(deconv4_4, pool3_index, shape_3, batch_size, name='unpool_3')
            deconv3_2 = conv_layer(deconv3_1, 'deconv3_2', [3, 3, 256, 256], is_training)
            deconv3_3 = conv_layer(deconv3_2, 'deconv3_3', [3, 3, 256, 256], is_training)
            deconv3_4 = conv_layer(deconv3_3, 'deconv3_4', [3, 3, 256, 128], is_training)

            if bayes:
                dropout6 = tf.layers.dropout(deconv3_4, rate=(1 - keep_prob), training=with_dropout, name='dropout6')
                deconv2_1 = up_sampling(dropout6, pool2_index, shape_2, batch_size, name='unpool_2')
            else:
                deconv2_1 = up_sampling(deconv3_4, pool2_index, shape_2, batch_size, name='unpool_2')
            deconv2_2 = conv_layer(deconv2_1, 'deconv2_2', [3, 3, 128, 128], is_training)
            deconv2_3 = conv_layer(deconv2_2, 'deconv2_3', [3, 3, 128, 64], is_training)

            deconv1_1 = up_sampling(deconv2_3, pool1_index, shape_1, batch_size, name='unpool_1')
            deconv1_2 = conv_layer(deconv1_1, 'deconv1_2', [3, 3, 64, 64], is_training)
            deconv1_3 = conv_layer(deconv1_2, 'deconv1_3', [3, 3, 64, 64], is_training)

            with tf.variable_scope('conv_classifier') as scope:
                kernel = tf.get_variable('weights', [1, 1, 64, Num_Classes], initializer=initialization(1, 64))
                conv = tf.nn.conv2d(deconv1_3, kernel, [1, 1, 1, 1], padding='VALID')
                bias = tf.get_variable('bias', Num_Classes, initializer=tf.constant_initializer(0.0))
                logits = tf.nn.bias_add(conv, bias)
            weight = tf.nn.softmax(logits)
            Img = []
            for i in range(3):
                img_fused = img1[:,:,:,i:i+1]*weight[:,:,:,0:1] + img2[:,:,:,i:i+1]*weight[:,:,:,1:2]
                Img.append(img_fused)
            Fused_Img = tf.concat(Img,-1)
    return Fused_Img











