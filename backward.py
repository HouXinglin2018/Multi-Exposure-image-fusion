# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 2019

@author: Junchao Zhang

"""
import tensorflow as tf
import model1 as model
import os
import numpy as np
import h5py
import data_augmentation as DA

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

BATCH_SIZE = 64
LEARNING_RATE_BASE = 0.0001
LEARNING_RATE_DECAY = 0.99
MAX_EPOCH = 10

MODEL_SAVE_PATH = './model-weight2_new/'
MODEL_NAME = 'Fusion'

IMG_SIZE = (60, 60)
IMG_CHANNEL = 3

def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)



def SSIM_LOSS(img1, img2, size=11, sigma=1.5):
    window = fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='SAME')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='SAME')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='SAME') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='SAME') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='SAME') - mu1_mu2

    v1 = 2*mu1_mu2+C1
    v2 = mu1_sq+mu2_sq+C1

    value = (v1*(2.0*sigma12 + C2))/(v2*(sigma1_sq + sigma2_sq + C2))

    # sigma1_sq = sigma1_sq/(mu1_sq+0.00000001)
    v = tf.zeros_like(sigma1_sq) + 0.001
    sigma1 = tf.where(sigma1_sq<0.001,v,sigma1_sq)
    return value, sigma1

def loss_func(y_,y):
    img1,img2 = tf.split(y_,2,3)
    
    Win = [11,9,7,5,3]
    loss = 0
    lossmae = 0
    for s in Win:
        for j in range(3):
            loss1, sigma1 = SSIM_LOSS(img1[:,:,:,j:j+1], y[:,:,:,j:j+1], s)
            loss2, sigma2 = SSIM_LOSS(img2[:,:,:,j:j+1], y[:,:,:,j:j+1], s)
            r = sigma1 / (sigma1 + sigma2 + 0.0000001)
            tmp = 1 - tf.reduce_mean(r * loss1) - tf.reduce_mean((1 - r) * loss2)
            loss = loss + tmp
    loss = loss/15.0
    return loss







def backward(train_data, train_num):
    with tf.Graph().as_default() as g:
        with tf.name_scope('input'):
            x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 6])
            y_ = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, IMG_SIZE[0], IMG_SIZE[1], 6])
            with_dropout = tf.placeholder(tf.bool, name="with_dropout")
            keep_prob = tf.placeholder(tf.float32, shape=None, name="keep_rate")
            batch_size = tf.placeholder(tf.int64, shape=[], name="batch_size")
        # forward
        y = model.forward(x,True,with_dropout,keep_prob,BATCH_SIZE)
        # learning rate
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step,
                                                   train_num // BATCH_SIZE,
                                                   LEARNING_RATE_DECAY, staircase=True)
        # loss function
        with tf.name_scope('loss'):
            loss = loss_func(y_,y)

        # Optimizer
        with tf.name_scope('train'):
            # Adam
            optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step=global_step)

        # Save model
        saver = tf.train.Saver(max_to_keep=50)
        epoch = 0

        config = tf.ConfigProto(log_device_placement=True)
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()

            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
                epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1].split('-')[-2])

            while epoch < MAX_EPOCH:
                max_step = train_num // BATCH_SIZE
                listtmp = np.random.permutation(train_num)
                j = 0
                for i in range(max_step):
                    file = open("loss.txt", 'a')
                    ind = listtmp[j:j + BATCH_SIZE]
                    j = j + BATCH_SIZE
                    xs = train_data[ind, :, :, :]
                    mode = np.random.permutation(8)
                    xs = DA.data_augmentation(xs,mode[0])


                    _, loss_v, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: xs,with_dropout: True,keep_prob: 0.5,batch_size: BATCH_SIZE})
                    file.write("Epoch: %d  Step is: %d After [ %d / %d ] training,  the batch loss is %g.\n" % (
                    epoch + 1, step, i + 1, max_step, loss_v))
                    file.close()
                    # print("Epoch: %d  After [ %d / %d ] training,  the batch loss is %g." % (epoch + 1, i + 1, max_step, loss_v))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME + '_epoch_' + str(epoch + 1)),
                           global_step=global_step)
                epoch += 1


if __name__ == '__main__':
    data = h5py.File('F:\Multi-Exposure Image Fusion\Data\TrainingPatches_Tensorflow\imdb_60_128.mat')
    input_data = data["inputs"]
    input_npy = np.transpose(input_data)

    print(input_npy.shape)
    train_num = input_npy.shape[0]
    backward(input_npy,  train_num)