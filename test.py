# -*- coding: utf-8 -*-
"""
Created on Wed May  9 09:17:21 2018

@author: Junchao Zhang

"""
import tensorflow as tf
import model1 as model
import numpy as np
import h5py
import scipy.io
from scipy.misc import imread, imsave, imfilter
from os import listdir, mkdir
from os.path import join, exists
from skimage import color as skco


L1_norm = lambda x: np.sum(np.abs(x))


MODEL_SAVE_PATH = './model-weight2_new/'
IMG_CHANNEL = 6


Mode = 'color'
Image_Path = 'F:/Multi-Exposure Image Fusion/Data/Data/'
Save_Path = 'Results'





def save_images(datas, save_path, prefix,mode):
    if not exists(save_path):
        mkdir(save_path)
    path = join(save_path, prefix + '.tif')
    # datas.astype('uint8')
    imsave(path, datas)

# def RGB2YCbCr(Img):
#     R = np.array(Img[:,:,0])
#     G = np.array(Img[:, :, 1])
#     B = np.array(Img[:, :, 2])
#
#     Y = 0.299*R + 0.587*G + 0.114*B
#     Cb = 2.0*(1.0-0.114)*(B-Y)
#     Cr = 2.0 * (1.0 - 0.299) * (R - Y)
#     return Y, Cb, Cr
#
# def YCbCr2RGB(Y,Cb,Cr):
#     kr = 0.7133
#     kb = 0.5643
#     R = Y + kr*Cr
#     B = Y + kb*Cb
#     G = (Y-0.299*R-0.114*B)/0.587
#     R = np.expand_dims(R, 2)
#     G = np.expand_dims(G, 2)
#     B = np.expand_dims(B, 2)
#
#     img = np.concatenate([R,G,B],-1)
#     # img = img*255
#     # img = img.astype('uint8')
#     return img


def test(test_data,IMG_SIZE):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32,[1,
                                       IMG_SIZE[0],
                                       IMG_SIZE[1],
                                       IMG_CHANNEL])

        y = model.forward(x,False,with_dropout=False,keep_prob=0.5,batch_size=1,bayes=False)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                ckpt.model_checkpoint_path = ckpt.all_model_checkpoint_paths[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                img = sess.run(y, feed_dict={x: test_data})
                Img = np.array(img)
                return Img
            else:
                print("No checkpoint is found.")
                return

if __name__=='__main__':
    data = h5py.File('F:\Multi-Exposure Image Fusion\Data\TrainingPatches_Tensorflow\Test_List.mat')
    List = data["Test_List"]
    Num = List.shape[0]
    for i in range(Num):
        tmp = List[i,0]
        tmp = int(tmp)
        img1 = Image_Path + 's' + str(tmp) + '/1.tif'
        img2 = Image_Path + 's' + str(tmp) + '/2.tif'
        color_img1 = imread(img1, mode='RGB')
        color_img2 = imread(img2, mode='RGB')
        IMG1 = color_img1
        IMG2 = color_img2
        ######## RGB
        color_img1 = color_img1.astype('float32')
        color_img2 = color_img2.astype('float32')


        img_y1 = color_img1 / 255.0
        img_y2 = color_img2 / 255.0
        dimension = img_y1.shape
        img_y1 = img_y1.reshape([1, dimension[0], dimension[1], dimension[2]])
        img_y2 = img_y2.reshape([1, dimension[0], dimension[1], dimension[2]])
        input_npy = np.concatenate([img_y1, img_y2], -1)
        Img = test(input_npy, dimension)
        Img = np.array(Img)
        Fused_Img = Img.reshape([dimension[0], dimension[1], dimension[2]]) * 255.0

        save_images(Fused_Img, Save_Path, 'fused' + str(tmp), Mode)
        save_images(IMG1, Save_Path,  str(tmp) + '_1', Mode)
        save_images(IMG2, Save_Path, str(tmp) + '_2', Mode)


