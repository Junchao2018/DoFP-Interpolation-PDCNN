# -*- coding: utf-8 -*-
"""
========================================================================
PDCNN for DoFP image interpolation, Version 1.0
Copyright(c) 2018  Junchao Zhang, Jianbo Shao, Haibo Luo, Xiangyue Zhang,
Bin Hui, Zheng Chang, and Rongguang Liang
All Rights Reserved.

----------------------------------------------------------------------
Permission to use, copy, or modify this software and its documentation
for educational and research purposes only and without fee is here
granted, provided that this copyright notice and the original authors'
names appear on all copies and supporting documentation. This program
shall not be used, rewritten, or adapted as the basis of a commercial
software or hardware product without first obtaining permission of the
authors. The authors make no representations about the suitability of
this software for any purpose. It is provided "as is" without express
or implied warranty.
----------------------------------------------------------------------
Please cite the following paper when you use it:

Junchao Zhang, Jianbo Shao, Haibo Luo, Xiangyue Zhang, Bin Hui,
Zheng Chang, and Rongguang Liang, "Learning a convolutional demosaicing
network for microgrid polarimeter imagery," Optics Letters 43(18),
4534-4537 (2018). 
========================================================================
"""
import tensorflow as tf
import numpy as np


template = [[-5,3,3],[-5,0,3],[-5,3,3]]
w0 = np.array([[template]])
w0 = np.transpose(w0,[2,3,1,0])

template1 = [[3,3,3],[-5,0,3],[-5,-5,3]]
w1 = np.array([[template1]])
w1 = np.transpose(w1,[2,3,1,0])

w2 = np.array([[np.rot90(template)]])
w2 = np.transpose(w2,[2,3,1,0])

w3 = np.array([[np.rot90(template1)]])
w3 = np.transpose(w3,[2,3,1,0])

w4 = np.array([[np.rot90(template,k=2)]])
w4 = np.transpose(w4,[2,3,1,0])

w5 = np.array([[np.rot90(template1,k=2)]])
w5 = np.transpose(w5,[2,3,1,0])

w6 = np.array([[np.rot90(template,k=3)]])
w6 = np.transpose(w6,[2,3,1,0])

w7 = np.array([[np.rot90(template1,k=3)]])
w7 = np.transpose(w7,[2,3,1,0])


template2 = [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]
w8 = np.array([[template2]])
w8 = np.transpose(w8,[2,3,1,0])


W = np.concatenate((w0,w1,w2,w3,w4,w5,w6,w7,w8),axis = 3)



def conv_layer(x, filter_num, name, is_training):
    with tf.name_scope('layer_%02d' %name):
        ch = x.get_shape().as_list()
        w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0/(9.0*ch[-1])))
        output = tf.layers.conv2d(x, filter_num, 3, (1,1), padding='SAME',activation=None,kernel_initializer = w_init)
        output = tf.layers.batch_normalization(output, training=is_training)
        output = tf.nn.relu(output)
        return output

def conv2d(x,w,padding = 'SAME'):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding=padding)


def Calculate_gradient(img,w):
    im0, im1, im2, im3 = tf.split(img, 4, 3)
    a = conv2d(im0, w)
    b = conv2d(im1, w)
    c = conv2d(im2, w)
    d = conv2d(im3, w)
    output = tf.concat([a,b,c,d],3)
    return output

def forward(input_x,y_,is_training):
    with tf.device("/gpu:0"):
        output_list = []
        output_Stokes = None
        output_Stokes_GT = None
        grad_output = None
        grad_output_GT = None
        
        output = input_x
        for i in range(6):
            output = conv_layer(output, 64, i, is_training)
            output_list.append(output)


        Mid_layer_output = []
        for j in range(1,6):
            tmp = output_list[-1-j]
            output = tf.concat([tmp,output],3)
            output = conv_layer(output, 64, j+i, is_training)
            Mid_layer_output.append(output)


        output_layer = []
        with tf.name_scope('layer_final'):
            w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0 / (9.0 * 64)))
            for mid in Mid_layer_output:
                output = tf.layers.conv2d(mid, 4, 3, (1,1), padding='SAME',use_bias=True,kernel_initializer=w_init)
                output_layer.append(output)


        output = input_x
        for k in output_layer:
            output = tf.concat([output, k], 3)


        Num = len(output_layer) + 1
        Num = Num * 4
        with tf.name_scope('new_layer'):
            w_init = tf.random_normal_initializer(stddev=np.sqrt(2.0 / (9.0 * Num)))
            output = tf.layers.conv2d(output, 4, 1, (1, 1), padding='VALID', use_bias=False, kernel_initializer=w_init)


        with tf.name_scope('recon_layer'):
            with tf.name_scope('weights'):
                conv_w = tf.Variable(initial_value=tf.constant(
                        [[[[ 0.5,  1.,   0. ],
                           [ 0.5,  0.,   1. ],
                           [ 0.5, -1.,   0. ],
                           [ 0.5,  0.,  -1. ]]]]),trainable=False)
                output_Stokes = tf.nn.conv2d(output,conv_w,strides=[1,1,1,1],padding='VALID')
                output_Stokes_GT = tf.nn.conv2d(y_,conv_w,strides=[1,1,1,1],padding='VALID')


        if is_training:
            with tf.name_scope('Gradient'):
                grad_w = tf.Variable(initial_value=tf.constant(W,dtype=tf.float32),trainable=False)
                grad_output = Calculate_gradient(output,grad_w)
                grad_output_GT = Calculate_gradient(y_,grad_w)



        return output,output_Stokes,output_Stokes_GT,grad_output,grad_output_GT
