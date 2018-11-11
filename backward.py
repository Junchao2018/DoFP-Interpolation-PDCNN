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
import model
import os
import numpy as np
import data_augmentation as DA
import h5py


BATCH_SIZE = 128
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
MAX_EPOCH = 100

MODEL_SAVE_PATH = './model/'
MODEL_NAME = 'DoFP_InterpModel'

IMG_SIZE = (40,40)
IMG_CHANNEL = 4

def backward(train_data,train_labels,train_num):
    with tf.Graph().as_default() as g:
        with tf.name_scope('input'):
            x = tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE,IMG_SIZE[0],IMG_SIZE[1],IMG_CHANNEL])
            y_ = tf.placeholder(dtype=tf.float32,shape=[BATCH_SIZE,IMG_SIZE[0],IMG_SIZE[1],IMG_CHANNEL])
        # forward
        y,output_Stokes,output_Stokes_GT,grad_output,grad_output_GT = model.forward(x,y_,True)
        # learning rate
        global_step = tf.Variable(0,trainable=False)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,
                                                   train_num//BATCH_SIZE,
                                                   LEARNING_RATE_DECAY, staircase=True)
        #loss function
        with tf.name_scope('loss'):
            mse = (1.0/BATCH_SIZE)*tf.nn.l2_loss(tf.subtract(y,y_))
            mse_stokes = (1.0/BATCH_SIZE)*tf.nn.l2_loss(tf.subtract(output_Stokes, output_Stokes_GT))
            mse_grad = (1.0/BATCH_SIZE)*tf.nn.l2_loss(tf.subtract(grad_output, grad_output_GT))
            loss_init = mse + mse_stokes
            loss = mse + mse_stokes + 0.1*mse_grad
        #Optimizer
        # GradientDescent
        with tf.name_scope('train'):
            # Adam
            optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op_init = optimizer.minimize(loss_init, global_step=global_step)
            train_op = optimizer.minimize(loss,global_step=global_step)
        
        # Save model

        variables = tf.contrib.framework.get_variables_to_restore()
        variables_to_resotre = [v for v in variables if v.name.split('/')[0] != 'Gradient']
        saver = tf.train.Saver(variables_to_resotre,max_to_keep=50)
        epoch = 0

        config = tf.ConfigProto()
        with tf.Session(config=config) as sess:
            tf.global_variables_initializer().run()

            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                epoch = int(ckpt.model_checkpoint_path.split('/')[-1].split('_')[-1].split('-')[-2])
            
            while epoch<MAX_EPOCH:
                max_step = train_num//BATCH_SIZE
                listtmp = np.random.permutation(train_num)
                j = 0
                for i in range(max_step):
                    file =open("loss.txt",'a')
                    ind = listtmp[j:j+BATCH_SIZE]
                    j = j + BATCH_SIZE
                    xs = train_data[ind,:,:,:]
                    ys = train_labels[ind,:,:,:]
                    mode = np.random.permutation(8)
                    xs = DA.data_augmentation(xs,mode[0])
                    ys = DA.data_augmentation(ys,mode[0])
                    if epoch <50:
                        _, loss_v, step = sess.run([train_op_init, loss_init, global_step], feed_dict={x: xs, y_: ys})
                        file.write("Epoch: %d  Step is: %d After [ %d / %d ] training,  the batch loss is %g.\n" % (epoch + 1, step, i + 1, max_step, loss_v))
                        file.close()
                    else:
                        _,loss_v,step = sess.run([train_op,loss,global_step],feed_dict={x:xs, y_:ys})
                        file.write("Epoch: %d  Step is: %d After [ %d / %d ] training,  the batch loss is %g.\n" % (epoch + 1, step, i + 1, max_step, loss_v))
                        file.close()
                    #print("Epoch: %d  After [ %d / %d ] training,  the batch loss is %g." % (epoch + 1, i + 1, max_step, loss_v))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME+'_epoch_'+str(epoch+1)),global_step = global_step)
                epoch +=1
    
if __name__=='__main__':
    input_data = np.load("/data/npys/input_image.npy")
    labels = np.load("/data/npys/output_image.npy")
    train_num = input_data.shape[0]
    backward(input_data,labels,train_num)
