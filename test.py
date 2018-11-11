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
import numpy as np
import math
import h5py

MODEL_SAVE_PATH = '/model/'
IMG_CHANNEL = 4
EPOCH = 100

def PSNR(target, ref, scale, bias):
    target_data = np.array(target)
    target_data = target_data[bias:-bias, bias:-bias]

    ref_data = np.array(ref)
    ref_data = ref_data[bias:-bias, bias:-bias]
    diff = ref_data.astype(np.float32) - target_data.astype(np.float32)
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(scale / rmse)


def Cal_Stokes(img):
    img = np.array(img)
    img0 = img[:, :, 0]
    img45 = img[:, :, 1]
    img90 = img[:, :, 2]
    img135 = img[:, :, 3]

    S0 = (img0.astype(np.float32) + img45.astype(np.float32) +
          img90.astype(np.float32) + img135.astype(np.float32)) * 0.5
    S1 = img0.astype(np.float32) - img90.astype(np.float32)
    S2 = img45.astype(np.float32) - img135.astype(np.float32)
    DoLP = np.sqrt(S1 ** 2 + S2 ** 2) / S0
    return S0, DoLP


def Cal_psnr(target, ref):
    Num = target.shape[0]
    psnrv = np.zeros([Num,6],dtype=np.float32)
    for k in range(Num):
        IM_T = target[k,:,:,:]
        IM_T = np.clip(255 * IM_T, 0, 255)
        IM_R = ref[k,:,:,:]
        for i in range(4):
            imt = IM_T[:, :, i]
            imr = IM_R[:, :, i]
            psnr = PSNR(imt, imr, 255.0, 5)
            psnrv[k,i] = psnr
        S0, DoLP = Cal_Stokes(IM_T)
        S0_ref, DoLP_ref = Cal_Stokes(IM_R)

        psnr_s0 = PSNR(S0, S0_ref, 255.0, 5)
        psnrv[k,i+1]=psnr_s0

        psnr_dolp = PSNR(DoLP, DoLP_ref, 1.0, 5)
        psnrv[k, i+2] = psnr_dolp
    output = psnrv.mean(axis=0)
    return output




def test(test_data,ground_truth = [],Flag = False):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32,[None,
                                       None,
                                       None,
                                       IMG_CHANNEL])
        tmp = ground_truth.astype(np.float32)
        y,_,_,_,_ = model.forward(x,tmp,False)
        saver = tf.train.Saver()
        PSNR_Array = np.zeros((EPOCH+1,6),dtype=np.float32)

        tmp1 = np.clip(255 * test_data, 0, 255)
        psnr = Cal_psnr(tmp1, ground_truth)
        PSNR_Array[0,:] = psnr

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt:
                for i in range(EPOCH):
                    ckpt.model_checkpoint_path = ckpt.all_model_checkpoint_paths[i]
                    saver.restore(sess,ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    output = sess.run(y,feed_dict={x:test_data})
                    psnr = Cal_psnr(output, ground_truth)
                    PSNR_Array[i+1,:] = psnr
                np.savetxt('psnr.txt',PSNR_Array)
            else:
                print("No checkpoint is found.")
                return

if __name__=='__main__':
    data = h5py.File('./ValidationData/validationimage.mat')
    input_data = data["valid_image"]
    output_data = data["GT_valid"]
    input_npy = np.transpose(input_data)
    print(input_npy.shape)
    output_npy = np.transpose(output_data)
    print(output_npy.shape)
    test(input_npy, output_npy,True)

