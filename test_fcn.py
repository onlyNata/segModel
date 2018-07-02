# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 16:41:52 2018

@author: LiHongWang
"""

import skimage.io as io
import skimage.transform as transform
import numpy as np
import tensorflow as tf
from model import fcn_resnet_v2
from data import input_data
import cv2

slim = tf.contrib.slim

num_classes=2


def test():
    logs_train_dir = '/home/Public/seg_project/slim_seg/um/lane/u_arg/'
   
    batchSize=1
    
    data_sources = 'D:/dataSet/kitti/road/uu_val30.tfrecord'
    num_samples = 30
    
    samples=input_data.get_images_labels(data_sources,2, num_samples,batch_size=1)
    batch_queue = slim.prefetch_queue.prefetch_queue(samples,capacity=128 )
    tra_batch = batch_queue.dequeue()
#    images = tra_batch['image']
    x = tf.placeholder(tf.float32, shape=[1,224, 224, 3])#batchSize,
          
    _,logit=fcn_resnet_v2.fcn_res101(x,2,is_training=False)
#    logit = tf.nn.softmax(logit)
                              
    saver = tf.train.Saver()
             
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())    
        print("Reading checkpoints...")
        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
        if ckpt and ckpt.model_checkpoint_path:           
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success')
        else:
            print('No checkpoint file found')
            
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        
        try:
            for step in range(1):
                if step%100==0:
                    print("step: %d done !"%step)
                if coord.should_stop():
                    break
#                images,labels = sess.run([image_batch, label_batch])
                
                
                
                
#                    name = tra_batch['image_name']
#                    height=tra_batch['height']
#                    width=tra_batch['width']
#                    
#                    print(images,name)
#                    sess.run(tra_batch)
#                image=sess.run([images])
#                print(image)
#                image=np.reshape(image,[1, 256, 256, 3])
                    
                image=cv2.imread('D:/dataSet/kitti/road/data_road/testing/uu/uu_000000.png')
#                io.imshow(image)
                image=cv2.resize(image,(224,224))
                print('here !')
                image=tf.cast(image,tf.float32)
                image=tf.image.per_image_standardization(image)
                image=sess.run(image)
                image=np.reshape(image,[1, 224, 224, 3])
                print('123 !')
                prediction = sess.run(logit, feed_dict={x: image})
                print('OK !')
#                prediction = np.squeeze(prediction, axis=3)
                for i in range(batchSize):                     
                    img=prediction[i]  
#                    image=transform.resize(image,(375,1242))
#                    img=img.astype(np.uint8)
#                    np.reshape(img,(256,256,2)).astype(np.uint8)
#                                      
##                    name=labels[i]
#                    
#                    io.imshow(img)
#                    io.imsave(dst_dir+'.png',img)
                return prediction,img
                      
        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()           
        coord.join(threads)
    
if __name__=='__main__':
    pred,img=test()
