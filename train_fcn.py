# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 16:34:21 2018

@author: LiHongWang
"""

import os
import tensorflow as tf
from model import fcn_vgg
from model import fcn_mobile
from model import fcn_resnet_v2
from data import input_data


slim = tf.contrib.slim



def main():
    
    num_classes=2
    tfRecorf_dir= 'D:/dataSet/kitti/road/sub_um_lane_tra66.tfrecord'
    
    train_dir = './fm2/'   
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        tf.logging.set_verbosity(tf.logging.INFO)
        
        with tf.device("/cpu:0"):
            samples=input_data.get_images_labels(tfRecorf_dir,num_classes,66,
                                                 crop_size=[224,224],
                                                 batch_size=4)
        
              

            batch_queue = slim.prefetch_queue.prefetch_queue(samples,
                                                             capacity=128 )
 
        
            tra_batch = batch_queue.dequeue()

        logit,prediction=fcn_mobile.fcn_mobv1(tra_batch['image'],num_classes)
#        logit,prediction=fcn_vgg.fcn_vgg16(tra_batch['image'],num_classes)
        
#        logit,prediction=fcn_resnet_v2.fcn_res101(tra_batch['image'],num_classes)
        
        
        
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,
                           labels=tf.squeeze(tra_batch['label'], squeeze_dims=[3]),name="entropy")
    

    
    
        loss = tf.reduce_mean(cross_entropy,name='loss')
        slim.losses.add_loss(loss) 
        
        total_loss = slim.losses.get_total_loss()


#        print("image", tra_batch['image'])
#        print("label", tf.cast(tra_batch['label']*255, tf.uint8))
#        print("prediction", tf.cast(prediction*255, tf.uint8))

    # Create some summaries to visualize the training process:
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.image("image", tra_batch['image'], max_outputs=4)
        tf.summary.image("label", tf.cast(tra_batch['label']*255, tf.uint8), max_outputs=4)
        tf.summary.image("prediction", tf.cast(prediction*255, tf.uint8), max_outputs=4)
        
        
        
  
        lr = tf.train.exponential_decay(0.001,
                                  global_step,
                                  10000,
                                  0.8,
                                  staircase=True)

        #lr = tf.constant(0.001, tf.float32)
        tf.summary.scalar('learning_rate', lr)
        for variable in slim.get_model_variables():
           tf.summary.histogram(variable.op.name, variable)
    
    
    # Specify the optimizer and create the train op:
        optimizer = tf.train.RMSPropOptimizer(lr,0.9)
        train_op = slim.learning.create_train_op(total_loss, optimizer)
    
    # Run the training:
    
        
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

        config=tf.ConfigProto(gpu_options=gpu_options)
        final_loss = slim.learning.train(train_op,
                                     logdir=train_dir,
                                     log_every_n_steps=100,
                                     save_summaries_secs=20,
                                     save_interval_secs=1800,
                                     init_fn=None,#fcn_mobile.get_init_fn(),
                                     session_config=config,
                                     number_of_steps=65000)
          
    print('Finished training. Last batch loss %f' % final_loss)
    
if __name__=='__main__':
    main()