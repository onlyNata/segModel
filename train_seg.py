# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 09:41:46 2018

@author: LiHongWang
"""

import os
import tensorflow as tf
#from datasets import flowers
from model import u_net
#from preprocessing import inception_preprocessing
from data import input_data


slim = tf.contrib.slim



def main():
    
    num_classes=21
    tfRecorf_dir= 'D:/dataSet/VOCdevkit/voc07_val190.tfrecord'
    
    train_dir = './logs_voc07/'    
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
        
    with tf.Graph().as_default():
#        global_step = tf.contrib.framework.get_or_create_global_step()
        tf.logging.set_verbosity(tf.logging.INFO)
        
        with tf.device("/cpu:0"):
            samples=input_data.get_images_labels(tfRecorf_dir,2, 190)
        
              
            batch_queue = slim.prefetch_queue.prefetch_queue(samples,capacity=128 )
 
        
            tra_batch = batch_queue.dequeue()
            
         
        logit,pred=u_net.unet_256(tra_batch['image'],num_classes,
                                 pool='max_pool',
                                 is_training=True)
        
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit,
                           labels=tf.squeeze(tra_batch['label'], squeeze_dims=[1]),name="entropy")
        
        loss = tf.reduce_mean(cross_entropy,name='cross_entropy_loss')
     
#        slim.losses.add_loss(loss) 
        tf.losses.add_loss(loss)
#        total_loss = slim.losses.get_total_loss()
        total_loss=tf.losses.get_total_loss()

        lr = tf.constant(0.001, tf.float32)
        
        # Create some summaries to visualize the training process:             
        tf.summary.scalar('losses/loss', loss)
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('learning_rate', lr)
        
        
    
    
    # Specify the optimizer and create the train op:
        optimizer = tf.train.MomentumOptimizer(lr,0.9)
        train_op = slim.learning.create_train_op(total_loss, optimizer)
    
    # Run the training:
        final_loss = slim.learning.train(train_op,
                                     logdir=train_dir,
                                     log_every_n_steps=1,
                                     save_summaries_secs=120,
                                     save_interval_secs=600,
                                     init_fn=None,
                                     number_of_steps=120000)
        
  
    print('Finished training. Last batch loss %f' % final_loss)
    
if __name__=='__main__':
    main()

