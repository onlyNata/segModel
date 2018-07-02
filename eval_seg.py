# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 16:00:33 2018

@author: LiHongWang
"""

import os
import math
import tensorflow as tf
from model import u_net
from model import fcn_mobile
from data import input_data
import numpy as np

slim = tf.contrib.slim


#%%
def main():
    num_classes=2
    num_samples=29
    batch_size=1
    moving_average =False
    
    checkpoint_path= './logs/fu_arg/'
    eval_dir= './eval/resize/'
    
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)

    tfRecorf_dir= 'D:/dataSet/kitti/road/sub_um_lane_val29.tfrecord'
    

  
    with tf.Graph().as_default():
        tf_global_step = tf.train.get_or_create_global_step()
        tf.logging.set_verbosity(tf.logging.INFO)
        
        with tf.device("/cpu:0"):
#            samples=input_data.get_images_labels(tfRecorf_dir,num_classes, 29,
#                                                 crop_size=[256,256],
#                                                     batch_size=batch_size,
#                                                     is_training=True)
            samples=input_data.eval_images_labels(
                                tfRecorf_dir,num_classes, 29,
                                crop_size=[256,256],
                                batch_size=batch_size,
                                eval_modal='resize')
                
            batch_queue = slim.prefetch_queue.prefetch_queue(samples,
                                                             capacity=128 ) 
        
            tra_batch = batch_queue.dequeue()
               
        labels=tf.squeeze(tra_batch['label'], squeeze_dims=[3])  
#        labels=tra_batch['label']
        logit,pred=u_net.unet_256(tra_batch['image'],num_classes,
#                                                  pool='max_pool',
                                                  is_training=False)
        print('asd_1:',labels)
        pred=tf.squeeze(pred, squeeze_dims=[3])
        pred=tf.cast(pred,tf.int32)
        
        
        print('asd_2:',pred)
        
        if moving_average:
            
            variable_averages = tf.train.ExponentialMovingAverage(0.99, tf_global_step)
            
            variables_to_restore = variable_averages.variables_to_restore(
                                                    slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
            
        else:
            variables_to_restore = slim.get_variables_to_restore()


    # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'mean_iou': slim.metrics.streaming_mean_iou(pred,labels,num_classes)})
  
 
    # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        num_batches = math.ceil(num_samples / float(batch_size))

        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        else:
            checkpoint_path = checkpoint_path

        tf.logging.info('Evaluating %s' % checkpoint_path)
        
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        
        slim.evaluation.evaluate_once(master='',
                                      checkpoint_path=checkpoint_path,
                                      logdir=eval_dir,
                                      num_evals=num_batches,
                                      eval_op=list(names_to_updates.values()),
                                      variables_to_restore=variables_to_restore,
                                      session_config=config)
#%%

if __name__ == '__main__':
    main()