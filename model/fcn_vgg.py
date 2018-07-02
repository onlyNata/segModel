# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:22:04 2018

@author: LiHongWang
"""


import tensorflow as tf
from nets import vgg 


slim = tf.contrib.slim

#%%
def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    
    checkpoint_exclude_scopes=["vgg_16/fc8",'deconv1','deconv2','deconv3']
    
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn('/home/Public/seg_project/slim_seg/ckpt/vgg_16.ckpt',
                                          variables_to_restore)


#%%
def fcn_vgg16(images,num_classes,is_training=True):
    

    with slim.arg_scope(vgg.vgg_arg_scope()):
        
        net, end_points = vgg.vgg_16(images, 
                                         num_classes,
                                         spatial_squeeze=False,
                                         fc_conv_padding='SAME',
                                         is_training=is_training)
        pool4=end_points['vgg_16/pool4']
            
        dconv1_out=pool4.get_shape().as_list()
               
        print ('pool4',pool4)    
        deconv1=slim.conv2d_transpose(net,dconv1_out[3],[4,4], stride=2,scope='deconv1')
                
        fu1=tf.add(deconv1,pool4)
 
            
        pool3=end_points['vgg_16/pool3']
        dconv2_out=pool3.get_shape().as_list()
        deconv2=slim.conv2d_transpose(fu1,dconv2_out[3],[4,4], stride=2,scope='deconv2')

        fu2=tf.add(deconv2,pool3)

        logit=slim.conv2d_transpose(fu2,2,[16,16], stride=8,scope='deconv3')
        prediction = tf.argmax(logit, dimension=3)
        
        return logit,tf.expand_dims(prediction, axis=3)
    
    
    
