# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 18:26:25 2018

@author: LiHongWang
"""

import tensorflow as tf
from nets import mobilenet_v1 
from nets.mobilenet import mobilenet_v2

slim = tf.contrib.slim

#%%
def get_init_fn_v1():
    """Returns a function run by the chief worker to warm-start the training."""
    
    checkpoint_exclude_scopes=["MobilenetV1/Conv2d_1c_1x1","deconv32",
                               "deconv16","deconv8"]
    
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

    return slim.assign_from_checkpoint_fn('/home/Public/seg_project/slim_seg/ckpt/mobilenet_v1.ckpt',
                                          variables_to_restore)



#%%
def fcn_mobv1(images,num_classes,is_training=True):
 
        
    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope()):
        _, end_points = mobilenet_v1.mobilenet_v1(images,
                                                   num_classes,
                                                   is_training=False,                                                   
                                                   spatial_squeeze=False)
        
#        for v,k in end_points.items():
#                print('{v}:{k}'.format(v = v, k = k))

        
#        pool4=end_points['resnet_v1_101/pool4']
#            
#        dconv1_out=pool4.get_shape().as_list()
#               
#            
#        deconv1=slim.conv2d_transpose(net,dconv1_out[3],[4,4], stride=2,scope='deconv1')
#                
#        fu1=tf.add(deconv1,pool4)
# 
#            
#        pool3=end_points['resnet_v1_101/pool3']
#        dconv2_out=pool3.get_shape().as_list()
#        deconv2=slim.conv2d_transpose(fu1,dconv2_out[3],[4,4], stride=2,scope='deconv2')
#
#        fu2=tf.add(deconv2,pool3)
        net_7=end_points['Conv2d_13_pointwise']        
        net_14=end_points['Conv2d_11_pointwise']        
        net_28=end_points['Conv2d_5_pointwise']
        
        
       
        up1=slim.conv2d_transpose(net_7,512,[4,4], stride=2,scope='deconv32')
        fu1=tf.add(up1,net_14,name='fu1')
        
        up2=slim.conv2d_transpose(fu1,256,[4,4], stride=2,scope='deconv16')
        fu2=tf.add(up2,net_28,name='fu2')
        
        logit=slim.conv2d_transpose(fu2,num_classes,[16,16], stride=8,scope='deconv8')
        
        prediction = tf.argmax(logit, dimension=3)#, name="prediction")
        
        print('logit',logit)
        
        return logit,tf.expand_dims(prediction, axis=3)
    
#%%
def get_init_fn_v2():
    """Returns a function run by the chief worker to warm-start the training."""
    
    checkpoint_exclude_scopes=["MobilenetV2/Conv2d_1c_1x1","deconv32",
                               "deconv16","deconv8"]
    
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

    return slim.assign_from_checkpoint_fn('/home/Public/seg_project/slim_seg/ckpt/mobilenet_v1.ckpt',
                                          variables_to_restore)



#%%
def fcn_mobv2(images,num_classes,is_training=True):
 
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope()):
        _, end_points= mobilenet_v2.mobilenet(images,num_classes)    
    
        
        
        for v,k in end_points.items():
                print('{v}:{k}'.format(v = v, k = k))

        
#        pool4=end_points['resnet_v1_101/pool4']
#            
#        dconv1_out=pool4.get_shape().as_list()
#               
#            
#        deconv1=slim.conv2d_transpose(net,dconv1_out[3],[4,4], stride=2,scope='deconv1')
#                
#        fu1=tf.add(deconv1,pool4)
# 
#            
#        pool3=end_points['resnet_v1_101/pool3']
#        dconv2_out=pool3.get_shape().as_list()
#        deconv2=slim.conv2d_transpose(fu1,dconv2_out[3],[4,4], stride=2,scope='deconv2')
#
#        fu2=tf.add(deconv2,pool3)
        net=end_points['layer_18']        
#        net_14=end_points['Conv2d_11_pointwise']        
#        net_28=end_points['Conv2d_5_pointwise']
        
        
       
#        up1=slim.conv2d_transpose(net_7,2,[4,4], stride=2,scope='deconv32')
#        fu1=tf.add(up1,net_14,name='fu1')
#        
#        up2=slim.conv2d_transpose(fu1,2,[4,4], stride=2,scope='deconv16')
#        fu2=tf.add(up2,net_28,name='fu2')
        
        logit=slim.conv2d_transpose(net,2,[64,64], stride=32,scope='deconv8')
        
        prediction = tf.argmax(logit, dimension=3)#, name="prediction")
        
        print('logit',logit)
        
        return logit,tf.expand_dims(prediction, axis=3)    