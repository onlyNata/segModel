# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 16:57:09 2018

@author: LiHongWang
"""


import tensorflow as tf
from nets import resnet_v2 


slim = tf.contrib.slim

def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    
    checkpoint_exclude_scopes=["resnet_v2_101/logits","resnet_v2_101/fc","deconv32"]
    
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

    return slim.assign_from_checkpoint_fn('/home/Public/seg_project/slim_seg/ckpt/resnet_v2_101.ckpt',
                                          variables_to_restore)



#%%
def fcn_res101(images,num_classes,is_training=True):
 
        
    with slim.arg_scope(resnet_v2.resnet_arg_scope()):
        net, end_points = resnet_v2.resnet_v2_101(images,
                                                   num_classes,
                                                   is_training=False,
                                                   global_pool=False,
                                                   spatial_squeeze=False,
                                                   output_stride=16)
        
#        nn.Conv2D(num_classes, kernel_size=1),
#        nn.Conv2DTranspose(num_classes, kernel_size=64, padding=16,strides=32)
        
       
        
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
        print('net',net)
        logit=slim.conv2d_transpose(net,2,[32,32], stride=16,scope='deconv32')
        prediction = tf.argmax(logit, axis=3)#, name="prediction")
        
        print('logit',logit)
        return logit,tf.expand_dims(prediction, axis=3)
