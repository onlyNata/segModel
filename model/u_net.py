# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 11:37:15 2018

@author: LiHongWang
"""

import tensorflow as tf
 
slim = tf.contrib.slim




#%%
def diceCoeff(label,logit,axis=None,smooth=1):
    
    if axis is None:
        axis=[1,2]
#    y_true_f = tf.cast(y_true, dtype=tf.float32)
#    y_pred_f = tf.cast(y_pred, dtype=tf.float32)
#    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=axis)
#    dice = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f, axis=axis)
#                                           + tf.reduce_sum(y_pred_f, axis=axis) + smooth)
#    return tf.reduce_mean(dice)
    
    
    label=tf.cast(label,tf.float32)
    logit=tf.cast(logit,tf.float32)
    
#    label = slim.flatten(label)
#    logit = slim.flatten(logit)
    
    
    intersection = tf.reduce_sum(label*logit,axis=axis)
    
    
    score=(2.*intersection+smooth)/(tf.reduce_sum(label,axis=axis)
                                    +tf.reduce_sum(logit,axis=axis)+smooth)
    print('score  ok')
    return  tf.reduce_mean(score)

def diceLoss(label,logit):
    loss = 1- diceCoeff(label, logit)
    
#    loss =tf.maximum(loss,0.0004)
    
    return loss

def pixel_wise_loss(label, logit, pixel_weights=False):
    """Calculates pixel-wise softmax cross entropy loss
    Args:
        pixel_logits (4-D Tensor): (N, H, W, 2)
        gt_pixels (3-D Tensor): Image masks of shape (N, H, W, 2)
        pixel_weights (3-D Tensor) : (N, H, W) Weights for each pixel
    Returns:
        scalar loss : softmax cross-entropy
    """
    print('label',label)
    print('logit',logit)
    
    logits = tf.reshape(logit, [-1, 2])
    labels = tf.reshape(label, [-1, 2])
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    if pixel_weights is not True:
        return tf.reduce_mean(loss)
    else:
        weights = tf.reshape(pixel_weights, [-1])
        return tf.reduce_sum(loss * weights) / tf.reduce_sum(weights)


#%%
def mask_prediction(pixel_logits):
    """
    Args:
        pixel_logits (4-D Tensor): (N, H, W, 2)
    Returns:
        Predicted pixel-wise probabilities (3-D Tensor): (N, H, W)
        Predicted mask (3-D Tensor): (N, H, W)
    """
    probs = tf.nn.softmax(pixel_logits)
    n, h, w, _ = probs.get_shape()
    masks = tf.reshape(probs, [-1, 2])
    masks = tf.argmax(masks, axis=1)
    masks = tf.reshape(masks, [n.value, h.value, w.value])
    probs = tf.slice(probs, [0, 0, 0, 1], [-1, -1, -1, 1])
    probs = tf.squeeze(probs, axis=-1)
    return probs, masks
    
       
#%%
def down_modual(net,num_kernel,name,pool='max_pool',rate=2,is_training=True):
    with tf.name_scope(name):
        net = slim.conv2d(net, num_kernel, [3, 3], scope=name+'/conv1')
        net = tf.nn.relu(net)
        net = slim.batch_norm(net,is_training=is_training,scope=name+'/bn1')
        net = slim.conv2d(net, num_kernel, [3, 3], scope=name+'/conv2')
        net = tf.nn.relu(net)
        net = slim.batch_norm(net,is_training=is_training,scope=name+'/bn2')
        if pool=='max_pool':
            net_pool = slim.max_pool2d(net, [2, 2], scope=name+'/pool')
            net_pool=slim.dropout(net_pool, 0.8,
                                  is_training=is_training,
                                  scope=name+'/dropout')
            print('max pooling !')
        elif pool=='atrous':
            net_pool = slim.conv2d(net, num_kernel, [3, 3],
                                   rate=rate,scope=name+'/atrous')
            net_pool = slim.conv2d(net_pool, num_kernel, [1, 1],
                                   stride=2,scope=name+'/conv_1x1')
            net_pool = tf.nn.relu(net_pool)
            net_pool = slim.batch_norm(net_pool,is_training=is_training,scope=name+'/bn_pool')
            print('atrousg')
        else:
            net_pool = net
            print('No pooling !')
            
        return net_pool,net

def up_modual(net_up,net_fu,num_kernel,name,is_training=True):
    with tf.name_scope(name):
       deconv=slim.conv2d_transpose(net_up,num_kernel,[4,4],stride=2,activation_fn=None,
                                                  normalizer_fn=None,scope=name+'/deconv')
       deconv = tf.nn.relu(deconv)
       deconv = slim.batch_norm(deconv,is_training=is_training,scope=name+'deconv/bn')
       
       net = tf.concat([deconv,net_fu],3 )
            
       net = slim.conv2d(net, num_kernel, [3, 3], scope=name+'/conv1')
       net = tf.nn.relu(net)
       net = slim.batch_norm(net,is_training=is_training,scope=name+'/bn1')            
       net = slim.conv2d(net, num_kernel, [3, 3], scope=name+'/conv2')
       net = tf.nn.relu(net)
       net = slim.batch_norm(net,is_training=is_training,scope=name+'/bn2')            
       net = slim.conv2d(net, num_kernel, [3, 3], scope=name+'/conv3')
       net = tf.nn.relu(net)
       net = slim.batch_norm(net,is_training=is_training,scope=name+'/bn3') 
       
       
       return net
#%%   
def unet_512(images,num_class,is_training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=None,
                        normalizer_fn=None,
                        weights_regularizer=slim.l2_regularizer(0.0004),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d], padding='SAME'): 
            down_0,fu_0=down_modual(images,16,'down_0',is_training=is_training)
            down_1,fu_1=down_modual(down_0,32,'down_1',is_training=is_training)
            down_2,fu_2=down_modual(down_1,64,'down_2',is_training=is_training)
            down_3,fu_3,=down_modual(down_2,128,'down_3',is_training=is_training)
            down_4,fu_4=down_modual(down_3,256,'down_4',is_training=is_training)
            down_5,fu_5=down_modual(down_4,512,'down_5',is_training=is_training)
            
            center,_=down_modual(down_5,1024,'center',is_training=is_training,pool='None')
            
            up_5=up_modual(center,fu_5,512,'up_5',is_training=is_training)
            up_4=up_modual(up_5,fu_4,256,'up_4',is_training=is_training)
            up_3=up_modual(up_4,fu_3,128,'up_3',is_training=is_training)
            up_2=up_modual(up_3,fu_2,64,'up_2',is_training=is_training)
            up_1=up_modual(up_2,fu_1,32,'up_1',is_training=is_training)
            up_0=up_modual(up_1,fu_0,16,'up_1',is_training=is_training)
            
            logit = slim.conv2d(up_0,num_class,[1, 1], scope='logit')
            
            return logit
        
def unet_256(images,num_class,pool='atrous',is_training=True):
    with slim.arg_scope([slim.conv2d],
                        activation_fn=None,
                        normalizer_fn=None,
                        weights_regularizer=slim.l2_regularizer(0.00004),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d,slim.max_pool2d], padding='SAME'):
            
            down_0,fu_0=down_modual(images,32,'down_0',
                                    pool=pool,rate=7,is_training=is_training)
            down_1,fu_1=down_modual(down_0,64,'down_1',
                                    pool=pool,rate=7,is_training=is_training)
            down_2,fu_2=down_modual(down_1,128,'down_2',
                                    pool=pool,rate=5,is_training=is_training)
            down_3,fu_3,=down_modual(down_2,256,'down_3',
                                     pool=pool,rate=3,is_training=is_training)
            down_4,fu_4=down_modual(down_3,512,'down_4',
                                    pool=pool,rate=2,is_training=is_training)
                       
            center,_=down_modual(down_4,1024,'center',
                                 pool='None',is_training=is_training)
            
            up_4=up_modual(center,fu_4,512,'up_4',is_training=is_training)
            up_3=up_modual(up_4,fu_3,256,'up_3',is_training=is_training)
            up_2=up_modual(up_3,fu_2,128,'up_2',is_training=is_training)
            up_1=up_modual(up_2,fu_1,64,'up_1',is_training=is_training)
            up_0=up_modual(up_1,fu_0,32,'up_0',is_training=is_training)
            
            up_0=slim.dropout(up_0, 0.8, is_training=is_training)
            
            logit = slim.conv2d(up_0,num_class,[1, 1], scope='logit')  
#            logit=slim.conv2d_transpose(up_1,num_class,[4,4],stride=2,activation_fn=None,
#                                                  normalizer_fn=None,scope='logit')
            prediction = tf.argmax(logit, axis=3)
            
#            logit = slim.conv2d(up_0,num_class,[1, 1], scope='logit')
            
#            pred= slim.conv2d(up_0,num_class,[1, 1], scope='logit')
#            probs = tf.nn.softmax(logit)
#            n, h, w, _ = probs.get_shape()
##            masks = tf.reshape(probs, [-1, 2])
##            masks = tf.argmax(masks, axis=1)
##            masks = tf.reshape(masks, [n.value, h.value, w.value])
#            probs = tf.slice(probs, [0, 0, 0, 1], [-1, -1, -1, 1])
#            probs = tf.squeeze(probs, axis=-1)
            
            return logit,tf.expand_dims(prediction, axis=3)
#%%
            
           
    
            
            
            
            
            
            
            
            
            
            
    