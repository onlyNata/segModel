# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:19:36 2018

@author: LiHongWang
"""

import tensorflow as tf
from utils import preprocess_utils

#%%
_PROB_OF_FLIP = 0.5

means = [123.68, 116.779, 103.939]


def preprocess_image_and_label(image,
                               label,                               
                               crop_height,
                               crop_width,
                               label_norm=True,
                               ignore_label=0,
                               is_training=True,                              
                               model_variant=None):
    """Preprocesses the image and label.

    Args:
        image: Input image.
        label: Ground truth annotation label.
        crop_height: The height value used to crop the image and label.
        crop_width: The width value used to crop the image and label.
        min_resize_value: Desired size of the smaller image side.
        max_resize_value: Maximum allowed size of the larger image side.
        resize_factor: Resized dimensions are multiple of factor plus one.
        min_scale_factor: Minimum scale factor value.
        max_scale_factor: Maximum scale factor value.
        scale_factor_step_size: The step size from min scale factor to max scale
          factor. The input is randomly scaled based on the value of
          (min_scale_factor, max_scale_factor, scale_factor_step_size).
        ignore_label: The label value which will be ignored for training and
          evaluation.
        is_training: If the preprocessing is used for training or not.
        model_variant: Model variant (string) for choosing how to mean-subtract the
          images. See feature_extractor.network_map for supported model variants.

    Returns:
        original_image: Original image (could be resized).
        processed_image: Preprocessed image.
        label: Preprocessed ground truth segmentation label.

    Raises:
        ValueError: Ground truth label not provided during training.
    """
    if is_training and label is None:
        raise ValueError('During training, label must be provided.')
    if model_variant is None:
        tf.logging.warning('Default mean-subtraction is performed. Please specify '
                       'a model_variant. See feature_extractor.network_map for '
                       'supported model variants.')

  # Keep reference to original image.
    original_image = image

    processed_image = tf.cast(image, tf.float32)
    ###c插入 数据增强的方法，和之前的传统方式一样
    processed_image=tf.image.per_image_standardization(processed_image)
    

    if label is not None:
        label = tf.cast(label, tf.int32)
    
    if label_norm :
        label=tf.div(label,255)
 
    original_image = tf.identity(processed_image)

 
    processed_image.set_shape([None, None, 3])

#   Pad image and label to have dimensions >= [crop_height, crop_width]
    image_shape = tf.shape(processed_image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    target_height = image_height + tf.maximum(crop_height - image_height, 0)
    target_width = image_width + tf.maximum(crop_width - image_width, 0)

#   Pad image with mean pixel value.
    mean_pixel = tf.reshape([127.5, 127.5, 127.5], [1, 1, 3])
    processed_image = preprocess_utils.pad_to_bounding_box(
                processed_image, 0, 0, target_height, target_width, mean_pixel)

#    if label is not None:
    label = preprocess_utils.pad_to_bounding_box(
                label, 0, 0, target_height, target_width, 0)#ignore_label

#   Randomly crop the image and label.
    if is_training and label is not None:
        processed_image, label = preprocess_utils.random_crop(
                [processed_image, label], crop_height, crop_width)
        
 
    processed_image.set_shape([crop_height, crop_width, 3])

    if label is not None:
        label.set_shape([crop_height, crop_width, 1])

    if is_training:
#    # Randomly left-right flip the image and label.
        processed_image, label, _ = preprocess_utils.flip_dim(
                [processed_image, label], 0.5, dim=1)
        
    
#    processed_image=mean_image_subtraction(processed_image, means)
    
    
    return original_image, processed_image, label

#%%
def preprocess_eval_image(image,label,crop_height,crop_width,
                          label_norm=True,ignore_label=0,
                          eval_model='center',model_variant=None):
  
#    if is_training and label is None:
#        raise ValueError('During training, label must be provided.')
#    if model_variant is None:
#        tf.logging.warning('Default mean-subtraction is performed. Please specify '
#                       'a model_variant. See feature_extractor.network_map for '
#                       'supported model variants.')

  # Keep reference to original image.
    original_image = image

    processed_image = tf.cast(image, tf.float32)
    ###c插入 数据增强的方法，和之前的传统方式一样
    processed_image=tf.image.per_image_standardization(processed_image)
    

#    if label is not None:
    label = tf.cast(label, tf.int32)
    if label_norm :
        label=tf.div(label,255)
 
    original_image = tf.identity(processed_image)

    processed_image.set_shape([None, None, 3])

#   Pad image and label to have dimensions >= [crop_height, crop_width]
    image_shape = tf.shape(processed_image)
    image_height = image_shape[0]
    image_width = image_shape[1]

    target_height = image_height + tf.maximum(crop_height - image_height, 0)
    target_width = image_width + tf.maximum(crop_width - image_width, 0)

#   Pad image with mean pixel value.
    mean_pixel = tf.reshape([127.5, 127.5, 127.5], [1, 1, 3])
    processed_image = preprocess_utils.pad_to_bounding_box(
                processed_image, 0, 0, target_height, target_width, mean_pixel)

    if label is not None:
        label = preprocess_utils.pad_to_bounding_box(
                label, 0, 0, target_height, target_width, 0)#ignore_label

#   Randomly crop the image and label.
    if eval_model=='center':   
        
        processed_image, label = preprocess_utils.center_crop(
                    [processed_image, label], crop_height, crop_width)
        
        print('eval_model: center')
    elif eval_model=='resize':    
                
        [processed_image, label] = (preprocess_utils.resize_image(
                                    image=processed_image,
                                    label=label,
                                    height_size=crop_height,
                                    width_size=crop_width,
                                    align_corners=True))
        print('eval_model: resize')
    else :
        
        raise ValueError(' eval_model: wrong !')
        
    processed_image.set_shape([crop_height, crop_width, 3])

    if label is not None:
        label.set_shape([crop_height, crop_width, 1])
    
    
    return original_image, processed_image, label


