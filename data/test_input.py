# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 17:21:31 2018

@author: LiHongWang
"""

import os
import tensorflow as tf


slim = tf.contrib.slim

tfexample_decoder = slim.tfexample_decoder

#%%
def preprocess_test_image(image,image_name,crop_height,crop_width):
   
  # Keep reference to original image.
    original_image = image

    processed_image = tf.cast(image, tf.float32)
    ###c插入 数据增强的方法，和之前的传统方式一样
    processed_image=tf.image.per_image_standardization(processed_image)
    

    if image_name is not None:
        image_name = tf.cast(image_name, tf.string)
#        label=tf.div(label,255)
  # Resize image and label to the desired range.
#    if min_resize_value is not None or max_resize_value is not None:
#        [processed_image, label] = (
#            preprocess_utils.resize_to_range(
#                image=processed_image,
#                label=label,
#                min_size=min_resize_value,
#                max_size=max_resize_value,
#                factor=resize_factor,
#                align_corners=True))
        # The `original_image` becomes the resized image.
    original_image = tf.identity(processed_image)

  # Data augmentation by randomly scaling the inputs.
#    scale = preprocess_utils.get_random_scale(
#      min_scale_factor, max_scale_factor, scale_factor_step_size)
#    processed_image, label = preprocess_utils.randomly_scale_image_and_label(
#      processed_image, label, scale)
#    processed_image.set_shape([None, None, 3])

#   Pad image and label to have dimensions >= [crop_height, crop_width]
#    image_shape = tf.shape(processed_image)
#    image_height = image_shape[0]
#    image_width = image_shape[1]

#    target_height = image_height + tf.maximum(crop_height - image_height, 0)
#    target_width = image_width + tf.maximum(crop_width - image_width, 0)
#
##   Pad image with mean pixel value.
#    mean_pixel = tf.reshape([127.5, 127.5, 127.5], [1, 1, 3])
#    processed_image = preprocess_utils.pad_to_bounding_box(
#                processed_image, 0, 0, target_height, target_width, mean_pixel)

    
    processed_image=tf.image.resize_images(processed_image,[256,256])
    
#    processed_image.set_shape([crop_height, crop_width, 3])

    
   
    

  

    return  processed_image,image_name

def get_test_TFrecord(num_classes, dataset_dir,num_samples):
    
    file_pattern = os.path.join(dataset_dir)#, file_pattern % split_name)

  # Specify how the TF-Examples are decoded.
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/filename': tf.FixedLenFeature(
          (), tf.string, default_value=''),        
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='png'),
      'image/height': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/width': tf.FixedLenFeature(
          (), tf.int64, default_value=0),}
     
    items_to_handlers = {
            'image': tfexample_decoder.Image(
                            image_key='image/encoded',
                            format_key='image/format',
                            channels=3),
            'image_name':tfexample_decoder.Tensor('image/filename'),        
            'height': tfexample_decoder.Tensor('image/height'),
            'width': tfexample_decoder.Tensor('image/width'),}

    decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,items_to_handlers)

    return  slim.dataset.Dataset(data_sources=file_pattern,
                                reader=tf.TFRecordReader,
                                decoder=decoder,
                                num_samples=num_samples,
                                items_to_descriptions=None,#ITEMS_TO_DESCRIPTIONS,
                                num_classes=num_classes,
                                labels_to_names=None)    
    
def get_test_images(dataset_dir,num_classes, num_samples,
                      crop_size=[256,256],
                      batch_size=8,
                      is_training=False):
#TODO ASD      
    dataset =get_test_TFrecord(num_classes, dataset_dir,num_samples)
          
    provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    shuffle=is_training,
                    num_readers=4,
                    num_epochs=None,
                    common_queue_capacity=20 * batch_size,
                    common_queue_min=10 *batch_size)
              
    image,image_name,height,width=provider.get(['image','image_name',
                                                'height','width'])
      
   
        
    image,image_name= preprocess_test_image(image,
                                            image_name,
                                            crop_height=crop_size[0],
                                            crop_width=crop_size[1])

    
    image=tf.image.resize_images(image,[256,256])
#    label=tf.reshape(label,[crop_size[0], crop_size[1]])   
    sample = {'image': image,'image_name':image_name,
              'height': height,'width': width}
    
    return tf.train.batch(sample,
                          batch_size=batch_size,
                          num_threads=4,
                          capacity=32 * batch_size,
                          allow_smaller_final_batch=not is_training,
                          dynamic_pad=True)      

    
#%%
    
import numpy as np 
import cv2   

def main():
    data_sources = 'D:/dataSet/kitti/road/uu100.tfrecord'
    num_samples = 100
    
    samples=get_test_images(data_sources,2, num_samples,batch_size=8)
    batch_queue = slim.prefetch_queue.prefetch_queue(samples,capacity=128 ) 
  
    with tf.Session()  as sess:
        with slim.queues.QueueRunners(sess):
            sess.run(tf.local_variables_initializer())
      
#        image_batch, label_batch = read_and_decode(tfrecords_file, batch_size=1)
        
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
      
            try:
                for step in range(1):
#                    batch_queue = slim.prefetch_queue.prefetch_queue(samples,capacity=128 )
                    tra_batch = batch_queue.dequeue()
                    
#                    images = tra_batch['image']
#                    name = tra_batch['image_name']
#                    height=tra_batch['height']
#                    width=tra_batch['width']
#                    
#                    print(images,name)
#                    sess.run(tra_batch)
#                    image,name,height,width=sess.run([images,name,height,width])
#                    
#                  
#                    img=np.reshape(image,(height,width,1))
#    #                
#                    
#                    print(img)
#                    print(name)
    ##                io.imshow(img)
    ##                io.imshow(lab)
    #                cv2.imshow("image",img)
    #                cv2.imshow("label",lab)
    #                cv2.waitKey()
                    
    #            return tra_batch,images,labels
           
              
            except tf.errors.OutOfRangeError:
                print('done!')
            finally:
                coord.request_stop()
            coord.join(threads) 
    