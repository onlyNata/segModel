# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 17:11:03 2018

@author: LiHongWang
"""

from data import preprocessingSeg
import os.path
import tensorflow as tf


slim = tf.contrib.slim

dataset = slim.dataset

tfexample_decoder = slim.tfexample_decoder

#%%
def get_TFrecord(num_classes, dataset_dir,num_samples):
    
    file_pattern = os.path.join(dataset_dir)#, file_pattern % split_name)

  # Specify how the TF-Examples are decoded.
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),      
      'image/format': tf.FixedLenFeature(
          (), tf.string, default_value='jpeg'),
      'image/height': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/width': tf.FixedLenFeature(
          (), tf.int64, default_value=0),
      'image/seg/encoded': tf.FixedLenFeature(
          (), tf.string, default_value=''),
      'image/seg/format': tf.FixedLenFeature(
          (), tf.string, default_value='png'),}
    items_to_handlers = {
            'image': tfexample_decoder.Image(
                            image_key='image/encoded',
                            format_key='image/format',
                            channels=3),                   
            'height': tfexample_decoder.Tensor('image/height'),
            'width': tfexample_decoder.Tensor('image/width'),
            'label': tfexample_decoder.Image(
                                image_key='image/seg/encoded',
                                format_key='image/seg/format',
                                channels=1),}

    decoder = tfexample_decoder.TFExampleDecoder(keys_to_features,items_to_handlers)

    return  slim.dataset.Dataset(data_sources=file_pattern,
                                reader=tf.TFRecordReader,
                                decoder=decoder,
                                num_samples=num_samples,
                                items_to_descriptions=None,#ITEMS_TO_DESCRIPTIONS,
                                num_classes=num_classes,
                                labels_to_names=None)

#%%    
def get_images_labels(dataset_dir,num_classes, num_samples,
                      crop_size=[256,256],
                      batch_size=8,
                      is_training=True):
        
    dataset =get_TFrecord(num_classes, dataset_dir,num_samples)
          
    provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    shuffle=is_training,
                    num_readers=4,
                    num_epochs=None if is_training else 1,
                    common_queue_capacity=20 * batch_size,
                    common_queue_min=10 *batch_size)
              
    image,height,width,label=provider.get(['image','height','width','label'])
      
    if label is not None:
        if label.shape.ndims == 2:
            label = tf.expand_dims(label, 2)
        elif label.shape.ndims == 3 and label.shape.dims[2] == 1:
            pass
        else:
            raise ValueError('Input label shape must be [height, width], or '
                       '[height, width, 1].')

        label.set_shape([None, None, 1])
        
    original_image, image, label = preprocessingSeg.preprocess_image_and_label(
                                      image,label,
                                      crop_height=crop_size[0],
                                      crop_width=crop_size[1],                                      
                                      ignore_label=0,
                                      is_training=is_training)

    

#    label=tf.reshape(label,[crop_size[0], crop_size[1]])   
    sample = {'image': image,'height': height,'width': width,'label':label}
    
    return tf.train.batch(sample,
                          batch_size=batch_size,
                          num_threads=4,
                          capacity=32 * batch_size,
                          allow_smaller_final_batch=not is_training,
                          dynamic_pad=True)     
    
#%%
def eval_images_labels(dataset_dir,num_classes, num_samples,
                      crop_size=[256,256],
                      batch_size=8,
                      eval_modal='center'):
        
    dataset =get_TFrecord(num_classes, dataset_dir,num_samples)
          
    provider = slim.dataset_data_provider.DatasetDataProvider(
                    dataset,
                    shuffle=False,#is_training,
                    num_readers=4,
                    num_epochs=1,
                    common_queue_capacity=20 * batch_size,
                    common_queue_min=10 *batch_size)
              
    image,height,width,label=provider.get(['image','height','width','label'])
      
    if label is not None:
        if label.shape.ndims == 2:
            label = tf.expand_dims(label, 2)
        elif label.shape.ndims == 3 and label.shape.dims[2] == 1:
            pass
        else:
            raise ValueError('Input label shape must be [height, width], or '
                       '[height, width, 1].')

        label.set_shape([None, None, 1])
        
#    original_image, image, label = preprocessingSeg.preprocess_image_and_label(
#                                      image,label,
#                                      crop_height=crop_size[0],
#                                      crop_width=crop_size[1],                                      
#                                      ignore_label=0,
#                                      is_training=is_training)

    original_image, image, label = preprocessingSeg.preprocess_eval_image(
                                image,label,
                                crop_height=crop_size[0],
                                crop_width=crop_size[1],
                                label_norm=True,
                                ignore_label=0,
                                eval_model=eval_modal)#,model_variant=None)

#    label=tf.reshape(label,[crop_size[0], crop_size[1]])   
    sample = {'image': image,'height': height,'width': width,'label':label}
    
    return tf.train.batch(sample,
                          batch_size=batch_size,
                          num_threads=4,
                          capacity=32 * batch_size,
                          allow_smaller_final_batch=False,#if true ,error !
                          dynamic_pad=True)     
        
    
#%%    
#import numpy as np 
#import cv2   
#
#def main():
#    data_sources = 'D:/dataSet/kitti/road/uu_val30.tfrecord'
#    num_samples = 30
#    
#    samples=get_images_labels(data_sources,2, num_samples,batch_size=8)
#    batch_queue = slim.prefetch_queue.prefetch_queue(samples,capacity=128 ) 
#  
#    with tf.Session()  as sess:
#        with slim.queues.QueueRunners(sess):
#            sess.run(tf.local_variables_initializer())
#      
##        image_batch, label_batch = read_and_decode(tfrecords_file, batch_size=1)
#        
#            coord = tf.train.Coordinator()
#            threads = tf.train.start_queue_runners(coord=coord)
#      
#            try:
#                for step in range(1):
##                    batch_queue = slim.prefetch_queue.prefetch_queue(samples,capacity=128 )
#                    tra_batch = batch_queue.dequeue()
                    
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
           
              
#            except tf.errors.OutOfRangeError:
#                print('done!')
#            finally:
#                coord.request_stop()
#            coord.join(threads) 

    
    
    
    
    
    