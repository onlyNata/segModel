# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 17:38:08 2018

@author: LiHongWang
"""



import os.path
import tensorflow as tf
import skimage.io as io
import kitti_road
import vocSeg


#%%
def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
def creat_tfrecord(images, labels, save_dir, name):
    '''convert all images and labels to one tfrecord file.
    Args:
        images: list of image directories, string type
        labels: list of labels, int type
        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
        name: the name of tfrecord file, string type, e.g.: 'train'
    Return:
        no return
    Note:
        converting needs some time, be patient...
    '''
    
    filename = os.path.join(save_dir, name + '.tfrecord')
    n_images = len(images)
    n_labels = len(labels)
    if n_images==n_labels :
        print('there are %d sample !'%n_labels)
    else:       
        raise RuntimeError('mismatched between image and label.')
      
    writer = tf.python_io.TFRecordWriter(filename)
    print('\n Transform start......')
    
#    image_reader = ImageReader('jpeg', channels=3)
#    label_reader = ImageReader('png', channels=1)
    
    for i in range(0, n_images):
        if i%1000==0 or i==n_images-1:
            print('step: %d done !'%i)
        try:    
#            image_data = tf.gfile.FastGFile(images[i], 'rb').read()
#            height, width = image_reader.read_image_dims(image_data)
               
#            seg_data = tf.gfile.FastGFile(labels[i], 'rb').read()
#            seg_height, seg_width = label_reader.read_image_dims(seg_data)
            
            image_data=io.imread(images[i])
            shape=image_data.shape
            height, width =shape[0],shape[1]
            
            seg_data = io.imread(labels[i],as_gray=True)
            shape=seg_data.shape
            seg_height, seg_width =shape[0],shape[1]
            
            image_data = image_data.tostring()
            seg_data = seg_data.tostring()
            
            if height != seg_height or width != seg_width:
                raise RuntimeError('Shape mismatched between image and label.')

            example = tf.train.Example(features=tf.train.Features(feature={
                        'image/encoded': bytes_feature(image_data),                        
#                        'image/format': bytes_feature(b'jpeg'),
                        'image/height': int64_feature(height),
                        'image/width': int64_feature(width),
                        'image/channels': int64_feature(3),
                        'image/seg/encoded':(bytes_feature(seg_data)),
#                        'image/seg/format': bytes_feature(b'png')
                        }))

            writer.write(example.SerializeToString())
            
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' %e)
            print('Skip it!\n')
    writer.close()
    
    print('Transform done!')  

def main():
    save_dir = 'D:/dataSet/kitti/road/'
    
    timg,tlab,vimg,vlab=kitti_road.um_road()
    
#    timg,tlab,vimg,vlab=vocSeg.voc2012()
    
    print("begin ~")
    return timg,tlab,vimg,vlab

#    creat_tfrecord(timg, tlab, save_dir, 'um_road_tra')
#    creat_tfrecord(vimg, vlab, save_dir, 'um_road_val')
        
#    creat_tfrecord(timg, tlab, save_dir, 'voc12_tra')

#    creat_tfrecord(vimg, vlab, save_dir, 'voc12_val')
    
    
    
    