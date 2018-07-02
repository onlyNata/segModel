# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:48:14 2018

@author: LiHongWang
"""

import os.path
import tensorflow as tf

import kitti_road
import vocSeg

#%%
class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""
    def __init__(self, image_format='jpeg', channels=3):
        """Class constructor.
        Args:
            image_format: Image format. Only 'jpeg', 'jpg', or 'png' are supported.
            channels: Image channels.
        """
        with tf.Graph().as_default():
            self._decode_data = tf.placeholder(dtype=tf.string)
            self._image_format = image_format
            self._session = tf.Session()
            if self._image_format in ('jpeg', 'jpg'):
                self._decode = tf.image.decode_jpeg(self._decode_data,
                                            channels=channels)
            elif self._image_format == 'png':
                self._decode = tf.image.decode_png(self._decode_data,
                                           channels=channels)
    def read_image_dims(self, image_data):
        """Reads the image dimensions.
        Args:
            image_data: string of image data.
        Returns:
            image_height and image_width.
        """
        image = self.decode_image(image_data)    
        return image.shape[:2]
    def decode_image(self, image_data):
        """Decodes the image data string.
        Args:
            image_data: string of image data.
        Returns:
            Decoded image data.
        Raises:
            ValueError: Value of image channels not supported.
        """
        image = self._session.run(self._decode,
                              feed_dict={self._decode_data: image_data})
        if len(image.shape) != 3 or image.shape[2] not in (1, 3):
            raise ValueError('The image channels not supported.')

        return image

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
    
    image_reader = ImageReader('jpg', channels=3)
    label_reader = ImageReader('png', channels=1)
    
    for i in range(0, n_images):
        if i%1000==0 or i==n_images-1:
            print('step: %d done !'%i)
        try:    
            image_data = tf.gfile.FastGFile(images[i], 'rb').read()
            height, width = image_reader.read_image_dims(image_data)
               
            seg_data = tf.gfile.FastGFile(labels[i], 'rb').read()
            seg_height, seg_width = label_reader.read_image_dims(seg_data)
            
            if height != seg_height or width != seg_width:
                raise RuntimeError('Shape mismatched between image and label.')

            example = tf.train.Example(features=tf.train.Features(feature={
                        'image/encoded': bytes_feature(image_data),                        
                        'image/format': bytes_feature(b'jpg'),
                        'image/height': int64_feature(height),
                        'image/width': int64_feature(width),
                        'image/channels': int64_feature(3),
                        'image/seg/encoded':(bytes_feature(seg_data)),
                        'image/seg/format': bytes_feature(b'png')}))

            writer.write(example.SerializeToString())
            
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' %e)
            print('Skip it!\n')
    writer.close()
    
    print('Transform done!')  
#%%
def creat_tfrecord_test(images,name,save_dir, save_name):
    '''convert all images and labels to one tfrecord file.
    Args:
        images: list of image directories, string type
        labels: list of labels, int type
        save_dir: the directory to save tfrecord file, e.g.: '/home/folder1/'
        save_name: the name of tfrecord file, string type, e.g.: 'train'
    Return:
        no return
    Note:
        converting needs some time, be patient...
    '''
    
    filename = os.path.join(save_dir, save_name + '.tfrecord')
    n_images = len(images)
    
      
    writer = tf.python_io.TFRecordWriter(filename)
    print('\n Transform start......')
    
    image_reader = ImageReader('png', channels=3)
    
    
    for i in range(0, n_images):
        if i%1000==0 or i==n_images-1:
            print('step: %d done !'%i)
        try:    
            image_data = tf.gfile.FastGFile(images[i], 'rb').read()
            height, width = image_reader.read_image_dims(image_data)
            image_name=bytes(name[i],encoding="utf-8")
           
              
             

            example = tf.train.Example(features=tf.train.Features(feature={
                        'image/encoded': bytes_feature(image_data),
                        'image/filename': bytes_feature(image_name),
                        'image/format': bytes_feature(b'png'),
                        'image/height': int64_feature(height),
                        'image/width': int64_feature(width),
                        'image/channels': int64_feature(3)}))

            writer.write(example.SerializeToString())
            
        except IOError as e:
            print('Could not read:', images[i])
            print('error: %s' %e)
            print('Skip it!\n')
    writer.close()
    
    print('Transform done!')      
    
#%%
    
def main():
    save_dir = 'D:/dataSet/kitti/road/'
    
#    timg,tlab,vimg,vlab=kitti_road.sub_um_road()
    
    timg,tlab,vimg,vlab=vocSeg.voc2012()
    
    print("begin ~")
    
#    image,name=kitti_road.get_test(flag='umm')
#    creat_tfrecord_test(image,name, save_dir, 'umm')
#    creat_tfrecord(timg, tlab, save_dir, 'sub_um_road_tra')
#    creat_tfrecord(vimg, vlab, save_dir, 'sub_um_road_val')
        
    creat_tfrecord(timg, tlab, save_dir, 'voc12_tra')

    creat_tfrecord(vimg, vlab, save_dir, 'voc12_val')
    
    
    
    