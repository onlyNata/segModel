# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 15:17:58 2018

@author: LiHongWang
"""

import os
import numpy as np
import cv2 
import tensorflow as tf
from PIL import Image


def read():
    
    dst_dir='D:/dataSet/VOCdevkit/VOC2007/label_SegClass/'
    
    
    for file in os.listdir(dst_dir):
        sample=dst_dir+file
        img=cv2.imread(sample)
        cv2.imshow("",img)
        cv2.waitKey()



import skimage.io as io

#%%

classes = ['background','aeroplane','bicycle','bird','boat',
           'bottle','bus','car','cat','chair','cow','diningtable',
           'dog','horse','motorbike','person','potted plant',
           'sheep','sofa','train','tv/monitor']
# RGB color for each class
colormap = [[0,0,0],[128,0,0],[0,128,0], [128,128,0], [0,0,128],
            [128,0,128],[0,128,128],[128,128,128],[64,0,0],[192,0,0],
            [64,128,0],[192,128,0],[64,0,128],[192,0,128],
            [64,128,128],[192,128,128],[0,64,0],[128,64,0],
            [0,192,0],[128,192,0],[0,64,128]]


cm2lbl = np.zeros(256**3)
for i,cm in enumerate(colormap):
    cm2lbl[(cm[0]*256+cm[1])*256+cm[2]] = i

def image2label(im):
    
    data = im.astype(np.int32)       
    idx = (data[:,:,0]*256+data[:,:,1])*256+data[:,:,2]
    return cm2lbl[idx].astype(np.uint8)


def get_label():
    label_dir='D:/dataSet/VOCdevkit/VOC2012/SegmentationClass/'
    dst_dir='D:/dataSet/VOCdevkit/VOC2012/label_SegClass/'
    labels=[]
    write_name=[]
    name=[]
    
    for file in os.listdir(label_dir):
        name=file.split('.')
        sample=label_dir+name[0]+'.png'
        labels.append(sample)
        write_name.append(dst_dir+name[0]+'.png')

    for i in range(len(labels)):#len(labels)
        label=io.imread(labels[i])
#        io.imshow(label)
        
        img=image2label(label)
        
        io.imsave(write_name[i],img)
        
     




#%%
def voc2012():
    
    image_dir='D:/dataSet/VOCdevkit/VOC2012/JPEGImages/'
    label_dir='D:/dataSet/VOCdevkit/VOC2012/label_SegClass/'
        
    samples=[]
    labels=[]
    
    name=[]
    
    for file in os.listdir(label_dir):
        name=file.split('.')
        
        sample=image_dir+name[0]+'.jpg'
                
        labels.append(label_dir+file)
        samples.append(sample)
    
    temp = np.array([samples, labels])
    temp = temp.transpose()
    np.random.shuffle(temp) 
     
    image_list = temp[:, 0]
    label_list = temp[:, 1]       
            
    image_list=list(image_list)
    label_list=list(label_list)
    
    train=int(len(image_list)*0.7)
    
    tra_img=image_list[0:train]
    tra_lab=label_list[0:train]
    
    val_img=image_list[train:]
    val_lab=label_list[train:]
    
       
    return tra_img,tra_lab,val_img,val_lab
#%%
def voc2007():
    
    image_dir='D:/dataSet/VOCdevkit/VOC2007/JPEGImages/'
    label_dir='D:/dataSet/VOCdevkit/VOC2007/label_SegClass/'
        
    samples=[]
    labels=[]
    
    name=[]
    
    for file in os.listdir(label_dir):
        name=file.split('.')
        
        sample=image_dir+name[0]+'.jpg'
                
        labels.append(label_dir+file)
        samples.append(sample)
    
    temp = np.array([samples, labels])
    temp = temp.transpose()
    np.random.shuffle(temp) 
     
    image_list = temp[:, 0]
    label_list = temp[:, 1]       
            
    image_list=list(image_list)
    label_list=list(label_list)
    
    train=int(len(image_list)*0.7)
    
    tra_img=image_list[0:train]
    tra_lab=label_list[0:train]
    
    val_img=image_list[train:]
    val_lab=label_list[train:]
    
       
    return tra_img,tra_lab,val_img,val_lab    

#%%
#def _remove_colormap(filename):
#      
#    return np.array(Image.open(filename))
#
#def _save_annotation(annotation, filename):
#  
#    pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
#    with tf.gfile.Open(filename, mode='w') as f:
#        pil_image.save(f, 'PNG')
#
#
#def main():
#  # Create the output directory if not exists.
##    if not tf.gfile.IsDirectory(FLAGS.output_dir):
##        tf.gfile.MakeDirs(FLAGS.output_dir)
#    annotation='D:/dataSet/VOCdevkit/VOC2007/SegmentationClass/001585.png'
#
#    raw_annotation = _remove_colormap(annotation)
##    filename = os.path.basename(annotation)[:-4]
#    _save_annotation(raw_annotation,os.path.join('./','16' + '.' + 'png'))
#
#
#if __name__ == '__main__':
#  main()






#
#img=cv2.imread(a[0])
#cv2.imshow("img",img)
##cv2.waitKey()
#
#img1=cv2.imread(b[0])
#cv2.imshow("img1",img1)
#cv2.waitKey()
