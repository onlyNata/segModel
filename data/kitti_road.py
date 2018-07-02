# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 14:38:10 2018

@author: LiHongWang
"""

import os
import numpy as np
import cv2 
#%%
def sub_um_road():
    
    image_dir='D:/dataSet/kitti/pre_road/um/sub_road_img/'
    label_dir='D:/dataSet/kitti/pre_road/um/sub_road_mask/'
        
    samples=[]
    labels=[]
    
      
    for file in os.listdir(image_dir):
              
        samples.append(image_dir+file)
        labels.append(label_dir+file)
    
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

def sub_um_lane():
    
    image_dir='D:/dataSet/kitti/pre_road/um/sub_lane_img/'
    label_dir='D:/dataSet/kitti/pre_road/um/sub_lane_mask/'
        
    samples=[]
    labels=[]
    
      
    for file in os.listdir(image_dir):
              
        samples.append(image_dir+file)
        labels.append(label_dir+file)
    
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
def sub_umm():
    
    image_dir='D:/dataSet/kitti/pre_road/umm/sub_umm_img/'
    label_dir='D:/dataSet/kitti/pre_road/umm/sub_umm_mask/'
        
    samples=[]
    labels=[]
    
      
    for file in os.listdir(image_dir):
              
        samples.append(image_dir+file)
        labels.append(label_dir+file)
    
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
def sub_uu():
    
    image_dir='D:/dataSet/kitti/pre_road/uu/sub_uu_img/'
    label_dir='D:/dataSet/kitti/pre_road/uu/sub_uu_mask/'
        
    samples=[]
    labels=[]
    
      
    for file in os.listdir(image_dir):
              
        samples.append(image_dir+file)
        labels.append(label_dir+file)
    
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
def lane_show():
    
    image_dir='D:/dataSet/kitti/road/data_road/training/uu/'
    label_dir='D:/dataSet/kitti/road/data_road/mask/uu/'
      
    sub_image_dir='D:/dataSet/kitti/pre_road/uu/sub_uu_img/'
    sub_label_dir='D:/dataSet/kitti/pre_road/uu/sub_uu_mask/'
#    sub_mix_dir='D:/dataSet/kitti/pre_road/um/sub_lane_mix/'
    
#    if not os.path.exists(sub_mix_dir):
#        os.makedirs(sub_mix_dir)
    
    if not os.path.exists(sub_image_dir):
        os.makedirs(sub_image_dir)
    
    if not os.path.exists(sub_label_dir):
        os.makedirs(sub_label_dir)
        
    samples=[]
    labels=[]
    
    name=[]
    
    for file in os.listdir(image_dir):
        name=file.split('_')
        
        label=label_dir+'uu_road_'+name[1]
                
        samples.append(image_dir+file)
        labels.append(label)
        
    for i in range(len(samples)) :
        image=cv2.imread(samples[i],1)
        img=cv2.imread(labels[i],0)
        shape=img.shape
        img1=cv2.flip(img,1)
        
        w1=0
        w2=0
        h=100
        for j in range(shape[1]):
            if np.max(img[:,j])>0:
                w1=j-128
                if w1<0:
                    w1=0
                break
        for k in range(shape[1]) :            
            if np.max(img1[:,k])>0:                
                w2=(shape[1]-k)+128
                if w2>shape[1]:
                    w2=shape[1]                
                break
          
        img_roi=image[h:shape[0],w1:w2]
        lab_roi=img[h:shape[0],w1:w2]
        
#        mix=cv2.addWeighted(img_roi,0.4,lab_roi,0.6,0)
#        overlapping = cv2.addWeighted(bottom, 0.8, top, 0.2, 0)
      
#        
        img_name=sub_image_dir+str(i)+'.png' 
        lab_name=sub_label_dir+str(i)+'.png'
#        mix_name=sub_mix_dir+str(i)+'.png'
        
#        cv2.imwrite(mix_name,mix)
        cv2.imwrite(img_name,img_roi)
        cv2.imwrite(lab_name,lab_roi)
        
#        cv2.waitKey()

#%%
def um_lane():
    
    image_dir='D:/dataSet/kitti/road/data_road/training/um/'
    label_dir='D:/dataSet/kitti/road/data_road/mask/um_lane/'
        
    samples=[]
    labels=[]
    
    name=[]
    
    for file in os.listdir(image_dir):
        name=file.split('_')
        
        label=label_dir+'um_lane_'+name[1]
                
        samples.append(image_dir+file)
        labels.append(label)
    
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
def um_road():
    
    image_dir='D:/dataSet/kitti/road/data_road/training/um/'
    label_dir='D:/dataSet/kitti/road/data_road/mask/um_road/'
        
    samples=[]
    labels=[]
    
    name=[]
    
    for file in os.listdir(image_dir):
        name=file.split('_')
        
        label=label_dir+'um_road_'+name[1]
                
        samples.append(image_dir+file)
        labels.append(label)
    
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
def umm():
    
    image_dir='D:/dataSet/kitti/road/data_road/training/umm/'
    label_dir='D:/dataSet/kitti/road/data_road/mask/umm/'
        
    samples=[]
    labels=[]
    
    name=[]
    
    for file in os.listdir(image_dir):
        name=file.split('_')
        
        label=label_dir+'umm_road_'+name[1]
                
        samples.append(image_dir+file)
        labels.append(label)
    
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
def uu():
    
    image_dir='D:/dataSet/kitti/road/data_road/training/uu/'
    label_dir='D:/dataSet/kitti/road/data_road/mask/uu/'
        
    samples=[]
    labels=[]
    
    name=[]
    
    for file in os.listdir(image_dir):
        name=file.split('_')
        
        label=label_dir+'uu_road_'+name[1]
                
        samples.append(image_dir+file)
        labels.append(label)
    
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
def get_test(flag='uu'):
    uu_dir='D:/dataSet/kitti/road/data_road/testing/uu/'
    um_dir='D:/dataSet/kitti/road/data_road/testing/um/'
    umm_dir='D:/dataSet/kitti/road/data_road/testing/umm/'
    
    
    
    if flag=='uu':
        image_dir=uu_dir        
    elif flag=='um':  
        image_dir=um_dir
    elif flag=='umm': 
        image_dir=umm_dir
    else:
        raise ValueError('The dataSet not supported.')
        
    samples=[]
    names=[]
    for file in os.listdir(image_dir):
#        name=file.split('.')                 
        samples.append(image_dir+file)
        names.append(file)
    
    return samples,names   
        

#import cv2 
#
#img=cv2.imread(ti[0])
#cv2.imshow("img",img)
#cv2.waitKey()
##
#img1=cv2.imread(vl[0])
#cv2.imshow("img1",img1)
#cv2.waitKey()
