# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:53:26 2018

@author: LiHongWang
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import skimage.io as io
import skimage.transform as transform
import numpy as np
import tensorflow as tf
from model import u_net
from data import input_data


slim = tf.contrib.slim

num_classes=2


#%%
def main():
    num_classes=10
    num_samples=8400
    batch_size=100
    moving_average =True
    
    checkpoint_path= './logs/mnist_v19/'
    eval_dir= './eval/mnist/'
    tfRecorf_dir='D:/dataSet/kaggle/mnist/val.tfrecords'
  
    with tf.Graph().as_default():
        tf_global_step = slim.get_or_create_global_step()
        tf.logging.set_verbosity(tf.logging.INFO)
        
        images, labels = input_data.train_decode(tfRecorf_dir,128)

        with slim.arg_scope(vgg_v1.vgg_arg_scope()):
            logits, _ = vgg_v1.vgg_a(images, num_classes, is_training=False)    


        if moving_average:
            variable_averages = tf.train.ExponentialMovingAverage(
                                        0.99, tf_global_step)
            variables_to_restore = variable_averages.variables_to_restore(
                    slim.get_model_variables())
            variables_to_restore[tf_global_step.op.name] = tf_global_step
        else:
            variables_to_restore = slim.get_variables_to_restore()

#        variables_to_restore = variable_averages.variables_to_restore()
#        saver = tf.train.Saver(variables_to_restore)


        predictions = tf.argmax(logits, 1)
        labels = tf.squeeze(labels)

    # Define the metrics:
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
#        'precision': slim.metrics.precision(predictions, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(logits, labels, 3),})

    # Print the summaries to screen.
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        num_batches = math.ceil(num_samples / float(batch_size))

        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        else:
            checkpoint_path = checkpoint_path

        tf.logging.info('Evaluating %s' % checkpoint_path)

        slim.evaluation.evaluate_once(master='',
                                      checkpoint_path=checkpoint_path,
                                      logdir=eval_dir,
                                      num_evals=num_batches,
                                      eval_op=list(names_to_updates.values()),
                                      variables_to_restore=variables_to_restore)
    
    
    
#%%



def try_test():
   
    num_classes=2
    num_samples=29
    batch_size=1
    moving_average =False
    
    checkpoint_path= './logs/'
    eval_dir= './eval/u_arg/'
    
    if not os.path.exists(eval_dir):
        os.makedirs(eval_dir)
        
    x = tf.placeholder(tf.float32, shape=[4,256, 256, 3]) 
    y = tf.placeholder(tf.int64, shape=[4,256, 256, 1]) 
    tfRecorf_dir= 'D:/dataSet/kitti/road/sub_um_lane_val29.tfrecord'
    
    samples=input_data.get_images_labels(tfRecorf_dir,num_classes, 29,
                                                 crop_size=[256,256],
                                                     batch_size=4)
                
    batch_queue = slim.prefetch_queue.prefetch_queue(samples,
                                                             capacity=128 ) 
        
    tra_batch = batch_queue.dequeue()
    image = tra_batch['image']
    label=tra_batch['label']
               
#    labels=tf.squeeze(tra_batch['label'], squeeze_dims=[3])  
#        labels=tra_batch['label']
    logit,pred=u_net.unet_256(x,num_classes,
#                                                  pool='max_pool',
                                                  is_training=False)
    print('here ~~')
#    pred=tf.squeeze(pred, squeeze_dims=[3])
    
    saver = tf.train.Saver()
#    config = tf.ConfigProto()
#    config.gpu_options.allow_growth = True
#    session = tf.Session(config=config, ...)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_path)
        if ckpt and ckpt.model_checkpoint_path:           
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Loading success')
        else:
            print('No checkpoint file found')
        prediction = sess.run(pred, feed_dict={x: [image]}) 
        labels=sess.run([label])
        for i in range(4):
        
#            a=sess.run(offset_height)
#            sess.run(b)
            print('run ~~')
            iou,conf_mat = tf.metrics.mean_iou(prediction[i], labels[i], 2)
            sess.run([conf_mat])
            miou = sess.run([iou])
            print(miou)
    
            
#%%
#def test():
#    logs_train_dir = './logs/'
#   
#    batchSize=1
#    
#    data_sources = 'D:/dataSet/kitti/road/uu_val30.tfrecord'
#    num_samples = 30
#    
#    samples=input_data.get_images_labels(data_sources,2, num_samples,batch_size=1)
#    batch_queue = slim.prefetch_queue.prefetch_queue(samples,capacity=128 )
#    tra_batch = batch_queue.dequeue()
##    images = tra_batch['image']
#    x = tf.placeholder(tf.float32, shape=[1,256, 256, 3])#batchSize,
#          
#    _,logit= u_net.unet_256(x,2,pool='atrous',is_training=False)
##    logit = tf.nn.softmax(logit)
#                              
#    saver = tf.train.Saver()
#             
#    with tf.Session() as sess:
#        sess.run(tf.local_variables_initializer())    
#        print("Reading checkpoints...")
#        ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#        if ckpt and ckpt.model_checkpoint_path:           
#            saver.restore(sess, ckpt.model_checkpoint_path)
#            print('Loading success')
#        else:
#            print('No checkpoint file found')
#            
#        coord = tf.train.Coordinator()
#        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
#        
#        try:
#            for step in range(1):
#                if step%100==0:
#                    print("step: %d done !"%step)
#                if coord.should_stop():
#                    break
##                images,labels = sess.run([image_batch, label_batch])
#                
#                
#                
#                
##                    name = tra_batch['image_name']
##                    height=tra_batch['height']
##                    width=tra_batch['width']
##                    
##                    print(images,name)
##                    sess.run(tra_batch)
##                image=sess.run([images])
##                print(image)
##                image=np.reshape(image,[1, 256, 256, 3])
#                    
#                image=io.imread('D:/dataSet/kitti/road/data_road/testing/um/um_000000.png')
#                io.imshow(image)
#                image=transform.resize(image,(256,256))
#                image=tf.cast(image,tf.float32)
#                image=tf.image.per_image_standardization(image)
#                image=sess.run(image)
#                image=np.reshape(image,[1, 256, 256, 3])
#                prediction = sess.run(logit, feed_dict={x: image})
#               
#                prediction = np.squeeze(prediction, axis=3)
#                for i in range(batchSize):                     
#                    img=prediction[i] *255 
##                    image=transform.resize(image,(375,1242))
#                    img=img.astype(np.uint8)
##                    np.reshape(img,(256,256,2)).astype(np.uint8)
##                                      
###                    name=labels[i]
##                    
##                    io.imshow(img)
##                    io.imsave(dst_dir+'.png',img)
#                return prediction,img
#                      
#        except tf.errors.OutOfRangeError:
#            print('Done training -- epoch limit reached')
#        finally:
#            coord.request_stop()           
#        coord.join(threads)
#    
##if __name__=='__main__':
##    pred,img=test()


