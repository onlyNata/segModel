# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 14:12:48 2018

@author: LiHongWang
"""

import tensorflow as tf 


def dice_coe(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):
    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)

    References
    -----------
    - `Wiki-Dice <https://en.wikipedia.org/wiki/Sørensen–Dice_coefficient>`__

    """
    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")
    dice = (2. * inse + smooth) / (l + r + smooth)
    dice = tf.reduce_mean(dice)
    return dice


#def dice_coeff(y_true, y_pred):
#    smooth = 1.
#    y_true_f = K.flatten(y_true)
#    y_pred_f = K.flatten(y_pred)
#    intersection = K.sum(y_true_f * y_pred_f)
#    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
#    return score
#
#
#def dice_loss(y_true, y_pred):
#    loss = 1 - dice_coeff(y_true, y_pred)
#    return loss
#
#
#def bce_dice_loss(y_true, y_pred):
#    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
#    return loss


def diceCoeff(label,logit,smooth=1.0):
    
    intersection = tf.sum(label * logit)
    
    score=(2.*intersection+smooth)/(tf.sum(label)+tf.sum(logit)+smooth)
    
    return score

def diceLoss(label,logit):
    loss = 1 - diceCoeff(label, logit)
    return loss
    
def bceDiceLoss(label, logit):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logit,label) \
            +diceLoss(label,logit)
    return loss    
    