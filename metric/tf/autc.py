import numpy as np
import tensorflow as tf
from tensorflow import Tensor



def autc(y_true, y_score, pos_label=None):
    """
    # Area under Threshold Curve
    param
    -----
        - y_true:  label of sample, when pos_label is None, positive label should be 1 
        - y_score: positive confience score of each sample
        - pos_label: if your positive label is not 1, you should specify this
        - return_fpr_fnr: if true, will return autc, fpr, fnr and threshold
    
    return
    ------
        (autc, fpr, fnr, thresh)
    """
    # check
    #if not isinstance(y_true, Tensor) or not isinstance(y_score, Tensor):
    #    raise TypeError('y_true and y_score should be tf.Tensor object')
    y_true, y_score = tf.constant(y_true), tf.constant(y_score, tf.double)
    unique_label = tf.unique(y_true)[0]
    if tf.size(y_true) != tf.size(y_score):
        raise ValueError('y_ture and y_score should has the same length')
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError('The ndim of y_ture and y_score should both be 1')
    if tf.size(unique_label) > 2:
        raise ValueError('Unique label of y_true should = 2')
    if tf.size(unique_label) == 1:
        raise ValueError('All sample are positive/negative is not allowed')
    if pos_label is None and not tf.reduce_any(y_true == 1):
        raise ValueError('if your positive label is not 1, you should specify pos_label')
    if tf.reduce_max(y_score) > 1 or tf.reduce_min(y_score) < 0:
        raise ValueError('y_score should be in [0, 1]')

    # sort
    sorted_idx = tf.argsort(y_score)
    y_true = tf.gather(y_true, sorted_idx)
    y_score = tf.gather(y_score, sorted_idx)

    # split into positive and negative samples
    pos_label = 1 if pos_label is None else pos_label
    pos_idx = y_true == pos_label
    y_pos = y_true[pos_idx]
    y_neg = y_true[~pos_idx]
    score_pos = y_score[pos_idx]
    score_neg = y_score[~pos_idx]

    # thresholds (padding)
    zero = tf.zeros(1, y_score.dtype)
    one = tf.ones(1, y_score.dtype)
    thresh = tf.concat((zero, y_score, one), axis=0)

    # fpr, fnr
    pos_hist = np.histogram(score_pos, thresh)[0] / tf.size(y_pos)
    neg_hist = np.histogram(score_neg, thresh)[0] / tf.size(y_neg)
    fnr = tf.concat((tf.cumsum(pos_hist), one), axis=0)
    fpr = tf.concat((1. - tf.cumsum(neg_hist), zero), axis=0)

    # autc
    aufnr = np.trapz(fnr, thresh)
    aufpr = np.trapz(fpr, thresh)
    autc = tf.constant((aufnr + aufpr) / 2.)

    return autc, fpr, fnr, thresh