import numpy as np



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
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    unique_label = np.unique(y_true)
    if y_true.size != y_score.size:
        raise ValueError('y_ture and y_score should has the same length')
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError('The ndim of y_ture and y_score should both be 1')
    if unique_label.size > 2:
        raise ValueError('Unique label of y_true should = 2')
    if unique_label.size == 1:
        raise ValueError('All sample are positive/negative is not allowed')
    if pos_label is None and not np.isin(1, y_true):
        raise ValueError('if your positive label is not 1, you should specify pos_label')
    if y_score.max() > 1 or y_score.min() < 0:
        raise ValueError('y_score should be in [0, 1]')

    # sort
    sorted_idx = np.argsort(y_score)
    y_true = y_true[sorted_idx]
    y_score = y_score[sorted_idx]

    # split into positive and negative samples
    pos_label = 1 if pos_label is None else pos_label
    pos_idx = y_true == pos_label
    y_pos = y_true[pos_idx]
    y_neg = y_true[~pos_idx]
    score_pos = y_score[pos_idx]
    score_neg = y_score[~pos_idx]

    # thresholds (padding)
    thresh = np.r_[0, y_score, 1]

    # fpr, fnr
    pos_hist = np.histogram(score_pos, thresh)[0] / y_pos.size
    neg_hist = np.histogram(score_neg, thresh)[0] / y_neg.size
    fnr = np.r_[np.cumsum(pos_hist), 1]
    fpr = np.r_[1. - np.cumsum(neg_hist), 0]

    # autc
    aufnr = np.trapz(fnr, thresh)
    aufpr = np.trapz(fpr, thresh)
    autc = (aufnr + aufpr) / 2.

    return autc, fpr, fnr, thresh
