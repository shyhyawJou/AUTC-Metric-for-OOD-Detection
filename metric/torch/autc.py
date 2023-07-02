import torch
from torch import Tensor



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
    if not isinstance(y_true, Tensor) or not isinstance(y_score, Tensor):
        raise TypeError('y_true and y_score should be torch.Tensor object')
    unique_label = torch.unique(y_true)
    if y_true.device != y_score.device:
        raise ValueError('y_ture and y_score should be on the same device')
    if y_true.numel() != y_score.numel():
        raise ValueError('y_ture and y_score should has the same length')
    if y_true.ndim != 1 or y_score.ndim != 1:
        raise ValueError('The ndim of y_ture and y_score should both be 1')
    if unique_label.numel() > 2:
        raise ValueError('Unique label of y_true should = 2')
    if unique_label.numel() == 1:
        raise ValueError('All sample are positive/negative is not allowed')
    if pos_label is None and not torch.isin(1, y_true):
        raise ValueError('if your positive label is not 1, you should specify pos_label')
    if y_score.max() > 1 or y_score.min() < 0:
        raise ValueError('y_score should be in [0, 1]')

    # sort
    sorted_idx = torch.argsort(y_score)
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
    zero = torch.zeros(1, dtype=y_score.dtype, device=y_score.device)
    one = torch.ones(1, dtype=y_score.dtype, device=y_score.device)
    thresh = torch.cat((zero, y_score, one))

    # fpr, fnr
    pos_hist = torch.histogram(score_pos, thresh)[0] / y_pos.numel()
    neg_hist = torch.histogram(score_neg, thresh)[0] / y_neg.numel()
    fnr = torch.cat((torch.cumsum(pos_hist, dim=0), one))
    fpr = torch.cat((1. - torch.cumsum(neg_hist, dim=0), zero))

    # autc
    aufnr = torch.trapz(fnr, thresh)
    aufpr = torch.trapz(fpr, thresh)
    autc = (aufnr + aufpr) / 2.

    return autc, fpr, fnr, thresh
