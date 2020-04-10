# -*- coding: utf-8 -*-
# @Time    : 2019/7/18 上午9:54
# @Author  : Lart Pang
# @FileName: metric.py
# @Project : Paper_Code
# @GitHub  : https://github.com/lartpang

import numpy as np


def cal_pr_mae_meanf(prediction, gt):
    assert prediction.dtype == np.uint8
    assert gt.dtype == np.uint8
    assert prediction.shape == gt.shape
    
    # 确保图片和真值相同 ##################################################
    # if prediction.shape != gt.shape:
    #     prediction = Image.fromarray(prediction).convert('L')
    #     gt_temp = Image.fromarray(gt).convert('L')
    #     prediction = prediction.resize(gt_temp.size)
    #     prediction = np.array(prediction)
    
    # 获得需要的预测图和二值真值 ###########################################
    if prediction.max() == prediction.min():
        prediction = prediction / 255
    else:
        prediction = ((prediction - prediction.min()) /
                      (prediction.max() - prediction.min()))
    hard_gt = np.zeros_like(gt)
    hard_gt[gt > 128] = 1

    # MAE ##################################################################
    mae = np.mean(np.abs(prediction - hard_gt))
    
    # MeanF ################################################################
    threshold_fm = 2 * prediction.mean()
    if threshold_fm > 1:
        threshold_fm = 1
    binary = np.zeros_like(prediction)
    binary[prediction >= threshold_fm] = 1
    tp = (binary * hard_gt).sum()
    if tp == 0:
        meanf = 0
    else:
        pre = tp / binary.sum()
        rec = tp / hard_gt.sum()
        meanf = 1.3 * pre * rec / (0.3 * pre + rec)
    
    # PR curve #############################################################
    t = np.sum(hard_gt)
    precision, recall = [], []
    for threshold in range(256):
        threshold = threshold / 255.
        hard_prediction = np.zeros_like(prediction)
        hard_prediction[prediction >= threshold] = 1
        
        tp = np.sum(hard_prediction * hard_gt)
        p = np.sum(hard_prediction)
        if tp == 0:
            precision.append(0)
            recall.append(0)
        else:
            precision.append(tp / p)
            recall.append(tp / t)
    
    return precision, recall, mae, meanf


# MaxF #############################################################
def cal_maxf(ps, rs):
    assert len(ps) == 256
    assert len(rs) == 256
    maxf = []
    for p, r in zip(ps, rs):
        if p == 0 or r == 0:
            maxf.append(0)
        else:
            maxf.append(1.3 * p * r / (0.3 * p + r))
    
    return max(maxf)

def cal_maxe(y_pred, y, num = 255):
    score = np.zeros(num)
    for i in range(num):
        fm = y_pred - y_pred.mean()
        gt = y - y.mean()
        align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
        enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
        score[i] = np.sum(enhanced) / (y.size - 1 + 1e-20)
    return score.max()

def cal_s(pred, gt):
    alpha, avg_q, img_num = 0.5, 0.0, 0.0
    pred = pred.astype(np.float32) / 255.0
    gt = gt.astype(np.float32) / 255.0

    y = gt.mean()
    if y == 0:
        x = pred.mean()
        Q = 1.0 - x
    elif y == 1 or y == 255:
        x = pred.mean()
        Q = x
    else:
        Q = alpha * _S_object(pred, gt) + (1-alpha) * _S_region(pred, gt)
        if Q.item() < 0:
            Q = np.zeros(1)
    return Q.item()

def _S_object(pred, gt):
    fg = np.where(gt==0, np.zeros_like(pred), pred)
    bg = np.where((gt==1) | (gt==255), np.zeros_like(pred), 1-pred)
    o_fg = _object(fg, gt)
    o_bg = _object(bg, 1-gt)
    u = gt.mean()
    Q = u * o_fg + (1-u) * o_bg
    return Q

def _object(pred, gt):
    temp = pred[(gt==1) | (gt==255)]
    x = temp.mean()
    sigma_x = temp.std()
    score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
    
    return score

def _S_region(pred, gt):
    X, Y = _centroid(gt)
    gt1, gt2, gt3, gt4, w1, w2, w3, w4 = _divideGT(gt, X, Y)
    p1, p2, p3, p4 = _dividePrediction(pred, X, Y)
    Q1 = _ssim(p1, gt1)
    Q2 = _ssim(p2, gt2)
    Q3 = _ssim(p3, gt3)
    Q4 = _ssim(p4, gt4)
    Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
    # print(Q)
    return Q

def _centroid(gt):
    rows, cols = gt.shape[-2:]
    gt = gt.reshape(rows, cols)
    if gt.sum() == 0:
        X = np.eye(1) * round(cols / 2)
        Y = np.eye(1) * round(rows / 2)
    else:
        total = gt.sum()
        i = np.arange(0,cols).astype(np.float32)
        j = np.arange(0,rows).astype(np.float32)
        X = np.round((gt.sum(axis=0)*i).sum() / total)
        Y = np.round((gt.sum(axis=1)*j).sum() / total)
    return X.astype(np.int64), Y.astype(np.int64)

def _divideGT(gt, X, Y):
    h, w = gt.shape[-2:]
    area = h*w
    gt = gt.reshape(h, w)
    LT = gt[:Y, :X]
    RT = gt[:Y, X:w]
    LB = gt[Y:h, :X]
    RB = gt[Y:h, X:w]
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    w1 = X * Y / area
    w2 = (w - X) * Y / area
    w3 = X * (h - Y) / area
    w4 = 1 - w1 - w2 - w3
    return LT, RT, LB, RB, w1, w2, w3, w4

def _dividePrediction(pred, X, Y):
    h, w = pred.shape[-2:]
    pred = pred.reshape(h, w)
    LT = pred[:Y, :X]
    RT = pred[:Y, X:w]
    LB = pred[Y:h, :X]
    RB = pred[Y:h, X:w]
    return LT, RT, LB, RB

def _ssim(pred, gt):
    gt = gt.astype(np.float32)
    h, w = pred.shape[-2:]
    N = h*w
    x = pred.mean()
    y = gt.mean()
    sigma_x2 = ((pred - x)*(pred - x)).sum() / (N - 1 + 1e-20)
    sigma_y2 = ((gt - y)*(gt - y)).sum() / (N - 1 + 1e-20)
    sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + 1e-20)
    
    aplha = 4 * x * y *sigma_xy
    beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

    if aplha != 0:
        Q = aplha / (beta + 1e-20)
    elif aplha == 0 and beta == 0:
        Q = 1.0
    else:
        Q = 0
    return Q
