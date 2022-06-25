import numpy as np


# 从显著图获取定位框
def getbb_from_heatmap(hmp, alpha):

    heatmap = np.copy(hmp)

    # 设置阈值参数
    threshold = alpha * heatmap.mean()
    heatmap[heatmap < threshold] = 0

    if (heatmap == 0).all():
        return [[1, 1, heatmap.shape[0], heatmap.shape[1]]]

    # 获取定位框
    x = np.where(heatmap.sum(0) > 0)[0] + 1
    y = np.where(heatmap.sum(1) > 0)[0] + 1
    return [[x[0], y[0], x[-1], y[-1]]]

# 计算 iou
def ious(pred, gt):
    numObj = len(gt)
    gt = np.tile(gt, [len(pred), 1])
    pred = np.repeat(pred, numObj, axis=0)
    bi = np.minimum(pred[:,2:],gt[:,2:]) - \
         np.maximum(pred[:,:2], gt[:,:2]) + 1
    area_bi = np.prod(bi.clip(0), axis=1)
    gt_area = (gt[:, 2] - gt[:, 0] + 1) * (gt[:, 3] - gt[:, 1] + 1)
    pred_area = (pred[:, 2] - pred[:, 0] + 1) * (pred[:, 3] - pred[:, 1] + 1)
    area_bu = gt_area + pred_area - area_bi
    return area_bi / area_bu
