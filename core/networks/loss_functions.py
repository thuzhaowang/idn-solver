from __future__ import division
import torch
from torch import nn
import numpy as np
import pdb

def compute_errors_test(gt, pred):
    gt = gt.numpy()
    pred = pred.numpy()
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_diff = np.mean(np.abs(gt - pred))
    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, abs_diff, sq_rel, rmse, rmse_log, a1, a2, a3

def compute_errors_numpy(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25   ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_diff = np.mean(np.abs(gt - pred))
    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, abs_diff, sq_rel, rmse, rmse_log, a1, a2, a3

def compute_errors_train(gt, pred, valid):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    for current_gt, current_pred, current_valid in zip(gt, pred, valid):
        valid_gt = current_gt[current_valid]
        valid_pred = current_pred[current_valid]

        if len(valid_gt) == 0:
            continue
        else:
            thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
            a1 += (thresh < 1.25).float().mean()
            a2 += (thresh < 1.25 ** 2).float().mean()
            a3 += (thresh < 1.25 ** 3).float().mean()

            abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
            abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

            sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)

    return [metric / batch_size for metric in [abs_rel, abs_diff, sq_rel, a1, a2, a3]]



def compute_angles(normal1, normal2, dim=2):
    dot = torch.sum(normal1*normal2, dim = dim)
    dot = torch.abs(dot)
    
    total_angles = torch.acos(torch.clamp(dot, min = -1, max = 1))/np.pi*180
    return total_angles

def compute_normal_stats(normal_error_in_angle):
    mean = torch.mean(normal_error_in_angle)
    median = torch.median(normal_error_in_angle)
    rmse = torch.sqrt(torch.sum(normal_error_in_angle * normal_error_in_angle) / normal_error_in_angle.shape[0])
    deg5 = torch.sum(normal_error_in_angle < 5).float() / normal_error_in_angle.shape[0]
    deg1125 = torch.sum(normal_error_in_angle < 11.25).float() / normal_error_in_angle.shape[0]
    deg225 = torch.sum(normal_error_in_angle < 22.5).float() / normal_error_in_angle.shape[0]
    deg30 = torch.sum(normal_error_in_angle < 30).float() / normal_error_in_angle.shape[0]
    return [mean, median, rmse, deg5, deg1125, deg225, deg30]

def cross_entropy(pred, label):
    loss = -label * torch.log(pred) - (1.0 - label) * torch.log(1.0 - pred)
    return loss.mean()

def truncate_angular_loss(pred, label, mask):
    prediction_error = torch.cosine_similarity(pred, label, dim=1, eps=1e-6)
    # Robust acos loss
    acos_mask = mask.float() \
            * (prediction_error.detach() < 0.9999).float() * (prediction_error.detach() > 0.0).float()
    cos_mask = mask.float() * (prediction_error.detach() <= 0.0).float()
    acos_mask = acos_mask > 0.0
    cos_mask = cos_mask > 0.0
    optimize_loss = torch.sum(torch.acos(prediction_error[acos_mask])) - torch.sum(prediction_error[cos_mask])
    loss = optimize_loss / (torch.sum(cos_mask) + torch.sum(acos_mask))
    return loss
