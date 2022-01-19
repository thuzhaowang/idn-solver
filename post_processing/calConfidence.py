import numpy as np
import torch
import torch.nn.functional as F
import cv2
from .utils_post import vis_colored_pointcloud, pcwrite, scale_intrinsics


def to_tensor(a):
    a_t = torch.from_numpy(a).float()
    return a_t.unsqueeze(0)

def depth2flow(depth, K, poses, grid):
    """Unproject-transform-project to induce correspondence
    depth: [B, 1, H, W]
    K: [B, 3, 3]
    poses: [B, N, 3, 4]
    grid: [B, 3, H, W] base homogeneous grid
    Return: xy coordinates in the tgt image, normalized xy coords, and projected depth and valid mask
    """
    b, _, h, w = grid.shape
    N = poses.shape[1]
    xy_new = torch.zeros(b, N, h, w, 2)
    depth_new = torch.zeros(b, N, 1, h, w)
    points_new = torch.zeros(b, N, 3, h*w)
    
    K_inv = torch.inverse(K)
    points = depth.reshape(b, 1, -1) * torch.matmul(K_inv, grid.reshape(b, 3, -1))
    for i in range(N):
        points_trans = torch.matmul(poses[:,i,:3,:3], points) + poses[:,i,:3,-1:]
        points_new[:, i, :, :] = points_trans
        xyz_tgt = torch.matmul(K, points_trans)
        depth = xyz_tgt[:,-1:,:]
        xyz_tgt = xyz_tgt / depth
        xy_new[:,i,:,:,:] = xyz_tgt[:,:2,:].reshape(b, 2, h, w).permute(0, 2, 3, 1)
        depth_new[:,i,:,:,:] = depth.reshape(b, 1, h, w)
    xy_new_norm = torch.stack([(xy_new[:,:,:,:,0] - w/2.0)/ (w/2.0), (xy_new[:,:,:,:,1] - h/2.0) / (h/2.0)], -1)
    valid_mask = (xy_new_norm[:,:,:,:,0] >= -1.0).float() * (xy_new_norm[:,:,:,:,0] <= 1.0).float()
    valid_mask  = valid_mask * (xy_new_norm[:,:,:,:,1] >= -1.0).float() * (xy_new_norm[:,:,:,:,1] <= 1.0).float()
    return xy_new, xy_new_norm, depth_new, points_new, valid_mask

class calConf():
    def __init__(self, b, h, w):
        super(calConf, self).__init__()
        self.b = b
        self.h = h
        self.w = w
        self.setGrid()
    
    def setGrid(self):
        x = np.linspace(0, self.w-1, self.w)
        y = np.linspace(0, self.h-1, self.h)
        xx, yy = np.meshgrid(x, y)
        filler = np.ones([self.h, self.w])
        xy = np.stack([xx, yy, filler], 0)
        xy = torch.from_numpy(xy).float()
        self.grid = xy.unsqueeze(0).repeat(self.b,1,1,1)
        return None

    def reprojConf(self, tgtDepth, refDepths, K, poses, ratio):
        """Use depth reprojection check to compute the confidence
        tgtDepth: [B, 1, H, W]
        refDepths: [B, 1, N, H, W]
        K: [B, 3, 3]
        poses: [B, N, 3, 4]
        thres: scaling parameters for confidence
        """
        N = poses.shape[1]
        xy, xy_norm, reconsDepths, points_new, valid_masks = depth2flow(tgtDepth, K, poses, self.grid)
        # There are two ways to calculate the depth reprojection confidence:
        # 1. Compare the reconstructed depth and the predicted depth (from Gipuma) (current)
        # 2. Use the tgt predicted depth to do another reprojection and calculate the coord shift (from COLMAP)
        refReconsDepths = F.grid_sample(refDepths.reshape(-1, 1, self.h, self.w), xy_norm.reshape(-1, self.h, self.w, 2), padding_mode='zeros')
        refReconsDepths = refReconsDepths.reshape(self.b, N, 1, self.h, self.w)
        depthLoss = torch.abs(refReconsDepths - reconsDepths).squeeze(2)
        loss2conf = (depthLoss <= ratio*tgtDepth).float() * (1.0 - depthLoss / (ratio*tgtDepth))
        depthConf = torch.where(valid_masks > 0, loss2conf, 1.0*torch.ones(depthLoss.shape))
        depthConf, _ = torch.min(depthConf, 1) # min over the N views
        return depthConf
    
    def visConfImg(self, conf, savePath):
        conf = conf / conf.max() * 255.0
        cv2.imwrite(savePath, conf)
        return None

    def visConfPoints(self, points, confs, savePath):
        color = (confs * 100).astype(np.int)
        pcwrite(savePath, points)
        return None


        

        


        



