from __future__ import division
import shutil
import numpy as np
import torch
from path import Path
import datetime
from collections import OrderedDict
import cv2
import os
import yaml
import matplotlib.pyplot as plt

def load_config_file(path):
    cfg = yaml.safe_load(open(path, 'r'))
    class pObject(object):
        def __init__(self):
            pass
    cfg_new = pObject()
    for attr in list(cfg.keys()):
        setattr(cfg_new, attr, cfg[attr])
    return cfg_new

def save_path_formatter(args, parser):
    def is_default(key, value):
        return value == parser.get_default(key)
    args_dict = vars(args)
    data_folder_name = str(Path(args_dict['data']).normpath().name)
    folder_string = [data_folder_name]
    if not is_default('epochs', args_dict['epochs']):
        folder_string.append('{}epochs'.format(args_dict['epochs']))
    keys_with_prefix = OrderedDict()
    keys_with_prefix['batch_size'] = 'b'
    keys_with_prefix['lr'] = 'lr'

    for key, prefix in keys_with_prefix.items():
        value = args_dict[key]
        if not is_default(key, value):
            folder_string.append('{}{}'.format(prefix, value))
    save_path = Path(','.join(folder_string))
    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    return save_path/timestamp


def tensor2array(tensor, max_value=255, colormap='rainbow'):
    if max_value is None:
        max_value = tensor.max()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        try:
            import cv2
            if cv2.__version__.startswith('2'): # 2.4
                color_cvt = cv2.cv.CV_BGR2RGB
            else:  
                color_cvt = cv2.COLOR_BGR2RGB
            if colormap == 'rainbow':
                colormap = cv2.COLORMAP_RAINBOW
            elif colormap == 'bone':
                colormap = cv2.COLORMAP_BONE
            array = (255*tensor.squeeze().numpy()/max_value).clip(0, 255).astype(np.uint8)
            colored_array = cv2.applyColorMap(array, colormap)
            array = cv2.cvtColor(colored_array, color_cvt).astype(np.float32)/255
            array = array.transpose(2, 0, 1)
        except ImportError:
            if tensor.ndimension() == 2:
                tensor.unsqueeze_(2)
            elif tensor.size(0) == 1:
                tensor = tensor.permute(2,1,0)
            array = (tensor.expand(tensor.size(0), tensor.size(1), 3).numpy()/max_value).clip(0,1)
            array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        #assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5

    return array


def save_checkpoint(save_path, dpsnet_state, epoch, filename='checkpoint.pth.tar', file_prefixes = ['dpsnet']):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    states = [dpsnet_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}_{}'.format(prefix,epoch,filename))


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def vis_save_depth(output_dir, name, depth):
    cv2.imwrite(os.path.join(output_dir, name+'_depth.png'), (depth*6000.0).astype(np.uint16))
    np.save(os.path.join(output_dir, name+'_depth.npy'), depth)

def gray2rgb(im, cmap='gray'):
    cmap = plt.get_cmap(cmap)
    rgba_img = cmap(im.astype(np.float32))
    rgb_img = np.delete(rgba_img, 3, 2)
    return rgb_img

def normalize_depth_for_display(depth, pc=98, crop_percent=0, normalizer=None, cmap='gray'):
    # convert to disparity
    vinds = depth>0
    depth = 1./(depth + 1)

    z1 = np.percentile(depth[vinds], pc)
    z2 = np.percentile(depth[vinds], 100-pc)

    depth = (depth - z2) / (z1 - z2)
    depth = np.clip(depth, 0, 1)

    depth = gray2rgb(depth, cmap=cmap)
    keep_H = int(depth.shape[0] * (1-crop_percent))
    depth = depth[:keep_H]
    return depth

def vis_depth(output_dir, name, depth):
    depth_nor = normalize_depth_for_display(depth)
    plt.imsave(os.path.join(output_dir, name+'_depth.png'), depth_nor)

def vis_save_normal(output_dir, name, normal):
    normal_n = normal / (np.linalg.norm(normal, axis=2, keepdims=True) + 1e-10)
    normal_img = ((normal_n + 1.0) / 2.0) * 255.0
    cv2.imwrite(os.path.join(output_dir, name+'_normal.png'), normal_img[:,:,::-1].astype(np.uint8))
    np.save(os.path.join(output_dir, name+'_normal.npy'), normal)

def vis_normal(output_dir, name, normal):
    normal_n = normal / (np.linalg.norm(normal, axis=2, keepdims=True) + 1e-10)
    normal_img = ((normal_n + 1.0) / 2.0) * 255.0
    cv2.imwrite(os.path.join(output_dir, name+'_normal.png'), normal_img[:,:,::-1].astype(np.uint8))

def vis_conf(output_dir, name, conf):
    conf = torch.clamp(conf, 0.0, 1.0)
    plt.imsave(os.path.join(output_dir, name+'_conf.png'), conf)

def vis_colored_pointcloud(image, points, path):
    xyzrgb = np.concatenate([points, np.reshape(image, (-1,3))], axis=-1)
    pcwrite(path, xyzrgb)
    return None

def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file.
    """
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename,'w')
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n"%(xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write("%f %f %f %d %d %d\n"%(
        xyz[i, 0], xyz[i, 1], xyz[i, 2],
        rgb[i, 0], rgb[i, 1], rgb[i, 2],
        ))

import time
import torch

class Profiler(object):
    def __init__(self, silent=False):
        self.silent = silent
        torch.cuda.synchronize()
        self.start = time.time()
        self.cache_time = self.start

    def reset(self, silent=None):
        if silent is None:
            silent = self.silent
        self.__init__(silent=silent)

    def report_process(self, process_name):
        if self.silent:
            return None
        torch.cuda.synchronize()
        now = time.time()
        print('{0}\t: {1:.4f}'.format(process_name, now - self.cache_time))
        self.cache_time = now

    def report_all(self, whole_process_name):
        if self.silent:
            return None
        torch.cuda.synchronize()
        now = time.time()
        print('{0}\t: {1:.4f}'.format(whole_process_name, now - self.start))