import os,sys
import numpy as np
from matplotlib import pyplot as plt
import cv2

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

def scale_intrinsics(K, scale):
    K_new = np.copy(K)
    K_new[0,0] *= scale
    K_new[1,1] *= scale
    K_new[0,2] = (K_new[0,2]+0.5) * scale - 0.5
    K_new[1,2] = (K_new[1,2]+0.5) * scale - 0.5
    return K_new

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

def vis_conf(output_dir, name, conf):
    conf = np.clip(conf, 0.0, 1.0)
    plt.imsave(os.path.join(output_dir, name+'_conf.png'), conf)

def vis_error(output_dir, name, error):
    z1 = np.percentile(error, 98)
    z2 = np.percentile(error, 2)

    error = (error - z2) / (z1 - z2)
    error = np.clip(error, 0.0, 1.0)
    color_error = gray2rgb(error, cmap='magma')
    plt.imsave(os.path.join(output_dir, name+'_error.png'), color_error)

def vis_normal(output_dir, name, normal):
    normal_n = normal / (np.linalg.norm(normal, axis=2, keepdims=True) + 1e-10)
    normal_img = ((normal_n + 1.0) / 2.0) * 255.0
    cv2.imwrite(os.path.join(output_dir, name+'_normal.png'), normal_img[:,:,::-1].astype(np.uint8))
