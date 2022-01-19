import os, sys
import numpy as np
import cv2
import argparse
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir")
parser.add_argument("--output_dir")
args = parser.parse_args()

root_path = args.data_dir
tgt_path = args.output_dir
sample_gap = 20

def copy_all_with_gap(src_dir, tgt_dir, gap):
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    imgs = os.listdir(src_dir)
    img_name = sorted([int(img[:-4]) for img in imgs])
    img_ext = imgs[0][-4:]
    i = 0
    for i in range(len(imgs)):
        if img_name[i] % gap == 0:
            command = 'cp ' + os.path.join(src_dir, str(img_name[i]) + img_ext) + ' ' + os.path.join(tgt_dir, "%04d"%(img_name[i]) + img_ext)
            os.system(command)

def copy_resize_all_with_gap(src_dir, tgt_dir, gap):
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    imgs = os.listdir(src_dir)
    img_name = sorted([int(img[:-4]) for img in imgs])
    img_ext = imgs[0][-4:]
    i = 0
    for i in range(len(imgs)):
        if img_name[i] % gap == 0:
            image = cv2.imread(os.path.join(src_dir, str(img_name[i]) + img_ext))
            image_r = cv2.resize(image, (640,480))
            save_name = os.path.join(tgt_dir, "%04d"%(img_name[i]) + img_ext)
            cv2.imwrite(save_name, image_r)

def copy_all(src_dir, tgt_dir):
    if not os.path.exists(tgt_dir):
        os.makedirs(tgt_dir)
    command = 'cp ' + os.path.join(src_dir, '*') + ' ' + tgt_dir
    os.system(command)

scenes = sorted(os.listdir(root_path))
for s in scenes:
    copy_resize_all_with_gap(os.path.join(root_path, s, 'color'), os.path.join(tgt_path, s, 'color'), sample_gap)
    copy_all_with_gap(os.path.join(root_path, s, 'depth'), os.path.join(tgt_path, s, 'depth'), sample_gap)
    copy_all_with_gap(os.path.join(root_path, s, 'pose'), os.path.join(tgt_path, s, 'pose'), sample_gap)
    copy_all(os.path.join(root_path, s, 'intrinsic'), os.path.join(tgt_path, s, 'intrinsic/'))
    print(s)

