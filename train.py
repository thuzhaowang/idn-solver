import os
import time
import csv
import numpy as np
from path import Path
import argparse
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

import torch
import torch.nn.functional as F
from core.dataset import custom_transforms
from core.networks.MVDNet_conf import MVDNet_conf
from core.networks.MVDNet_joint import MVDNet_joint

from core.utils.utils import load_config_file, save_checkpoint, adjust_learning_rate
from core.networks.loss_functions import compute_errors_test, compute_angles, cross_entropy

from core.utils.logger import AverageMeter
from core.dataset.data_loader import SequenceFolder

def main(cfg):
    global n_iter
    save_path = Path(cfg.output_dir)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print('=> will save everything to {}'.format(save_path))

    training_writer = SummaryWriter(save_path)
    output_writers = []
    for i in range(3):
        output_writers.append(SummaryWriter(save_path/'valid'/str(i)))

    # Loading data
    normalize = custom_transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                            std=[0.5, 0.5, 0.5])
    
    train_transform = custom_transforms.Compose([
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])
    print("=> fetching scenes in '{}'".format(cfg.dataset_path))
    
    if cfg.dataset == 'scannet':
        train_set = SequenceFolder(cfg.dataset_path, transform=train_transform, ttype=cfg.train_list)        
        test_set = SequenceFolder(cfg.dataset_path, transform=valid_transform, ttype=cfg.test_list)
    else:
        raise NotImplementedError

    train_set.samples = train_set.samples[:len(train_set) - len(train_set)%cfg.batch_size]

    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} test scenes'.format(len(test_set), len(test_set.scenes)))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True)

    epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    if cfg.model_name == 'MVDNet_conf':
        mvdnet = MVDNet_conf(cfg).cuda()
    elif cfg.model_name == 'MVDNet_joint':
        mvdnet = MVDNet_joint(cfg).cuda()
    else:
        raise NotImplementedError
    
    mvdnet.init_weights()
    if cfg.pretrained_mvdn:
        print("=> using pre-trained weights for MVDNet")
        weights = torch.load(cfg.pretrained_mvdn)   
        mvdnet.load_state_dict(weights['state_dict'], strict=True)
    
    print('=> setting adam solver')

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mvdnet.parameters()), cfg.learning_rate, betas=(cfg.momentum, cfg.beta),
                                     weight_decay=cfg.weight_decay)

    torch.backends.cudnn.benchmark = True
    mvdnet = torch.nn.DataParallel(mvdnet)

    print(' ==> setting log files')
    with open(save_path/'log_summary.txt', 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_abs_rel', 'validation_abs_diff','validation_sq_rel', 'validation_a1', 'validation_a2', 'validation_a3', 'mean_angle_error'])

    print(' ==> main Loop')
    for epoch in range(cfg.epochs):
        adjust_learning_rate(cfg, optimizer, epoch)

        # train for one epoch
        train_loss = train_epoch(cfg, train_loader, mvdnet, optimizer, epoch_size, training_writer, epoch)
        
        errors, error_names = validate_with_gt(cfg, test_loader, mvdnet, epoch, output_writers)
        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[0]
        with open(save_path/'log_summary.txt', 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error, errors[1], errors[2], errors[3], errors[4], errors[5], errors[6]])
        save_checkpoint(os.path.join(save_path, 'checkpoints'), {'epoch': epoch + 1, 'state_dict': mvdnet.module.state_dict()},
            epoch, file_prefixes = ['mvdnet'])
        

def train_epoch(cfg, train_loader, mvdnet, optimizer, epoch_size, train_writer, epoch):
    global n_iter
    batch_time = AverageMeter()
    data_time = AverageMeter()
    total_losses = AverageMeter(precision=4)
    d_losses = AverageMeter(precision=4)
    nmap_losses = AverageMeter(precision=4)
    dconf_losses = AverageMeter(precision=4)
    nconf_losses = AverageMeter(precision=4)
    
    mvdnet.train()

    print("Training")
    end = time.time()

    for i, (tgt_img, ref_imgs, gt_nmap, ref_poses, intrinsics, intrinsics_inv, tgt_depth, ref_depths) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img_var = tgt_img.cuda()
        ref_imgs_var = [img.cuda() for img in ref_imgs]
        gt_nmap_var = gt_nmap.cuda()
        ref_poses_var = [pose.cuda() for pose in ref_poses]
        intrinsics_var = intrinsics.cuda()
        intrinsics_inv_var = intrinsics_inv.cuda()
        tgt_depth_var = tgt_depth.cuda()

        # compute output
        pose = torch.cat(ref_poses_var,1)

        # get mask
        mask = (tgt_depth_var <= 10.0) & (tgt_depth_var >= 0.5) & (tgt_depth_var == tgt_depth_var)
        mask.detach_()
        if mask.any() == 0:
            continue
        
        if cfg.model_name == 'MVDNet_conf':
            outputs = mvdnet(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var)
        elif cfg.model_name == 'MVDNet_joint':
            outputs = mvdnet(tgt_img_var, ref_imgs_var, pose, tgt_depth_var, gt_nmap_var, intrinsics_var, intrinsics_inv_var)
        else:
            raise NotImplementedError
        
        depth0, depth1 = outputs[0], outputs[1]
        nmap0 = outputs[2]
        dconf, nconf = outputs[-2], outputs[-1]
        
        # Loss
        d_loss = cfg.d_weight * F.smooth_l1_loss(depth0[mask], tgt_depth_var[mask]) + \
            F.smooth_l1_loss(depth1[mask], tgt_depth_var[mask])
        gt_dconf = 1.0 - cfg.conf_dgamma * torch.abs(depth0 - tgt_depth_var) / (tgt_depth_var + 1e-6)
        gt_dconf = torch.clamp(gt_dconf, 0.01, 1.0).detach_()
        dconf_loss = cross_entropy(dconf[mask], gt_dconf[mask])
        
        n_mask = mask.unsqueeze(1).expand(-1,3,-1,-1)
        nmap_loss = F.smooth_l1_loss(nmap0[n_mask], gt_nmap_var[n_mask])
        gt_nconf = 1.0 - cfg.conf_ngamma * compute_angles(nmap0, gt_nmap_var, dim=1) / 180.0
        gt_nconf = torch.clamp(gt_nconf, 0.01, 1.0).detach_()
        nconf_loss = cross_entropy(nconf[mask], gt_nconf[mask])

        loss = d_loss + cfg.n_weight * nmap_loss + cfg.dc_weight * dconf_loss + cfg.nc_weight * nconf_loss

        if i > 0 and n_iter % cfg.print_freq == 0:
            train_writer.add_scalar('total_loss', loss.item(), n_iter)
        
        # record loss and EPE
        total_losses.update(loss.item(), n=cfg.batch_size)
        d_losses.update(d_loss.mean().item(), n=cfg.batch_size)
        nmap_losses.update(nmap_loss.mean().item(), n=cfg.batch_size)
        dconf_losses.update(dconf_loss.mean().item(), n=cfg.batch_size)
        nconf_losses.update(nconf_loss.mean().item(), n=cfg.batch_size)
        
        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if cfg.log_mode == 'full':
            with open(cfg.output_dir/'log_full.txt', 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter='\t')
                writer.writerow([loss.item()])
        if i % cfg.print_freq == 0:
            print('Train: Time {} Loss {} NLoss {} DLoss {} DCLoss {} NCLoss {} Iter {}/{} Epoch {}/{}'.format(batch_time, total_losses, nmap_losses, 
                        d_losses, dconf_losses, nconf_losses, i, len(train_loader), epoch, cfg.epochs))
            
        if i >= epoch_size - 1:
            break

        n_iter += 1
    return total_losses.avg[0]


def validate_with_gt(cfg, test_loader, mvdnet, epoch, output_writers=[]):
    batch_time = AverageMeter()
    test_error_names = ['abs_rel','abs_diff','sq_rel','rms','log_rms','a1','a2','a3', 'dconf', 'nconf', 'mean_angle']
    test_errors = AverageMeter(i=len(test_error_names))
    log_outputs = len(output_writers) > 0

    mvdnet.eval()

    end = time.time()
    with torch.no_grad(): 
        for i, (tgt_img, ref_imgs, gt_nmap, ref_poses, intrinsics, intrinsics_inv, tgt_depth, ref_depths) in enumerate(test_loader):
            tgt_img_var = tgt_img.cuda()
            ref_imgs_var = [img.cuda() for img in ref_imgs]
            gt_nmap_var = gt_nmap.cuda()
            ref_poses_var = [pose.cuda() for pose in ref_poses]
            intrinsics_var = intrinsics.cuda()
            intrinsics_inv_var = intrinsics_inv.cuda()
            tgt_depth_var = tgt_depth.cuda()

            pose = torch.cat(ref_poses_var,1)
            if (pose != pose).any():
                continue
            
            if cfg.model_name == 'MVDNet_conf':
                outputs = mvdnet(tgt_img_var, ref_imgs_var, pose, intrinsics_var, intrinsics_inv_var)
            elif cfg.model_name == 'MVDNet_joint':
                outputs = mvdnet(tgt_img_var, ref_imgs_var, pose, tgt_depth_var, gt_nmap_var, intrinsics_var, intrinsics_inv_var)
            else:
                raise NotImplementedError
            
            output_depth = outputs[0].data.cpu()
            nmap = outputs[1].permute(0,2,3,1)
            dconf, nconf = outputs[-2], outputs[-1]
            
            mask = (tgt_depth <= 10) & (tgt_depth >= 0.5) & (tgt_depth == tgt_depth)

            if not mask.any():
                continue

            test_errors_ = list(compute_errors_test(tgt_depth[mask], output_depth[mask]))
            gt_dconf = 1.0 - cfg.conf_dgamma * torch.abs(tgt_depth - output_depth) / (tgt_depth + 1e-6)
            dconf_e = torch.abs(dconf.cpu()[mask] - gt_dconf[mask]).mean()
            test_errors_.append(dconf_e.item())

            n_mask = (gt_nmap_var.permute(0,2,3,1)[0,:,:] != 0)
            n_mask = n_mask[:,:,0] | n_mask[:,:,1] | n_mask[:,:,2]

            total_angles_m = compute_angles(gt_nmap_var.permute(0,2,3,1)[0], nmap[0])
            gt_nconf = 1.0 - cfg.conf_ngamma * total_angles_m / 180.0
            nconf_e = torch.abs(nconf[0][n_mask] - gt_nconf[n_mask]).mean()
            test_errors_.append(nconf_e.item())
            
            mask_angles = total_angles_m[n_mask]
            total_angles_m[~ n_mask] = 0
            test_errors_.append(torch.mean(mask_angles).item())
            
            test_errors.update(test_errors_)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % cfg.print_freq == 0 or i == len(test_loader)-1:
                print('valid: Time {} Rel Error {:.4f} ({:.4f}) DConf Error {:.4f} ({:.4f}) Iter {}/{}'.format(batch_time, test_errors.val[0], test_errors.avg[0], test_errors.val[-3], test_errors.avg[-3], i, len(test_loader)))
            if cfg.save_samples:
                output_dir = Path(os.path.join(cfg.output_dir, 'vis'))
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
                plt.imsave(output_dir/'{:04d}_depth.png'.format(i), output_depth.numpy()[0], cmap='rainbow')
    return test_errors.avg, test_error_names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Iterative solver for multi-view depth and normal',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('config_file', metavar='DIR', help='path to config file')
    args = parser.parse_args()
    cfg = load_config_file(args.config_file)
    
    n_iter = 0
    main(cfg)
