import os
import argparse
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from core.networks.MVDNet_conf import MVDNet_conf
from core.networks.solver import Solver
from post_processing.calConfidence import calConf
from core.utils.utils import load_config_file, vis_depth, vis_normal
from core.networks.loss_functions import compute_errors_numpy
from core.utils.logger import AverageMeter

def resize_intr(K, raw_hw, new_hw):
    new_K = np.copy(K)
    scale_h = 1.0 * new_hw[0] / raw_hw[0]
    scale_w = 1.0 * new_hw[1] / raw_hw[1]
    new_K[0,:] *= scale_w
    new_K[1,:] *= scale_h
    return new_K

def img_to_tensor(img):
    img_nor = (img / 255.0 - 0.5) / 0.5
    img_t = torch.from_numpy(img_nor).permute(2,0,1).float()
    return img_t.unsqueeze(0)

def to_tensor(data):
    data = torch.from_numpy(data).float()
    return data.unsqueeze(0)

def read_folder(path, input_size=(480,640)):
    img_dir = os.path.join(path, "color")
    depth_dir = os.path.join(path, "depth")
    pose_dir = os.path.join(path, "pose")
    K_dir = os.path.join(path, "intrinsic")
    if (not os.path.exists(img_dir)) or (not os.path.exists(pose_dir)):
        print("Should have a color & pose folder under the given path")
        raise NotImplementedError
    if not os.path.exists(depth_dir):
        raise NotImplementedError
    img_names = sorted(os.listdir(img_dir))
    depth_names = sorted(os.listdir(depth_dir))
    pose_names = sorted(os.listdir(pose_dir))
    imgs, depths, poses = [], [], []
    for img_name, depth_name, pose_name in zip(img_names, depth_names, pose_names):
        img = cv2.cvtColor(cv2.imread(os.path.join(img_dir, img_name)), cv2.COLOR_BGR2RGB).astype(np.float)
        raw_hw = img.shape[:2]
        img = cv2.resize(img, (input_size[1], input_size[0]))
        depth = cv2.imread(os.path.join(depth_dir, depth_name), -1) / 1000.0
        assert depth.shape[0] == input_size[0]
        assert depth.shape[1] == input_size[1]
        pose = np.loadtxt(os.path.join(pose_dir, pose_name))
        imgs.append(img)
        depths.append(depth)
        poses.append(pose)
    intrinsics = np.loadtxt(os.path.join(K_dir, "intrinsic_depth.txt"))[:3,:3]
    return imgs, depths, poses, intrinsics

def find_ref(imgs, gts, poses, K, gap):
    if len(imgs) < 3 * gap:
        print("The gap is too large or too few images")
        raise NotImplementedError
    batches = []
    assert len(imgs) == len(poses)
    for i in range(len(imgs)):
        if i < gap:
            ref1, ref2 = i + gap, i + gap * 2
        elif len(imgs) - 1 - i < gap:
            ref1, ref2 = i - gap, i - gap * 2
        else:
            ref1, ref2 = i - gap, i + gap
        batch = {}
        batch["ref_idx"] = [ref1, ref2]
        batch["tgt_img"] = imgs[i]
        batch["gt_depth"] = gts[i]
        batch["tgt_img_t"] = img_to_tensor(imgs[i])
        batch["ref_imgs"] = [imgs[ref1], imgs[ref2]]
        batch["ref_imgs_t"] = [img_to_tensor(imgs[ref1]), img_to_tensor(imgs[ref2])]
        nan_pose_tgt = np.sum(np.isnan(poses[i]) | np.isinf(poses[i]))
        if nan_pose_tgt > 0:
            continue
        ref1_pose = np.linalg.inv(poses[ref1]) @ poses[i]
        ref1_pose = ref1_pose[:3,:].astype(np.float)
        ref2_pose = np.linalg.inv(poses[ref2]) @ poses[i]
        ref2_pose = ref2_pose[:3,:].astype(np.float)
        nan_pose = np.sum(np.isnan(ref1_pose)) + np.sum(np.isinf(ref1_pose)) + \
            np.sum(np.isnan(ref2_pose)) + np.sum(np.isinf(ref2_pose))
        if nan_pose > 0:
            continue
        batch["tgt_pose"] = poses[i]
        batch["ref_poses"] = [ref1_pose, ref2_pose]
        batch["ref_poses_t"] = [to_tensor(ref1_pose), to_tensor(ref2_pose)]
        batch["ref_poses_t"] = torch.stack(batch["ref_poses_t"], 1)
        batch["intrinsics"] = np.copy(K)
        batch["intrinsics_t"] = to_tensor(K)
        batch["intrinsics_inv_t"] = to_tensor(np.linalg.inv(K))
        batches.append(batch)
    return batches

def get_initial_geo(net, batches):
    net.eval()
    new_batches = []
    with torch.no_grad():
        for batch in batches:
            tgt_img = batch["tgt_img_t"].cuda()
            ref_imgs = [ref_img.cuda() for ref_img in batch["ref_imgs_t"]]
            ref_poses = batch["ref_poses_t"].cuda()
            K, K_inv = batch["intrinsics_t"].cuda(), batch["intrinsics_inv_t"].cuda()
            output = net(tgt_img, ref_imgs, ref_poses, K, K_inv)
            depth, normal, dconf, nconf = output
            batch["init_depth"], batch["init_normal"] = depth.detach().cpu(), normal.detach().cpu()
            batch["dconf"], batch["nconf"] = dconf.detach().cpu(), nconf.detach().cpu()
            new_batches.append(batch)
    return new_batches

def refine_geo(solver, confCal, batches, iters, gap):
    depths, normals = [], []
    depth_gts = []
    for i in range(len(batches)):
        batch = batches[i]
        # find new ref depths and ref poses
        # Cannot directly use the image ref due to some invalid samples
        ref_depths = []
        if i < gap:
            ref1, ref2 = i + gap, i + gap * 2
        elif len(batches) - 1 - i < gap:
            ref1, ref2 = i - gap, i - gap * 2
        else:
            ref1, ref2 = i - gap, i + gap
        ref_depths.append(batches[ref1]["init_depth"])
        ref_depths.append(batches[ref2]["init_depth"])
        ref1_pose = np.linalg.inv(batches[ref1]["tgt_pose"]) @ batches[i]["tgt_pose"]
        ref1_pose = ref1_pose[:3,:].astype(np.float)
        ref2_pose = np.linalg.inv(batches[ref2]["tgt_pose"]) @ batches[i]["tgt_pose"]
        ref2_pose = ref2_pose[:3,:].astype(np.float)
        ref_poses_t = torch.stack([to_tensor(ref1_pose), to_tensor(ref2_pose)], 1)
        
        ref_depths = torch.stack(ref_depths, 1)
        reproj_conf = confCal.reprojConf(batch["init_depth"], ref_depths, batch["intrinsics_t"], ref_poses_t, ratio=0.20)
        pred_dconf = torch.where(batch["dconf"] < 0.30, torch.zeros_like(batch["dconf"]), batch["dconf"])
        pred_dconf = torch.sigmoid(10.0 * (pred_dconf - pred_dconf.mean()))
        confD = torch.clamp(reproj_conf * pred_dconf, 0.01, 1.0).unsqueeze(0).cuda()
        confN = batch["nconf"].unsqueeze(0).cuda()
        cur_depth, cur_normal = batch["init_depth"].unsqueeze(0).cuda(), batch["init_normal"].cuda()
        tgt_img = torch.from_numpy(batch["tgt_img"]).float().permute(2,0,1).unsqueeze(0).cuda()
        for i in range(iters):
            cur_depth, cur_normal = solver(cur_depth, cur_normal, tgt_img, confD, confN, batch["intrinsics_t"].cuda())
        
        depths.append(cur_depth[0,0].cpu().numpy())
        normals.append(cur_normal[0].permute(1,2,0).cpu().numpy())
        depth_gts.append(batch["gt_depth"])
        
    return depths, normals, depth_gts

def vis_geo(save_dir, depths, normals):
    for i in range(len(depths)):
        vis_depth(save_dir, str(i), depths[i])
        vis_normal(save_dir, str(i), normals[i])

def compute_seq_error(depths, gts, test_errors):
    assert len(depths) == len(gts)
    for depth, gt in zip(depths, gts):
        mask = (gt >= 0.5) & (gt <= 10)
        if not mask.any():
            continue
        errors = list(compute_errors_numpy(gt[mask], depth[mask]))
        test_errors.update(errors)

def main(args, cfg):
    # create model
    print("=> creating model")

    mvdnet = MVDNet_conf(cfg).cuda()
    solver = Solver(h=cfg.input_size[0], w=cfg.input_size[1], check_offsets=cfg.check_offsets, alpha1=cfg.solver_alpha1, \
                    alpha2=cfg.solver_alpha2, sigma1=cfg.solver_sigma1, sigma2=cfg.solver_sigma2)
    confCal = calConf(1, h=cfg.input_size[0], w=cfg.input_size[1])
    mvdnet.init_weights()
    
    if cfg.pretrained_mvdn is not None:
        print("=> using pre-trained weights for MVDNet")
        weights = torch.load(cfg.pretrained_mvdn)   
        mvdnet.load_state_dict(weights['state_dict'])
    else:
        print("Must provide a checkpoint model")
        raise NotImplementedError
    
    cudnn.benchmark = True
    mvdnet = torch.nn.DataParallel(mvdnet)

    print("=> evaluate model on scannet test set '{}'".format(args.data_dir))
    test_error_names = ['abs_rel','abs_diff','sq_rel','rms','log_rms','a1','a2','a3']
    test_errors = AverageMeter(i=len(test_error_names))
    
    seq_names = sorted(os.listdir(args.data_dir))
    total_num = 0
    for seq in seq_names:
        seq_dir = os.path.join(args.data_dir, seq)
        if not os.path.isdir(seq_dir):
            continue
        imgs, gt_depths, poses, K = read_folder(seq_dir, cfg.input_size)
        batches = find_ref(imgs, gt_depths, poses, K, gap=cfg.reference_gap)
        total_num += len(batches)

        print("=> predict initial geometry")
        new_batches = get_initial_geo(mvdnet, batches)
        print("=> refine geometry")
        final_depths, final_normals, gts = refine_geo(solver, confCal, new_batches, cfg.refine_iter, cfg.reference_gap)
        print("=> calculate loss")
        compute_seq_error(final_depths, gts, test_errors)
        print(seq)
        print(len(batches))
    print("Total test num: " + str(total_num))
    print(test_error_names)
    print(test_errors.avg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Iterative solver for deep mvs',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--data_dir', type=str, help='path to the scannet test dir')
    parser.add_argument('--config', type=str, help='path to the config file')
    args = parser.parse_args()
    cfg = load_config_file(args.config)
    main(args, cfg)
