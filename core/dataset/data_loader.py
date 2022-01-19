import numpy as np
from path import Path
import random
import pickle
import torch	
import os
import cv2

def load_as_float(path):
    """Loads image"""
    im =  cv2.imread(path)
    im =  cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) 
    return im

class SequenceFolder(torch.utils.data.Dataset):
	"""Creates a pickle file for ScanNet scene loading, and corresponding dataloader"""

	def __init__(self, root, ttype, seed=None, seq_length=3, seq_gap=20, transform=None):
		np.random.seed(seed)
		random.seed(seed)
		self.root = Path(root)

		scene_list_path = ttype
		self.scene_list_path = scene_list_path[:-4]
		fold_root = 'scans_test_sample' if 'test' in ttype else 'scannet_nas' 
		#fold_root = 'scannet_nas'
		scenes = [self.root/fold_root/folder[:-1] for folder in open(scene_list_path)]
		self.ttype = ttype
		self.scenes = sorted(scenes)

		self.seq_gap = seq_gap
		self.seq_length = seq_length

		self.transform = transform
		file_pickle = self.scene_list_path+ '_len_'+str(self.seq_length)+ '_gap_'+str(self.seq_gap)+'.pickle'
		if os.path.exists(file_pickle):
			with open(file_pickle, 'rb') as handle:
				sequence_set = pickle.load(handle)
				self.samples = sequence_set
		else:
			self.crawl_folders()


	def crawl_folders(self):
		sequence_set = []
		isc = 0
		cnt = 0
		for scene in self.scenes:
			#print(isc, len(self.scenes))
			isc += 1
			frames = os.listdir(os.path.join(scene, "color"))
			frames = [int(os.path.splitext(frame)[0]) for frame in frames]
			frames =  sorted(frames)
			intrinsics = np.genfromtxt(os.path.join(scene, "intrinsic", "intrinsic_depth.txt")).astype(np.float32).reshape((4, 4))[:3,:3]

			# The index from scannet nas is already sampled
			if len(frames) < (self.seq_gap // 20) * self.seq_length:
				continue
			cnt += len(frames)
			end_idx = len(frames) * 20

			path_split = scene.split('/')

			for i in range(len(frames)):
				idx = frames[i]
				img = os.path.join(scene, "color", "%04d.jpg" % idx)
				if 'test' in self.ttype:
					depth = os.path.join(scene, "depth", "%04d.png" % idx)
					# do not require normal when test
					normal = ""
				else:
					depth = os.path.join(scene, "depth", "%04d.npy" % idx)
					normal = os.path.join(scene, "normal", "%04d_normal.npy" % idx)

				pose_tgt = np.loadtxt(os.path.join(scene, "pose", "%04d.txt" % idx))

				do_nan_tgt = False
				nan_pose_tgt = np.sum(np.isnan(pose_tgt) | np.isinf(pose_tgt))
				if nan_pose_tgt>0:
					do_nan_tgt = True

				sample = {'intrinsics': intrinsics, 'tgt': img, 'tgt_depth': depth, 'tgt_normal': normal, 'ref_depths': [], 'ref_imgs': [], 'ref_poses': [], 'path': []}				
				sample['path'] = os.path.join(scene , img[:-4])

				if idx < self.seq_gap:
					shifts = list(range(idx,idx+(self.seq_length-1)*self.seq_gap+1,self.seq_gap))
					shifts.remove(idx) #.pop(i)
				elif idx >= end_idx - self.seq_gap:
					shifts = list(range(idx,end_idx,self.seq_gap))
					shifts = list(range(idx-(self.seq_length-1)*self.seq_gap,idx+1,self.seq_gap))
					shifts.remove(idx)
				else:
					if self.seq_length%2 == 1:
						demi_length = self.seq_length//2
						if (idx>=demi_length*self.seq_gap) and (idx<end_idx- demi_length*self.seq_gap):
							shifts = list(range(idx- (demi_length)*self.seq_gap, idx+(demi_length)*self.seq_gap+1,self.seq_gap))
						elif idx<demi_length*self.seq_gap:
							
							diff_demi = (demi_length-idx//self.seq_gap)
							shifts = list(range(idx- (demi_length-diff_demi)*self.seq_gap, idx+(demi_length+diff_demi)*self.seq_gap+1,self.seq_gap))
						elif idx>=end_idx- demi_length*self.seq_gap:
						   
							diff_demi = (demi_length-(end_idx-idx-1)//self.seq_gap)
							shifts = list(range(idx- (demi_length+diff_demi)*self.seq_gap, idx+(demi_length-diff_demi)*self.seq_gap+1,self.seq_gap))
						else:
							print('Error')
						shifts.remove(idx)
					else:
						#2 scenarios
						demi_length = self.seq_length//2
						if (idx >= demi_length*self.seq_gap) and (idx < end_idx- demi_length*self.seq_gap):
							shifts = list(range(idx - demi_length*self.seq_gap, idx + (demi_length-1)*self.seq_gap+1, self.seq_gap))
						elif idx < demi_length*self.seq_gap:	
							diff_demi = (demi_length-idx//self.seq_gap)
							shifts = list(range(idx- (demi_length-diff_demi)*self.seq_gap, idx+(demi_length+diff_demi-1)*self.seq_gap+1,self.seq_gap))
						elif idx>=end_idx- demi_length*self.seq_gap:
						   
							diff_demi = (demi_length-(end_idx-idx-1)//self.seq_gap)
							shifts = list(range(idx- (demi_length+diff_demi-1)*self.seq_gap, idx+(demi_length-diff_demi)*self.seq_gap+1,self.seq_gap))
						else:
							print('Error')
						shifts.remove(idx)

				do_nan = False
				try:
					for j in shifts:
						pose_src = np.loadtxt(os.path.join(scene, "pose", "%04d.txt" % j))
						pose_rel =  np.linalg.inv(pose_src) @ pose_tgt
						pose = pose_rel[:3,:].reshape((1,3,4)).astype(np.float32)
						sample['ref_poses'].append(pose)

						sample['ref_imgs'].append(os.path.join(scene, "color", "%04d.jpg" % j))
						if 'test' in self.ttype:
							sample['ref_depths'].append(os.path.join(scene, "depth", "%04d.png" % j))
						else:
							sample['ref_depths'].append(os.path.join(scene, "depth", "%04d.npy" % j))

						nan_pose = np.sum(np.isnan(pose)) + np.sum(np.isinf(pose))
						if nan_pose>0:
							do_nan = True
						
					if not do_nan_tgt and not do_nan:
						sequence_set.append(sample)
				except:
					continue
				
		file_pickle = self.scene_list_path+ '_len_'+str(self.seq_length)+ '_gap_'+str(self.seq_gap)+'.pickle'
		with open(file_pickle, 'wb') as handle:
			pickle.dump(sequence_set, handle, protocol=pickle.HIGHEST_PROTOCOL)

		self.samples = sequence_set

	def __getitem__(self, index):
		sample = self.samples[index]
		tgt_img = load_as_float(sample['tgt'])
		if 'test' in self.ttype:
			tgt_depth = cv2.imread(sample['tgt_depth'],-1).astype(np.float32) / 1000.0
			tgt_normal = np.tile(np.expand_dims(np.ones_like(tgt_depth), -1), (1,1,3))
		else:
			tgt_depth = np.load(sample['tgt_depth']).astype(np.float32) / 1000.0
			tgt_normal = np.load(sample['tgt_normal']).astype(np.float32)
			tgt_normal = 1.0 - tgt_normal * 2.0 # [-1, 1]
			tgt_normal[:,:,2] = np.abs(tgt_normal[:,:,2]) * -1.0

		ref_poses = sample['ref_poses']

		ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
		if 'test' in self.ttype:
			ref_depths = [cv2.imread(depth_img,-1).astype(np.float32)/1000.0 for depth_img in sample['ref_depths']]
		else:
			ref_depths = [np.load(depth_img).astype(np.float32)/1000.0 for depth_img in sample['ref_depths']]

		if self.transform is not None:
			imgs, depths, normals, intrinsics = self.transform([tgt_img] + ref_imgs, [tgt_depth] + ref_depths, [tgt_normal], np.copy(sample['intrinsics']))
			tgt_img = imgs[0]	 
			tgt_depth = depths[0]
			tgt_normal = normals[0]
			ref_imgs = imgs[1:]
			ref_depths = depths[1:]
		else:
			intrinsics = np.copy(sample['intrinsics'])
		intrinsics_inv = np.linalg.inv(intrinsics)
		return tgt_img, ref_imgs, tgt_normal, ref_poses, intrinsics, intrinsics_inv, tgt_depth, ref_depths


	def __len__(self):
		return len(self.samples)
