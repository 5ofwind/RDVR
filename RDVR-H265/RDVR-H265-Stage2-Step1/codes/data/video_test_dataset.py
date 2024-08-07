import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import numpy as np

from .Guassian import Guassian_downsample

class VideoTestDataset(data.Dataset):
	"""
	A video test dataset. Support:
	Vid4
	REDS4
	Vimeo90K-Test

	no need to prepare LMDB files
	"""

	def __init__(self, opt):
		super(VideoTestDataset, self).__init__()
		self.opt = opt
		self.cache_data = opt['cache_data']
		self.half_N_frames = opt['N_frames'] // 2
		self.GT_root, self.LQ_root = opt['dataroot_GT'], opt['dataroot_LQ']
		self.data_type = self.opt['data_type']
		self.data_info = {'path_LQ': [], 'path_GT': [], 'folder': [], 'idx': [], 'border': []}

		if self.data_type == 'lmdb':
			raise ValueError('No need to use LMDB during validation/test.')
		#### Generate data info and cache data
		self.imgs_LQ, self.imgs_GT = {}, {}
		if opt['name'].lower() in ['vid4', 'reds4']:
			subfolders_LQ = util.glob_file_list(self.LQ_root)
			subfolders_GT = util.glob_file_list(self.GT_root)
			for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
				subfolder_name = osp.basename(subfolder_GT)
				img_paths_LQ = util.glob_file_list(subfolder_LQ)
				#print(subfolder_GT)
				img_paths_GT = util.glob_file_list(subfolder_GT)
				#print(img_paths_GT)
				max_idx = len(img_paths_LQ)
				assert max_idx == len(img_paths_GT), 'Different number of images in LQ and GT folders'

				if opt['pred_interval'] == 0:
					self.data_info['path_LQ'].extend(img_paths_LQ)
					self.data_info['path_GT'].extend(img_paths_GT)
					self.data_info['folder'].extend([subfolder_name] * max_idx)
					for i in range(max_idx):
						self.data_info['idx'].append('{}/{}'.format(i, max_idx))
				elif opt['pred_interval'] < 0:
					LQs = []
					GTs = []
					LQs.append(img_paths_LQ)
					GTs.append(img_paths_GT)

					#print(GTs)
					self.data_info['path_LQ'].extend(LQs)
					self.data_info['path_GT'].extend(GTs)
					self.data_info['folder'].extend([subfolder_name])
					self.data_info['idx'].append('{}/{}'.format(1, 1))
				else:
					self.pred_interval = opt['pred_interval']
					LQs = []
					GTs = []
					if max_idx % self.pred_interval == 1 or max_idx % self.pred_interval == 0:
						num_clip = max_idx // self.pred_interval
					else:
						num_clip = max_idx // self.pred_interval + 1

					for i in range(num_clip):
						if i != max_idx // self.pred_interval:
							LQs.append(img_paths_LQ[i * self.pred_interval: (i+1) * self.pred_interval + 1])
							GTs.append(img_paths_GT[i * self.pred_interval: (i+1) * self.pred_interval + 1])
						else:
							LQs.append(img_paths_LQ[i * self.pred_interval:])
							GTs.append(img_paths_GT[i * self.pred_interval:])

					self.data_info['path_LQ'].extend(LQs)
					self.data_info['path_GT'].extend(GTs)
					self.data_info['folder'].extend([subfolder_name] * num_clip)
					for i in range(max_idx // self.pred_interval + 1):
						self.data_info['idx'].append('{}/{}'.format(i, num_clip))

				if self.cache_data:
					######print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')                    
					self.imgs_LQ[subfolder_name] = util.read_img_seq(img_paths_LQ)
					self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT)
		elif opt['name'].lower() in ['vimeo90k-test']:

			subfolders_LQ0 = util.glob_file_list(self.LQ_root)
			subfolders_GT0 = util.glob_file_list(self.GT_root)
			for subfolder_LQ0, subfolder_GT0 in zip(subfolders_LQ0, subfolders_GT0):
				subfolders_LQ = util.glob_file_list(subfolder_LQ0)
				subfolders_GT = util.glob_file_list(subfolder_GT0)
				for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
					subfolder_name = osp.basename(subfolder_GT)
					img_paths_LQ = util.glob_file_list(subfolder_LQ)
					img_paths_GT = util.glob_file_list(subfolder_GT)
					max_idx = len(img_paths_LQ)
					assert max_idx == len(img_paths_GT), 'Different number of images in LQ and GT folders'

					if opt['pred_interval'] == 0:
						self.data_info['path_LQ'].extend(img_paths_LQ)
						self.data_info['path_GT'].extend(img_paths_GT)
						self.data_info['folder'].extend([subfolder_name] * max_idx)
						for i in range(max_idx):
							self.data_info['idx'].append('{}/{}'.format(i, max_idx))
					elif opt['pred_interval'] < 0:
						LQs = []
						GTs = []
						LQs.append(img_paths_LQ)
						GTs.append(img_paths_GT)
						self.data_info['path_LQ'].extend(LQs)
						self.data_info['path_GT'].extend(GTs)
						#print()

						#print(self.data_info['path_GT'])
						self.data_info['folder'].extend([subfolder_name])
						self.data_info['idx'].append('{}/{}'.format(1, 1))
					else:
						self.pred_interval = opt['pred_interval']
						LQs = []
						GTs = []
						if max_idx % self.pred_interval == 1 or max_idx % self.pred_interval == 0:
							num_clip = max_idx // self.pred_interval
						else:
							num_clip = max_idx // self.pred_interval + 1

						for i in range(num_clip):
							if i != max_idx // self.pred_interval:
								LQs.append(img_paths_LQ[i * self.pred_interval: (i+1) * self.pred_interval + 1])
								GTs.append(img_paths_GT[i * self.pred_interval: (i+1) * self.pred_interval + 1])
							else:
								LQs.append(img_paths_LQ[i * self.pred_interval:])
								GTs.append(img_paths_GT[i * self.pred_interval:])

						self.data_info['path_LQ'].extend(LQs)
						self.data_info['path_GT'].extend(GTs)
						self.data_info['folder'].extend([subfolder_name] * num_clip)
						for i in range(max_idx // self.pred_interval + 1):
							self.data_info['idx'].append('{}/{}'.format(i, num_clip))

					#self.data_info['path_LQ'][:]=self.data_info['path_LQ'][:][1:6]
					#self.data_info['path_GT'][:]=self.data_info['path_GT'][:][1:6]

					if self.cache_data:
						self.imgs_LQ[subfolder_name] = util.read_img_seq(img_paths_LQ)
						self.imgs_GT[subfolder_name] = util.read_img_seq(img_paths_GT)

			#pass  # TODO
		else:
			raise ValueError(
				'Not support video test dataset. Support Vid4, REDS4 and Vimeo90k-Test.')

	def __getitem__(self, index):
		path_LQ = self.data_info['path_LQ'][index]#[1:6]######################for vimeo: [1:6]
		path_GT = self.data_info['path_GT'][index]#[1:6]
		
		folder = self.data_info['folder'][index]
		idx, max_idx = self.data_info['idx'][index].split('/')
		idx, max_idx = int(idx), int(max_idx)

		if self.cache_data:
			# select_idx = util.index_generation(idx, max_idx, self.opt['N_frames'], padding=self.opt['padding'])
			# select_idx = [i for i in range(max_idx)]
			# imgs_LQ = self.imgs_LQ[folder].index_select(0, torch.LongTensor(select_idx))

			imgs_LQ = self.imgs_LQ[folder][idx]
			img_GT = self.imgs_GT[folder][idx]
		else:
            
            
            
			######print('BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB')  
			img_GT = util.read_img_seq(path_GT)
			#print(img_GT.shape)                        
			######imgs_LQ = util.read_img_seq(path_LQ)
			img_GT2=img_GT.numpy()         
			imgs_LQ=np.random.rand(img_GT2.shape[0],img_GT2.shape[1],img_GT2.shape[2]//2,img_GT2.shape[3]//2)#img_GT2            
			for i in range(img_GT.shape[0]):
				pp=img_GT2[i,:,:,:]#np.squeeze(img_GT2[i,:,:,:],0)
                
                
				#pp=np.transpose(pp,(1,2,0))
				#print(pp.shape)
				#print(img_GT2[i,:,:,:].shape)                                     
				#ppp=np.transpose(util.imresize_np(pp, 1 / 2, True),(2,0,1))
                
				pp=torch.tensor(pp).unsqueeze(1)
				#print(pp.shape)
				ppp=Guassian_downsample(pp,scale=2).squeeze(1).numpy()
                
                
                
                
				imgs_LQ[i,:,:,:] = ppp #np.expand_dims(ppp,0)     
                
			imgs_LQ=torch.tensor(imgs_LQ)            
            


		#imgs_LQ=imgs_LQ[1:6,:,:,:]
		#img_GT=img_GT[1:6,:,:,:]
		#print(img_GT.shape)

		return {
			'LQ': imgs_LQ,
			'GT': img_GT,
			'folder': folder,
			'idx': self.data_info['idx'][index],
			'LQ_path': path_LQ,
			'GT_path': path_GT
		}

	def __len__(self):
		return len(self.data_info['path_GT'])
