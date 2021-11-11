import torch
import os
import os.path as osp
from torch.utils.data import Dataset, DataLoader, dataset
# from torchvision import transforms, utils
from scipy.ndimage import affine_transform
import itertools
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
import cv2
import nrrd
from PIL import Image
from PIL import ImageEnhance

# functrion to load the data
# function to split the data into training and validation and testing

def intensity_jitter(img):
	if np.random.uniform(0, 1) > 0.5:
		img = img + img.max()*np.random.uniform(0, 0.2)
	if np.random.uniform(0, 1) > 0.5:
		contrastfactor = np.random.uniform(0.9, 1.1)
		img = img*contrastfactor
	return img

def get_random_rot(deg):
	deg = np.deg2rad(deg)
	theta_x = np.random.uniform(low=-1*deg, high=deg)
	theta_y = np.random.uniform(low=-1*deg, high=deg)
	theta_z = np.random.uniform(low=-1*deg, high=deg)
	R1 = np.eye(4)
	R1[1, 1] = np.cos(theta_x)
	R1[2, 2] = np.cos(theta_x)
	R1[1, 2] = -1*np.sin(theta_x)
	R1[2, 1] = np.sin(theta_x)
	R2 = np.eye(4)
	R2[0, 0] = np.cos(theta_y)
	R2[2, 2] = np.cos(theta_y)
	R2[2, 0] = -1*np.sin(theta_y)
	R2[0, 2] = np.sin(theta_y)
	R3 = np.eye(4)
	R3[1, 1] = np.cos(theta_z)
	R3[2, 2] = np.cos(theta_z)
	R3[1, 2] = -1*np.sin(theta_z)
	R3[2, 1] = np.sin(theta_z)
	R = np.matmul(np.matmul(R1, R2), R3)
	return R

def get_affine(deg, range):
	R = get_random_rot(deg)
	del_x = np.random.uniform(low=-1*range, high=range)
	del_y = np.random.uniform(low=-1*range, high=range)
	del_z = np.random.uniform(low=-1*range, high=range)
	T = np.eye(4)
	R[0, 3] = del_x
	R[1, 3] = del_y
	R[2, 3] = del_z
	return R

def get_dataset(model_type, data_dir, batch_size, file_type, data_type, noise):

	if model_type == "2d":
		dataset = image_data_2d(data_dir, file_type=file_type, data_type=data_type, noise=noise)
	if data_type == "train":
		shuffle = True
	else:
		shuffle = False

	return DataLoader(
	  dataset,
	  batch_size=batch_size,
	  shuffle=shuffle,
	  pin_memory=True,
	  drop_last=True
	)



class image_data_2d(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath, file_type="npy", data_type='train',  noise=False):
		'''
		Dataloader for 2D image datasets --> this is for training and validation
		--> file_type denotes the type in which images are stored, it can be "npy" or "png/jpg"
		--> data_type can be train validation or test
		The directory should have three directories --> train validation or test
		'''

		self.inDir = dirpath
		self.noise = noise
		folder_path = osp.join(dirpath, data_type)
		self.data = [osp.join(folder_path, filep) for filep in os.listdir(folder_path)]
		self.size = len(self.data)
		if self.noise:
			self.noiseVars = np.random.uniform(low=0.1, high=2, size=[len(self.data), 2])

	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, index):

		tmpid = torch.randperm(self.size)
		img_path_s = self.data[index % self.size]
		img_path_t = self.data[tmpid[0] % self.size]
		imgS = np.load(img_path_s)
		imgT = np.load(img_path_t)
		if self.noise:
			curNoise = self.noiseVars[index % self.size, :]
			imgS = imgS + curNoise[0]*np.random.normal(size=imgS.shape)
			imgT = imgT + curNoise[1]*np.random.normal(size=imgT.shape)

		imgS = torch.from_numpy(imgS.astype(np.float64))
		imgS = imgS.permute(2, 0, 1)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(2, 0, 1)
		# combine imgS and imgT 
		img = torch.cat([imgS, imgT], 0)

		return img




class cranioData3D_Pairs(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath, train=None, noise=False, indiv=False):
		self.inDir = dirpath
		self.noise = noise
		self.indiv = indiv
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs1') if train else osp.join(dirpath, 'TrainPairs1')
		self.data = sorted([osp.join(folder_path, filep) for filep in os.listdir(folder_path)])
		np.save('trainData.npy', self.data)
		if self.indiv:
			self.pairIndices = np.zeros([len(self.data), 2])
			for i in range(len(self.data)):
				self.pairIndices[i, 0] = 0
				self.pairIndices[i, 1] = i
		else:
			self.pairIndices = np.zeros([len(self.data)**2, 2])
			tempPrd = itertools.product(np.arange(len(self.data)), np.arange(len(self.data)))
			count = 0
			if self.noise:
				self.noiseVars = np.random.uniform(low=0.1, high=2, size=[len(self.data)**2, 2])
			for i in tempPrd:
				self.pairIndices[count, 0] = i[0]
				self.pairIndices[count, 1] = i[1]
				count += 1

	def __len__(self):
		return len(self.pairIndices)
	
	def __getitem__(self, index):
		idx = self.pairIndices[index % len(self.pairIndices), :]
		img_path_s = self.data[int(idx[0])]
		img_path_t = self.data[int(idx[1])]
		[imgS,f] = nrrd.read(img_path_s)
		[imgT,f] = nrrd.read(img_path_t) 
		imgS = imgS.reshape(imgS.shape[0], imgS.shape[1], imgS.shape[2], 1)
		imgT = imgT.reshape(imgT.shape[0], imgT.shape[1], imgS.shape[2], 1)
		if self.noise:
			curNoise = self.noiseVars[index % len(self.pairIndices), :]
			imgS = imgS + curNoise[0]*np.random.normal(size=imgS.shape)
			imgT = imgT + curNoise[1]*np.random.normal(size=imgT.shape)
		mean = 49.9
		std = 163.9
		imgS = (imgS - mean)/std
		imgT = (imgT - mean)/std
		imgS = torch.from_numpy(imgS.astype(np.float64))
		# # print(imgS.shape)
		imgS = imgS.permute(3, 0, 1, 2)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(3, 0, 1, 2)
		img = torch.cat([imgS, imgT], 0)
		return img

class cardiacData3D_Pairs(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath, mean, std,train=None, noise=False, indiv=False):
		self.imgmean = mean
		self.imgstd = std
		self.inDir = dirpath
		self.noise = noise
		self.indiv = indiv
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs') if train else osp.join(dirpath, 'ValPairs')
		self.data = sorted([osp.join(folder_path, filep) for filep in os.listdir(folder_path)])
		np.save('trainData.npy', self.data)
		if self.indiv:
			self.pairIndices = np.zeros([len(self.data), 2])
			for i in range(len(self.data)):
				self.pairIndices[i, 0] = 0
				self.pairIndices[i, 1] = i
		else:
			self.pairIndices = np.zeros([len(self.data)**2, 2])
			tempPrd = itertools.product(np.arange(len(self.data)), np.arange(len(self.data)))
			count = 0
			if self.noise:
				self.noiseVars = np.random.uniform(low=0.1, high=2, size=[len(self.data)**2, 2])
			for i in tempPrd:
				self.pairIndices[count, 0] = i[0]
				self.pairIndices[count, 1] = i[1]
				count += 1

	def __len__(self):
		return len(self.pairIndices)
	
	def __getitem__(self, index):
		idx = self.pairIndices[index % len(self.pairIndices), :]
		img_path_s = self.data[int(idx[0])]
		img_path_t = self.data[int(idx[1])]
		[imgS,f] = nrrd.read(img_path_s)
		[imgT,f] = nrrd.read(img_path_t) 
		imgS = imgS.reshape(imgS.shape[0], imgS.shape[1], imgS.shape[2], 1)
		imgT = imgT.reshape(imgT.shape[0], imgT.shape[1], imgS.shape[2], 1)
		if self.noise:
			curNoise = self.noiseVars[index % len(self.pairIndices), :]
			imgS = imgS + curNoise[0]*np.random.normal(size=imgS.shape)
			imgT = imgT + curNoise[1]*np.random.normal(size=imgT.shape)
		imgS = (imgS - self.imgmean)/self.imgstd
		imgT = (imgT - self.imgmean)/self.imgstd
		imgS = torch.from_numpy(imgS.astype(np.float64))
		# # print(imgS.shape)
		imgS = imgS.permute(3, 0, 1, 2)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(3, 0, 1, 2)
		img = torch.cat([imgS, imgT], 0)
		return img

class femurs_ss(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath,  mean, std, train=None, noise=False, indiv=False):
		self.imgmean = mean
		self.imgstd = std
		self.inDir = dirpath
		self.noise = noise
		self.indiv = indiv
		
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs') if train else osp.join(dirpath, 'NormalPairs')
		self.data = sorted([osp.join(folder_path, filep) for filep in os.listdir(folder_path)])
		self.data_fnm = sorted(os.listdir(folder_path))
		self.seg_dict = self.get_seg()
		# np.save('trainData.npy', self.data)
		if self.indiv:
			self.pairIndices = np.zeros([len(self.data), 2])
			for i in range(len(self.data)):
				self.pairIndices[i, 0] = 0
				self.pairIndices[i, 1] = i
		else:
			self.pairIndices = np.zeros([len(self.data)**2, 2])
			tempPrd = itertools.product(np.arange(len(self.data)), np.arange(len(self.data)))
			count = 0
			if self.noise:
				self.noiseVars = np.random.uniform(low=0.1, high=2, size=[len(self.data)**2, 2])
			for i in tempPrd:
				self.pairIndices[count, 0] = i[0]
				self.pairIndices[count, 1] = i[1]
				count += 1

	def get_seg(self):
		seg_path = osp.join(self.inDir, 'DTFull')
		self.seg_files = sorted([osp.join(seg_path, filep) for filep in os.listdir(seg_path)])
		self.seg_fnms = np.array(sorted(os.listdir(seg_path)))
		seg_dict = {}
		for i in range(len(self.data)):
			fnm = self.data_fnm[i]
			if np.isin(fnm, self.seg_fnms):
				idx = np.where(self.seg_fnms == fnm)[0][0]
				seg_dict[self.data[i]] = self.seg_files[idx]
			else:
				seg_dict[self.data[i]] = 'none'
		return seg_dict

		
	def __len__(self):
		return len(self.pairIndices)
	
	def __getitem__(self, index):
		idx = self.pairIndices[index % len(self.pairIndices), :]
		img_path_s = self.data[int(idx[0])]
		img_path_t = self.data[int(idx[1])]
		seg_path_s = self.seg_dict[img_path_s]
		seg_path_t = self.seg_dict[img_path_t]
		
		[imgS,f] = nrrd.read(img_path_s)
		[imgT,f] = nrrd.read(img_path_t) 
		imgS = imgS.reshape(imgS.shape[0], imgS.shape[1], imgS.shape[2], 1)
		imgT = imgT.reshape(imgT.shape[0], imgT.shape[1], imgS.shape[2], 1)
		if self.noise:
			curNoise = self.noiseVars[index % len(self.pairIndices), :]
			imgS = imgS + curNoise[0]*np.random.normal(size=imgS.shape)
			imgT = imgT + curNoise[1]*np.random.normal(size=imgT.shape)
		
		imgS = (imgS - self.imgmean)/self.imgstd
		imgT = (imgT - self.imgmean)/self.imgstd
		imgS = torch.from_numpy(imgS.astype(np.float64))
		# # print(imgS.shape)
		imgS = imgS.permute(3, 0, 1, 2)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(3, 0, 1, 2)
		img = torch.cat([imgS, imgT], 0)

		if seg_path_s != 'none' and seg_path_t != 'none':
			[segS,f] = nrrd.read(seg_path_s)
			[segT,f] = nrrd.read(seg_path_t) 
			
			try:
				segS = segS.reshape(segS.shape[0], segS.shape[1], segS.shape[2], 1)
				segT = segT.reshape(segT.shape[0], segT.shape[1], segS.shape[2], 1)
			except:
				import pdb; pdb.set_trace()
			segS = torch.from_numpy(segS.astype(np.float64))
			segS = segS.permute(3, 0, 1, 2)
			segS = torch.clamp(segS, 0, 1)
			segT = torch.from_numpy(segT.astype(np.float64))
			segT = segT.permute(3, 0, 1, 2)
			segT = torch.clamp(segT, 0, 1)
			seg = torch.cat([segS, segT], 0)
			msk = 1
		else:
			segS = torch.zeros(imgS.shape[0], imgS.shape[1], imgS.shape[2], imgS.shape[3]) + torch.rand(imgS.shape[0], imgS.shape[1], imgS.shape[2], imgS.shape[3])*0.001
			segT = torch.zeros(imgS.shape[0], imgT.shape[1], imgS.shape[2], imgS.shape[3]) + torch.rand(imgS.shape[0], imgS.shape[1], imgS.shape[2], imgS.shape[3])*0.001
			seg = torch.cat([segS, segT], 0)
			msk = 0
		return [img, seg, msk]

class cardiacData3D_Pairs_SS(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath,  mean, std, train=None, noise=False, indiv=False):
		self.imgmean = mean
		self.imgstd = std
		self.inDir = dirpath
		self.noise = noise
		self.indiv = indiv
		self.kernel = np.ones((3,3),np.float32)/27
		self.train = train
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs') if train else osp.join(dirpath, 'images')
		self.data = sorted([osp.join(folder_path, filep) for filep in os.listdir(folder_path)])
		self.data_fnm = sorted(os.listdir(folder_path))
		self.seg_dict = self.get_seg()
		np.save('trainData_lge_images.npy', self.data)
		if self.indiv:
			self.pairIndices = np.zeros([len(self.data), 2])
			for i in range(len(self.data)):
				self.pairIndices[i, 0] = 0
				self.pairIndices[i, 1] = i
		else:
			self.pairIndices = np.zeros([len(self.data)**2, 2])
			tempPrd = itertools.product(np.arange(len(self.data)), np.arange(len(self.data)))
			count = 0
			if self.noise:
				self.noiseVars = np.random.uniform(low=0.1, high=2, size=[len(self.data)**2, 2])
			for i in tempPrd:
				self.pairIndices[count, 0] = i[0]
				self.pairIndices[count, 1] = i[1]
				count += 1

	def get_seg(self):
		seg_path = osp.join(self.inDir, 'DT_all')
		self.seg_files = sorted([osp.join(seg_path, filep) for filep in os.listdir(seg_path)])
		self.seg_fnms = np.array(sorted(os.listdir(seg_path)))
		seg_dict = {}
		for i in range(len(self.data)):
			fnm = self.data_fnm[i]
			if np.isin(fnm, self.seg_fnms):
				idx = np.where(self.seg_fnms == fnm)[0][0]
				seg_dict[self.data[i]] = self.seg_files[idx]
			else:
				seg_dict[self.data[i]] = 'none'
		return seg_dict

		
	def __len__(self):
		return len(self.pairIndices)
	
	def __getitem__(self, index):
		idx = self.pairIndices[index % len(self.pairIndices), :]
		img_path_s = self.data[int(idx[0])]
		img_path_t = self.data[int(idx[1])]
		seg_path_s = self.seg_dict[img_path_s]
		seg_path_t = self.seg_dict[img_path_t]
		# import pdb; pdb.set_trace()
		self.affine_mat_s = get_affine(5, 2)
		self.affine_mat_t = get_affine(5, 2)
		[imgS,f] = nrrd.read(img_path_s)
		[imgT,f] = nrrd.read(img_path_t) 
		imgS = imgS.reshape(imgS.shape[0], imgS.shape[1], imgS.shape[2], 1)
		imgT = imgT.reshape(imgT.shape[0], imgT.shape[1], imgS.shape[2], 1)
		if self.noise:
			curNoise = self.noiseVars[index % len(self.pairIndices), :]
			imgS = imgS + curNoise[0]*np.random.normal(size=imgS.shape)
			imgT = imgT + curNoise[1]*np.random.normal(size=imgT.shape)
		
		# imgS = (imgS - self.imgmean)/self.imgstd
		# imgT = (imgT - self.imgmean)/self.imgstd
		imgS = torch.from_numpy(imgS.astype(np.float64))
		# # print(imgS.shape)
		imgS = imgS.permute(3, 0, 1, 2)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(3, 0, 1, 2)
		img = torch.cat([imgS, imgT], 0)		

		if seg_path_s != 'none' and seg_path_t != 'none':
			[segS,f] = nrrd.read(seg_path_s)
			[segT,f] = nrrd.read(seg_path_t) 
			if np.random.uniform(0, 1) > 0.5 and self.train:
				imgT = affine_transform(imgT, self.affine_mat_t)
				segT = affine_transform(segT, self.affine_mat_t)
				# imgT = img_jitter(imgT)
				# print('tgt')
				# import pdb; pdb.set_trace()
			if np.random.uniform(0, 1) > 0.5 and self.train:
				# imgS = img_jitter(imgT)
				imgS = affine_transform(imgS, self.affine_mat_s)
				segS = affine_transform(segS, self.affine_mat_s)
			#	 # import pdb; pdb.set_trace()
			# imgS = intensity_jitter(imgS)
			# imgT = intensity_jitter(imgT)
			segS = cv2.filter2D(segS,-1,self.kernel)
			segT = cv2.filter2D(segT,-1,self.kernel)
			try:
				segS = segS.reshape(segS.shape[0], segS.shape[1], segS.shape[2], 1)
				segT = segT.reshape(segT.shape[0], segT.shape[1], segS.shape[2], 1)
			except:
				import pdb; pdb.set_trace()
			segS = torch.from_numpy(segS.astype(np.float64))
			segS = segS.permute(3, 0, 1, 2)
			segS = torch.clamp(segS, 0, 1)
			segT = torch.from_numpy(segT.astype(np.float64))
			segT = segT.permute(3, 0, 1, 2)
			segT = torch.clamp(segT, 0, 1)
			seg = torch.cat([segS, segT], 0)
		else:
			segS = torch.zeros(imgS.shape[0], imgS.shape[1], imgS.shape[2], 1).float()
			segT = torch.zeros(imgS.shape[0], imgT.shape[1], imgS.shape[2], 1).float()
			seg = torch.cat([segS, segT], 0)
		
		
		return [img, seg]

class cardiacData3D_Pairs_SS_DT(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath,  mean, std, train=None, noise=False, indiv=False):
		self.imgmean = mean
		self.imgstd = std
		self.inDir = dirpath
		self.noise = noise
		self.indiv = indiv
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs') if train else osp.join(dirpath, 'ValPairs')
		self.data = sorted([osp.join(folder_path, filep) for filep in os.listdir(folder_path)])
		self.data_fnm = sorted(os.listdir(folder_path))
		self.dt_dict = self.get_seg()
		np.save('trainData.npy', self.data)
		if self.indiv:
			self.pairIndices = np.zeros([len(self.data), 2])
			for i in range(len(self.data)):
				self.pairIndices[i, 0] = 0
				self.pairIndices[i, 1] = i
		else:
			self.pairIndices = np.zeros([len(self.data)**2, 2])
			tempPrd = itertools.product(np.arange(len(self.data)), np.arange(len(self.data)))
			count = 0
			if self.noise:
				self.noiseVars = np.random.uniform(low=0.1, high=2, size=[len(self.data)**2, 2])
			for i in tempPrd:
				self.pairIndices[count, 0] = i[0]
				self.pairIndices[count, 1] = i[1]
				count += 1

	def get_seg(self):
		dt_path = osp.join(self.inDir, 'DT')
		self.dt_files = sorted([osp.join(dt_path, filep) for filep in os.listdir(dt_path)])
		self.dt_fnms = np.array(sorted(os.listdir(dt_path)))
		dt_dict = {}
		for i in range(len(self.data)):
			fnm = self.data_fnm[i]
			if np.isin(fnm, self.dt_fnms):
				idx = np.where(self.dt_fnms == fnm)[0][0]
				dt_dict[self.data[i]] = self.dt_files[idx]
			else:
				dt_dict[self.data[i]] = 'none'
		return dt_dict

		
	def __len__(self):
		return len(self.pairIndices)
	
	def __getitem__(self, index):
		idx = self.pairIndices[index % len(self.pairIndices), :]
		img_path_s = self.data[int(idx[0])]
		img_path_t = self.data[int(idx[1])]
		dt_path_s = self.dt_dict[img_path_s]
		dt_path_t = self.dt_dict[img_path_t]
		
		[imgS,f] = nrrd.read(img_path_s)
		[imgT,f] = nrrd.read(img_path_t) 
		imgS = imgS.reshape(imgS.shape[0], imgS.shape[1], imgS.shape[2], 1)
		imgT = imgT.reshape(imgT.shape[0], imgT.shape[1], imgS.shape[2], 1)
		if self.noise:
			curNoise = self.noiseVars[index % len(self.pairIndices), :]
			imgS = imgS + curNoise[0]*np.random.normal(size=imgS.shape)
			imgT = imgT + curNoise[1]*np.random.normal(size=imgT.shape)
		imgS = (imgS - self.imgmean)/self.imgstd
		imgT = (imgT - self.imgmean)/self.imgstd
		imgS = torch.from_numpy(imgS.astype(np.float64))
		# # print(imgS.shape)
		imgS = imgS.permute(3, 0, 1, 2)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(3, 0, 1, 2)
		img = torch.cat([imgS, imgT], 0)

		if dt_path_s != 'none':
			[segS,f] = nrrd.read(dt_path_s)
			# import pdb; pdb.set_trace()
			segS = segS.reshape(segS.shape[0], segS.shape[1], segS.shape[2], 1)
			segS = torch.from_numpy(segS.astype(np.float64))
			segS = segS.permute(3, 0, 1, 2)
		else:
			segS = torch.zeros(imgS.shape[0], imgS.shape[1], imgS.shape[2], imgS.shape[3])
		
		if dt_path_t != 'none':
			[segT,f] = nrrd.read(dt_path_t) 
			# import pdb; pdb.set_trace()
			segT = segT.reshape(segT.shape[0], segT.shape[1], segT.shape[2], 1)
			segT = torch.from_numpy(segT.astype(np.float64))
			segT = segT.permute(3, 0, 1, 2)
		else:
			segT = torch.zeros(imgS.shape[0], imgT.shape[1], imgS.shape[2], imgS.shape[3])
		
		seg = torch.cat([segS.float(), segT.float()], 0)
		return [img, seg]

class cranio3D_2D_projectionsData(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath, train=None, noise=False, indiv=False):
		self.inDir = dirpath
		self.noise = noise
		self.indiv = indiv
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs1') if train else osp.join(dirpath, 'ValPairs')
		self.data = sorted([osp.join(folder_path, filep) for filep in os.listdir(folder_path)])
		if train is not None:
			folder_path_2d = osp.join(dirpath, '2DProject/TrainPairs1') if train else osp.join(dirpath, '2DProject/ValPairs')
		self.data_2d = sorted([osp.join(folder_path_2d, filep) for filep in os.listdir(folder_path_2d)])
		# np.save('trainData.npy', self.data)
		if self.indiv:
			self.pairIndices = np.zeros([len(self.data), 2])
			for i in range(len(self.data)):
				self.pairIndices[i, 0] = 0
				self.pairIndices[i, 1] = i
		else:
			self.pairIndices = np.zeros([len(self.data)**2, 2])
			tempPrd = itertools.product(np.arange(len(self.data_2d)), np.arange(len(self.data_2d)))
			count = 0
			if self.noise:
				self.noiseVars = np.random.uniform(low=0.1, high=2, size=[len(self.data_2d)**2, 2])
			for i in tempPrd:
				self.pairIndices[count, 0] = i[0]
				self.pairIndices[count, 1] = i[1]
				count += 1

	def __len__(self):
		return len(self.pairIndices)
	
	def __getitem__(self, index):
		idx = self.pairIndices[index % len(self.pairIndices), :]
		img_path_s_2d = self.data_2d[int(idx[0])]
		img_path_t_2d = self.data_2d[int(idx[1])]
		img_path_s = self.data[int(idx[0])]
		img_path_t = self.data[int(idx[1])]
		[imgS,f] = nrrd.read(img_path_s)
		[imgT,f] = nrrd.read(img_path_t) 
		imgS_2d = np.load(img_path_s_2d)
		imgT_2d = np.load(img_path_t_2d) 
		imgS = imgS.reshape(imgS.shape[0], imgS.shape[1], imgS.shape[2], 1)
		imgT = imgT.reshape(imgT.shape[0], imgT.shape[1], imgS.shape[2], 1)
		# imgS_2d = imgS_2d.reshape(imgS_2d.shape[0], imgS_2d.shape[1], imgS_2d.shape[2], 1)
		# imgT_2d = imgT_2d.reshape(imgT_2d.shape[0], imgT_2d.shape[1], imgS_2d.shape[2], 1) 
		if self.noise:
			curNoise = self.noiseVars[index % len(self.pairIndices), :]
			imgS_2d = imgS_2d + curNoise[0]*np.random.normal(size=imgS_2d.shape)
			imgT_2d = imgT_2d + curNoise[1]*np.random.normal(size=imgT_2d.shape)
		mean = 49.9
		std = 163.9
		imgS = (imgS - mean)/std
		imgT = (imgT - mean)/std
		imgS = torch.from_numpy(imgS.astype(np.float64))
		imgS = imgS.permute(3, 0, 1, 2)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(3, 0, 1, 2)
		img = torch.cat([imgS, imgT], 0)

		imgS_2d = torch.from_numpy(imgS_2d.astype(np.float64))
		imgS_2d = imgS_2d.permute(2, 0, 1)
		imgT_2d = torch.from_numpy(imgT_2d.astype(np.float64))
		imgT_2d = imgT_2d.permute(2, 0, 1)
		img_2d = torch.cat([imgS_2d, imgT_2d], 0)
		return [img, img_2d]