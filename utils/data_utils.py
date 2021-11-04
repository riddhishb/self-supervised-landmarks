import torch
import os
import os.path as osp
from torch.utils.data import Dataset, DataLoader
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

def get_dataset(dataType, datamean, datastd, dataDirPath, batchSz, cv=None, train=True,  noise=False, indiv=False):

	if dataType == 'phantom_atlas':
		dataset = phantomImageDataset(dataDirPath, train=train)
	if dataType == 'phantom_pairs':
		dataset = phantomImageDataset_Pairs(dataDirPath, train=train,  noise=noise)
	if dataType == 'phantom_noisy':
		dataset = phantomImageDataset_Noisy(dataDirPath, train=train, )
	if dataType == 'boxbump_pairs':
		dataset = boxbumpDataset_Pairs(dataDirPath, train=train)
	if dataType == 'diatoms_pairs':
		dataset = diatomsDataset_Pairs(dataDirPath, cv, train=train, indiv=indiv)
	if dataType == 'diatoms_pairs_ss':
		dataset = diatomsDataset_Pairs_SS(dataDirPath, train=train, indiv=indiv)
	if dataType == 'diatoms_pairs_ss_dt':
		dataset = diatomsDataset_Pairs_SS_DT(dataDirPath, train=train, indiv=indiv)
	if dataType == 'cine_pairs':
		dataset = cineData_Pairs(dataDirPath, train=train)
	if dataType == 'brain2d_pairs':
		dataset = brainData2D_Pairs(dataDirPath, train=train)
	if dataType == 'brain2d_pairs_ss':
		dataset = brainData2D_Pairs_SS(dataDirPath, train=train)
	if dataType == 'brain2d_pairs_dt':
		dataset = brainData2D_Pairs_DT(dataDirPath, datamean, datastd, train=train)
	if dataType == 'cranio2d_pairs':
		dataset = cranioData2D_Pairs(dataDirPath, train=train)
	if dataType == 'cranio3d_pairs':
		dataset = cranioData3D_Pairs(dataDirPath, train=train, indiv=indiv)
	if dataType == 'cardiac3d_pairs':
		dataset = cardiacData3D_Pairs(dataDirPath, datamean, datastd, train=train, indiv=indiv)
	if dataType == 'cardiac3d_pairs_semisupervised':
		dataset = cardiacData3D_Pairs_SS(dataDirPath, datamean, datastd, train=train, indiv=indiv)
	if dataType == 'femurs_ss':
		dataset = femurs_ss(dataDirPath, datamean, datastd, train=train, indiv=indiv)
	if dataType == 'cardiac3D_pairs_semisupervised_dt':
		dataset = cardiacData3D_Pairs_SS_DT(dataDirPath, datamean, datastd, train=train, indiv=indiv)
	if dataType == 'cranio3d_pairs_projections':
		dataset = cranio3D_2D_projectionsData(dataDirPath, train=train, indiv=indiv)
		# transform = ransforms.Normalize((0.1307,), (0.3081,))

	return DataLoader(
	  dataset,
	  batch_size=batchSz,
	  shuffle=train,
	  pin_memory=True,
	  drop_last=True
	#   normalize = 
	)

class phantomImageDataset(Dataset):
	'''
	Data loader class for the 
	'''
	def __init__(self, dirpath, train=None):
		self.inDir = dirpath
		if train is not None:
			folder_path = osp.join(dirpath, 'Train') if train else osp.join(dirpath, 'Val')
		self.data = [osp.join(folder_path, filep) for filep in os.listdir(folder_path)]
	
	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, index):
		img_path = self.data[index % len(self.data)]
		img = np.load(img_path)
		# print(img.shape)
		# img = np.array(Image.open(img_path))
		# img = img.reshape([img.shape[0], img.shape[1], 1])
		# img = np.array([img])
		img = torch.from_numpy(img.astype(np.double))
		img = img.permute(2, 0, 1)
		# img = transforms.ToTensor()(img)
		return img

class phantomImageDataset_Pairs(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath, train=None,  noise=False):
		self.inDir = dirpath
		self.noise = noise
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs') if train else osp.join(dirpath, 'ValPairs')
		self.data = [osp.join(folder_path, filep) for filep in os.listdir(folder_path)]
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
		imgS = np.load(img_path_s)
		imgT = np.load(img_path_t)
		
		if self.noise:
			curNoise = self.noiseVars[index % len(self.pairIndices), :]
			imgS = imgS + curNoise[0]*np.random.normal(size=imgS.shape)
			imgT = imgT + curNoise[1]*np.random.normal(size=imgT.shape)
		imgS = torch.from_numpy(imgS.astype(np.float64))
		imgS = imgS.permute(2, 0, 1)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(2, 0, 1)
		# combine imgS and imgT 
		img = torch.cat([imgS, imgT], 0)

		return img


class phantomImageDataset_Noisy(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath, train=None):
		self.inDir = dirpath
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs') if train else osp.join(dirpath, 'ValPairs')
		self.data = sorted([osp.join(folder_path, filep) for filep in os.listdir(folder_path)])
		
		self.pairIndices = np.zeros([len(self.data)**2, 2])
		# compute sigma from noise range 
		self.noiseVars = np.random.uniform(low=0.1, high=2, size=[len(self.data)**2, 2])
		tempPrd = itertools.product(np.arange(len(self.data)), np.arange(len(self.data)))
		count = 0
		for i in tempPrd:
			self.pairIndices[count, 0] = i[0]
			self.pairIndices[count, 1] = i[1]
			count += 1

	def __len__(self):
		return len(self.pairIndices)
	
	def __getitem__(self, index):
		idx = self.pairIndices[index % len(self.pairIndices), :]
		curNoise = self.noiseVars[index % len(self.pairIndices), :]
		img_path_s = self.data[int(idx[0])]
		img_path_t = self.data[int(idx[1])]
		imgS = np.load(img_path_s)
		imgT = np.load(img_path_t)
		imgSnoisy = imgS + curNoise[0]*np.random.normal(size=imgS.shape)
		imgTnoisy = imgT + curNoise[1]*np.random.normal(size=imgT.shape)
		curNoise = torch.from_numpy(curNoise.astype(np.float64))
		imgS = torch.from_numpy(imgS.astype(np.float64))
		imgS = imgS.permute(2, 0, 1)
		imgSnoisy = torch.from_numpy(imgSnoisy.astype(np.float64))
		imgSnoisy = imgSnoisy.permute(2, 0, 1)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(2, 0, 1)
		imgTnoisy = torch.from_numpy(imgTnoisy.astype(np.float64))
		imgTnoisy = imgTnoisy.permute(2, 0, 1)
		# combine imgS and imgT 
		img = torch.cat([imgS, imgT], 0)
		imgnoise = torch.cat([imgSnoisy, imgTnoisy], 0)
		return [img, imgnoise, curNoise]

class boxbumpDataset_Pairs(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath, train=None):
		self.inDir = dirpath
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs') if train else osp.join(dirpath, 'ValPairs')
		self.data = [osp.join(folder_path, filep) for filep in os.listdir(folder_path)]
		self.pairIndices = np.zeros([len(self.data)**2, 2])
		tempPrd = itertools.product(np.arange(len(self.data)), np.arange(len(self.data)))
		count = 0
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
		imgS = np.load(img_path_s)
		imgT = np.load(img_path_t)
		imgS = imgS.reshape(imgS.shape[0], imgS.shape[1], 1)
		imgT = imgT.reshape(imgT.shape[0], imgT.shape[1], 1)
		imgS = torch.from_numpy(imgS.astype(np.float64))
		# print(imgS.shape)
		imgS = imgS.permute(2, 0, 1)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(2, 0, 1)
		# combine imgS and imgT 
		img = torch.cat([imgS, imgT], 0)
		return img

class diatomsDataset_Pairs(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath, cvid, train=None, noise=False, indiv=False):
		self.inDir = dirpath
		self.noise = noise
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs') if train else osp.join(dirpath, 'ValPairs')

		self.data = sorted([osp.join(folder_path, filep) for filep in os.listdir(folder_path)])
		if cvid is not None:
			# import pdb; pdb.set_trace()
			self.data = np.array(self.data)
			self.data = self.data[cvid]

		np.save('diatoms.npy', self.data)
		if indiv:
			self.pairIndices = np.zeros([len(self.data), 2])
			for i in range(len(self.data)):
				self.pairIndices[i, 0] = 0
				self.pairIndices[i, 1] = i
		else:
			self.pairIndices = np.zeros([len(self.data)**2, 2])
			tempPrd = itertools.product(np.arange(len(self.data)), np.arange(len(self.data)))
			count = 0
			if self.noise:
				self.noiseVars = np.random.uniform(low=5, high=10, size=[len(self.data)**2, 2])
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
		imgS = np.load(img_path_s) 
		imgT = np.load(img_path_t)
		imgS = imgS.reshape(imgS.shape[0], imgS.shape[1], 1)
		imgT = imgT.reshape(imgT.shape[0], imgT.shape[1], 1)
		imgS = imgS * (1/(np.max(imgS) - np.min(imgS))) -  np.min(imgS) / (np.max(imgS) - np.min(imgS))
		imgS = imgS * 255
		imgT = imgT * (1/(np.max(imgT) - np.min(imgT))) -  np.min(imgT) / (np.max(imgT) - np.min(imgT))
		imgT = imgT * 255
		if self.noise:
			curNoise = self.noiseVars[index % len(self.pairIndices), :]
			imgS = imgS + curNoise[0]*np.random.normal(size=imgS.shape)
			imgT = imgT + curNoise[1]*np.random.normal(size=imgT.shape)
		# transform = transforms.Compose([transforms.ToTensor(),
		#						 transforms.Normalize([32.6], [69.4])])
		# imgS = transform(imgS)
		# transform = transforms.Compose([transforms.ToTensor(),
		#						 transforms.Normalize([32.6], [69.4])])
		# imgT = transform(imgT)

		imgS = torch.from_numpy(imgS.astype(np.float64))
		# print(imgS.shape)
		imgS = imgS.permute(2, 0, 1)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(2, 0, 1)
		# combine imgS and imgT 
		img = torch.cat([imgS, imgT], 0)
		return img

class diatomsDataset_Pairs_SS(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath, train=None, noise=False, indiv=False):
		self.inDir = dirpath
		self.noise = noise
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs') if train else osp.join(dirpath, 'ValPairs')
		self.data = sorted([osp.join(folder_path, filep) for filep in os.listdir(folder_path)])
		self.data_fnm = sorted(os.listdir(folder_path))
		np.save('diatoms.npy', self.data)
		self.seg_dict = self.get_seg()
		if indiv:
			self.pairIndices = np.zeros([len(self.data), 2])
			for i in range(len(self.data)):
				self.pairIndices[i, 0] = 0
				self.pairIndices[i, 1] = i
		else:
			self.pairIndices = np.zeros([len(self.data)**2, 2])
			tempPrd = itertools.product(np.arange(len(self.data)), np.arange(len(self.data)))
			count = 0
			if self.noise:
				self.noiseVars = np.random.uniform(low=5, high=10, size=[len(self.data)**2, 2])
			for i in tempPrd:
				self.pairIndices[count, 0] = i[0]
				self.pairIndices[count, 1] = i[1]
				count += 1

	def get_seg(self):
		seg_path = osp.join(self.inDir, 'Segmentations')
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
		imgS = np.load(img_path_s) 
		imgT = np.load(img_path_t)
		imgS = imgS.reshape(imgS.shape[0], imgS.shape[1], 1)
		imgT = imgT.reshape(imgT.shape[0], imgT.shape[1], 1)
		imgS = imgS * (1/(np.max(imgS) - np.min(imgS))) -  np.min(imgS) / (np.max(imgS) - np.min(imgS))
		imgS = imgS * 255
		imgT = imgT * (1/(np.max(imgT) - np.min(imgT))) -  np.min(imgT) / (np.max(imgT) - np.min(imgT))
		imgT = imgT * 255
		if self.noise:
			curNoise = self.noiseVars[index % len(self.pairIndices), :]
			imgS = imgS + curNoise[0]*np.random.normal(size=imgS.shape)
			imgT = imgT + curNoise[1]*np.random.normal(size=imgT.shape)
		# transform = transforms.Compose([transforms.ToTensor(),
		#						 transforms.Normalize([32.6], [69.4])])
		# imgS = transform(imgS)
		# transform = transforms.Compose([transforms.ToTensor(),
		#						 transforms.Normalize([32.6], [69.4])])
		# imgT = transform(imgT)

		imgS = torch.from_numpy(imgS.astype(np.float64))
		# print(imgS.shape)
		imgS = imgS.permute(2, 0, 1)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(2, 0, 1)
		# combine imgS and imgT 
		img = torch.cat([imgS, imgT], 0)
		if seg_path_s != 'none' and seg_path_t != 'none':
			segS = np.load(seg_path_s)
			segT = np.load(seg_path_t) 
			try:
				segS = segS.reshape(segS.shape[0], segS.shape[1], 1)
				segT = segT.reshape(segT.shape[0], segT.shape[1], 1)
			except:
				import pdb; pdb.set_trace()
			segS = torch.from_numpy(segS.astype(np.float64))
			segS = segS.permute(2, 0, 1)
			segT = torch.from_numpy(segT.astype(np.float64))
			segT = segT.permute(2, 0, 1)
			seg = torch.cat([segS, segT], 0).float()
			# import pdb; pdb.set_trace()
		else:
			segS = torch.zeros(imgS.shape[0], imgS.shape[1], imgS.shape[2])
			segT = torch.zeros(imgS.shape[0], imgT.shape[1], imgS.shape[2])
			seg = torch.cat([segS, segT], 0)
		return [img, seg]

class diatomsDataset_Pairs_SS_DT(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath, train=None, noise=False, indiv=False):
		self.inDir = dirpath
		self.noise = noise
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs') if train else osp.join(dirpath, 'ValPairs')
		self.data = sorted([osp.join(folder_path, filep) for filep in os.listdir(folder_path)])
		np.save('diatoms.npy', self.data)
		self.seg_dict = self.get_seg()
		if indiv:
			self.pairIndices = np.zeros([len(self.data), 2])
			for i in range(len(self.data)):
				self.pairIndices[i, 0] = 0
				self.pairIndices[i, 1] = i
		else:
			self.pairIndices = np.zeros([len(self.data)**2, 2])
			tempPrd = itertools.product(np.arange(len(self.data)), np.arange(len(self.data)))
			count = 0
			if self.noise:
				self.noiseVars = np.random.uniform(low=5, high=10, size=[len(self.data)**2, 2])
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
		dt_path_s = self.seg_dict[img_path_s]
		dt_path_t = self.seg_dict[img_path_t]
		imgS = np.load(img_path_s) 
		imgT = np.load(img_path_t)
		imgS = imgS.reshape(imgS.shape[0], imgS.shape[1], 1)
		imgT = imgT.reshape(imgT.shape[0], imgT.shape[1], 1)
		imgS = imgS * (1/(np.max(imgS) - np.min(imgS))) -  np.min(imgS) / (np.max(imgS) - np.min(imgS))
		imgS = imgS * 255
		imgT = imgT * (1/(np.max(imgT) - np.min(imgT))) -  np.min(imgT) / (np.max(imgT) - np.min(imgT))
		imgT = imgT * 255
		if self.noise:
			curNoise = self.noiseVars[index % len(self.pairIndices), :]
			imgS = imgS + curNoise[0]*np.random.normal(size=imgS.shape)
			imgT = imgT + curNoise[1]*np.random.normal(size=imgT.shape)
		# transform = transforms.Compose([transforms.ToTensor(),
		#						 transforms.Normalize([32.6], [69.4])])
		# imgS = transform(imgS)
		# transform = transforms.Compose([transforms.ToTensor(),
		#						 transforms.Normalize([32.6], [69.4])])
		# imgT = transform(imgT)

		imgS = torch.from_numpy(imgS.astype(np.float64))
		# print(imgS.shape)
		imgS = imgS.permute(2, 0, 1)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(2, 0, 1)
		# combine imgS and imgT 
		img = torch.cat([imgS, imgT], 0)
		if dt_path_s != 'none':
			segS = np.load(dt_path_s)
			segS = segS.reshape(segS.shape[0], segS.shape[1], 1)
			segS = torch.from_numpy(segS.astype(np.float64))
			segS = segS.permute(2, 0, 1)
		else:
			segS = torch.zeros(imgS.shape[0], imgS.shape[1], imgS.shape[2])
		
		if dt_path_t != 'none':
			segT = np.load(dt_path_t) 
			segT = segT.reshape(segT.shape[0], segT.shape[1], 1)
			segT = torch.from_numpy(segT.astype(np.float64))
			segT = segT.permute(2, 0, 1)
		else:
			segT = torch.zeros(imgS.shape[0], imgT.shape[1], imgS.shape[2])
		
		seg = torch.cat([segS, segT], 0)
		return [img, seg]

class cineData_Pairs(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath, train=None):
		self.inDir = dirpath
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs') if train else osp.join(dirpath, 'ValPairs')
		self.data = [osp.join(folder_path, filep) for filep in os.listdir(folder_path)]
		self.pairIndices = np.zeros([len(self.data)**2, 2])
		tempPrd = itertools.product(np.arange(len(self.data)), np.arange(len(self.data)))
		count = 0
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
		imgS = np.load(img_path_s)
		imgT = np.load(img_path_t) 
		imgS = imgS.reshape(imgS.shape[0], imgS.shape[1], 1)
		imgT = imgT.reshape(imgT.shape[0], imgT.shape[1], 1)
		imgS = torch.from_numpy(imgS.astype(np.float64))
		# print(imgS.shape)
		imgS = imgS.permute(2, 0, 1)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(2, 0, 1)
		# combine imgS and imgT 
		img = torch.cat([imgS, imgT], 0)
		return img

class brainData2D_Pairs(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath, train=None):
		self.inDir = dirpath
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs') if train else osp.join(dirpath, 'ValPairs')
		self.data = [osp.join(folder_path, filep) for filep in os.listdir(folder_path)]
		self.pairIndices = np.zeros([len(self.data)**2, 2])
		tempPrd = itertools.product(np.arange(len(self.data)), np.arange(len(self.data)))
		count = 0
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
		imgS = np.load(img_path_s)[:, :, 0]
		# import pdb; pdb.set_trace()
		# imgS = ndimage.gaussian_filter(imgS, sigma=(1, 1), order=0)
		imgT = np.load(img_path_t)[:, :, 0]
		# imgT = ndimage.gaussian_filter(imgT, sigma=(1, 1), order=0)
		imgS = imgS.reshape(imgS.shape[0], imgS.shape[1], 1)
		imgT = imgT.reshape(imgT.shape[0], imgT.shape[1], 1)
		imgS = imgS + 0.1*np.random.normal(size=imgS.shape)
		imgT = imgT + 0.1*np.random.normal(size=imgT.shape)
		imgS = torch.from_numpy(imgS.astype(np.float64))
		# print(imgS.shape)
		imgS = imgS.permute(2, 0, 1)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(2, 0, 1)
		# combine imgS and imgT 
		img = torch.cat([imgS, imgT], 0)
		return img

class brainData2D_Pairs_SS(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath, train=None):
		self.inDir = dirpath
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs') if train else osp.join(dirpath, 'ValPairs')
		self.data = [osp.join(folder_path, filep) for filep in os.listdir(folder_path)]
		self.data_fnm = sorted(os.listdir(folder_path))
		self.seg_dict = self.get_seg()
		self.kernel = np.ones((3,3),np.float32)/9
		self.pairIndices = np.zeros([len(self.data)**2, 2])
		tempPrd = itertools.product(np.arange(len(self.data)), np.arange(len(self.data)))
		count = 0
		for i in tempPrd:
			self.pairIndices[count, 0] = i[0]
			self.pairIndices[count, 1] = i[1]
			count += 1
	
	def get_seg(self):
		seg_path = osp.join(self.inDir, 'SegmentationsAll')
		print("THIS IS USING THIS: ", seg_path)
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
		imgS = np.load(img_path_s)[:, :, 0]
		# import pdb; pdb.set_trace()
		# imgS = ndimage.gaussian_filter(imgS, sigma=(1, 1), order=0)
		imgT = np.load(img_path_t)[:, :, 0]
		# imgT = ndimage.gaussian_filter(imgT, sigma=(1, 1), order=0)
		imgS = imgS.reshape(imgS.shape[0], imgS.shape[1], 1)
		imgT = imgT.reshape(imgT.shape[0], imgT.shape[1], 1)
		imgS = imgS + 0.01*np.random.normal(size=imgS.shape)
		imgT = imgT + 0.01*np.random.normal(size=imgT.shape)
		imgS = torch.from_numpy(imgS.astype(np.float64))
		# print(imgS.shape)
		imgS = imgS.permute(2, 0, 1)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(2, 0, 1)
		# combine imgS and imgT 
		img = torch.cat([imgS, imgT], 0)
		if seg_path_s != 'none' and seg_path_t != 'none':
			segS = np.load(seg_path_s)[:, :, 0].astype(np.float64)/255.0
			segT = np.load(seg_path_t)[:, :, 0].astype(np.float64)/255.0
			# import pdb; pdb.set_trace()
			segS = cv2.filter2D(segS,-1,self.kernel)
			segT = cv2.filter2D(segT,-1,self.kernel)
			try:
				segS = segS.reshape(segS.shape[0], segS.shape[1], 1)
				segS = segS + 0.01*np.random.normal(size=segS.shape)
				segT = segT.reshape(segT.shape[0], segT.shape[1], 1)
				segT = segT + 0.01*np.random.normal(size=segT.shape)
			except:
				import pdb; pdb.set_trace()
			segS = torch.from_numpy(segS.astype(np.float64))
			segS = segS.permute(2, 0, 1)
			segT = torch.from_numpy(segT.astype(np.float64))
			segT = segT.permute(2, 0, 1)
			seg = torch.cat([segS, segT], 0).float()
			# msk = torch.ones(imgS.shape[0], imgS.shape[1], imgS.shape[2])
			msk = 1
			# import pdb; pdb.set_trace()
		else:
			# msk = torch.zeros(imgS.shape[0], imgS.shape[1], imgS.shape[2])
			msk = 0
			segS = torch.zeros(imgS.shape[0], imgS.shape[1], imgS.shape[2])
			segT = torch.zeros(imgS.shape[0], imgT.shape[1], imgS.shape[2])
			seg = torch.cat([segS, segT], 0)

		return [img, seg, msk]

class brainData2D_Pairs_DT(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath, mean, std, train=None):
		self.inDir = dirpath
		self.imgmean = mean
		self.imgstd = std
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs') if train else osp.join(dirpath, 'ValTestPairs')
		self.data = [osp.join(folder_path, filep) for filep in os.listdir(folder_path)]
		self.data_fnm = sorted(os.listdir(folder_path))
		# np.save('brain2d_alltestval.npy', self.data)
		self.seg_dict = self.get_seg()
		self.kernel = np.ones((3,3),np.float32)/9
		self.pairIndices = np.zeros([len(self.data)**2, 2])
		tempPrd = itertools.product(np.arange(len(self.data)), np.arange(len(self.data)))
		count = 0
		for i in tempPrd:
			self.pairIndices[count, 0] = i[0]
			self.pairIndices[count, 1] = i[1]
			count += 1
	
	def get_seg(self):
		seg_path = osp.join(self.inDir, 'DT_cropped')
		print("THIS IS USING THIS: ", seg_path)
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
		imgS = np.load(img_path_s)[:, :, 0]
		# import pdb; pdb.set_trace()
		# imgS = ndimage.gaussian_filter(imgS, sigma=(1, 1), order=0)
		imgT = np.load(img_path_t)[:, :, 0]
		# imgT = ndimage.gaussian_filter(imgT, sigma=(1, 1), order=0)
		imgS = imgS.reshape(imgS.shape[0], imgS.shape[1], 1)
		imgT = imgT.reshape(imgT.shape[0], imgT.shape[1], 1)
		imgS = imgS + 0.01*np.random.normal(size=imgS.shape)
		imgT = imgT + 0.01*np.random.normal(size=imgT.shape)

		# imgS = (imgS - self.imgmean)/self.imgstd
		# imgT = (imgT - self.imgmean)/self.imgstd

		imgS = torch.from_numpy(imgS.astype(np.float64))
		# print(imgS.shape)
		imgS = imgS.permute(2, 0, 1)
		imgT = torch.from_numpy(imgT.astype(np.float64))
		imgT = imgT.permute(2, 0, 1)
		
		# combine imgS and imgT 
		img = torch.cat([imgS, imgT], 0)
		if seg_path_s != 'none' and seg_path_t != 'none':
			segS = np.load(seg_path_s)[:, :].astype(np.float64)
			segT = np.load(seg_path_t)[:, :].astype(np.float64)
			# import pdb; pdb.set_trace()
			segS = cv2.filter2D(segS,-1,self.kernel)
			segT = cv2.filter2D(segT,-1,self.kernel)
			try:
				segS = segS.reshape(segS.shape[0], segS.shape[1], 1)
				
				# segS = segS + 0.01*np.random.normal(size=segS.shape)
				segT = segT.reshape(segT.shape[0], segT.shape[1], 1)
				
				# segT = segT + 0.01*np.random.normal(size=segT.shape)
			except:
				import pdb; pdb.set_trace()
			segS = torch.from_numpy(segS.astype(np.float64))
			segS = segS.permute(2, 0, 1)
			segT = torch.from_numpy(segT.astype(np.float64))
			segT = segT.permute(2, 0, 1)
			segS = torch.clamp(segS, 0, 1)
			segT = torch.clamp(segT, 0, 1)
			seg = torch.cat([segS, segT], 0).float()
			# msk = torch.ones(imgS.shape[0], imgS.shape[1], imgS.shape[2])
			msk = 1
			# import pdb; pdb.set_trace()
		else:
			# msk = torch.zeros(imgS.shape[0], imgS.shape[1], imgS.shape[2])
			msk = 0
			segS = torch.zeros(imgS.shape[0], imgS.shape[1], imgS.shape[2]) + torch.rand(imgS.shape[0], imgS.shape[1], imgS.shape[2])*0.0001
			segT = torch.zeros(imgS.shape[0], imgT.shape[1], imgS.shape[2]) + torch.rand(imgS.shape[0], imgS.shape[1], imgS.shape[2])*0.0001
			seg = torch.cat([segS, segT], 0)

		return [img, seg, msk]


class cranioData2D_Pairs(Dataset):
	'''
	Data Loader class for Phantom Data Pairs
	'''
	def __init__(self, dirpath, train=None, noise=False):
		self.noise = noise
		self.inDir = dirpath
		if train is not None:
			folder_path = osp.join(dirpath, 'TrainPairs') if train else osp.join(dirpath, 'ValPairs')
		self.data = [osp.join(folder_path, filep) for filep in os.listdir(folder_path)]
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
		mask_path_s = img_path_s.replace('img', 'mask')
		mask_path_t = img_path_t.replace('img', 'mask')
		# imgnorm_path_s = img_path_s.replace('img', 'imgnorm')
		# imgnorm_path_t = img_path_t.replace('img', 'imgnorm')
		imgS = np.load(img_path_s)
		maskS = np.load(mask_path_s)
		# imgnormS = np.load(imgnorm_path_s)
		# imgS = ndimage.gaussian_filter(imgS, sigma=(1, 1), order=0)
		imgT = np.load(img_path_t) 
		maskT = np.load(mask_path_t)
		if self.noise:
			curNoise = self.noiseVars[index % len(self.pairIndices), :]
			imgS = imgS + curNoise[0]*np.random.normal(size=imgS.shape)
			imgT = imgT + curNoise[1]*np.random.normal(size=imgT.shape)
		# imgnormT = np.load(imgnorm_path_t)
		# imgT = ndimage.gaussian_filter(imgT, sigma=(1, 1), order=0)
		imgS = imgS.reshape(imgS.shape[0], imgS.shape[1], 1)
		imgT = imgT.reshape(imgT.shape[0], imgT.shape[1], 1)
		maskS = maskS.reshape(imgS.shape[0], imgS.shape[1], 1)
		maskT = maskT.reshape(imgT.shape[0], imgT.shape[1], 1)
		# imgnormS = imgnormS.reshape(imgnormS.shape[0], imgnormS.shape[1], 1)
		# imgnormT = imgnormT.reshape(imgnormT.shape[0], imgnormT.shape[1], 1)
		transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize([43.44], [141.5])])
		imgS = transform(imgS)
		transform = transforms.Compose([transforms.ToTensor(),
								transforms.Normalize([43.44], [141.5])])
		imgT = transform(imgT)
		# imgS = torch.from_numpy(imgS.astype(np.float64))
		# # print(imgS.shape)
		# imgS = imgS.permute(2, 0, 1)
		# imgT = torch.from_numpy(imgT.astype(np.float64))
		# imgT = imgT.permute(2, 0, 1)
		maskS = torch.from_numpy(maskS.astype(np.float64))
		maskS = maskS.permute(2, 0, 1)
		maskT = torch.from_numpy(maskT.astype(np.float64))
		maskT = maskT.permute(2, 0, 1)
		# combine imgS and imgT 
		img = torch.cat([imgS, imgT], 0)
		# imgnorm = torch.cat([imgnormS, imgnormT])
		mask = torch.cat([maskS, maskT], 0)
		return [img, mask]


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