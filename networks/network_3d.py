import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from utils.network_utils import grid_sample, atanh


class self_supervised_model_3d(nn.Module):
	'''
	[1] Network which has a fixed source image and predicts the target
	landmarks.
	[2] This will be the first instance with the batched input.
	TODO: fix the dimensions for a geenral matrix -- refer to DeepSSM
	'''

	def __init__(self, config, device):
		super(self_supervised_model_3d, self).__init__()
		self.device = device
		self.numL = config['num_landmarks']
		self.imgH = config['image_height']
		self.imgW = config['image_width']
		self.batchSz = config['batch_size']
		self.grad_img = config['grad_img']
		self.add_corners = config['add_corners']
		self.input_channels = config['input_channels']
		# input is a stack of 2 images
		self.conv1_1 = nn.Conv2d(3, 32, kernel_size=(3,3), stride=(1, 1), padding=1) #256x256
		self.conv1_2 = nn.Conv2d(32, 32, kernel_size=(3,3), stride=(1, 1), padding=1) #256x256
		self.conv2_1 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1, 1), padding=1) #128x128
		self.conv2_2 = nn.Conv2d(64, 64, kernel_size=(3,3), stride=(1, 1), padding=1) #128x128
		self.conv3_1 = nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1, 1), padding=1) #64x64
		self.conv3_2 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=(2, 2), padding=1) #64x64
		self.conv3_3 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=(1, 1), padding=1) #32x32
		self.conv3_4 = nn.Conv2d(128, 128, kernel_size=(3,3), stride=(2, 2), padding=1) #32x32
		self.conv4_1 = nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1, 1), padding=1) # 8x8
		self.conv4_2 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(2, 2), padding=1) # 8x8
		self.conv4_3 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(1, 1), padding=1) # 4x4
		self.conv4_4 = nn.Conv2d(256, 256, kernel_size=(3,3), stride=(2, 2), padding=1) # 4x4
		self.fc1 = nn.Linear(256*2*2, 512) # 2x2
		self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
		if self.add_corners:
			self.fc2 = nn.Linear(512, (self.numL - 4)*2)
			# uniform initialization of the points
			randomPts = np.random.uniform(-1, 1, (self.numL - 4)*2)
		else:
			self.fc2 = nn.Linear(512, (self.numL)*2)
			randomPts = np.random.uniform(-1, 1, (self.numL)*2)
		
		randomPts = torch.from_numpy(randomPts).to(device)
		self.fc2.bias.data.copy_(randomPts)
		self.fc2.weight.data.zero_()

	def CNN_landS(self, x):
		x = F.relu(self.conv1_1(x))
		x = F.relu(self.conv1_2(x))
		x = self.pool(x)
		x = F.relu(self.conv2_1(x))
		x = F.relu(self.conv2_2(x))
		x = self.pool(x)
		x = F.relu(self.conv3_1(x))
		x = F.relu(self.conv3_2(x))
		x = F.relu(self.conv3_3(x))
		x = F.relu(self.conv3_4(x))
		x = self.pool(x)
		x = F.relu(self.conv4_1(x))
		x = F.relu(self.conv4_2(x))
		x = F.relu(self.conv4_3(x))
		x = F.relu(self.conv4_4(x))
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		output = F.tanh(self.fc2(x))
		return output
	
	def createKmat(self, input_points, control_points):
		N = input_points.size(1)
		M = control_points.size(1)
		pairwise_diff_batch = input_points.view(self.batchSz, N, 1, 2) - control_points.view(self.batchSz, 1, M, 2)
		pairwise_diff_square = pairwise_diff_batch * pairwise_diff_batch
		pairwise_dist = pairwise_diff_square[:, :, :, 0] + pairwise_diff_square[:, :, :, 1]
		repr_matrix = pairwise_dist * torch.log(torch.sqrt(pairwise_dist+0.0001))
		# repr_matrix = torch.exp(-1*pairwise_dist*10)
		# fix numerical error for 0 * log(0), substitute all nan with 0
		mask = repr_matrix != repr_matrix
		repr_matrix.masked_fill_(mask, 0)
		return repr_matrix

	def TPS_solve(self, Lt, Ls, epsilon):
		N = Ls.size(1)
		Ls = Ls.float()
		# first create the tensor (these would be in pytorch)
		tempZeros = torch.zeros(3, 1).to(self.device)
		tempZeros = tempZeros.repeat(self.batchSz, 1, 1)
		b1 = torch.cat([tempZeros, Ls[:, :,0].view(self.batchSz, -1, 1)], dim=1)
		b2 = torch.cat([tempZeros, Ls[:, :,1].view(self.batchSz, -1, 1)], dim=1)
		# this part is fixed as the Lt are supervised
		b = torch.cat([b1, b2], dim=1) # dimension 2(N+3)
		A = torch.zeros(self.batchSz, (N+3), (N+3))
		matR = self.createKmat(Lt, Lt)
		A[:, :2, :N].copy_(Lt.transpose(2, 1))
		A[:, 2, :N].fill_(1)
		A[:, 3:, :N].copy_(matR)
		A[:, 3:, N:N+2].copy_(Lt)
		A[:, 3:, N+2].fill_(1)
		Abig = torch.zeros(self.batchSz, 2*(N+3), 2*(N+3))
		Abig[:, :(N+3), :(N+3)] = A
		Abig[:, (N+3):, (N+3):] = A
		condMat = torch.eye(Abig.shape[1])
		condMat = condMat.reshape(1, Abig.shape[1], Abig.shape[1])
		condMat = condMat.repeat(self.batchSz, 1, 1)
		Acond = Abig + epsilon*condMat
		Acond = Acond.to(self.device)
		w, LU = torch.solve(b, Acond)
		return w, Abig

	def imgGrid(self, Is, Ls):
		# define a grid then warp it by tps
		# Is = 1xCxHxW image
		targetCoord = list(itertools.product(range(self.imgH), range(self.imgW)))
		targetCoord =  torch.tensor(targetCoord).to(self.device)
		targetCoord = targetCoord.float()
		Y, X = targetCoord.split(1, dim=1)
		Y = Y * 2/ (self.imgH - 1) - 1
		X = X * 2/ (self.imgW - 1) - 1
		targetCoord = torch.cat([X, Y], dim=1)
		targetCoord = targetCoord.reshape(1, targetCoord.shape[0], targetCoord.shape[1])
		targetCoord = targetCoord.repeat(self.batchSz, 1, 1)
		matT = self.createKmat(targetCoord, Ls)
		# this needs to change according to our formulation 
		targetCoord_full = torch.cat([matT, targetCoord.float(), torch.ones(self.batchSz, self.imgH*self.imgW, 1).to(self.device).float()], dim = 2)
		return targetCoord_full, targetCoord
	
	def forward(self, inImg, epsilon, inland, landS, landT):
		
		[imgS, imgT] = inImg.split([self.input_channels, self.input_channels], 1)
		# this will get the batched output 
		
		if inland:
			self.landS = landS
			self.landT = landT
			self.numL = self.landS.size(1)
		else:
			self.landT = self.CNN_landS(imgT)
			self.landS = self.CNN_landS(imgS)

			if self.add_corners:
				self.numL = int(self.landT.size(1)*0.5 + 4)
				# we need to preserve the batching 
				self.landT = self.landT.view(self.batchSz, self.numL-4, 2)
				self.landS = self.landS.view(self.batchSz, self.numL-4, 2)
				# adding the corners
				temp = torch.from_numpy(np.array([[-1,-1],[-1,1],[1,1],[1,-1]]))
				temp = temp.repeat(self.batchSz, 1, 1).to(self.device)
				self.landT = torch.cat([self.landT, temp.float()], 1)
				self.landS = torch.cat([self.landS, temp.float()], 1)
			else:
				self.landT = self.landT.view(self.batchSz, self.numL, 2)
				self.landS = self.landS.view(self.batchSz, self.numL, 2)

		self.w, self.A = self.TPS_solve(self.landT, self.landS, epsilon)
		self.w = self.w.float()
		self.targetCoord, origgrid = self.imgGrid(imgT, self.landT)

		# self.sourceCoord = torch.matmul(self.targetCoord, self.w.view(self.batchSz, 2, self.numL + 3).transpose(0, 1))
		# with batched data we need to use torch bmm 
		self.sourceCoord = torch.bmm(self.targetCoord, self.w.view(self.batchSz, 2, self.numL + 3).transpose(1, 2))
		grid = self.sourceCoord.view(self.batchSz, self.imgH, self.imgW, 2).float()
		transformed_x = F.grid_sample(imgS.float(), grid, 'bilinear')
		return transformed_x, self.landT, self.landS, self.A