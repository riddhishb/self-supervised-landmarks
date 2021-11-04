from torch.serialization import save
import numpy as np
import torch
import json
import networks.network_2d as nt_2d
import networks.network_3d as nt_3d
import utils.data_utils as dt
import utils.network_utils as nt
import utils.losses as ls
from torch import optim
from argparse import ArgumentParser
import os
import os.path as osp 

def train(model, config, device):
	'''
	Training  for the Phantom Data
	'''
	# first load the data
	
	lr = config['learning_rate']
	dataDir = config['dataDir']
	save_dir = config['save_dir']
	num_epochs = config['num_epochs']
	print_iter = config['print_iter']
	save_iter = config['save_iter']
	batchSz = config['batch_size']
	start_epsilon = config['start_epsilon']
	loadStep = config['load_step']
	reg_alpha = config['reg_alpha']
	input_channels = config["input_channels"]
	optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
	# model, optimizer = nt.load_checkpoint(model, optimizer, saveDir, str(loadStep), loadStep)
	num_land = config['num_landmarks']
	model.train()
	data_loader = dt.get_dataset('phantom_pairs', 0, 0,dataDir, batchSz, train=True,  noise=True)
	val_loader = dt.get_dataset('phantom_pairs', 0, 0, dataDir, batchSz, train=False, noise=False)
	
	epoch = loadStep
	best_val = float('inf')
	while epoch < num_epochs:
		
		avgLoss = 0
		avgRelLoss = 0
		count = 0
		model = model.float()
		if epoch < 1:
			noweps = np.linspace(start_epsilon, 0, len(data_loader))
		else:
			noweps = np.linspace(0, 0, len(data_loader))
		model.train()
		for x in data_loader:
			# print(x.shape)
			x = x.to(device)
			epsilon = noweps[count]
			outImg, landT, landS, A = model(x.float(), epsilon, False, 0, 0)
			[imgS, imgT] = x.split([input_channels, input_channels], 1)
			reg_loss = ls.cond_num_loss_v2(A, device)
			l2_loss, rel_loss, denon = ls.l2_loss(imgT.float(), outImg, device)
			loss = l2_loss + reg_alpha*reg_loss
			# if count % print_iter == 0:
			#	 visualEvol(model, dataDir, batchSz, device, saveDir, len(data_loader)*epoch + count)
			#	 parentDir = saveDir + '/Train/'
			#	 imgS = imgS[0, ...].reshape(1, 3, 256, 256)
			#	 imgT = imgT[0, ...].reshape(1, 3, 256, 256)
			#	 imgS = imgS.permute(0, 2, 3 ,1)
			#	 imgS = imgS.squeeze().cpu().detach().numpy()
			#	 imgT = imgT.permute(0, 2, 3 ,1)
			#	 imgT = imgT.squeeze().cpu().detach().numpy()
			#	 plt.imsave(parentDir + 'imgS' + str(count) + '.png', imgS.astype(np.uint8))
			#	 plt.imsave(parentDir + 'imgT' + str(count) + '.png', imgT.astype(np.uint8))
			#	 # np.save(parentDir + 'inImg' + str(count) + '.npy', inImg)
			#	 outImg = outImg[0, ...].reshape(1, 3, 256, 256)
			#	 finalOut = outImg.permute(0, 2, 3, 1)
			#	 finalOut = finalOut.squeeze().cpu().detach().numpy()
			#	 plt.imsave(parentDir + 'regImg' + str(count) + '.png', finalOut.astype(np.uint8))
			#	 # np.save(parentDir + 'regImg' + str(count) + '.npy', finalOut)
			#	 outPoints = landS.cpu().detach().numpy()
			#	 outPoints = outPoints[0, ...]
			#	 outPoints[:, 0] = (outPoints[:, 0]+1)*0.5*(255)
			#	 outPoints[:, 1] = (outPoints[:, 1]+1)*0.5*(255)
			#	 np.save(parentDir + 'landS' + str(count) + '.npy', outPoints)
			#	 fig, ax = plt.subplots()
			#	 plt.imshow(imgS.astype(np.uint8))
			#	 for i in range(outPoints.shape[0]):
			#		 ax.scatter(outPoints[i, 0], outPoints[i, 1], c='c', edgecolors='k')
			#		 ax.text(outPoints[i, 0]+0.3, outPoints[i, 1]+0.3, str(i), fontsize=9, c='m')
			#	 plt.savefig(parentDir + 'overlayImg_source' + str(count) + '.png')
			#	 plt.clf()

			#	 outPoints = landT.cpu().detach().numpy()
			#	 outPoints = outPoints[0, ...]
			#	 outPoints[:, 0] = (outPoints[:, 0]+1)*0.5*(255)
			#	 outPoints[:, 1] = (outPoints[:, 1]+1)*0.5*(255)
			#	 np.save(parentDir + 'landT' + str(count) + '.npy', outPoints)
			#	 fig, ax = plt.subplots()
			#	 plt.imshow(imgT.astype(np.uint8))
			#	 for i in range(outPoints.shape[0]):
			#		 ax.scatter(outPoints[i, 0], outPoints[i, 1], c='c', edgecolors='k')
			#		 ax.text(outPoints[i, 0]+0.3, outPoints[i, 1]+0.3, str(i), fontsize=9, c='m')
			#	 plt.savefig(parentDir + 'overlayImg_target' + str(count) + '.png')
			#	 plt.clf()

			#	 # also need to plot the image difference 
			#	 fig, axis = plt.subplots(figsize=(5, 5))
			#	 axis.imshow(imgT.astype(np.uint8), cmap='Reds', alpha=0.6)
			#	 axis.imshow(finalOut.astype(np.uint8), cmap='Blues', alpha=0.6)
			#	 axis.set_title('Images overlayed')
			#	 plt.savefig(parentDir + 'regAcc' + str(count) + '.png')
			#	 plt.clf()

			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			avgLoss += loss.item()
			avgRelLoss += rel_loss.item()
			if count % print_iter == 0:
				print('Loss = ', loss.item(), rel_loss.item(), reg_loss.item(), len(data_loader), x.shape, batchSz, count)

			count +=1
		epoch += 1

		model.eval()
		with torch.no_grad():
			tot_loss = 0
			tot_rel_loss = 0
			count_new = 0
			for x in val_loader:
				x = x.to(device)
				outImg, landT, landS, A = model(x.float(), 0, False, 0, 0)
				[imgS, imgT] = x.split([input_channels, input_channels], 1)
				loss, rel_loss, denon = ls.l2_loss(imgT.float(), outImg, device)
				tot_loss += loss.item()
				tot_rel_loss += rel_loss.item()
				count_new += 1
				# these lines are for spending less time during validation
				if count_new > 10:
					break
				# break
			tot_loss = tot_loss/count_new
			tot_rel_loss = tot_rel_loss/count_new
			print('Epoch, count = ', epoch, count, ' Loss = ', tot_loss, tot_rel_loss)
		# add a part to save some test images and their predicted things
		if best_val > tot_loss:
			best_val = tot_loss
			torch.save(model, save_dir + '/best_model.pt')

	torch.save(model, save_dir + '/final_model.pt')

def runFunction(args):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	config = json.load(open(args.config_file))
	# create the save directory
	save_dir = config['save_dir']
	if osp.exists(save_dir):
		os.makedirs(save_dir)

	# TODO: add an option to print to a log file
	print("///////////////////////////////////////")
	print("Defining Model")
	print("///////////////////////////////////////")

	if args.model == "2d":
		model = nt_2d.self_supervised_model_2d(config, device).to(device)
	else:
		model = nt_3d.self_supervised_model_3d(config, device).to(device)

	train(model, config, device)
	

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--model", type=str, help="2d or 3d")
	parser.add_argument("--config_file", type=str, default="CAE", help="configFile for the parameters")
	args = parser.parse_args()
	runFunction(args)