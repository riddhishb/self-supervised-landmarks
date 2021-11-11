from torch.serialization import save
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import networks.network_2d as nt_2d
import networks.network_3d as nt_3d
import utils.data_utils as dt
import utils.network_utils as nt
import utils.losses as ls
from torch import optim
from argparse import ArgumentParser
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import os.path as osp 


def redundancy_removal(model, config, device, red_metric=None):
	'''
	Identifies the points with less threshold and get rid of them
	'''

	optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.999))
	dataDir = config['dataDir']
	saveDir = config['saveDir']
	num_epochs = config['num_epochs']
	print_iter = config['print_iter']
	save_iter = config['save_iter']
	loadStep = config['load_step']
	batchSz = config['batch_size']
	N = config['num_landmarks']
	model, optimizer = nt.load_checkpoint(model, optimizer, saveDir, str(loadStep), loadStep)
	model.eval()
	model = model.float()
	data_loader = ut.get_dataset('phantom_pairs', dataDir, batchSz, train=True, noise=False)
	red_metric = np.zeros([N, 200])
	count = 0
	for x in data_loader:
		x = x.to(device)
		# base loss 
		outImg, landT, landS, A = model(x.float(), 0, False, 0, 0)
		[imgS, imgT] = x.split([3, 3], 1)
		loss, rel_loss, denon = nt.l2_Loss_2D(imgT.float(), outImg, device)
		landS = landS.cpu().detach().numpy()
		landT = landT.cpu().detach().numpy()
		# Now compute the things for different landmarks and stuff
		print(count, len(data_loader))
		for i in range(N):
			idx = np.arange(N)
			idx = np.delete(idx, i)
			newlandT = landT[:, idx, :]
			newlandS = landS[:, idx, :]
			# now run the TPS warp with this parameters
			[imgS, imgT] = x.split([3, 3], 1)
			outImg, t, s, A = model(x.float(), 0, True, torch.from_numpy(newlandS).to(device), torch.from_numpy(newlandT).to(device))
			loss_i, rel_loss_i, denon = nt.l2_Loss_2D(imgT.float(), outImg, device)
			temp = (loss - loss_i)**2
			red_metric[i, count] = temp.detach().cpu().numpy()
			t = t.detach().cpu().numpy()
			s = s.detach().cpu().numpy()
			A = A.detach().cpu().numpy()
			imgS = imgS.detach().cpu().numpy()
			imgT = imgT.detach().cpu().numpy()
		count += 1
		if count > 199:
			break
	# red_metric = red_metric/count
	return red_metric

def test(model, model_type, loader, config, device, red_metric=None):
	'''
	Testing 
	'''
	
	save_dir = config['save_dir']
	img_h = config['image_height']
	img_w = config['image_width']
	if model_type == '3d':
		img_l = config['image_length']
	
	num_land = config['num_landmarks']
	
	input_channels = config['input_channels']
	
	model.eval()
	model = model.float()
	count = 0
	
	for x in loader:
		x = x.to(device)
		# the loader is constructed with batch size of 1 ==> to travel through
		# all the images in the dataset we need to get the source image
		# for the first size.loader times and then break the process.
		
		outImg, landT, landS, A = model(x.float(), 0, False, 0, 0)
		[imgS, imgT] = x.split([input_channels, input_channels], 1)
		if red_metric is not None:
			prj = red_metric
		else:
			prj = np.arange(num_land)
		
		parent_dir = save_dir + '/Test/'
		if not os.path.exists(parent_dir):
			os.makedirs(parent_dir)

		if model_type == '2d':
			imgS = imgS[0, ...].reshape(1, input_channels, img_h, img_w)
			imgT = imgT[0, ...].reshape(1, input_channels, img_h, img_w)
			imgS = imgS.permute(0, 2, 3 ,1)
			imgS = imgS.squeeze().cpu().detach().numpy()
			imgT = imgT.permute(0, 2, 3 ,1)
			imgT = imgT.squeeze().cpu().detach().numpy()
			plt.imsave(parent_dir + 'imgS' + str(count) + '.png', imgS.astype(np.uint8))
			plt.imsave(parent_dir + 'imgT' + str(count) + '.png', imgT.astype(np.uint8))
		
			outImg = outImg[0, ...].reshape(1, input_channels, img_h, img_w)
			finalOut = outImg.permute(0, 2, 3, 1)
			finalOut = finalOut.squeeze().cpu().detach().numpy()
			plt.imsave(parent_dir + 'regImg' + str(count) + '.png', finalOut.astype(np.uint8))
		
			outPoints = landS.cpu().detach().numpy()
			outPoints = outPoints[0, ...]
			outPoints = outPoints[prj, :]
			outPoints[:, 0] = (outPoints[:, 0]+1)*0.5*(img_w - 1)
			outPoints[:, 1] = (outPoints[:, 1]+1)*0.5*(img_h - 1)
			np.save(parent_dir + 'landS' + str(count) + '.npy', outPoints)
			fig, ax = plt.subplots()
			plt.imshow(imgS.astype(np.uint8))
			for i in range(outPoints.shape[0]):
				ax.scatter(outPoints[i, 0], outPoints[i, 1], c='c', edgecolors='k')
				ax.text(outPoints[i, 0]+0.3, outPoints[i, 1]+0.3, str(prj[i]), fontsize=9, c='m')
					
			plt.savefig(parent_dir + 'overlayImg_source' + str(count) + '.png')
			plt.clf()

			fig, axis = plt.subplots(figsize=(5, 5))
			axis.imshow(imgT.astype(np.uint8), cmap='Reds', alpha=0.6)
			axis.imshow(finalOut.astype(np.uint8), cmap='Blues', alpha=0.6)
			axis.set_title('Images overlayed')
			plt.savefig(parent_dir + 'regAcc' + str(count) + '.png')
			plt.clf()

		if count < len(loader):
			count += 1	
		else:
			break
	
def runFunction(args):

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	
	config = json.load(open(args.config_file))
	# create the save directory
	save_dir = config['save_dir']
	if not osp.exists(save_dir):
		os.makedirs(save_dir)

	# TODO: add an option to print to a log file
	print("///////////////////////////////////////")
	print("Defining Model")
	print("///////////////////////////////////////")
	model_type = args.model
	if model_type == "2d":
		model = nt_2d.self_supervised_model_2d(config, device).to(device)
	else:
		model = nt_3d.self_supervised_model_3d(config, device).to(device)
	
	# get the loader
	data_dir = config['data_dir']
	save_dir = config['save_dir']

	if args.use_best:
		model_path = save_dir + '/best_model.pt'
	else:
		model_path = save_dir + '/final_model.pt'
	
	model = torch.load(model_path).to(device)
	loader = dt.get_dataset_temp(model_type, data_dir, 1, file_type='npy', data_type='test', noise=False)

	if args.redu_remove:
		red_metric = redundancy_removal(model, config, device)
		test(model, model_type, loader, config, device, red_metric=red_metric)
	else:
		test(model, model_type, loader, config, device)

	

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--model", type=str, help="2d or 3d")
	parser.add_argument("--redu_remove", type=bool, default=False, help="True or False, basically means that the testing is to be does using the reduced points or the same.")
	parser.add_argument("--use_best", type=bool, default=True, help="True or false, if false it uses the final model")
	parser.add_argument("--config_file", type=str, help="configFile for the parameters")
	args = parser.parse_args()
	runFunction(args)