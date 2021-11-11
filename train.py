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
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import os.path as osp 

def train(model, model_type, config, device):
	'''
	Training  for the Phantom Data
	'''
	# first load the data
	
	lr = config['learning_rate']
	data_dir = config['data_dir']
	save_dir = config['save_dir']
	num_epochs = config['num_epochs']
	print_iter = config['print_iter']
	save_iter = config['save_iter']
	batch_size = config['batch_size']
	start_epsilon = config['start_epsilon']
	loadStep = config['load_step']
	reg_alpha = config['reg_alpha']
	input_channels = config["input_channels"]
	optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
	# model, optimizer = nt.load_checkpoint(model, optimizer, saveDir, str(loadStep), loadStep)
	num_land = config['num_landmarks']
	model.train()
	train_loader = dt.get_dataset(model_type, data_dir, batch_size, file_type='npy', data_type='train', noise=False)
	val_loader = dt.get_dataset(model_type, data_dir, batch_size, file_type='npy', data_type='validation', noise=False)
	
	epoch = loadStep
	best_val = float('inf')
	logger = open(save_dir + "train_log.csv", "w+")
	nt.log_print(logger, ["Epoch", "train_loss", "train_reg_loss", "val_loss"])

	while epoch < num_epochs:
		
		train_loss = 0
		train_reg_loss = 0
		count = 0
		model = model.float()
		if epoch < 1:
			noweps = np.linspace(start_epsilon, 0, len(train_loader))
		else:
			noweps = np.linspace(0, 0, len(train_loader))
		model.train()
		for x in train_loader:
			# print(x.shape)
			x = x.to(device)
			epsilon = noweps[count]
			outImg, landT, landS, A = model(x.float(), epsilon, False, 0, 0)
			[imgS, imgT] = x.split([input_channels, input_channels], 1)
			reg_loss = ls.cond_num_loss_v2(A, device)
			# TODO: loss type
			l2_loss, rel_loss, denon = ls.l2_loss(imgT.float(), outImg, device)
			loss = l2_loss + reg_alpha*reg_loss			
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			train_loss += loss.item()
			train_reg_loss += reg_loss.item()
			count += 1
		epoch += 1
		train_loss = train_loss / count
		train_reg_loss = train_reg_loss / count
		model.eval()
		with torch.no_grad():
			val_loss = 0
			count_new = 0
			for x in val_loader:
				x = x.to(device)
				outImg, landT, landS, A = model(x.float(), 0, False, 0, 0)
				[imgS, imgT] = x.split([input_channels, input_channels], 1)
				# TODO: loss type
				loss, rel_loss, denon = ls.l2_loss(imgT.float(), outImg, device)
				val_loss += loss.item()
				count_new += 1
				# these lines are for spending less time during validation
				if count_new > 10:
					break
				# break
			val_loss = val_loss/count_new
			nt.log_print(logger, [epoch, train_loss, train_reg_loss, val_loss] )
			# print('Epoch = ', epoch, ' Train Loss = ', ' Val Loss = ', tot_loss, tot_rel_loss)
		# add a part to save some test images and their predicted things
		if best_val > val_loss:
			best_val = val_loss
			torch.save(model, save_dir + '/best_model.pt')

	torch.save(model, save_dir + '/final_model.pt')

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

	if args.model == "2d":
		model = nt_2d.self_supervised_model_2d(config, device).to(device)
	else:
		model = nt_3d.self_supervised_model_3d(config, device).to(device)

	train(model, args.model, config, device)
	

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--model", type=str, help="2d or 3d")
	parser.add_argument("--config_file", type=str, default="CAE", help="configFile for the parameters")
	args = parser.parse_args()
	runFunction(args)