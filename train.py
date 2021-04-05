#!/home/tony/anaconda3/envs/pytorch17_102/bin/python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torch.utils.tensorboard import SummaryWriter

import madgrad

import time
import copy
from collections import defaultdict
import socket    
import os
import datetime
import glob

### custom imports

import pascal
import helper
from loss import dice_loss
from custom_utils import sanity_check
from metrics import eval_metrics
import time
import config
DATE_TIME_STAMP = str(time.time()).split('.')[0]
### Reproducability  (Not possible, because of torch.nn.bilinear..)
np.random.seed(1234)
torch.manual_seed(1234)
# torch.set_deterministic(True)  ### biliner interpolation has no deterministic implementation so commenting

hostname = socket.gethostname()

import sys
userinp_optimizer = sys.argv[1]

args = dict()



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
args['resume_chkpt'] = None # "../models_2021-02-08_18:46:11.040400/checkpoint.pth"#None
args["device"] = device
print("Training using {}".format(args["device"]))
##### update values ########
args['verbose']=True
args['pid'] = os.getpid()# kill -9 pid
args['cwd'] = os.getcwd().split('/')[-2]### maindir/src (we need maindir)
args['hostname'] = hostname
args['batch_size'] = 64 ## used only for val
args['n_work'] = 6
args['optimizer_choice'] =userinp_optimizer#"MADGRAD" ### "Adam", "SGD"
args['lr_scheduler'] = "CyclicLR"
args['initialization'] = "kaiming" ## xavier
args['pin'] =  True
args['two_pow'] =  4 ## configure model width size
args['tensorboard_logs_dir'] = config.tensorboard_path
args['extra_tag'] = "var_bsize"

if args["optimizer_choice"]=="SGD":
	cycle_momentum_sgd = True
elif args["optimizer_choice"] == "Adam":
	cycle_momentum_sgd = False
elif args["optimizer_choice"] == "MADGRAD":
	cycle_momentum_sgd = False

run_identifiers = ['cwd','amp','pid','two_pow','optimizer_choice','lr_scheduler','initialization','extra_tag']

identifier_for_pth = 'Optim{o}_lrsc_{r}'.format(o=args['optimizer_choice'],r=args['lr_scheduler'])
run_timestamp = str(datetime.datetime.now()).replace(' ','_')

### folder for saving models
os.mkdir('models_{ts}'.format(ts=run_timestamp))

pth_save_step = 50

run_id = run_timestamp
for k,v in args.items():
	if k in run_identifiers:
		run_id = run_id + '_' + k+'_'+str(v)
run_id = run_id.replace('_','').replace('-','').replace(':','')


args['run_id'] = run_id


print(args)
print("#################################")
## For testing
if hostname == config.myhostname:
	args['tensorboard_logs_dir'] = config.local_tensorboard_path
	args["batch_size"] = 2
	args["n_work"] = 0

##################################
writer = SummaryWriter(args['tensorboard_logs_dir'])
#### load the custom model(according to choice on maxpool)


from customNet import CustomNet





def calc_loss(pred, target, metrics ):

	loss = F.cross_entropy(pred, target, weight=None,ignore_index=255, reduction='mean')

	metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

	return loss


def train_model(model, optimizer, scheduler,last_epoch, num_epochs,dataloaders):
	best_epoch_loss = np.inf
	cumIOU = 0
	correctPixels = 0
	total_labeled = 0
	for epoch in range(last_epoch,num_epochs):
		print('Epoch {}/{}'.format(epoch, num_epochs - 1))
		print('-' * 10)
		since = time.time()

		# Each epoch has a training and validation phase

		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			metrics = defaultdict(float)
			epoch_samples = 0

			k = 0 # there should only be one param_group
			for param_group in optimizer.param_groups:
				epoch_lr = param_group['lr']
				if k>0:
					exit("need to check multiple param group")
				k += 1
				print("LR", param_group['lr'])

			for data in dataloaders[phase]: ## labels is mask
				input = data['image']
				labels = data['label']

			

				# # ######### visualize #########
				# f, axarr = plt.subplots(2)
				# axarr[0].imshow(input[0,:].permute(1,2,0))
				# axarr[1].imshow(labels[0,:])
				# plt.show()
				# continue

				# # # #########################

				

				input = input.to(device, dtype=torch.float)

				labels = labels.to(device, dtype=torch.int64)




				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(input)
					# print((outputs.shape,labels.shape))


					loss = calc_loss(outputs, labels, metrics)



					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				epoch_samples += input.size(0)

				correct, num_labeled, inter, union  = eval_metrics(outputs,labels,args['num_class'])

				correctPixels += correct
				total_labeled += num_labeled

				# a = inter/(union+0.00001)
				cumIOU += np.sum(inter/(union+1e-06))


			epoch_loss = metrics['loss'] / epoch_samples

			epoch_mean_iou =cumIOU/ epoch_samples

			epoch_pixel_accuracy = correctPixels/ total_labeled


			if phase == 'val': ## will be useful in next epoch
				to_lr_loss = copy.deepcopy(epoch_loss)
			if (phase == 'train') & (epoch > 0):
				#scheduler.step(to_lr_loss)
				scheduler.step()
				
			# deep copy the model
			if phase == 'val':
				print("saving model")
				if epoch_loss < best_epoch_loss:
					# torch.save(model.state_dict(),'bestModel_{ide}_{l}_{e}.pth'.format(ide=args['run_id'],l=epoch_loss, e=epoch))
					best_epoch_loss = epoch_loss
					print('Saved bestmodel with best loss {}'.format(best_epoch_loss))
					best_model_path = 'models_{ts}/bestmodel.pth'.format(ts = run_timestamp)
					os.system('mv models_{ts}/bestmodel.pth models_{ts}/bestmodel_prev.pth'.format(ts=run_timestamp))
					print(best_model_path)

					torch.save({
					'epoch': epoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'loss': best_epoch_loss,
					'scheduler_state_dict' : scheduler.state_dict()
					}, best_model_path)
				if True: # overwrite every chekcpt

					model_path = 'models_{ts}/checkpoint.pth'.format(ts=run_timestamp)

					os.system('mv models_{ts}/checkpoint.pth models_{ts}/checkpoint_prev.pth'.format(ts=run_timestamp))

					torch.save({
					'epoch': epoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'loss': best_epoch_loss,
					'scheduler_state_dict' : scheduler.state_dict()
					}, model_path)
					
			### tensorboard updation

			if phase == 'train':
				print( {'lr':epoch_lr,'loss':epoch_loss,'meanIOU':epoch_mean_iou,'pixelAccuracy':epoch_pixel_accuracy})
				writer.add_scalars('{}/trainloss'.format(run_id), {'lr':epoch_lr,'loss':epoch_loss,'meanIOU':epoch_mean_iou,'pixelAccuracy':epoch_pixel_accuracy}, epoch)			
			elif phase == 'val':
				print({'lr':epoch_lr,'loss':epoch_loss,'meanIOU':epoch_mean_iou,'pixelAccuracy':epoch_pixel_accuracy})
				writer.add_scalars('{}/valloss'.format(run_id), {'lr':epoch_lr,'loss':epoch_loss,'meanIOU':epoch_mean_iou,'pixelAccuracy':epoch_pixel_accuracy}, epoch)
				writer.flush()

		time_elapsed = time.time() - since
		print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__ == "__main__":


	if hostname == config.myhostname:
		train_set  = pascal.VOCSegmentation(resolution=320,base_dir=config.VOC2012_datapath_local, split='train') # path should end with '/VOCdevkit/VOC2012/'
		val_set  = pascal.VOCSegmentation(resolution=320,base_dir=config.VOC2012_datapath_local, split='val') # path should end with '/VOCdevkit/VOC2012/'

	else:
		train_set  = pascal.VOCSegmentation(resolution=320,base_dir=config.VOC2012_datapath, split='train') # path should end with '/VOCdevkit/VOC2012/'
		val_set  = pascal.VOCSegmentation(resolution=320,base_dir=config.VOC2012_datapath, split='val') # path should end with '/VOCdevkit/VOC2012/'

	image_datasets = {
		'train': train_set, 'val': val_set
	}

	args['num_class'] = train_set.NUM_CLASSES

	for k,v in image_datasets.items():
		print((k,len(v)))

	if hostname == config.myhostname:
		train_set = torch.utils.data.Subset(train_set, np.random.choice(len(train_set), 2, replace=False))
		val_set = train_set


	dataloaders = {
		'train': DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['n_work'], pin_memory=args['pin']),
		'val': DataLoader(val_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['n_work'], pin_memory=args['pin'])
	}



	
	model = CustomNet(args['num_class'],args['two_pow']).to(device)


	if args['initialization'] == 'xavier':
		model._initialize_()
	elif args['initialization'] == 'kaiming':
		model._kaiming_initialize_()


	sanity_check(model)
	
	if args['optimizer_choice'] == "SGD":
		optimizer_ft =  torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

	elif args['optimizer_choice'] == "Adam":
		optimizer_ft = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2)
	elif args['optimizer_choice'] == 'MADGRAD':
		optimizer_ft = madgrad.MADGRAD(model.parameters(), lr=0.001, momentum= 0.9, weight_decay = 0, eps= 1e-06)
	else:
		exit('Wrong optimizer')


	if args['lr_scheduler'] == "ReduceLROnPlateau":
		exp_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=3, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True)

	elif args['lr_scheduler'] == "CyclicLR":
		exp_lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer_ft,base_lr=0.01, max_lr=0.0000000000001, step_size_up=15, step_size_down=15,
			 mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=cycle_momentum_sgd, base_momentum=0.8, 
			 max_momentum=0.9, last_epoch=-1, verbose=False) ## base_lr and max_lr are swapped(diff from pytorch documentation, verify if it create a problems)
		if args['verbose']:
			print("base_lr and max_lr are swapped(diff from pytorch documentation, verify if it create a problems)")

	else:
		exit('Wrong lr_scheduler')

	last_epoch = 0
	if args['resume_chkpt']: ## none if start from scratch
		print("Loading from checkpoint")
		checkpoint = torch.load(args['resume_chkpt'],map_location='cpu')
		last_epoch = checkpoint['epoch']
		
		model.load_state_dict(checkpoint['model_state_dict'])
		model.to(device)

		optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])

		exp_lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
		print("all loaded")




	train_model(model, optimizer_ft, exp_lr_scheduler,last_epoch, num_epochs=10000, dataloaders=dataloaders)

	writer.close()





