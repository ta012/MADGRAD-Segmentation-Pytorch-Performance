#!/home/tony/anaconda3/envs/pytorch17_102/bin/python


from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import matplotlib.pyplot as plt
import config

optim_tag = {'SGD':'20210405043845.763865ampFalsepid30353cwdSegmentationOptimizersComparisionoptimizerchoiceSGDlrschedulerCyclicLRinitializationkaimingtwopow4extratagvarbsizeleakReluFalse_',
'ADAM':'20210405043148.998989ampFalsepid29049cwdSegmentationOptimizersComparisionoptimizerchoiceAdamlrschedulerCyclicLRinitializationkaimingtwopow4extratagvarbsizeleakReluFalse_',
'MADGRAD':'20210405070700.466992ampFalsepid27599cwdSegmentationOptimizersComparisionoptimizerchoiceMADGRADlrschedulerCyclicLRinitializationkaimingtwopow4extratagvarbsizeleakReluFalse_'}

tensorboard_log_dir = config.tmp


for optimizer,tag in optim_tag.items():
	phase_l = ['trainloss','valloss']

	metric_l = ['_pixelAccuracy','_loss']


	i = 0 
	for phase in phase_l:
		for metric in  metric_l:
			if i== 0:
				pth = tensorboard_log_dir + tag + phase + metric
				print(pth)

				ea = event_accumulator.EventAccumulator(pth)
				ea.Reload()

				df = pd.DataFrame(ea.Scalars(ea.Tags()['scalars'][0]))	
				df.columns = ['wall_time','step',phase+metric]
				df = df[['step',phase+metric]]	
				i +=1
			else:
				pth = tensorboard_log_dir + tag + phase + metric
				print(pth)
				ea = event_accumulator.EventAccumulator(pth)
				ea.Reload()

				temp = pd.DataFrame(ea.Scalars(ea.Tags()['scalars'][0]))
				temp.columns = ['wall_time','step',phase+metric]
				temp = temp[['step',phase+metric]]

				df = pd.merge(df,temp,on='step',how='inner')
				# df.to_csv('test.csv',index=False)
				print(df.shape)


				i +=1
	print(df.head())

	df  = df.iloc[:,1:].head(250)
	df.columns = [i.replace('loss_','_') for i in df.columns]

	# f, axarr = plt.subplots(2)
	# axarr[0] = plt.plot(df.index,df['train_pixelAccuracy'])
	# # df.plot.line()





	df[[i for i in df.columns if '_pixelAccuracy' in i ]].plot.line(title='Optimizer = {}'.format(optimizer),ylim=(0.3,1))
	plt.savefig('figures/pixelaccuracy_{}.png'.format(optimizer))



	df[[i for i in df.columns if '_loss' in i ]].plot.line(title='Optimizer = {}'.format(optimizer),ylim=(0,2))
	plt.savefig('figures/crossentropyloss_{}.png'.format(optimizer))
