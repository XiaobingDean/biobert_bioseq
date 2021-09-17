
from argparse import ArgumentParser
import time
import os.path

# some global lists
train_Plink = []
train_Acc1 = []
train_Acc5 =[]
dev_Plink = []
dev_Acc1 = []
dev_Acc5 = []
train_loss = []
dev_loss = []

def get_args():
	# CUDA_VISIBLE_DEVICES=0,1,2,3 to select GPU

	parser=ArgumentParser(description='BioMatch')
	
	# data arguments
	parser.add_argument('--dataset',type=str,default='cl') # {'cl','go','doid','chebi'}
	# parser.add_argument('--max-length',type=int,default=1) # max-length of phrases, <=512 as BERT input

	# model arguments
	parser.add_argument('--lp-gnn',action='store_true',default=False) # whether to use lp_gnn
	parser.add_argument('--load-model',type=str,default=None) # model path
	
	# optimization arguments
	parser.add_argument('--batch-size',type=int,default=32)
	parser.add_argument('--bs-mul',type=int,default=1) # update by every bus_mul batches 	
	parser.add_argument('--epoch',type=int,default=10) # total training epochs
	parser.add_argument('--lp-pretrain',type=int,default=10) # link prediction pretrain epochs
	
	parser.add_argument('--optim',type=int,default=0) # 0 for Adam, 1 for SGD+Nesterov
	parser.add_argument('--lr',type=float,default=0.03) # initial learning rate
	parser.add_argument('--mom',type=float,default=0.9)
	parser.add_argument('--wd',type=float,default=0)
	parser.add_argument('--dp',type=float,default=0.1) # dropout rate {0.0, 0.2, 0.4, 0.5}
	parser.add_argument('--alpha',type=float,default=0.5) # cooeficient of matching loss againest link prediction loss
	parser.add_argument('--save-acc',type=float,default=2.0) # save the model when dev Acc1> save-acc, used only when record plot data.
	
	# general arguments
	parser.add_argument('--seed',type=int,default=1)
	parser.add_argument('--mode',type=str,default='') # {'train','test'}
	parser.add_argument('--save',action='store_true',default=False) # whether to save the final model 
	parser.add_argument('--name',type=str,default='') # used in as saved filenames to distinct different configurations
	
	args=parser.parse_args()
	return args

def show_time(seconds,show_hour=False):
	m=seconds//60 # m is an integer
	s=seconds%60   # s is a real number
	if show_hour:
		h=m//60  # h is an integer
		m=m%60    # m is an integer
		return '%02d:%02d:%05.2f'%(h,m,s)
	else:
		return '%02d:%05.2f'%(m,s)

def logging(s:str,model_name='',log_=True,print_=True):
	if print_:
		print(s)
	if log_:
		with open(os.path.join('Log','log_'+model_name+'.txt'), 'a+') as f_log:
			f_log.write(s + '\n')

def len_dist(names,train,dev,test):
	seq_len_dist = []
	for (i,seq) in (names+train+dev+test):
		diff = len(seq_len_dist)-1-len(seq)
		if diff>=0:
			seq_len_dist[len(seq)] += 1
		else:
			seq_len_dist = seq_len_dist+[0]*(-diff-1)+[1]
	
	num_seq = sum(seq_len_dist)
	assert num_seq==len(names+train+dev+test)

	max_seq_len = len(seq_len_dist)-1
	seq_len_cul = [sum(seq_len_dist[:length]) for length in range(1,max_seq_len+2)]
	assert seq_len_cul[-1]==num_seq

	percentile_999 = 0
	while (seq_len_cul[percentile_999]/num_seq)<0.999:
		percentile_999 += 1
	return max_seq_len,percentile_999,seq_len_dist


