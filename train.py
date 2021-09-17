import time
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import logging,show_time
from utils import train_Plink,train_Acc1,train_Acc5,dev_Plink,dev_Acc1,dev_Acc5,train_loss,dev_loss


def train(batchloader,devloader,model, # model & data
	optimizer,bs_mul,alpha,       # optimization arguments
	PTepochs,max_Acc1,max_epoch,cur_epoch,name:str):  # arguments for saving model
	
	# train through the whole dataset for one time
	P_link = 0
	Acc_1 = 0
	Acc_5 = 0
	epoch_loss = 0

	# In each epoch, shuffle the whole dataset to get different mini-batch splits.
	batchloader.reshuffle()

	# B = batchloader.batch_size

	# criterion = nn.MarginRankingLoss(margin=margin,reduction='sum')

	model.train()
	optimizer.zero_grad()

	for batch_idx,batch in enumerate(batchloader):
		
		if batch_idx%bs_mul==0:
			start=time.time()
			update_loss=torch.cuda.FloatTensor(1).zero_()
			update_bs = 0
		
		(x,x_len,names,names_len,y,y_link,edge_idx,edge_idx_plus)=batch
		B = x.size(0)
		input_batch = (x.cuda(),x_len,names.cuda(),names_len,edge_idx.cuda(),edge_idx_plus.cuda())
		scores_lp,scores = model(input_batch,cur_epoch<PTepochs) # (B,name_size)

		sorted_scores,_ = torch.sort(scores,dim=1,descending=False) # (B,name_size) on gpu
		r_hat = scores.size(1)-torch.searchsorted(sorted_scores,scores[range(B),y].view(-1,1),right=False).squeeze(1) # (B,1)->(B) rank in [1,2,...,name_size]

		# r_hat = []
		# sample_set = [list(onedtensor) for onedtensor in list(rank_i)] # [[name_size],[],...,[]]
		# for i,rank in enumerate(sample_set):
		# 	r_hat.append(rank.index(y[i])+1)
		

		loss_m = F.nll_loss(F.log_softmax(scores,dim=1),y.cuda(),reduction='sum')
		# loss_lp = F.binary_cross_entropy_with_logits(scores_lp,y_link.cuda(),reduction='sum') #/names.size(0)
		loss_lp = F.binary_cross_entropy(torch.sigmoid(scores_lp),y_link.cuda(),reduction='sum')
		batch_loss = alpha*loss_m+loss_lp

		update_loss += batch_loss
		update_bs += B

		if (batch_idx+1)%bs_mul==0 or (batch_idx+1)==batchloader.n_batches:
			optimizer.zero_grad()
			update_loss = update_loss/update_bs # update_loss will be zero in the next for loop
			update_loss.backward()

			# gradient clipping on gat
			# if model.gat_lp is not None:
			# 	nn.utils.clip_grad_value_(model.gat_lp.parameters(),1.0)
			
			# grad_bert0 = model.bert.encoder.layer[0].intermediate.dense.weight.grad.view(-1)

			if cur_epoch>=PTepochs:
				grad_rnn = torch.cat([param.grad.view(-1) for param in model.encoder_m.lstm.parameters()])
				grad_gat = torch.zeros(1) if model.gat_m is None else torch.cat([param.grad.view(-1) for param in model.gat_m.parameters()])
			else:
				grad_rnn = torch.cat([param.grad.view(-1) for param in model.encoder_lp.lstm.parameters()])
				grad_gat = torch.zeros(1) if model.gat_lp is None else torch.cat([param.grad.view(-1) for param in model.gat_lp.parameters()])
			
			grad_avg = [grad.sum().item()/len(grad) for grad in [grad_rnn,grad_gat]]

			optimizer.step()

			logging('{:05d}, loss(e-4):{:.6f}, '.format(batch_idx,update_loss.item()*1e4)+show_time(time.time()-start),name)
			logging('gradient(e-6) lstm:{:.4f}, gat:{:.4f}'.format(grad_avg[0]*1e6,grad_avg[1]*1e6),name)

		
		P_link += ((torch.sigmoid(scores_lp.cpu())*y_link).sum(dim=1)/y_link.sum(dim=1)).sum().item()
		Acc_1 += r_hat.cpu().long().eq(torch.ones(B)).sum().item()
		Acc_5 += (r_hat.cpu()<=5).sum().item()
		epoch_loss += batch_loss.cpu().item()

		

	P_link /= batchloader.size
	Acc_1 /= batchloader.size
	Acc_5 /= batchloader.size
	epoch_loss /= batchloader.size 
	
	train_Plink.append(P_link)
	train_Acc1.append(Acc_1)
	train_Acc5.append(Acc_5)
	train_loss.append(epoch_loss)

	logging('epoch {:2d}'.format(cur_epoch),name)
	test(devloader,model,alpha,name,is_dev=True)	
	logging('Train Plink'+str(train_Plink),name)
	logging('Dev Plink'+str(dev_Plink),name)
	logging('Train Acc@1 '+str(train_Acc1),name)
	logging('Dev Acc@1 '+str(dev_Acc1),name)
	logging('Train Acc@5 '+str(train_Acc5),name)
	logging('Dev Acc@5 '+str(dev_Acc5),name)
	logging('Train loss '+str(train_loss),name)
	logging('Dev loss '+str(dev_loss),name)
	logging('',name)

	if dev_Acc1[-1]>max_Acc1[0]:
		max_Acc1[0] = dev_Acc1[-1]
		max_epoch[0] = cur_epoch
		torch.save(model.state_dict(),os.path.join('Log','model_max_{:d}_'.format(cur_epoch)+name+'.net'))
	return

def test(batchloader,model,alpha,name:str,is_dev:bool):
	
	# test through the whole dataset for one time
	P_link = 0
	Acc_1 = 0
	Acc_5 = 0
	epoch_loss = 0

	batchloader.reshuffle() # set self.batch_idx=0

	# B = batchloader.batch_size

	# criterion = nn.MarginRankingLoss(margin=margin,reduction='sum')

	model.eval()  # model.training=False eval mode

	with torch.no_grad():
		for batch_idx,batch in enumerate(batchloader):
			(x,x_len,names,names_len,y,y_link,edge_idx,edge_idx_plus)=batch
			B = x.size(0)
			input_batch = (x.cuda(),x_len,names.cuda(),names_len,edge_idx.cuda(),edge_idx_plus.cuda())
			scores_lp,scores = model(input_batch) # (B,name_size)

			sorted_scores,_ = torch.sort(scores,dim=1,descending=False) # (B,name_size) on gpu
			r_hat = scores.size(1)-torch.searchsorted(sorted_scores,scores[range(B),y].view(-1,1),right=False).squeeze(1) # (B,1)->(B)

			# _,rank_i = torch.sort(scores,dim=1,descending=True) # (B,name_size) on gpu
			# r_hat = []
			# sample_set = [list(onedtensor) for onedtensor in list(rank_i)] # [[name_size],[],...,[]]
			# for i,rank in enumerate(sample_set):
			# 	r_hat.append(rank.index(y[i])+1)
			
			loss_m = F.nll_loss(F.log_softmax(scores,dim=1),y.cuda(),reduction='sum')
			loss_lp = F.binary_cross_entropy_with_logits(scores_lp,y_link.cuda(),reduction='sum') # /names.size(0)
			batch_loss = alpha*loss_m+loss_lp

			# logging('scores='+str(scores),name)
			# logging('y='+str(y),name)
			# logging('neg='+str(neg),name)
			# logging('r_hat='+str(r_hat),name)

			P_link += ((torch.sigmoid(scores_lp.cpu())*y_link).sum(dim=1)/y_link.sum(dim=1)).sum().item()
			Acc_1 += r_hat.cpu().long().eq(torch.ones(B)).sum().item()
			Acc_5 += (r_hat.cpu()<=5).sum().item()
			epoch_loss += batch_loss.cpu().item()

		P_link /= batchloader.size
		Acc_1 /= batchloader.size
		Acc_5 /= batchloader.size
		epoch_loss /= batchloader.size 
	
	if is_dev:
		dev_Plink.append(P_link)
		dev_Acc1.append(Acc_1)
		dev_Acc5.append(Acc_5)
		dev_loss.append(epoch_loss)
	else:
		logging('Test Plink: {:.8f}'.format(P_link),name)
		logging('Test Acc@1: {:.8f}'.format(Acc_1),name)
		logging('Test Acc@5: {:.8f}'.format(Acc_5),name)
		logging('Test loss: {:.8f}'.format(epoch_loss),name)

