import os.path
import math

import numpy as np
import scipy.sparse as sp
import networkx
# import obonet

import torch
import tensorflow as tf # v2.3

from Ref import tokenization,read


class Vocabulary(object):
	def __init__(self,vocab_file=None):
		super(Vocabulary,self).__init__()
		self.word2index = {}
		self.index2word = []
		self.size = 0
		if vocab_file is not None:
			with open(vocab_file,'r',encoding='utf-8') as f:
				while True:
					token = f.readline()
					if not token:
						break
					token = token.strip()
					self.word2index[token] = self.size
					self.index2word.append(token)
					self.size += 1

	def add_sentence(self,sentence):
		for word in sentence:
			if word not in self.word2index:
				self.word2index[word] = self.size
				self.index2word.append(word)
				self.size += 1


def create_vocab(namelist,dataset):
	''' create vocab from data '''
	vocab = Vocabulary()
	vocab.add_sentence(['[PAD]']) # vocab.word2index['[PAD]']=0, consistent with LSTM pad and nn.Embedding()
	vocab.add_sentence(['[UNK]'])
	for name in namelist:
		vocab.add_sentence(name[1])
	for example in dataset:
		vocab.add_sentence(example[1])
	return vocab


def get_biobert_wordvec():
	'''BioBERT-Large v1.1 (+ PubMed 1M) - based on BERT-large-Cased (custom 30k vocabulary)'''
	wordvec_idx = {}
	with open(os.path.join('..','pretrain','biobert_large','vocab_cased_pubmed_pmc_30k.txt'),'r',encoding='utf-8') as f:
		for i,word in enumerate(f.readlines()):
			wordvec_idx[word.strip()] = i

	reader = tf.train.load_checkpoint('../pretrain/biobert_large/bio_bert_large_1000k.ckpt')
	wordvec = reader.get_tensor('bert/embeddings/word_embeddings') # numpy.ndarray, wordvec.shape = (58996,1024)
	return wordvec_idx,wordvec



def Obo2Netx(obo_filename):

	obo_path = os.path.join('..','data',obo_filename)

	netx = read.read_obo(obo_path) # read_obo() does not contain obsolete nodes
	synonym_pairs = []

	
	# remove None name nodes
	id2name = {id_:data.get('name') for id_,data in netx.nodes(data=True)} # id_ is a string like 'CL:0000001', data is a dict
	for id_ in id2name:
		if id2name[id_] is None:
			netx.remove_node(id_)

	# id is unique, names may be duplicate
	id2name = {id_:data.get('name') for id_,data in netx.nodes(data=True)}
	idx2id = list(id2name.keys()) # determines the idx of all the id_ s.
	id2idx = {id_:idx for idx,id_ in enumerate(idx2id)}

	# using is_a edges to build adjacency matrix
	nnodes = len(idx2id)
	e_source = []
	e_dest = []
	for u,v in netx.edges(): # u,v are id strings like 'CL:0000001'
		if 'is_a' in netx.get_edge_data(u,v):
			e_source.append(id2idx[u])
			e_dest.append(id2idx[v])
	adj = sp.coo_matrix((np.ones(len(e_source)),(e_source,e_dest)),shape=(nnodes,nnodes),dtype=np.float32)
	
	# remove duplicate eadges
	adj = adj-adj.multiply(adj>1)+(adj>1)

	# build undirected adj by getting the larger element of adj and adj.T at each position
	undirect_adj = adj+adj.T.multiply(adj.T>adj)-adj.multiply(adj.T>adj) 

	# don't forget to add self-connections!
	undirect_adj = undirect_adj+sp.eye(nnodes)

	for id_,data in netx.nodes(data=True):
		if data.get('synonym')!=None:
			for synonym in data.get('synonym'):
				synonym_pairs.append((id_,synonym.split('"')[1]))

	return synonym_pairs,id2name,id2idx,undirect_adj


def split_and_save_dataset(ontology_name:str,split:tuple):
	'''
	ontology_name in {'cl','go','doid','chebi'}, split=(train,dev,test)
	split_and_save_dataset('cl',(20080,2000,2000))
	split_and_save_dataset('go',(20080,2000,2000))
	split_and_save_dataset('doid',(20080,2000,2000))
	split_and_save_dataset('chebi',(20080,2000,2000))
	'''

	synonym_pairs,id2name,id2idx,adj = Obo2Netx(ontology_name+'.obo')
	print(ontology_name+' synonym pairs: {:d}  train/dev/test: {:d}/{:d}/{:d}'.format(len(synonym_pairs),split[0],split[1],split[2]))
	
	shuffle = torch.randperm(len(synonym_pairs))
	train_pairs = [synonym_pairs[i] for i in shuffle[:split[0]]]
	dev_pairs = [synonym_pairs[i] for i in shuffle[split[0]:split[0]+split[1]]]
	test_pairs = [synonym_pairs[i] for i in shuffle[split[0]+split[1]:]]

	with open(os.path.join('..','data',ontology_name+'_train.tsv'),'w',encoding='utf-8') as f:
		f.write('LABEL\tSYNONYM\n')
		for i in shuffle[:split[0]]:
			f.write(synonym_pairs[i][0]+'\t'+synonym_pairs[i][1]+'\n')

	with open(os.path.join('..','data',ontology_name+'_dev.tsv'),'w',encoding='utf-8') as f:
		f.write('LABEL\tSYNONYM\n')
		for i in shuffle[split[0]:split[0]+split[1]]:
			f.write(synonym_pairs[i][0]+'\t'+synonym_pairs[i][1]+'\n')

	with open(os.path.join('..','data',ontology_name+'_test.tsv'),'w',encoding='utf-8') as f:
		f.write('LABEL\tSYNONYM\n')
		for i in shuffle[split[0]+split[1]:]:
			f.write(synonym_pairs[i][0]+'\t'+synonym_pairs[i][1]+'\n')

	with open(os.path.join('..','data',ontology_name+'_names.tsv'),'w',encoding='utf-8') as f:
		f.write('ID\tNAME\tNODE_INDEX\n')
		for id_ in id2name:
			f.write(id_+'\t'+id2name[id_]+'\t'+str(id2idx[id_])+'\n')

	sp.save_npz(os.path.join('..','data',ontology_name+'_adj.npz'),adj)


def load_dataset(ontology_name:str):
	# ontology_name in {'cl','go','doid','chebi'}

	tokenizer = tokenization.FullTokenizer(os.path.join('..','pretrain','biobert_large','vocab_cased_pubmed_pmc_30k.txt'),do_lower_case=False)

	id2idx = {}
	namelist = []
	with open(os.path.join('..','data',ontology_name+'_names.tsv'),'r',encoding='utf-8') as f:
		f.readline()
		for line in f.readlines():
			(id_,name,idx) = line.strip().split('\t')
			id2idx[id_] = int(idx)
			namelist.append((int(idx),tokenizer.tokenize(name)))
			
	trainset = []
	devset = []
	testset = []
	with open(os.path.join('..','data',ontology_name+'_train.tsv'),'r',encoding='utf-8') as f:
		f.readline()
		for line in f.readlines():
			(id_,syn) = line.strip().split('\t')
			trainset.append((id2idx[id_],tokenizer.tokenize(syn)))
	with open(os.path.join('..','data',ontology_name+'_dev.tsv'),'r',encoding='utf-8') as f:
		f.readline()
		for line in f.readlines():
			(id_,syn) = line.strip().split('\t')
			devset.append((id2idx[id_],tokenizer.tokenize(syn)))
	with open(os.path.join('..','data',ontology_name+'_test.tsv'),'r',encoding='utf-8') as f:
		f.readline()
		for line in f.readlines():
			(id_,syn) = line.strip().split('\t')
			testset.append((id2idx[id_],tokenizer.tokenize(syn)))

	adj = sp.load_npz(os.path.join('..','data',ontology_name+'_adj.npz'))

	return namelist,trainset,devset,testset,adj


def pad_and_index_dataset(rdataset,vocab,max_length):
	'''truncate, add [CLS],[SEP], pad to max_length
	   prepare input for a BERT tokenizer'''
	dataset=[]
	for (nameidx,wordseq) in rdataset:
		max_name_length = max_length-2
		if len(wordseq)>max_name_length:
			wordseq = wordseq[:max_name_length]
		wordseq = ['[CLS]']+wordseq+['[SEP]']+['[PAD]']*(max_name_length-len(wordseq)) # max_length
		wordseq = [vocab.word2index[word] for word in wordseq] 
		# [PAD] index is 0 in vocab, pad index is 0 in nn.utils.rnn.pad_packed_sequence and nn.Embedding() 
		dataset.append((nameidx,wordseq))
	return dataset


def index_dataset(rdataset,vocab):
	dataset = []
	for (nameidx,wordseq) in rdataset:
		wordseq = [vocab.word2index.get(word,vocab.word2index['[UNK]']) for word in wordseq]
		dataset.append((nameidx,wordseq))
	return dataset


class BatchLoader(object):   
	# an iterator/generator object
	def __init__(self,adj,namelist,dataset,batch_size,pad:int):
		super(BatchLoader,self).__init__()

		self.dataset=dataset
		self.batch_size=batch_size
		self.pad=pad
		self.size=len(dataset)
		self.n_batches=math.ceil(self.size/batch_size)

		self.batch_idx=0
		self.shuffle=torch.randperm(self.size) 
		# last mini-batch is less or equal to batch_size, no cycle.

		self.names = []
		for i,example in enumerate(namelist):
			assert(i==example[0])
			self.names.append(example[1]) # name order consistent with id2idx

		self.names_len = [len(wordseq) for wordseq in self.names]
		names_max = max(self.names_len)
		self.names = [s+[self.pad]*(names_max-len(s)) for s in self.names]
		self.names = torch.LongTensor(self.names) # (name_size,name_max_len), torch.LongTensor([[...],[...]]) is a 2d tensor

		self.adj = torch.FloatTensor(np.array(adj.todense())) # (name_size,name_size)
		self.edge_idx = torch.nonzero(self.adj).t() # (2,E) LongTensor, including both (i,j) and (j,i)

		name_size = self.adj.size(0)
		plus = torch.LongTensor([list(range(name_size))+(name_size+1)*[name_size],name_size*[name_size]+list(range(name_size+1))]) # (2,2*name_size+1)
		self.edge_idx_plus = torch.cat([self.edge_idx,plus],1) # (2,E+2*name_size+1)

		self.adj2 = ((self.adj+torch.mm(self.adj,self.adj))>0).float() # 2-hop neighborhood

	def __len__(self):
		return self.size

	def __iter__(self):
		return self

	def __next__(self):
		if self.batch_idx>=self.n_batches:
			raise StopIteration
		else:
			shuffle=self.shuffle[self.batch_idx*self.batch_size:min((self.batch_idx+1)*self.batch_size,self.size)]
			x=[]
			y=[]
			for idx in shuffle:
				example=self.dataset[idx]
				x.append(example[1])
				y.append(example[0])
			
			y=torch.LongTensor(y) # (N)
			y_link=self.adj2[y,:] # (N,name_size), need to be float for nn.BCEWithLogits()
			
			x_len=[len(wordseq) for wordseq in x] 
			# x_len=[length if length>=12 else 12 for length in x_len]
			x_max=max(x_len)
			x=[s+[self.pad]*(x_max-len(s)) for s in x]
			x=torch.LongTensor(x) # (N,x_max_len), torch.LongTensor([[...],[...]]) is 2d tensor

			self.batch_idx+=1
			return (x,x_len,self.names,self.names_len,y,y_link,self.edge_idx,self.edge_idx_plus)

	def reshuffle(self):
		self.batch_idx=0
		self.shuffle=torch.randperm(self.size)