import os
import csv
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from gensim.models import KeyedVectors

from sklearn.metrics import f1_score

label_dict = {'Arts':0, 'World':1, 'Sports':2, 'Opinion':3, 'New York and Region':4, 'Business':5, 'U.S.':6}

def freeze_seed(seed=42):
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)


class Config:

	 def __init__(self, num_classes, keep_dropout, emb_size, hidden_dims, rnn_layers, learning_rate, epoch, batch_size, len_seq, device, is_dev, is_test):
	 	self.num_classes = num_classes
	 	self.keep_dropout = keep_dropout
	 	self.emb_size = emb_size
	 	self.hidden_dims = hidden_dims
	 	self.rnn_layers = rnn_layers

	 	self.learning_rate = learning_rate
	 	self.epoch = epoch
	 	self.batch_size = batch_size
	 	self.len_seq = len_seq

	 	self.device = device

	 	self.is_dev = is_dev
	 	self.is_test = is_test


class TextBiLSTM(nn.Module):

	def __init__(self, config):
		super(TextBiLSTM, self).__init__()
		self.num_classes = config.num_classes
		self.keep_dropout = config.keep_dropout
		self.emb_size = config.emb_size
		self.hidden_dims = config.hidden_dims
		self.rnn_layers = config.rnn_layers

		self.build_model()


	def build_model(self):
		# Bi-LSTM
		self.lstm_net = nn.LSTM(self.emb_size, self.hidden_dims, num_layers=self.rnn_layers, dropout=self.keep_dropout, bidirectional=True)
		
		# Attention
		self.attention_layer = nn.Sequential(
			nn.Linear(self.hidden_dims, self.hidden_dims),
			nn.ReLU(inplace=True)
		)

		# Dense
		self.fc_out = nn.Sequential(
			nn.Dropout(self.keep_dropout),
			nn.Linear(self.hidden_dims, self.hidden_dims),
			nn.ReLU(inplace=True),
			nn.Dropout(self.keep_dropout),
			nn.Linear(self.hidden_dims, self.num_classes)
		)


	def attention_net_with_w(self, lstm_out, lstm_hidden):
		'''
		params:
			lstm_out:		[batch_size, len_seq, n_hidden*2]
			lstm_hidden:	[batch_size, num_layers*num_directions, n_hidden]
		return:
			[batch_size, n_hidden]
		'''

		lstm_tmp_out = torch.chunk(lstm_out, 2, -1) # 2 * [batch_size, len_seq, n_hidden]

		h = lstm_tmp_out[0] + lstm_tmp_out[1] # [batch_size, len_seq, n_hidden]

		lstm_hidden = torch.sum(lstm_hidden, dim=1) # [batch_size, 1, n_hidden]
		lstm_hidden = lstm_hidden.unsqueeze(1) # [batch_size, n_hidden]

		attn_w = self.attention_layer(lstm_hidden) # [batch_size, n_hidden]

		m = nn.Tanh()(h) # [batch_size, len_seq, n_hidden]

		attn_context = torch.bmm(attn_w, m.transpose(1, 2)) # [batch_size, 1, len_seq]

		softmax_w = F.softmax(attn_context, dim=-1) # [batch_size, 1, len_seq]

		context = torch.bmm(softmax_w, h) # [batch_size, 1, n_hidden]

		result = context.squeeze(1) # [batch_size, n_hidden]

		return result


	def forward(self, batch_embs):
		'''
		params:
			batch_embs:		[batch_size, len_seq, emb_size]
		return:
			[batch_size, num_classes]
		'''

		sen_input = batch_embs.permute(1, 0, 2) # [len_seq, batch_size, emb_size]

		output, (final_hidden_state, final_cell_state) = self.lstm_net(sen_input)
		'''
			output:					[len_seq, batch_size, n_hidden*num_directions]
			final_hidden_state:		[num_layers*num_directions, batch_size, hidden_size]
			final_cell_state:		[num_layers*num_directions, batch_size, hidden_size]
		'''

		output = output.permute(1, 0, 2) # [batch_size, len_seq, n_hidden*num_diections]

		final_hidden_state = final_hidden_state.permute(1, 0, 2) # [batch_size, num_layers*num_directions, hidden_size]

		attn_out = self.attention_net_with_w(output, final_hidden_state) # [batch_size, n_hidden]

		return self.fc_out(attn_out) # [batch_size, num_classes]


def build_dataset(datadir, sep, embdir, time_list, config):

	dataset = {}
	for t in time_list:
		dataset[t] = {'data':[], 'label':[]}

		# load pretrained embedding
		emb_fn = os.path.join(embdir, str(t)+'.w2v')
		wv = KeyedVectors.load_word2vec_format(emb_fn)

		# load data
		data_fn = os.path.join(datadir, 'sep_class_text', sep, str(t)+'.txt')
		with open(data_fn, 'r') as fin:
			for line in fin:
				line = line.strip().split(' ')[:config.len_seq]
				data = []
				for word in line:
					if word in wv:
						data.append(wv[word])
					else:
						data.append(unk_vec)
				# padding
				while len(data) < config.len_seq:
					data.append(pad_vec)

				dataset[t]['data'].append(np.array(data))

		# load label
		label_fn = os.path.join(datadir, 'sep_class_label', sep, str(t)+'.txt')
		with open(label_fn, 'r') as fin:
			for line in fin:
				label = line.strip()
				dataset[t]['label'].append(label_dict[label])

	return dataset


def build_dataset_t(datadir, sep, embdir, time_list, t, config):

	dataset_t = {'data':[], 'label':[]}

	# load pretrained embedding
	emb_fn = os.path.join(embdir, str(t)+'.w2v')
	wv = KeyedVectors.load_word2vec_format(emb_fn)

	# load data
	data_fn = os.path.join(datadir, 'sep_class_text', sep, str(t)+'.txt')
	with open(data_fn, 'r') as fin:
		for line in fin:
			line = line.strip().split(' ')[:config.len_seq]
			data = []
			for word in line:
				if word in wv:
					data.append(wv[word])
				else:
					data.append(unk_vec)
			# padding
			while len(data) < config.len_seq:
				data.append(pad_vec)

			dataset_t['data'].append(np.array(data))

	# load label
	label_fn = os.path.join(datadir, 'sep_class_label', sep, str(t)+'.txt')
	with open(label_fn, 'r') as fin:
		for line in fin:
			label = line.strip()
			dataset_t['label'].append(label_dict[label])

	return dataset_t


def test_time_slice(model, dataset_t, config):

	model.eval()
	with torch.no_grad():
		data = dataset_t['data']
		label = dataset_t['label']

		y_true = []
		y_pred = []

		sample_num = len(data)
		correct = 0
		for idx in range(0, sample_num, config.batch_size):
			idx_list = range(idx, min(sample_num, idx+config.batch_size))
			batch_data = np.array([data[i] for i in idx_list])
			batch_label = np.array([label[i] for i in idx_list])

			data_tensor = torch.from_numpy(batch_data).float().to(config.device)
			label_tensor = torch.from_numpy(batch_label).long().to(config.device)

			outputs = model(data_tensor)
			pred = torch.max(outputs, 1)[1]

			y_true.extend(list(batch_label))
			y_pred.extend(list(pred.cpu().numpy()))

			#print('pred.shape = {}'.format(pred.shape))

			correct += ((pred == label_tensor).sum().item())

	result = {}
	result['acc'] = correct/sample_num
	result['micro'] = f1_score(y_true, y_pred, average='micro')
	result['macro'] = f1_score(y_true, y_pred, average='macro')
	
	return result


def test(model, dataset, time_list, config):

	model.eval()
	results = []
	for t in time_list:
		acc_t = test_time_slice(model, dataset[t], config)
		results.append(acc_t)
	return results



def train(datadir, embdir, dev_dataset, test_dataset, time_list, config):

	dev_results = []
	test_results = []

	model = TextBiLSTM(config).to(config.device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

	for t in time_list:
		start_time = time.time()
		print('processing year:{}'.format(t))

		dataset_t = build_dataset_t(datadir, 'train', embdir, time_list, t, config)
		sample_num = len(dataset_t['data'])
		idx_list = np.arange(sample_num)

		for ep in range(config.epoch):
			ep_start_time = time.time()
			print('processing epoch:{}'.format(ep))

			# build batch
			np.random.shuffle(idx_list)

			total_loss = 0
			for idx in range(0, sample_num, config.batch_size):
				model.train()

				idx_list_t = idx_list[idx : min(sample_num, idx+config.batch_size)]
				batch_data = np.array([dataset_t['data'][i] for i in idx_list_t])
				batch_label = np.array([dataset_t['label'][i] for i in idx_list_t])

				data_tensor = torch.from_numpy(batch_data).float().to(config.device)
				label_tensor = torch.from_numpy(batch_label).long().to(config.device)

				optimizer.zero_grad()
				
				outputs = model(data_tensor)
				#print(outputs[0], label_tensor[0])
				#pred = torch.max(outputs, 1)

				loss = criterion(outputs, label_tensor)
				loss.backward()
				optimizer.step()

				total_loss += loss.item()
				'''
				if (idx/config.batch_size) % 100 == 0:
					print('[year/epoch/step] {}/{}/{} loss:{:.4f}'.format(t, ep, (idx + 1)/config.batch_size, loss.item()/len(idx_list_t)))
					dev_acc_list = test(model, dev_dataset, time_list, config)
					info = '[dev]'
					for pt, dev_acc in zip(time_list, dev_acc_list):
						info += ' {}:{:.2f}'.format(pt, dev_acc)
					print(info)
				'''
			print('process epoch:{} finish loss:{:4f} used:{:.2f} min'.format(ep, total_loss/sample_num, (time.time() - ep_start_time)/60.0))
		print('process year:{} finish used:{:.2f} min'.format(t, (time.time() - start_time)/60.0))

		# dev
		if config.is_dev:
			dev_acc_list = test(model, dev_dataset, time_list, config)
			dev_results.append(dev_acc_list)
			info = '[dev]'
			for m in ['acc', 'micro', 'macro']:
				info += '\n{}\n'.format(m)
				for pt, dev_acc in zip(time_list, dev_acc_list):
					info += ' {}:{:.2f}'.format(pt, dev_acc[m])
			print(info)

		if config.is_test:
			# test
			test_acc_list = test(model, test_dataset, time_list, config)
			test_results.append(test_acc_list)
			info = '[test]'
			for m in ['acc', 'micro', 'macro']:
				info += '\n{}\n'.format(m)
				for pt, test_acc in zip(time_list, test_acc_list):
					info += ' {}:{:.2f}'.format(pt, test_acc[m])
			print(info)

	return dev_results, test_results


def save_result(savefn, results, m):
	with open(savefn, 'w', newline='') as fout:
		writer = csv.writer(fout)
		for result in results:
			writer.writerow([r[m] for r in result])



if __name__=='__main__':
	freeze_seed()

	parser = argparse.ArgumentParser()
	parser.add_argument('--datadir', default='../corpus/nyt/forClassification')
	parser.add_argument("--embdir", default="model")
	parser.add_argument("--result", default="result")

	parser.add_argument('--start', type=int, default=1990)
	parser.add_argument('--end', type=int, default=2016)

	parser.add_argument('--classes', type=int, default=7)
	parser.add_argument('--dropout', type=float, default=0.1)
	parser.add_argument("--embsize", type=int, default=50)
	parser.add_argument('--hidden', type=int, default=50)
	parser.add_argument('--layers', type=int, default=1)
	parser.add_argument('--eta', type=float, default=0.001)
	parser.add_argument('--epoch', type=int, default=10)
	parser.add_argument('--batch', type=int, default=16)
	parser.add_argument('--seq', type=int, default=60)

	parser.add_argument('--cuda', type=int, default=0)

	parser.add_argument('--dev', action='store_true')
	parser.add_argument('--test', action='store_true')

	args = parser.parse_args()
	datadir = args.datadir
	embdir = args.embdir
	result = args.result

	time_list = range(args.start, args.end+1)

	num_classes = args.classes
	keep_dropout = args.dropout
	emb_size = args.embsize 
	hidden_dims = args.hidden 
	rnn_layers = args.layers 
	learning_rate = args.eta 
	epoch = args.epoch
	batch_size = args.batch
	len_seq = args.seq

	cuda_id = args.cuda
	if cuda_id < 0:
		device = torch.device('cpu')
	else:
		device = torch.device('cuda:'+str(cuda_id) if torch.cuda.is_available() else 'cpu')

	is_dev = args.dev 
	is_test = args.test

	config = Config(num_classes=num_classes, 
					keep_dropout=keep_dropout,
					emb_size=emb_size,
					hidden_dims=hidden_dims,
					rnn_layers=rnn_layers, 
					learning_rate=learning_rate, 
					epoch=epoch, 
					batch_size=batch_size,
					len_seq=len_seq, 
					device=device,
					is_dev=is_dev,
					is_test=is_test)

	pad_vec = np.random.rand(config.emb_size)
	unk_vec = np.random.rand(config.emb_size)

	print('building dataset ...')
	#train_dataset = build_dataset(datadir, 'train', embdir, time_list, config)
	if is_dev:
		dev_dataset = build_dataset(datadir, 'dev', embdir, time_list, config)
	else:
		dev_dataset = None

	if is_test:
		test_dataset = build_dataset(datadir, 'test', embdir, time_list, config)
	else:
		test_dataset = None
	
	print('built dataset! [dev={}] [test={}]'.format(is_dev, is_test))

	dev_results, test_results = train(datadir, embdir, dev_dataset, test_dataset, time_list, config)

	for m in ['acc', 'micro', 'macro']:
		if is_dev:
			dev_savefn = os.path.join(result+'-'+m+'-dev.csv')
			save_result(dev_savefn, dev_results, m)

		if is_test:
			test_savefn = os.path.join(result+'-'+m+'-test.csv')
			save_result(test_savefn, test_results, m)
