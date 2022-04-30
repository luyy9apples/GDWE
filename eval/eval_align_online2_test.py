import argparse
import os
import sys
from gensim.models import KeyedVectors
from eval_metrics2 import evalMRR_online, evalMP_online
import csv
from tqdm import tqdm

def readTestData(test_data_fn):
	src_words = []
	src_years = []
	tar_words = []
	tar_years = []

	years_list = list(range(1990, 2017))

	total_row = 0
	invalid_row = 0
	with open(test_data_fn, 'r', newline='') as f:
		reader = csv.reader(f)
		for row in reader:
			total_row += 1
			if len(row) != 2:
				invalid_row += 1
				continue
			data0 = row[0].strip().split('-')
			data1 = row[1].strip().split('-')
			if len(data0) != 2 or len(data1) != 2 or int(data0[1]) not in years_list or int(data1[1]) not in years_list:
				invalid_row += 1
				continue
			src_words.append(data0[0].strip())
			src_years.append(int(data0[1]))
			tar_words.append(data1[0].strip())
			tar_years.append(int(data1[1]))

	print('invalid_row / total_row:', invalid_row, '/', total_row)
	return src_words, src_years, tar_words, tar_years


def findInterNeighbors(src_vec, wv, topn=20):
	nebs = []
	if len(src_vec) == 0:
		return nebs
	nebs = wv.similar_by_vector(src_vec, topn=topn)
	return nebs


# delete my missing words
def build_my_testset2(save_path, src_words, src_years, tar_words, tar_years, src_nebs_list):
	missing_count = 0
	del_idx = []
	with open(save_path, 'w', newline='') as fout:
		writer = csv.writer(fout)
		for i in range(0, len(src_words)):
			src_w = src_words[i]
			src_y = src_years[i]
			tar_w = tar_words[i]
			tar_y = tar_years[i]

			nebs = src_nebs_list[src_y][src_w][tar_y]
			if len(nebs) == 0:
				missing_count += 1
				del_idx.append(i)
				continue

			writer.writerow([str(src_w)+'-'+str(src_y), str(tar_w)+'-'+str(tar_y)])

	total_count = len(src_words)
	print('missing count={}/{}'.format(missing_count, total_count))

	del_src_words = [src_words[i] for i in range(total_count) if i not in del_idx]
	del_src_years = [src_years[i] for i in range(total_count) if i not in del_idx]
	del_tar_words = [tar_words[i] for i in range(total_count) if i not in del_idx]
	del_tar_years = [tar_years[i] for i in range(total_count) if i not in del_idx]

	return del_src_words, del_src_years, del_tar_words, del_tar_years


def save_results(src_words, src_years, tar_words, tar_years, src_nebs_list, results_fn):
	with open(results_fn, 'w') as f:
		for i in range(0, len(src_words)):
			src_w = src_words[i]
			src_y = src_years[i]
			tar_w = tar_words[i]
			tar_y = tar_years[i]

			nebs = src_nebs_list[src_y][src_w][tar_y]
			if len(nebs) == 0:
				continue
			nebs = nebs[:10]
			nebs_w = [tup[0] for tup in nebs]
			if tar_w not in nebs_w:
				f.write('-1\t')
			else:
				f.write(str(nebs_w.index(tar_w)+1)+'\t')
			f.write(src_w+'-'+str(src_y)+'\t')
			f.write(tar_w+'-'+str(tar_y)+'\t')
			f.write(','.join(nebs_w)+'\n')


def save_temporal_result(results_fn, results):
	with open(results_fn, 'w', newline='') as fout:
		writer = csv.writer(fout)
		for row in results:
			writer.writerow([str(data) for data in row])


if __name__=="__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=int, default=1)
	parser.add_argument("--embdir", default="model")
	parser.add_argument("--result", default="")

	args = parser.parse_args()
	dataset = args.dataset
	embdir = args.embdir
	result = args.result

	# load test data
	test_data_fn = os.path.join('../corpus/nyt/Testset', 'testset_2_test.csv')
	src_words, src_years, tar_words, tar_years = readTestData(test_data_fn)

	# get src word embedding
	src_emb_list = {}
	for i in range(1990, 2017):
		# load temporal embedding
		emb_fn = os.path.join(embdir, str(i)+'.w2v') # proposed model
		#emb_fn = os.path.join(embdir, 'enriched_'+str(i)+'.txt') # CW2V
		if not os.path.exists(emb_fn):
			print(emb_fn, 'Not Found')
			break

		src_emb_i = {}
		wv = KeyedVectors.load_word2vec_format(emb_fn)
		for src_w, src_y in zip(src_words, src_years):
			if src_y != i:
				continue
			if src_w in src_emb_i:
				continue

			if src_w in wv:
				src_emb_i[src_w] = wv[src_w]
			else:
				src_emb_i[src_w] = []
		src_emb_list[i] = src_emb_i

	# get src word nebs
	src_nebs_list = {}
	for i in range(1990, 2017):
		src_nebs_list[i] = {}

	for i in range(1990, 2017):
		# load temporal embedding
		emb_fn = os.path.join(embdir, str(i)+'.w2v') # proposed model
		#emb_fn = os.path.join(embdir, 'enriched_'+str(i)+'.txt') # CW2V
		if not os.path.exists(emb_fn):
			print(emb_fn, 'Not Found')
			break

		wv = KeyedVectors.load_word2vec_format(emb_fn)
		# get nebs of src words(1990,i-1) in (i)
		for j in tqdm(range(1990, 2017), desc='building nebs in '+str(i)):
			if i == j:
				continue
			src_emb_j = src_emb_list[j]
			for src_w in src_emb_j.keys():
				src_v = src_emb_j[src_w]
				nebs = findInterNeighbors(src_v, wv)
				if src_w not in src_nebs_list[j]:
					src_nebs_list[j][src_w] = {}
				src_nebs_list[j][src_w][i] = sorted(nebs, key=lambda tup: tup[1], reverse=True)

	#results_fn = 'result_log/results_' + str(dataset) + '-' + result + '.txt'
	#save_results(src_words, src_years, tar_words, tar_years, src_nebs_list, results_fn)

	#save_fn = os.path.join('../corpus/nyt/Testset', 'my_testset_2('+str(dataset)+').csv')
	#src_words, src_years, tar_words, tar_years = build_my_testset2(save_fn, src_words, src_years, tar_words, tar_years, src_nebs_list)

	# calculate metrics
	MRR, MRR_sep = evalMRR_online(src_words, src_years, tar_words, tar_years, src_nebs_list)
	MP1, MP1_sep = evalMP_online(src_words, src_years, tar_words, tar_years, src_nebs_list, 1)
	MP3, MP3_sep = evalMP_online(src_words, src_years, tar_words, tar_years, src_nebs_list, 3)
	MP5, MP5_sep = evalMP_online(src_words, src_years, tar_words, tar_years, src_nebs_list, 5)
	MP10, MP10_sep = evalMP_online(src_words, src_years, tar_words, tar_years, src_nebs_list, 10)

	if not os.path.exists('online_result'):
		os.mkdir('online_result')

	results_fn = 'online_result/results_' + str(dataset) + '-' + result + '.csv'
	# save_temporal_result(results_fn, [MRR, MP1, MP3, MP5, MP10, MRR_sep, MP1_sep, MP3_sep, MP5_sep, MP10_sep])
	save_temporal_result(results_fn, [MRR_sep, MP1_sep, MP3_sep, MP5_sep, MP10_sep])
