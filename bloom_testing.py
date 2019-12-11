import bloom
import pandas as pd
import pyhash
import numpy as np
import func
import nnn
import math
import random
import oracle
import data_structs
import matplotlib.pyplot as plt

def read_data(filename = "nyc_abb.csv", TRAIN_PROP = .20):
	global train_feats
	global train_labels
	global key_IDs
	global key_feats
	global T_data
	global Q_data
	global Q_data_ids
	global layers

	all_abb = np.array(pd.read_csv(filename))

	# print all_abb

	all_abb_ids = all_abb[:, 0]

	all_abb_labels_city = all_abb[:, 4]
	all_abb_labels = np.array(list(map(func.loc_to_bool, all_abb_labels_city)))

	all_abb_feats = all_abb[:, [6, 7, 9, 10, 11, 13, 14, 15]]
	# all_abb_feats = all_abb[:, [6, 7]]


	for i in range(len(all_abb_feats)):
		for j in range(len(all_abb_feats[i])):
			if math.isnan(all_abb_feats[i][j]):
				all_abb_feats[i][j] = 0


	train_feats = [] # mix of TRAIN_PROP of positives and TRAIN_PROP of negatives, only for training oracle
	train_labels = [] # labels for training data
	key_IDs = [] # IDs for positives to be inserted into bloom filter
	key_feats = []


	T_data = [] # only negatives, used to determine empirical false positive rate (1 - TRAIN_PROP) / 2

	Q_data = [] # query set (only negatives) (1 - TRAIN_PROP) / 2
	Q_data_ids = [] # query set unique values to fool bloom filters

	for i in range(len(all_abb)):
		val = random.random()
		if all_abb_labels[i]: # it's a positive
			if val < TRAIN_PROP:
				train_feats.append(all_abb_feats[i])
				train_labels.append(all_abb_labels[i])
				key_IDs.append(all_abb_ids[i])
				key_feats.append(all_abb_feats[i])
		else: # it's a negative
			if val < TRAIN_PROP:
				train_feats.append(all_abb_feats[i])
				train_labels.append(all_abb_labels[i])
			elif val < TRAIN_PROP + (1 - TRAIN_PROP) / 2:
				T_data.append(all_abb_feats[i])
			else:
				Q_data.append(all_abb_feats[i])
				Q_data_ids.append(all_abb_ids[i])


	train_feats = np.array(train_feats)
	train_labels = np.array(train_labels)
	key_IDs = np.array(key_IDs)
	key_feats = np.array(key_feats)
	T_data = np.array(T_data)
	Q_data = np.array(Q_data)
	Q_data_ids = np.array(Q_data_ids)

	layers = [len(train_feats[0]), 100, 100, 50]

	'''

	print train_feats
	print train_labels
	print key_IDs
	print T_data
	print Q_data
	print Q_data_ids
	'''


def test_lbf(tau, m):
	learned_bloom_filter = data_structs.Learned_Bloom_Filter(tau, len(key_IDs), m)
	learned_bloom_filter.train_learned_function(train_feats, train_labels, layers)
	learned_bloom_filter.insert_many(key_IDs)
	learned_bloom_filter.measure_learned_Fp(T_data)
	print "Learned Bloom Filter oracle empirical false positive rate:", learned_bloom_filter.learned_Fp

	(query_results, _, _) = learned_bloom_filter.query_many(Q_data, Q_data_ids)

	pos_count = 0
	for val in query_results:
		if val:
			pos_count += 1

	print "Learned Bloom Filter backup false positive rate:", learned_bloom_filter.backup_filter.Fp
	print "Learned Bloom Filter false positive rate:", float(pos_count) / len(query_results)
	print "Theoretical False Positive rate:", learned_bloom_filter.learned_Fp + (1 - learned_bloom_filter.learned_Fp) * learned_bloom_filter.backup_filter.Fp

	return float(pos_count) / len(query_results)


if __name__ == "__main__":

	read_data()
	# test_lbf(.6, 10000)

	b_vals = []

	trials = 3
	tau_arr = [0.4, 0.6, 0.8, 0.9] #cant go up to tau = 1

	theoretical_fpr = [[] for i in range(len(tau_arr))]
	actual_fpr = [[] for i in range(len(tau_arr))]

	b_nums = [0.2, 0.4, 0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]

	for b in b_nums: # b has to start out large enough

		b_vals.append(b)
		print b

		tf = [0 for i in range(len(tau_arr))]
		af = [0 for i in range(len(tau_arr))]

		for t in range(trials):
			read_data()
			learned_bloom_filter = data_structs.Sandwiched_Bloom_Filter(0)
			learned_bloom_filter.train_learned_function(train_feats, train_labels, layers)


			for i in range(len(tau_arr)):
				learned_bloom_filter.set_tau(tau_arr[i])
				learned_bloom_filter.measure_learned_Fp(T_data)
				learned_bloom_filter.measure_learned_Fn(key_feats)

				learned_bloom_filter.init_filters(b, len(key_IDs))
				learned_bloom_filter.insert_many(key_IDs, key_feats)

				tf[i] += learned_bloom_filter.initial_filter.Fp * (learned_bloom_filter.learned_Fp + (1 - learned_bloom_filter.learned_Fp) * learned_bloom_filter.backup_filter.Fp)

				(query_results, initial_fpr, learned_fpr, backup_fpr) = learned_bloom_filter.query_many(Q_data, Q_data_ids)


				pos_count = 0
				for val in query_results:
					if val:
						pos_count += 1
				af[i] += float(pos_count) / len(query_results)

		for i in range(len(tau_arr)):
			theoretical_fpr[i].append(tf[i] / trials)
			actual_fpr[i].append(af[i] / trials)

	cols = ["Red", "Blue", "Green", "Orange", "Purple"]

	for i in range(len(tau_arr)):
		plt.plot(b_vals, actual_fpr[i], c = cols[i], marker = ".", label = "tau = " + str(tau_arr[i]))
		plt.scatter(b_vals, theoretical_fpr[i], c = cols[i], marker = "+")

	plt.xlabel("b")
	plt.ylabel("Average false positive rate")
	plt.title("False Positive Rate vs b")
	plt.legend()
	plt.show()
















