import oracle
import bloom
import math

class Learned_Bloom_Filter():
	def __init__(self, t, n, m):
		self.learned_function = oracle.nn_oracle(t)
		k = int(math.ceil(0.6931 * m / n))
 		self.backup_filter = bloom.bloom_filter(k, m)

	def set_tau(self, tau):
		self.learned_function.set_tau(tau)

	def set_learned_function(self, f):
		self.learned_function = f

	def train_learned_function(self, train_feats, train_labels, layers):
		self.learned_function.train(train_feats, train_labels, layers)

	def insert_many(self, key_IDs, key_feats):
		key_predictions = self.learned_function.predict(key_feats)
		neg = 0
		neg_keys = []
		for i in range(len(key_predictions)):
			if not key_predictions[i]:
				neg += 1
				neg_keys.append(key_IDs[i])

		new_k = int(math.ceil(0.6931 * self.backup_filter.m / len(neg_keys)))
		self.backup_filter.set_k(new_k)
		self.backup_filter.insert_many(neg_keys)
		self.learned_Fn = float(neg) / len(key_predictions)

	def measure_learned_Fp(self, T):
		self.learned_Fp = self.learned_function.measure_Fp_rate(T)
		return self.learned_Fp

	def clear_bloom(self):
		self.backup_filter.clear()

	def query_many(self, query_feats, query_keys):
		learned_results = self.learned_function.predict(query_feats)
		backup_results = self.backup_filter.query_many(query_keys)


		final_results = [False for i in range(len(learned_results))]

		for i in range(len(learned_results)):
			final_results[i] = learned_results[i] or backup_results[i]

		# print "all res:"
		# for i in range(100):
		# 	print learned_results[i], backup_results[i], final_results[i]

		learned_pos_ct = 0
		for val in learned_results:
			if val:
				learned_pos_ct += 1

		learned_fpr = float(learned_pos_ct) / len(learned_results)

		backup_pos_ct = 0
		for val in backup_results:
			if val:
				backup_pos_ct += 1

		backup_fpr = float(backup_pos_ct) / len(backup_results)


		
		
		if float(learned_pos_ct) / len(learned_results) > .9 or float(backup_pos_ct) / len(backup_results) > .9:
			print "learned results:", learned_results[:40]
			print "backup results:", backup_results[:40]

		print "Learned Bloom Filter experimental learned false positive rate:", learned_fpr
		print "Learned Bloom Filter experimental backup false positive rate:", backup_fpr
	

		return (final_results, learned_fpr, backup_fpr)
		

class Sandwiched_Bloom_Filter():
	def __init__(self, t): #storing n keys, b_1n bits for the initial filter, b_2n for backup
		self.learned_function = oracle.nn_oracle(t)

	def set_tau(self, tau):
		self.learned_function.set_tau(tau)

	def set_learned_function(f):
		self.learned_function = f

	def train_learned_function(self, train_feats, train_labels, layers):
		self.learned_function.train(train_feats, train_labels, layers)

	def init_filters(self, b, n):

		alpha = 0.6185

		if self.learned_Fp == 1:
			self.learned_Fp -= 0.001
			self.learned_Fn += 0.001
			print "adj down"

		elif self.learned_Fp == 0:
			self.learned_Fp += 0.001
			self.learned_Fn -= 0.001

		big_expr = (self.learned_Fp) / (1 - self.learned_Fp) / (1 / self.learned_Fn - 1)

		b_2 = min(b, self.learned_Fn * math.log(big_expr, alpha))
		b_1 = b - b_2

		print "b1, b2", b_1, b_2

		initial_k = int(math.ceil(0.6931 * b_1))
		backup_k = int(math.ceil(0.6931 * b_2))

		self.initial_filter = bloom.bloom_filter(initial_k, b_1 * n)
		self.backup_filter = bloom.bloom_filter(backup_k, b_2 * n)

	def insert_many(self, key_IDs, key_feats):
		self.initial_filter.insert_many(key_IDs)

		key_predictions = self.learned_function.predict(key_feats)
		neg = 0
		neg_keys = []
		for i in range(len(key_predictions)):
			if not key_predictions[i]:
				neg += 1
				neg_keys.append(key_IDs[i])

		if len(neg_keys) == 0:
			new_k = 1
		else:
			new_k = int(math.ceil(0.6931 * self.backup_filter.m / len(neg_keys)))
		self.backup_filter.set_k(new_k)
		self.backup_filter.insert_many(neg_keys)
		self.learned_Fn = float(neg) / len(key_predictions)

	def measure_learned_Fp(self, T):
		self.learned_Fp = self.learned_function.measure_Fp_rate(T)
		return self.learned_Fp

	def measure_learned_Fn(self, key_feats):
		self.learned_Fn = self.learned_function.measure_Fn_rate(key_feats)
		return self.learned_Fn

	def query_many(self, query_feats, query_keys):
		initial_results = self.initial_filter.query_many(query_keys)
		learned_results = self.learned_function.predict(query_feats)
		backup_results = self.backup_filter.query_many(query_keys)

		final_results = [False for i in range(len(learned_results))]

		for i in range(len(learned_results)):
			final_results[i] = initial_results[i] and (learned_results[i] or backup_results[i])

		initial_pos_ct = 0
		for val in initial_results:
			if val:
				initial_pos_ct += 1

		initial_fpr = float(initial_pos_ct) / len(initial_results)

		learned_pos_ct = 0
		for val in learned_results:
			if val:
				learned_pos_ct += 1

		learned_fpr = float(learned_pos_ct) / len(learned_results)

		backup_pos_ct = 0
		for val in backup_results:
			if val:
				backup_pos_ct += 1

		backup_fpr = float(backup_pos_ct) / len(backup_results)

		
		'''
		if float(learned_pos_ct) / len(learned_results) > .9 or float(backup_pos_ct) / len(backup_results) > .9:
			print "learned results:", learned_results
			print "backup results:", backup_results

		print "Sandwiched Filter experimental initial false positive rate:", initial_fpr
		print "Sandwiched Filter experimental learned false positive rate:", learned_fpr
		print "Sandwiched Filter experimental backup false positive rate:", backup_fpr
		'''

		return (final_results, initial_fpr, learned_fpr, backup_fpr)


