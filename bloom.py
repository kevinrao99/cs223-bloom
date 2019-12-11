import pyhash
import random
import math

class bloom_filter():
	def __init__(self, k, m): # we have k hash functions into the range [0, m - 1]
		self.k = max(0, int(math.ceil(k)))
		self.m = max(1, int(m))

		print self.k, self.m

		self.bits = [False for i in range(self.m)]
		self.hasher = pyhash.murmur3_32()
		sample_range = []
		for i in range(k ** 3 + 2):
			sample_range.append(i)
		if k > 0:
			self.seeds = random.sample(sample_range, k)
		else:
			self.seeds = []

	def insert(self, key):
		key = str(key)
		for hash_seed in self.seeds:
			hash_value = self.hasher(key, seed = hash_seed) % self.m
			self.bits[hash_value] = True

	def clear(self):
		self.bits = [False for i in range(self.m)]

	def set_k(self, k):
		self.k = k
		self.hasher = pyhash.murmur3_32()
		sample_range = []
		for i in range(k ** 3):
			sample_range.append(i)
		self.seeds = random.sample(sample_range, k)

	def insert_many(self, key_arr):
		for key in key_arr:
			self.insert(key)

		self.Fp = (1 - math.e ** (-1.0 * float(len(key_arr)) * self.k / self.m)) ** self.k

	def query(self, key):
		key = str(key)
		ans = True
		for hash_seed in self.seeds:
			hash_value = self.hasher(key, seed = hash_seed) % self.m
			ans = ans and self.bits[hash_value]
		return ans

	def query_many(self, key_arr):
		queries = [False for i in range(len(key_arr))]
		for i in range(len(key_arr)):
			queries[i] = self.query(key_arr[i])
		return queries

