import bloom
import matplotlib.pyplot as plt







if __name__ == "__main__":
	trials = 10
	m = 10000
	n = 1000
	keys = list(range(n))
	u = 10000
	queries = list(range(n, u + n))

	k_arr = []
	fpr_arr = []
	fpr_max = 0

	for k in range(1, 13):
		for t in range(trials):
			bloom_filter = bloom.bloom_filter(k, m)
			bloom_filter.insert_many(keys)
			res = bloom_filter.query_many(queries)
			pos = 0
			for val in res:
				if val:
					pos += 1

			
			k_arr.append(k)
			fpr_arr.append(float(pos) / len(res))
			fpr_max = max(fpr_max, float(pos) / len(res))

	plt.scatter(k_arr, fpr_arr, c = "Blue", label = "False Positive Rate")
	plt.title("Effect of k On False Positive Rate")
	plt.xlabel("Number of hash functions")
	plt.ylabel("False Positive Rates")
	plt.vlines(0.6931 * m / n, 0, fpr_max, colors = "Red", label = "Optimal k")
	plt.legend()
	plt.show()












