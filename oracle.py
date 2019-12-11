import nnn
import numpy as np

class nn_oracle():
	def __init__(self, t):
		self.oracle = None
		self.tau = t

	def train(self, train_dat, train_labels, layers):
		val_size = len(train_dat) / 10

		train_actual = train_dat[val_size:]
		train_labels_actual = train_labels[val_size:]

		val_feats = train_dat[:val_size]
		val_labels = train_labels[:val_size]

		'''

		print train_actual.shape
		print train_labels_actual.shape
		print val_feats.shape
		print val_labels.shape

		'''

		self.oracle = nnn.NN(train_actual, train_labels_actual, (val_feats, val_labels))
		self.oracle.fit(layers)

	def predict(self, x_test):
		nn_confidence = self.oracle.predict(x_test) # for each row, col 0 corresponds to negative and col 1 corresponds to positive
		predictions = [False for i in range(len(nn_confidence))]
		for i in range(len(nn_confidence)):
			predictions[i] = nn_confidence[i][1] > self.tau # predicts positive (true) for a query if the confidence in positive is high

		return np.array(predictions)

	def set_tau(self, t):
		self.tau = t

	def measure_Fp_rate(self, T):
		predictions = self.predict(T)
		tot = 0
		for val in predictions:
			if val:
				tot += 1

		self.Fp = float(tot) / len(T)

		return self.Fp

	def measure_Fn_rate(self, key_feats):
		predictions = self.predict(key_feats)
		tot = 0
		for val in predictions:
			if not val:
				tot += 1

		self.Fn = float(tot) / len(key_feats)

		return self.Fn


class lr_oracle():
	def __init__(self):
		self.oracle = None

	def train(self, train_dat, train_labels, layers = None):
		self.oracle = nnn.LR(train_dat, train_labels)
		self.oracle.fit()

	def predict(self, x_test):
		predictions = self.oracle.predict(x_test)
		return predictions

	def measure_Fp_rate(self, T):
		predictions = self.predict(T)
		tot = 0
		for val in predictions:
			if val == 1:
				tot += 1

		self.Fp = float(tot) / len(T)

		return self.Fp







