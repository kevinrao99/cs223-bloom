import tensorflow as tf
from keras.utils import np_utils
from sklearn.linear_model import LogisticRegression

class NN():
	def __init__(self, x_train, t_train, val_data = None):
		#self.x_train = tf.keras.utils.normalize(x_train, axis=1)
		self.x_train = x_train
		self.t_train = np_utils.to_categorical(t_train)
		#self.x_test = tf.keras.utils.normalize(x_test, axis=1)
		self.val_data = val_data
		self.model = tf.keras.models.Sequential()

	def fit(self, layers):
		self.model.add(tf.keras.layers.Flatten())
		self.model.add(tf.keras.layers.Dense(layers[1], input_dim = layers[0], activation = 'relu'))
		for i in layers[2:]:
			self.model.add(tf.keras.layers.Dense(i, activation='relu'))
		self.model.add(tf.keras.layers.Dense(2, activation='softmax'))

		self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

		print "Fitting..."

		print "DEBUGGING MODE: EPOCHS ARE REDUCED"

		self.model.fit(self.x_train, self.t_train, epochs = 20, batch_size = 20, validation_data=self.val_data, verbose = 0)

	def predict(self, x_test):
		return self.model.predict(x_test)


class LR(): # logistic regression classifier
	def __init__(self, x_train, t_train, val_data = None):
		self.model = LogisticRegression()
		self.x_train = x_train
		self.t_train = t_train
		self.val_data = val_data

	def fit(self):
		self.model.fit(self.x_train, self.t_train)

	def predict(self, x_test):
		return self.model.predict(x_test)

