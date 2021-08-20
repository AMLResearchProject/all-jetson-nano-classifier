#!/usr/bin/env python3
""" AI Model class.

Provides the AI Model functionality.

MIT License

Copyright (c) 2021 Asociación de Investigacion en Inteligencia Artificial
Para la Leucemia Peter Moss

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Contributors:
- Adam Milton-Barker

"""

import cv2
import json
import os
import pathlib
import requests
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from PIL import Image
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from mlxtend.plotting import plot_confusion_matrix

from modules.AbstractModel import AbstractModel

plt.switch_backend('Agg')


class model(AbstractModel):
	""" AI Model class """

	def prepare_data(self):
		""" Creates/sorts dataset. """

		self.data.remove_testing()
		self.data.process()

		self.helpers.logger.info("Data preperation complete.")

	def prepare_network(self):
		""" Builds the network. """

		self.tf_model = tf.keras.models.Sequential([
			tf.keras.layers.InputLayer(input_shape=(self.data.X_train.shape[1:])),
			tf.keras.layers.AveragePooling2D(
				pool_size=(2, 2), strides=None, padding='valid'),
			tf.keras.layers.Conv2D(30, (5, 5), strides=1,
				padding="valid", activation='relu'),
			tf.keras.layers.DepthwiseConv2D(30, (1, 1),
				padding="valid", activation='relu'),
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(2),
			tf.keras.layers.Activation('softmax', name='softmax')
		],
		"AllJetsonNano")
		self.tf_model.summary()

		self.helpers.logger.info("Network initialization complete.")

	def train(self):
		""" Trains the model

		Compiles and fits the model.
		"""

		self.helpers.logger.info("Using Adam Optimizer.")
		optimizer = tf.keras.optimizers.Adam(learning_rate=self.confs["train"]["learning_rate_adam"],
											 decay = self.confs["train"]["decay_adam"])

		self.helpers.logger.info("Using Early Stopping.")
		callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
													patience=3,
													verbose=0,
													mode='auto',
													restore_best_weights=True)

		self.tf_model.compile(optimizer=optimizer,
							  loss='binary_crossentropy',
							  metrics=[tf.keras.metrics.BinaryAccuracy(name='acc'),
									   tf.keras.metrics.Precision(name='precision'),
									   tf.keras.metrics.Recall(name='recall'),
									   tf.keras.metrics.AUC(name='auc') ])

		self.history = self.tf_model.fit(self.data.X_train, self.data.y_train,
										 validation_data=(self.data.X_test, self.data.y_test),
										 validation_steps=self.confs["train"]["val_steps"],
										 epochs=self.confs["train"]["epochs"], callbacks=[callback])

		print(self.history)
		print("")

		self.save_model_as_json()
		self.save_weights()
		self.convert_to_tfrt()
		self.convert_to_onnx()

	def save_model_as_json(self):
		""" Saves the model as JSON """

		with open(self.json_model_path, "w") as file:
			file.write(self.tf_model.to_json())

		self.tf_model.save(self.saved_model_path)

		self.helpers.logger.info("Model JSON saved " + self.json_model_path)

	def save_weights(self):
		""" Saves the model weights """

		self.tf_model.save_weights(self.weights_file_path)
		self.helpers.logger.info("Weights saved " + self.weights_file_path)

	def convert_to_tfrt(self):
		""" Converts the model to TFRT format """

		converter = trt.TrtGraphConverterV2(
			input_saved_model_dir=self.saved_model_path)
		converter.convert()
		converter.save(self.tfrt_model_path)

	def convert_to_onnx(self):
		""" Converts the model to ONNX format """

		os.system('python3 -m tf2onnx.convert --saved-model ' \
			+ self.saved_model_path \
			+ ' --output ' + self.onnx_model_path \
			+ ' --tag serve --signature_def serving_default')

	def predictions(self):
		""" Gets a prediction for an image. """

		self.train_preds = self.tf_model.predict(self.data.X_train)
		self.test_preds = self.tf_model.predict(self.data.X_test)

	def evaluate(self):
		""" Evaluates the model """

		self.predictions()

		metrics = self.tf_model.evaluate(
			self.data.X_test, self.data.y_test, verbose=0)
		for name, value in zip(self.tf_model.metrics_names, metrics):
			self.helpers.logger.info("Metrics: " + name + " " + str(value))
		print()

		self.plot_accuracy()
		self.plot_loss()
		self.plot_auc()
		self.plot_precision()
		self.plot_recall()
		self.confusion_matrix()
		self.figures_of_merit()

	def plot_accuracy(self):
		""" Plots the accuracy. """

		plt.plot(self.history.history['acc'])
		plt.plot(self.history.history['val_acc'])
		plt.title('Model Accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.ylim((0, 1))
		plt.legend(['Train', 'Validate'], loc='upper left')
		plt.savefig('model/plots/accuracy.png')
		plt.show()
		plt.clf()

	def plot_loss(self):
		""" Plots the loss. """

		plt.plot(self.history.history['loss'])
		plt.plot(self.history.history['val_loss'])
		plt.title('Model Loss')
		plt.ylabel('loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validate'], loc='upper left')
		plt.savefig('model/plots/loss.png')
		plt.show()
		plt.clf()

	def plot_auc(self):
		""" Plots the AUC. """

		plt.plot(self.history.history['auc'])
		plt.plot(self.history.history['val_auc'])
		plt.title('Model AUC')
		plt.ylabel('AUC')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validate'], loc='upper left')
		plt.savefig('model/plots/auc.png')
		plt.show()
		plt.clf()

	def plot_precision(self):
		""" Plots the precision. """

		plt.plot(self.history.history['precision'])
		plt.plot(self.history.history['val_precision'])
		plt.title('Model Precision')
		plt.ylabel('Precision')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validate'], loc='upper left')
		plt.savefig('model/plots/precision.png')
		plt.show()
		plt.clf()

	def plot_recall(self):
		""" Plots the recall. """

		plt.plot(self.history.history['recall'])
		plt.plot(self.history.history['val_recall'])
		plt.title('Model Recall')
		plt.ylabel('Recall')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validate'], loc='upper left')
		plt.savefig('model/plots/recall.png')
		plt.show()
		plt.clf()

	def confusion_matrix(self):
		""" Plots the confusion matrix. """

		self.matrix = confusion_matrix(self.data.y_test.argmax(axis=1),
									   self.test_preds.argmax(axis=1))

		self.helpers.logger.info("Confusion Matrix: " + str(self.matrix))
		print("")

		plot_confusion_matrix(conf_mat=self.matrix)
		plt.savefig('model/plots/confusion-matrix.png')
		plt.show()
		plt.clf()

	def figures_of_merit(self):
		""" Calculates/prints the figures of merit.

		https://homes.di.unimi.it/scotti/all/
		"""

		test_len = len(self.data.X_test)

		TP = self.matrix[1][1]
		TN = self.matrix[0][0]
		FP = self.matrix[0][1]
		FN = self.matrix[1][0]

		TPP = (TP * 100)/test_len
		FPP = (FP * 100)/test_len
		FNP = (FN * 100)/test_len
		TNP = (TN * 100)/test_len

		specificity = TN/(TN+FP)

		misc = FP + FN
		miscp = (misc * 100)/test_len

		self.helpers.logger.info(
			"True Positives: " + str(TP) + "(" + str(TPP) + "%)")
		self.helpers.logger.info(
			"False Positives: " + str(FP) + "(" + str(FPP) + "%)")
		self.helpers.logger.info(
			"True Negatives: " + str(TN) + "(" + str(TNP) + "%)")
		self.helpers.logger.info(
			"False Negatives: " + str(FN) + "(" + str(FNP) + "%)")

		self.helpers.logger.info("Specificity: " + str(specificity))
		self.helpers.logger.info("Misclassification: " +
						str(misc) + "(" + str(miscp) + "%)")

	def load(self):
		""" Loads the model """

		with open(self.json_model_path) as file:
			m_json = file.read()

		self.tf_model = tf.keras.models.model_from_json(m_json)
		self.tf_model.load_weights(self.weights_file_path)

		self.helpers.logger.info("Model loaded ")

		self.tf_model.summary()

	def predict(self, img):
		""" Gets a prediction for an image. """

		predictions = self.tf_model.predict(img)
		prediction = np.argmax(predictions, axis=-1)

		return prediction

	def reshape(self, img):
		""" Reshapes an image. """

		dx, dy, dz = img.shape
		input_data = img.reshape((-1, dx, dy, dz))
		input_data = input_data / 255.0

		return input_data

	def test(self):
		""" Test mode

		Loops through the test directory and classifies the images.
		"""

		files = 0
		tp = 0
		fp = 0
		tn = 0
		fn = 0
		totaltime = 0

		for testFile in os.listdir(self.testing_dir):
			if os.path.splitext(testFile)[1] in self.valid:
				files += 1
				fileName = self.testing_dir + "/" + testFile

				img = cv2.imread(fileName).astype(np.float32)
				self.helpers.logger.info("Loaded test image " + fileName)

				img = cv2.resize(img, (self.data.dim, self.data.dim))
				img = self.reshape(img)

				start = time.time()
				prediction = self.predict(img)
				end = time.time()
				benchmark = end - start
				totaltime += benchmark

				msg = ""
				if prediction == 1 and "_1." in testFile:
					tp += 1
					msg = "Acute Lymphoblastic Leukemia correctly detected (True Positive) in " + str(benchmark) + " seconds."
				elif prediction == 1 and "_0." in testFile:
					fp += 1
					msg = "Acute Lymphoblastic Leukemia incorrectly detected (False Positive) in " + str(benchmark) + " seconds."
				elif prediction == 0 and "_0." in testFile:
					tn += 1
					msg = "Acute Lymphoblastic Leukemia correctly not detected (True Negative) in " + str(benchmark) + " seconds."
				elif prediction == 0 and "_1." in testFile:
					fn += 1
					msg = "Acute Lymphoblastic Leukemia incorrectly not detected (False Negative) in " + str(benchmark) + " seconds."
				self.helpers.logger.info(msg)

		self.helpers.logger.info("Images Classified: " + str(files))
		self.helpers.logger.info("True Positives: " + str(tp))
		self.helpers.logger.info("False Positives: " + str(fp))
		self.helpers.logger.info("True Negatives: " + str(tn))
		self.helpers.logger.info("False Negatives: " + str(fn))
		self.helpers.logger.info("Total Time Taken: " + str(totaltime))

	def http_request(self, img_path):
		""" Sends image to the inference API endpoint. """

		self.helpers.logger.info("Sending request for: " + img_path)

		_, img_encoded = cv2.imencode('.jpg', cv2.imread(img_path))
		response = requests.post(self.addr, data=img_encoded.tostring(), headers=self.headers)
		response = json.loads(response.text)

		return response

	def test_http(self):
		"""Server test mode

		Loops through the test directory and sends the images to the
		classification server.
		"""

		totaltime = 0
		files = 0

		tp = 0
		fp = 0
		tn = 0
		fn = 0

		self.addr = "http://" + self.helpers.get_ip_addr() + \
			':'+str(self.helpers.confs["agent"]["port"]) + '/Inference'
		self.headers = {'content-type': 'image/jpeg'}

		for testFile in os.listdir(self.testing_dir):
			if os.path.splitext(testFile)[1] in self.valid:

				start = time.time()
				prediction = self.http_request(self.testing_dir + "/" + testFile)
				end = time.time()
				benchmark = end - start
				totaltime += benchmark

				msg = ""
				status = ""
				outcome = ""

				if prediction["Diagnosis"] == "Positive" and "_1." in testFile:
					tp += 1
					status = "correctly"
					outcome = "(True Positive)"
				elif prediction["Diagnosis"] == "Positive" and "_0." in testFile:
					fp += 1
					status = "incorrectly"
					outcome = "(False Positive)"
				elif prediction["Diagnosis"] == "Negative" and "_0." in testFile:
					tn += 1
					status = "correctly"
					outcome = "(True Negative)"
				elif prediction["Diagnosis"] == "Negative" and "_1." in testFile:
					fn += 1
					status = "incorrectly"
					outcome = "(False Negative)"

				files += 1
				self.helpers.logger.info("Acute Lymphoblastic Leukemia " + status +
								" detected " + outcome + " in " + str(benchmark) + " seconds.")

		self.helpers.logger.info("Images Classified: " + str(files))
		self.helpers.logger.info("True Positives: " + str(tp))
		self.helpers.logger.info("False Positives: " + str(fp))
		self.helpers.logger.info("True Negatives: " + str(tn))
		self.helpers.logger.info("False Negatives: " + str(fn))
		self.helpers.logger.info("Total Time Taken: " + str(totaltime))

	def load_tfrt(self):
		""" Loads the tfrt model """

		self.tfrt_model = tf.saved_model.load(self.tfrt_model_path)

	def predict_tfrt(self, img):
		""" Gets a prediction for an image. """

		inference = self.tfrt_model.signatures["serving_default"]
		prediction = inference(tf.constant(img, dtype=float))['softmax']
		prediction = self.data_labels[int(tf.argmax(prediction, axis=1))]

		return prediction

	def test_tfrt(self):
		"""TFRT test mode

		Loops through the test directory and classifies the images
		usin the TFRT model.
		"""

		files = 0
		tp = 0
		fp = 0
		tn = 0
		fn = 0
		totaltime = 0

		for testFile in os.listdir(self.testing_dir):
			if os.path.splitext(testFile)[1] in self.valid:
				files += 1
				fileName = self.testing_dir + "/" + testFile

				img = cv2.imread(fileName).astype(np.float32)
				self.helpers.logger.info("Loaded test image " + fileName)

				img = cv2.resize(img, (self.data.dim, self.data.dim))
				img = self.reshape(img)

				start = time.time()
				prediction = self.predict_tfrt(img)
				end = time.time()
				benchmark = end - start
				totaltime += benchmark

				msg = ""
				if prediction == 1 and "_1." in testFile:
					tp += 1
					msg = "Acute Lymphoblastic Leukemia correctly detected (True Positive) in " + str(benchmark) + " seconds."
				elif prediction == 1 and "_0." in testFile:
					fp += 1
					msg = "Acute Lymphoblastic Leukemia incorrectly detected (False Positive) in " + str(benchmark) + " seconds."
				elif prediction == 0 and "_0." in testFile:
					tn += 1
					msg = "Acute Lymphoblastic Leukemia correctly not detected (True Negative) in " + str(benchmark) + " seconds."
				elif prediction == 0 and "_1." in testFile:
					fn += 1
					msg = "Acute Lymphoblastic Leukemia incorrectly not detected (False Negative) in " + str(benchmark) + " seconds."
				self.helpers.logger.info(msg)

		self.helpers.logger.info("Images Classified: " + str(files))
		self.helpers.logger.info("True Positives: " + str(tp))
		self.helpers.logger.info("False Positives: " + str(fp))
		self.helpers.logger.info("True Negatives: " + str(tn))
		self.helpers.logger.info("False Negatives: " + str(fn))
		self.helpers.logger.info("Total Time Taken: " + str(totaltime))