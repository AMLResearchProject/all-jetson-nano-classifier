""" TensorRT engine

Provides the TensorRT engine functionality.

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

Reference: https://github.com/jkjung-avt/keras_imagenet/
"""

import cv2
import numpy as np
import os.path
import tensorrt as trt
import time

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = [1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)]


class engine():
	""" TensorRT engine """

	def __init__(self, helpers):
		""" Initializes the TensorRT engine class. """

		self.helpers = helpers
		self.confs = helpers.confs
		self.onnx_model_path = self.confs["model"]["onnx"]
		self.tensorrt_model_path = self.confs["model"]["tensorrt"]
		self.testing_dir = self.confs["data"]["test"]
		self.valid = self.confs["data"]["valid_types"]
		self.labels = self.confs["data"]["labels"]

		if not os.path.isfile(self.tensorrt_model_path):
			self.save_engine(self.build_engine())
			self.helpers.logger.info("TensorRT model generated.")

		self.helpers.logger.info("Engine class initialization complete.")

	def build_engine(self):
		""" Builds the TensorRT engine. """

		with trt.Builder(TRT_LOGGER) as builder, builder.create_network(*EXPLICIT_BATCH) \
				as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
			builder.max_workspace_size = 1 << 30
			builder.max_batch_size = 1
			builder.fp16_mode = False
			with open(self.onnx_model_path, 'rb') as model:
				if not parser.parse(model.read()):
					self.helpers.logger.info("ERROR: Failed to parse the ONNX file.")
					for error in range(parser.num_errors):
						self.helpers.logger.info(parser.get_error(error))
					return None
			shape = list(network.get_input(0).shape)
			shape[0] = 1
			network.get_input(0).shape = shape
			return builder.build_cuda_engine(network)

		self.helpers.logger.info("Engine build complete.")

	def save_engine(self, engine):
		""" Saves the TensorRT engine. """

		with open(self.tensorrt_model_path, 'wb') as f:
			f.write(engine.serialize())

		self.helpers.logger.info("Engine save complete.")

	def load_engine(self):
		""" Loads the TensorRT engine. """

		with open(self.tensorrt_model_path, 'rb') as f:
			engine_data = f.read()
		self.engine = trt.Runtime(TRT_LOGGER).deserialize_cuda_engine(engine_data)

		self.helpers.logger.info("Engine load complete.")

	def init_trt_buffers(self, cuda):
		""" Initialize host buffers and cuda buffers for the engine."""

		size = trt.volume((1, 100, 100, 3)) * self.engine.max_batch_size
		host_input = cuda.pagelocked_empty(size, np.float32)
		cuda_input = cuda.mem_alloc(host_input.nbytes)
		size = trt.volume((1, 2)) * self.engine.max_batch_size
		host_output = cuda.pagelocked_empty(size, np.float32)
		cuda_output = cuda.mem_alloc(host_output.nbytes)
		return host_input, cuda_input, host_output, cuda_output

		self.helpers.logger.info("Engine buffers initialized.")

	def predict(self, img):
		""" Inference the image with TensorRT engine."""

		import pycuda.autoinit
		import pycuda.driver as cuda

		with open(self.tensorrt_model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
			engine = runtime.deserialize_cuda_engine(f.read())

		host_input, cuda_input, host_output, cuda_output = self.init_trt_buffers(
			cuda)
		stream = cuda.Stream()

		context = self.engine.create_execution_context()
		context.set_binding_shape(0, (1, 100, 100, 3))

		np.copyto(host_input, img.ravel())
		cuda.memcpy_htod_async(cuda_input, host_input, stream)

		context.execute_async_v2(bindings=[int(cuda_input), int(cuda_output)],
									stream_handle=stream.handle)

		cuda.memcpy_dtoh_async(host_output, cuda_output, stream)
		stream.synchronize()

		return host_output

	def reshape(self, img):
		""" Reshapes an image. """

		dx, dy, dz = img.shape
		input_data = img.reshape((-1, dx, dy, dz))
		input_data = input_data / 255.0

		return input_data

	def test(self):
		"""TensorRT test mode

		Loops through the test directory and classifies the images
		using the TensorRT model.
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

				img = cv2.resize(img, (100,100))
				img = self.reshape(img)

				start = time.time()
				predictions = self.predict(img)
				predictions = predictions.argsort()[::-1]
				prediction = self.labels[predictions[0]]
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