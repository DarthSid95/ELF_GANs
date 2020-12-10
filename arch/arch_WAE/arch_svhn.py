from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
from matplotlib.backends.backend_pgf import PdfPages

from sklearn.manifold import TSNE


'''
REPLACE GENERATOR WITH COMMENTED OUT AUTOENCODER LATENT SPACE GENERATOR!!!!!,,,, FIX TRAIN_STEP ACCORDINGLY
'''

### MINIBATHC DISCRIMINATOR!?!?
"""def minibatch(input, num_kernels=5, kernel_dim=3):
x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
diffs = tf.expand_dims(activation, 3) - \
tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
return tf.concat([input, minibatch_features], 1)"""


class ARCH_svhn():

	def __init__(self):
		print("CREATING ARCH_AAE CLASS")
		return

	def encdec_model_svhn(self):

		def ama_relu(x):
			x = tf.clip_by_value(x, clip_value_min = -6., clip_value_max = 6.)
			return x

		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.001, seed=None)
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.output_size,self.output_size,3)) #64x64x3

		enc1 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(inputs) #32x32x64
		enc1 = tf.keras.layers.BatchNormalization()(enc1)
		# enc1 = tf.keras.layers.Dropout(0.1)(enc1)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)
		# if self.loss == 'FS':
		# 	enc1 = tf.keras.layers.Activation( activation = 'tanh')(enc1)


		enc2 = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc1) #16x16x128
		enc2 = tf.keras.layers.BatchNormalization()(enc2)
		# enc2 = tf.keras.layers.Dropout(0.1)(enc2)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)
		# if self.loss == 'FS':
		# 	enc2 = tf.keras.layers.Activation( activation = 'tanh')(enc2)




		enc3 = tf.keras.layers.Conv2D(512, 4, strides=1, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc2) #8x8x256
		enc3 = tf.keras.layers.BatchNormalization()(enc3)
		# enc3 = tf.keras.layers.Dropout(0.1)(enc3)
		enc3 = tf.keras.layers.LeakyReLU()(enc3)
		# if self.loss == 'FS':
		# 	enc3 = tf.keras.layers.Activation( activation = 'tanh')(enc3)


		enc4 = tf.keras.layers.Conv2D(1024, 4, strides=1, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc3) #4x4x128
		enc4 = tf.keras.layers.BatchNormalization()(enc4)
		# enc4 = tf.keras.layers.Dropout(0.5)(enc4)
		enc4 = tf.keras.layers.LeakyReLU()(enc4)
		# if self.loss == 'FS':
		# 	enc4 = tf.keras.layers.Activation( activation = 'tanh')(enc4)


		dense = tf.keras.layers.Flatten()(enc4)
		dense = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(dense)

		enc = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(dense)
		# enc =  tf.keras.layers.Activation( activation = 'sigmoid')(enc)
		# enc =  tf.keras.layers.Activation( activation = 'tanh')(enc)
		if self.loss in ['FS','KL']:
			enc  = tf.keras.layers.Lambda(ama_relu)(enc)
		# enc = tf.math.scalar_mul(10., enc)
		# enc = tf.keras.layers.ReLU(max_value = 5.)(enc)
		# enc =  tf.keras.layers.Activation( activation = 'sigmoid')(enc)

		encoded = tf.keras.Input(shape=(self.latent_dims,))

		den = tf.keras.layers.Dense(1024*int(self.output_size/4)*int(self.output_size/4), kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(encoded)
		enc_res = tf.keras.layers.Reshape([int(self.output_size/4),int(self.output_size/4),1024])(den)
		# enc_res = tf.keras.layers.Reshape([1,1,int(self.latent_dims)])(den) #1x1xlatent

		denc5 = tf.keras.layers.Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc_res) #2x2x128
		denc5 = tf.keras.layers.BatchNormalization()(denc5)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc5)
		denc5 = tf.keras.layers.LeakyReLU()(denc5)
		# if self.loss == 'FS':
		# 	denc5 = tf.keras.layers.Activation( activation = 'tanh')(denc5)

		denc4 = tf.keras.layers.Conv2DTranspose(256, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True, bias_initializer = bias_init_fn)(denc5) #4x4x128
		denc4 = tf.keras.layers.BatchNormalization()(denc4)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc4)
		denc1 = tf.keras.layers.LeakyReLU()(denc4)

		# denc3 = tf.keras.layers.Conv2DTranspose(128, 4, strides=1,padding='same',kernel_initializer=init_fn,use_bias=True)(denc4) #8x8x256
		# denc3 = tf.keras.layers.BatchNormalization()(denc3)
		# # denc3 = tf.keras.layers.Dropout(0.5)(denc3)
		# denc3 = tf.keras.layers.LeakyReLU()(denc3)


		out = tf.keras.layers.Conv2DTranspose(3, 4,strides=1,padding='same', kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(denc1) #64x64x3
		out =  tf.keras.layers.Activation( activation = 'tanh')(out)


		self.Decoder = tf.keras.Model(inputs=encoded, outputs=out)
		self.Encoder = tf.keras.Model(inputs=inputs, outputs=enc)

		return


	def encdec_model_svhn_dense(self):  # FOR BASE AE

		# init_fn = tf.keras.initializers.glorot_uniform()
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.5, seed=None)
		# init_fn = tf.function(init_fn, autograph=False)

		def ama_relu(x):
			x = tf.clip_by_value(x, clip_value_min = -6., clip_value_max = 6.)
			return x

		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None)
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)


		# tiny_init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.5, seed=None)
		# tiny_init_fn = tf.function(tiny_init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.output_size,self.output_size,3))

		inputs_res = tf.keras.layers.Reshape([int(self.output_size*self.output_size),])(inputs)

		enc0 = tf.keras.layers.Dense(128, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(inputs_res)
		enc0 = tf.keras.layers.BatchNormalization()(enc0)
		# enc0 = tf.keras.layers.Dropout(0.3)(enc0)
		# enc0 = tf.keras.layers.LeakyReLU()(enc0)
		# enc0 = tf.keras.layers.Activation( activation = 'tanh')(enc0)

		enc1 = tf.keras.layers.Dense(64, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn )(enc0)
		enc1 = tf.keras.layers.BatchNormalization()(enc1)
		# enc1 = tf.keras.layers.Dropout(0.3)(enc1)
		# enc1 = tf.keras.layers.LeakyReLU()(enc1)
		# enc1 = tf.keras.layers.Activation( activation = 'tanh')(enc1)

		enc2 = tf.keras.layers.Dense(32, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(enc1)
		enc2 = tf.keras.layers.BatchNormalization()(enc2)
		# enc2 = tf.keras.layers.Dropout(0.3)(enc2)
		# enc2 = tf.keras.layers.LeakyReLU()(enc2)
		# enc2 = tf.keras.layers.Activation( activation = 'tanh')(enc2)

		enc3 = tf.keras.layers.Dense(32, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(enc2)
		enc3 = tf.keras.layers.BatchNormalization()(enc3)
		# enc3 = tf.keras.layers.Dropout(0.3)(enc3)
		# enc3 = tf.keras.layers.LeakyReLU()(enc3)
		# enc3 = tf.keras.layers.Activation( activation = 'tanh')(enc3)

		enc4 = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc3)
		# enc4 = tf.keras.activations.relu(enc4, threshold = -2., max_value = 2.)
		# enc4 = tf.math.scalar_mul(10., enc4)
		# enc4 = tf.keras.layers.Activation( activation = 'sigmoid')(enc4)
		# enc4 = tf.keras.layers.Activation( activation = 'tanh')(enc4)
		# enc4 = tf.math.scalar_mul(2., enc4)
		# enc4 = tf.keras.layers.ReLU(threshold = -2., max_value = 2.)(enc4)

		enc = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias=False)(enc4)
		if self.loss != 'base':
			enc  = tf.keras.layers.Lambda(ama_relu)(enc)

		# enc4 = tf.keras.layers.ReLU( threshold = -2.)(enc4)
		# enc4 = tf.math.scalar_mul(-1., enc4)
		# enc4 = tf.keras.layers.ReLU( threshold = -2.)(enc4)
		# enc4 = tf.math.scalar_mul(-1., enc4)

		# enc_out = tf.keras.layers.Add()([enc4p,enc4n])
		
		# enc4 = tf.keras.layers.BatchNormalization()(enc4)
		# enc4 = tf.keras.layers.ReLU(max_value = 5.)(enc4)
		### 11022020 - 05 onwards, tanh. sigmoid before that.

		encoded = tf.keras.Input(shape=(self.latent_dims,))

		dec0 = tf.keras.layers.Dense(32, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(encoded)
		dec0 = tf.keras.layers.BatchNormalization()(dec0)
		# dec0 = tf.keras.layers.Dropout(0.3)(dec0)
		# dec0 = tf.keras.layers.LeakyReLU()(dec0)
		# dec0 = tf.keras.layers.Activation( activation = 'tanh')(dec0)

		dec1 = tf.keras.layers.Dense(32, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(dec0)
		dec1 = tf.keras.layers.BatchNormalization()(dec1)
		# dec1 = tf.keras.layers.Dropout(0.3)(dec1)
		# dec1 = tf.keras.layers.LeakyReLU()(dec1)
		# dec1 = tf.keras.layers.Activation( activation = 'tanh')(dec1)

		dec2 = tf.keras.layers.Dense(64, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(dec1)
		dec2 = tf.keras.layers.BatchNormalization()(dec2)
		# dec2 = tf.keras.layers.Dropout(0.3)(dec2)
		# dec2 = tf.keras.layers.LeakyReLU()(dec2)
		# dec2 = tf.keras.layers.Activation( activation = 'tanh')(dec2)

		dec3 = tf.keras.layers.Dense(128, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(dec2)
		dec3 = tf.keras.layers.BatchNormalization()(dec3)
		# dec3 = tf.keras.layers.Dropout(0.3)(dec3)
		# dec3 = tf.keras.layers.LeakyReLU()(dec3)
		# dec3 = tf.keras.layers.Activation( activation = 'tanh')(dec3)

		out_enc = tf.keras.layers.Dense(int(self.output_size*self.output_size*3), kernel_initializer=init_fn)(dec3)
		out_enc = tf.keras.layers.Activation( activation = 'tanh')(out_enc)


		out = tf.keras.layers.Reshape([int(self.output_size),int(self.output_size),3])(out_enc)
		# out = tf.keras.layers.ReLU(max_value = 1.)(out)

		self.Encoder = tf.keras.Model(inputs = inputs, outputs = enc)
		self.Decoder = tf.keras.Model(inputs = encoded, outputs = out)
		
		return self.Encoder



	def discriminator_model_svhn(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		model = tf.keras.Sequential()
		model.add(layers.Dense(256, use_bias=False, input_shape=(self.latent_dims,), kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		# model.add(layers.Activation(activation = 'tanh'))
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(512, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(128, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1))
		# model.add(layers.Softmax())
		return model
		# model = tf.keras.Sequential()

		# model.add(layers.Dense(256, use_bias=False, input_shape=(self.latent_dims,), kernel_initializer=init_fn))
		# # model.add(layers.BatchNormalization())
		# model.add(layers.Activation(activation = 'tanh'))
		# model.add(layers.LeakyReLU())

		# model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn, input_shape=[int(self.output_size), int(self.output_size), 1]))
		# model.add(layers.LeakyReLU())
		# model.add(layers.BatchNormalization())
		# # model.add(layers.Dropout(0.3))

		# model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn, input_shape=[int(self.output_size), int(self.output_size), 1]))
		# model.add(layers.LeakyReLU())
		# model.add(layers.BatchNormalization())

		# # model.add(layers.Conv2D(32, (5, 5), strides=(1, 1), padding='same', kernel_initializer=init_fn, input_shape=[int(self.output_size), int(self.output_size), 1]))
		# # model.add(layers.LeakyReLU())

		# model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', kernel_initializer=init_fn))
		# model.add(layers.LeakyReLU())
		# model.add(layers.BatchNormalization())
		# # model.add(layers.Dropout(0.3))

		# model.add(layers.Flatten())
		
		# model.add(layers.Dense(50))
		# model.add(layers.LeakyReLU())

		# model.add(layers.Dense(1))

		# return model


	# def show_result_svhn(self, images=None, num_epoch=0, show = False, save = False, path = 'result.png'):

		# print("Gaussian Stats : True mean {} True Sigma {} \n Fake mean {} Fake Sigma {}".format(np.mean(self.fakes_enc,axis = 0), np.cov(self.fakes_enc,rowvar = False), np.mean(self.reals_enc, axis = 0), np.cov(self.reals_enc,rowvar = False) ))
		# if self.res_flag:
		# 	self.res_file.write("Gaussian Stats : True mean {} True Sigma {} \n Fake mean {} Fake Sigma {}".format(np.mean(self.fakes_enc,axis = 0), np.cov(self.fakes_enc,rowvar = False), np.mean(self.reals_enc, axis = 0), np.cov(self.reals_enc,rowvar = False) ))

		# size_figure_grid = 5
		# images = tf.reshape(images, [images.shape[0],self.output_size,self.output_size,1])
		# fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
		# for i in range(size_figure_grid):
		# 	for j in range(size_figure_grid):
		# 		ax[i, j].get_xaxis().set_visible(False)
		# 		ax[i, j].get_yaxis().set_visible(False)

		# for k in range(size_figure_grid*size_figure_grid):
		# 	i = k // size_figure_grid
		# 	j = k % size_figure_grid
		# 	ax[i, j].cla()
		# 	im = images[k,:,:,0]
		# 	ax[i, j].imshow(im, cmap='gray')

		# label = 'Epoch {0}'.format(num_epoch)
		# fig.text(0.5, 0.04, label, ha='center')
		# if save:
		# 	plt.savefig(path)
		# if show:
		# 	plt.show()
		# else:
		# 	plt.close()

		# fig1, ax1 = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
		# for i in range(size_figure_grid):
		# 	for j in range(size_figure_grid):
		# 		ax1[i, j].get_xaxis().set_visible(False)
		# 		ax1[i, j].get_yaxis().set_visible(False)

		# for k in range(size_figure_grid*size_figure_grid):
		# 	i = k // size_figure_grid
		# 	j = k % size_figure_grid
		# 	ax1[i, j].cla()
		# 	gt = self.reals[k,:,:,0]
		# 	ax1[i, j].imshow(gt, cmap='gray')

		# label = 'Epoch {0}'.format(num_epoch)
		# fig1.text(0.5, 0.04, label, ha='center')
		# if save:
		# 	plt.savefig(path.split('.')[0]+'gt.png')

		# if show:
		# 	plt.show()
		# else:
		# 	plt.close()

	def show_result_svhn(self, images=None, num_epoch=0, show = False, save = False, path = 'result.png'):

		# if num_epoch%2 == 0 and num_epoch>self.AE_count:
		print("Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.fakes_enc,axis = 0), np.cov(self.fakes_enc,rowvar = False), np.mean(self.reals_enc, axis = 0), np.cov(self.reals_enc,rowvar = False) ))
		if self.res_flag:# and num_epoch>self.AE_count:
			self.res_file.write("Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.reals_enc, axis = 0), np.cov(self.reals_enc,rowvar = False), np.mean(self.fakes_enc, axis = 0), np.cov(self.fakes_enc,rowvar = False) ))
		size_figure_grid = 5

		images = tf.reshape(images, [images.shape[0],self.output_size,self.output_size,3])
		images_on_grid = self.image_grid(input_tensor = images, grid_shape = (self.num_to_print,self.num_to_print),image_shape=(self.output_size,self.output_size),num_channels=3)
		fig = plt.figure(figsize=(7,7))
		ax1 = fig.add_subplot(111)
		ax1.cla()
		ax1.axis("off")
		ax1.imshow(np.clip(images_on_grid,0.,1.))

		label = 'Epoch {0}'.format(num_epoch)
		plt.title(label, fontsize=8)
		if save:
			plt.tight_layout()
			plt.savefig(path)
		if show:
			plt.show()
		else:
			plt.close()

		reals_to_display = (self.reals[0:self.num_to_print*self.num_to_print] + 1.0)/2.0
		images_on_grid = self.image_grid(input_tensor = reals_to_display, grid_shape = (self.num_to_print,self.num_to_print),image_shape=(self.output_size,self.output_size),num_channels=3)
		fig1 = plt.figure(figsize=(7,7))
		ax1 = fig1.add_subplot(111)
		ax1.cla()
		ax1.axis("off")
		ax1.imshow(np.clip(images_on_grid,0.,1.))

		label = 'Epoch {0}'.format(num_epoch)
		plt.title(label, fontsize=8)
		if save:
			plt.tight_layout()
			plt.savefig(path.split('.')[0]+'gt.png')

		if show:
			plt.show()
		else:
			plt.close()


		if self.latent_dims == 2:

			print("Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.fakes_enc,axis = 0), np.cov(self.fakes_enc,rowvar = False), np.mean(self.reals_enc, axis = 0), np.cov(self.reals_enc,rowvar = False) ))
			if self.res_flag:
				self.res_file.write("Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.reals_enc, axis = 0), np.cov(self.reals_enc,rowvar = False), np.mean(self.fakes_enc, axis = 0), np.cov(self.fakes_enc,rowvar = False) ))

			if self.colab == 1:
				from matplotlib.backends.backend_pdf import PdfPages
				plt.rc('text', usetex=False)
			else:
				from matplotlib.backends.backend_pgf import FigureCanvasPgf
				matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
				from matplotlib.backends.backend_pgf import PdfPages
				plt.rcParams.update({
					"pgf.texsystem": "pdflatex",
					"font.family": "serif",  # use serif/main font for text elements
					"font.size":10,	
					"font.serif": [], 
					"text.usetex": True,     # use inline math for ticks
					"pgf.rcfonts": False,    # don't setup fonts from rc parameters
				})

			with PdfPages(path+'_distribution.pdf') as pdf:

				fig1 = plt.figure(figsize=(3.5, 3.5))
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.get_xaxis().set_visible(False)
				ax1.get_yaxis().set_visible(False)
				ax1.set_xlim([-3,3])
				ax1.set_ylim([-3,3])
				if num_epoch>self.AE_count:
					ax1.scatter(self.fakes_enc[:,0], self.fakes_enc[:,1], c='r', linewidth = 1.5, label='Target Class Data', marker = '.')
				ax1.scatter(self.reals_enc[:,0], self.reals_enc[:,1], c='b', linewidth = 1.5, label='Source Class Data', marker = '.')
				# ax1.scatter(images[:,0], images[:,1], c='g', linewidth = 1.5, label='Fake Data', marker = '.')
				ax1.legend(loc = 'upper right')
				fig1.tight_layout()
				pdf.savefig(fig1)
				plt.close(fig1)


	def SVHN_Classifier(self):
		self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(299,299,3), classes=1000)

	def FID_svhn(self):

		def data_preprocess(image):
			with tf.device('/CPU'):
				image = tf.image.resize(image,[299,299])
				# This will convert to float values in [0, 1]

				# image = tf.divide(image,255.0)
				# image = tf.image.grayscale_to_rgb(image)
				# image = tf.subtract(image,0.5)
				# image = tf.scalar_mul(2.0,image)
				# image = tf.image.convert_image_dtype(image, tf.float16)
			return image


		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1	

			random_points = tf.keras.backend.random_uniform([self.FID_num_samples], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			print(random_points)
			self.fid_train_images = self.fid_train_images[random_points]
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images)
			if self.FID_kind != 'latent':
				self.fid_image_dataset = self.fid_image_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset = self.fid_image_dataset.batch(self.fid_batch_size)

			# self.fid_train_images = tf.image.resize(self.fid_train_images, [80,80])
			# self.fid_train_images = tf.image.grayscale_to_rgb(self.fid_train_images)
			# self.fid_images = self.fid_train_images.numpy()

			if self.FID_kind != 'latent':
				self.SVHN_Classifier()


		if self.mode == 'fid':
			print(self.checkpoint_dir)
			# print('logs/130919_ELeGANt_svhn_lsgan_base_01/130919_ELeGANt_svhn_lsgan_base_Results_checkpoints')
			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			print('Models Loaded Successfully')

		with tf.device(self.device):
			for image_batch in self.fid_image_dataset:

				if self.FID_kind == 'latent':
					## Measure FID on the latent Gaussians 
					preds = self.Encoder(image_batch)
					act1 = self.get_noise(tf.constant(100))
					act2 = preds
				else:
					# print(self.fid_train_images.shape)
					preds = self.Decoder(self.get_noise(tf.constant(100)), training=False)
					# preds = preds[:,:,:].numpy()		
					preds = tf.image.resize(preds, [299,299])
					# preds = tf.image.grayscale_to_rgb(preds)
					# preds = tf.subtract(preds,0.50)
					# preds = tf.scalar_mul(2.0,preds)
					preds = preds.numpy()

					act1 = self.FID_model.predict(image_batch)
					act2 = self.FID_model.predict(preds)
					
				try:
					self.act1 = np.concatenate([self.act1,act1], axis = 0)
					self.act2 = np.concatenate([self.act2,act2], axis = 0)
				except:
					self.act1 = act1
					self.act2 = act2
			# print(self.act1.shape, self.act2.shape)
			self.eval_FID()
			return


		# with tf.device('/CPU'):
		# 	# print(self.fid_train_images.shape)
		# 	preds = self.Decoder(self.get_noise(self.fid_batch_size), training=False)
		# 	# preds = preds[:,:,:].numpy()		
		# 	preds = tf.image.resize(preds, [80,80])
		# 	preds = tf.image.grayscale_to_rgb(preds)
		# 	preds = preds.numpy()

		# 	self.act1 = self.FID_model.predict(self.fid_images)
		# 	self.act2 = self.FID_model.predict(preds)
		# 	self.eval_FID()
		# 	return


	# def FID_svhn(self):
	# 	self.MNIST_Classifier()
	# 	if self.mode == 'fid':
	# 		print(self.checkpoint_dir)
	# 		# print('logs/130919_ELeGANt_svhn_lsgan_base_01/130919_ELeGANt_svhn_lsgan_base_Results_checkpoints')
	# 		self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
	# 		print('Models Loaded Successfully')
	# 	else:
	# 		print('Evaluating FID Score ')

	# 	with tf.device('/CPU'):
	# 		(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.svhn.load_data()
	# 		train_images = train_images.reshape(train_images.shape[0], self.output_size,self.output_size,1).astype('float64')
	# 		train_images = (train_images - 0.) / 255.0

	# 		if self.testcase == 'single':
	# 			t_images = train_images[np.where(train_labels == self.number)[0][0:5000]]
	# 			train_images = t_images
	# 		if self.testcase == 'even':
	# 			train_images = train_images[np.where(train_labels%2 == 0)[0]]
	# 		if self.testcase == 'odd':
	# 			train_images = train_images[np.where(train_labels%2 != 0)[0]]

	# 			# for i in range(self.reps-1):
	# 			# 	train_images = np.concatenate([train_images, t_images])

	# 		noise = tf.random.normal([train_images.shape[0], self.noise_dims])
	# 		preds = self.generator(noise, training = False)
	# 		preds = self.Decoder(preds, training = False)
	# 		print(preds)
	# 		preds = preds[:,:,:,0].numpy()
	# 		# calculate latent representations
	# 		self.act1 = self.FID_model.predict(train_images)
	# 		self.act2 = self.FID_model.predict(np.expand_dims(preds,axis=3))

	# 		# prd_data_1 = prd.compute_prd_from_embedding(self.act2, self.act1)


	# 		# self.tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
	# 		# plot_only = 500
	# 		# low_dim_embs = tsne.fit_transform(self.act1)
	# 		# for i in range(10):
	# 		# 		x, y = low_dim_embs[i*6000:((i+1)*6000), :]
	# 		# 		print(x,x.shape)
	# 		# 		print(y,y.shape)
	# 		# 		plt.scatter(x, y)
	# 		# 		# plt.annotate(label,xy=(x, y),xytext=(5, 2),textcoords='offset points',ha='right',va='bottom')
	# 		# plt.savefig(self.tsne_path)


	# 		#### COMMENTED FOR GPU0 ISSUE 
	# 		self.eval_FID()
	# 		if self.testcase != 'single':
	# 			print("FID SCORE: {:>2.4f}\n".format(self.fid))
	# 			if self.res_flag:
	# 				self.res_file.write("FID SCORE: {:>2.4f}\n".format(self.fid))
	# 		else:
	# 			print("FID SCORE Full: {:>2.4f}\n".format(self.fid))
	# 			if self.res_flag:
	# 				self.res_file.write("FID SCORE Full: {:>2.4f}\n".format(self.fid))

	# 			self.reps = 500
	# 			t_images = train_images[np.where(train_labels == self.number)[0][0:10]]
	# 			train_images = t_images
	# 			for i in range(self.reps-1):
	# 				train_images = np.concatenate([train_images, t_images])
	# 			self.act1 = self.FID_model.predict(train_images)
	# 			self.eval_FID()
	# 			print("FID SCORE Small: {:>2.4f}\n".format(self.fid))
	# 			if self.res_flag:
	# 				self.res_file.write("FID SCORE Small: {:>2.4f}\n".format(self.fid))

	# 		# prd_data_2 = prd.compute_prd_from_embedding(self.act2, self.act1)
	# 		# prd.plot([prd_data_1, prd_data_2], ['GAN_1', 'GAN_2'], out_path = self.tsne_path)

	# def MNIST_Classifier(self):

	# 	if os.path.exists('MNIST_FID.h5'):
	# 		print('Existing MNIST encoder model is being loaded')
	# 		# self.FID_model = tf.keras.models.load_model('MNIST_FID.h5')
	# 		with tf.device(self.device):
	# 			if not self.FID_load_flag:
	# 				self.FID_model = tf.keras.models.load_model('MNIST_FID.h5')
	# 				self.FID_load_flag = 1
	# 		# self.FID_model.layers.pop()
	# 		# self.FID_model.layers.pop()

	# 		# model = tf.keras.Sequential()
	# 		# model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
	# 		#                  activation='relu',
	# 		#                  input_shape=(28,28,1)))
	# 		# model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
	# 		# model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
	# 		# model.add(tf.keras.layers.Dropout(0.25))
	# 		# model.add(tf.keras.layers.Flatten())
	# 		# model.add(tf.keras.layers.Dense(128, activation='relu'))

	# 		# model.set_weights(self.FID_model.get_weights()[0:6])

	# 		# model.summary()
	# 		# model.save('MNIST_FID.h5')
	# 		# self.FID_model.build()
	# 		print(self.FID_model.summary())
	# 		return

	# 	(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.svhn.load_data()
	# 	train_images = train_images.reshape(train_images.shape[0], self.output_size,self.output_size,1).astype('float64')
	# 	train_labels = train_labels.reshape(train_images.shape[0], 1).astype('float64')
	# 	train_images = (train_images  -127.5) / 255.0

	# 	train_cat = tf.keras.utils.to_categorical(train_labels,num_classes=10,dtype='float64')


	# 	init_fn = tf.keras.initializers.glorot_uniform()
	# 	init_fn = tf.function(init_fn, autograph=False)


	# 	''' MNIST Classifier for FID'''
	# 	inputs = tf.keras.Input(shape=(self.output_size,self.output_size,1))

	# 	x1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu')(inputs)
	# 	x2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x1)
	# 	x3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x2)
	# 	x4 = tf.keras.layers.Flatten()(x3)
	# 	x5 = tf.keras.layers.Dense(128, activation='relu')(x4)
	# 	x6 = tf.keras.layers.Dense(100, activation='relu')(x5)
	# 	x7 = tf.keras.layers.Dense(50, activation='relu')(x6)
	# 	Cla = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x7)



	# 	FID = tf.keras.Model(inputs=inputs, outputs=x7)
	# 	Classifier = tf.keras.Model(inputs=inputs, outputs=Cla)

	# 	''' Autoencoder Architecture for FID'''
	# 	# inputs = tf.keras.Input(shape=(self.output_size,self.output_size,))
	# 	# reshape = tf.keras.layers.Reshape(target_shape=(self.output_size*self.output_size,))(inputs)
	# 	# x1 = tf.keras.layers.Dense(50, activation=tf.nn.relu)(reshape)
	# 	# # x2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(x1)
	# 	# Enc = tf.keras.layers.Dense(8, activation=tf.nn.relu)(x1)
	# 	# Encoder = tf.keras.Model(inputs=inputs, outputs=Enc)
	# 	# # y1 = tf.keras.layers.Dense(8, activation=tf.nn.relu)(Enc)
	# 	# # y2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)(y1)
	# 	# y1 = tf.keras.layers.Dense(50, activation=tf.nn.relu)(Enc)
	# 	# Decoder = tf.keras.layers.Dense(784, activation=tf.nn.relu)(y1)
	# 	# Decoder = tf.keras.layers.Reshape(target_shape=(self.output_size,self.output_size,))(Decoder)
	# 	# Autoencoder = tf.keras.Model(inputs=inputs, outputs=Decoder)

	# 	print('FID training model made')
	# 	print(FID.summary(),Classifier.summary())

	# 	Classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	# 	Classifier.fit(train_images, train_cat,epochs=10,batch_size=100,shuffle=True,)
	# 	Classifier.save("MNIST_FID_FULL.h5")
	# 	FID.save("MNIST_FID.h5")

	# 	self.MNIST_Classifier()
	# 	return

