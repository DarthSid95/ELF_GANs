from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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


class ARCH_cifar10():

	def __init__(self):
		print("CREATING ARCH_AAE CLASS")
		return



	def encdec_model_cifar10(self):

		def ama_relu(x):
			x = tf.clip_by_value(x, clip_value_min = -8., clip_value_max = 8.)
			return x

		# if self.loss == 'FS':
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.001, seed=None)
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)
		# else:
		# 	init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.001, seed=None)
		# 	init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.output_size,self.output_size,3)) #64x64x3

		enc1 = tf.keras.layers.Conv2D(128, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(inputs) #32x32x64
		enc1 = tf.keras.layers.BatchNormalization()(enc1)
		# enc1 = tf.keras.layers.Dropout(0.1)(enc1)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)
		# enc1 = tf.keras.layers.Activation( activation = 'tanh')(enc1)

		enc2 = tf.keras.layers.Conv2D(256, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc1) #16x16x128
		enc2 = tf.keras.layers.BatchNormalization()(enc2)
		# enc2 = tf.keras.layers.Dropout(0.1)(enc2)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)
		# enc2 = tf.keras.layers.Activation( activation = 'tanh')(enc2)

		enc3 = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc2) #8x8x256
		enc3 = tf.keras.layers.BatchNormalization()(enc3)
		# enc3 = tf.keras.layers.Dropout(0.1)(enc3)
		enc3 = tf.keras.layers.LeakyReLU()(enc3)
		# enc3 = tf.keras.layers.Activation( activation = 'tanh')(enc3)

		enc4 = tf.keras.layers.Conv2D(1024, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True, bias_initializer = bias_init_fn)(enc3) #4x4x128
		enc4 = tf.keras.layers.BatchNormalization()(enc4)
		# enc4 = tf.keras.layers.Dropout(0.5)(enc4)
		enc4 = tf.keras.layers.LeakyReLU()(enc4)
		# enc4 = tf.keras.layers.Activation( activation = 'tanh')(enc4)

		# enc5 = tf.keras.layers.Conv2D(512, 4, strides=2, padding='same',kernel_initializer=init_fn, use_bias=True)(enc4) #2x2x128
		# enc5 = tf.keras.layers.BatchNormalization()(enc5)
		# # enc5 = tf.keras.layers.Dropout(0.5)(enc5)
		# enc5 = tf.keras.layers.LeakyReLU()(enc5)
		# enc5 = tf.keras.layers.Activation( activation = 'tanh')(enc5)

		# enc6 = tf.keras.layers.Conv2D(1024, 4, strides=1, padding='same',kernel_initializer=init_fn, use_bias=True)(enc5) #1x1xlatent
		# enc6 = tf.keras.layers.BatchNormalization()(enc6)
		# enc6 = tf.keras.layers.LeakyReLU()(enc6)
		# # enc6 = tf.keras.layers.Activation( activation = 'tanh')(enc6)


		dense = tf.keras.layers.Flatten()(enc4)

		dense = tf.keras.layers.Dense(self.latent_dims, use_bias = True, kernel_initializer=init_fn, bias_initializer = bias_init_fn)(dense)

		enc = tf.keras.layers.Dense(self.latent_dims, use_bias = True, kernel_initializer=init_fn, bias_initializer = bias_init_fn)(dense)

		if self.loss == 'FS':
			enc  = tf.keras.layers.Lambda(ama_relu)(enc)
		# enc =  tf.keras.layers.Activation(activation = 'sigmoid')(enc)

		# if self.noise_kind == 'gaussian_1m1':
		# 	enc =  tf.keras.layers.Activation(activation = 'tanh')(enc)
		# elif self.noise_kind == 'gaussian_02':
		# 	enc = tf.keras.layers.ReLU(max_value = 2.)(enc)
		# 	# enc =  tf.keras.layers.Activation(activation = 'sigmoid')(enc)
		# 	# enc = tf.math.scalar_mul(5., enc)
		# elif self.noise_kind == 'gaussian_01':
		# 	enc =  tf.keras.layers.Activation(activation = 'sigmoid')(enc)
		# elif self.noise_kind == 'gaussian_s2':
		# 	enc =  tf.keras.layers.Activation(activation = 'tanh')(enc)
		# 	enc = tf.math.scalar_mul(6., enc)
		# elif self.noise_kind == 'gaussian':
		# 	enc  = tf.keras.layers.Lambda(ama_relu)(enc)


		# enc = tf.math.scalar_mul(10., enc)
		# enc = tf.keras.layers.ReLU(max_value = 1., threshold=-1.)(enc)
		# enc =  tf.keras.layers.Activation( activation = 'sigmoid')(enc)

		encoded = tf.keras.Input(shape=(self.latent_dims,))

		den = tf.keras.layers.Dense(1024*int(self.output_size/8)*int(self.output_size/8),use_bias=True, bias_initializer = bias_init_fn)(encoded)
		enc_res = tf.keras.layers.Reshape([int(self.output_size/8),int(self.output_size/8),1024])(den)
		# enc_res = tf.keras.layers.Reshape([1,1,int(self.latent_dims)])(den) #1x1xlatent

		denc5 = tf.keras.layers.Conv2DTranspose(512, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True, bias_initializer = bias_init_fn)(enc_res) #2x2x128
		denc5 = tf.keras.layers.BatchNormalization()(denc5)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc5)
		denc5 = tf.keras.layers.LeakyReLU()(denc5)
		# denc5 = tf.keras.layers.Activation( activation = 'tanh')(denc5)

		denc4 = tf.keras.layers.Conv2DTranspose(256, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True, bias_initializer = bias_init_fn)(denc5) #4x4x128
		denc4 = tf.keras.layers.BatchNormalization()(denc4)
		# denc4 = tf.keras.layers.Dropout(0.5)(denc4)
		denc4 = tf.keras.layers.LeakyReLU()(denc4)
		# denc4 = tf.keras.layers.Activation( activation = 'tanh')(denc4)

		denc3 = tf.keras.layers.Conv2DTranspose(128, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True, bias_initializer = bias_init_fn)(denc4) #8x8x256
		denc3 = tf.keras.layers.BatchNormalization()(denc3)
		# denc3 = tf.keras.layers.Dropout(0.5)(denc3)
		denc1 = tf.keras.layers.LeakyReLU()(denc3)
		# denc1 = tf.keras.layers.Activation( activation = 'tanh')(denc3)

		# denc2 = tf.keras.layers.Conv2DTranspose(128, 4, strides=2,padding='same',kernel_initializer=init_fn,use_bias=True)(denc3) #16x16x128
		# denc2 = tf.keras.layers.BatchNormalization()(denc2)
		# # denc2 = tf.keras.layers.Dropout(0.5)(denc2)
		# denc2 = tf.keras.layers.LeakyReLU()(denc2)
		# # denc2 = tf.keras.layers.Activation( activation = 'tanh')(denc2)

		# denc1 = tf.keras.layers.Conv2DTranspose(64, 4, strides=1,padding='same',kernel_initializer=init_fn,use_bias=True)(denc2) #32x32x64
		# denc1 = tf.keras.layers.BatchNormalization()(denc1)
		# # denc1 = tf.keras.layers.Dropout(0.5)(denc1)
		# denc1 = tf.keras.layers.LeakyReLU()(denc1)
		# # denc1 = tf.keras.layers.Activation( activation = 'tanh')(denc1)

		out = tf.keras.layers.Conv2DTranspose(3, 4,strides=1,padding='same', kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(denc1) #64x64x3
		out =  tf.keras.layers.Activation(activation = 'tanh')(out)

	
		self.Decoder = tf.keras.Model(inputs=encoded, outputs=out)
		self.Encoder = tf.keras.Model(inputs=inputs, outputs=enc)

		return


	def discriminator_model_cifar10(self):
		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(256, use_bias=False, input_shape=(self.latent_dims,), kernel_initializer=init_fn))
		model.add(layers.BatchNormalization())
		# model.add(layers.Activation(activation = 'tanh'))
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(512, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1025, use_bias=False, kernel_initializer=init_fn))
		# model.add(layers.BatchNormalization())
		model.add(layers.LeakyReLU())

		model.add(layers.Dense(1))
		# model.add(layers.Softmax())
		return model



	def show_result_cifar10(self, images=None, num_epoch=0, show = False, save = False, path = 'result.png'):
		
		from sklearn.covariance import EmpiricalCovariance

		# if num_epoch%1 == 0 and num_epoch>self.AE_count:
		# print("Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.fakes_enc,axis = 0), np.cov(self.fakes_enc,rowvar = False), np.mean(self.reals_enc, axis = 0), np.cov(self.reals_enc,rowvar = False) ))
		if self.res_flag:# and num_epoch>self.AE_count:
			self.res_file.write("Gaussian Stats : True mean {} True Cov {} \n Fake mean {} Fake Cov {}".format(np.mean(self.reals_enc, axis = 0), np.cov(self.reals_enc,rowvar = False), np.mean(self.fakes_enc, axis = 0), np.cov(self.fakes_enc,rowvar = False) ))
		size_figure_grid = 5
		# print(EmpiricalCovariance().fit(self.fakes_enc).covariance_)
		# exit(0)
		images = tf.reshape(images, [images.shape[0],self.output_size,self.output_size,3])
		images_on_grid = self.image_grid(input_tensor = images, grid_shape = (self.num_to_print,self.num_to_print),image_shape=(self.output_size,self.output_size),num_channels=3)
		fig = plt.figure(figsize=(7,7))
		ax1 = fig.add_subplot(111)
		ax1.cla()
		ax1.axis("off")
		ax1.imshow(np.clip(images_on_grid,0.,1.))

		# fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
		# for i in range(size_figure_grid):
		# 	for j in range(size_figure_grid):
		# 		ax[i, j].get_xaxis().set_visible(False)
		# 		ax[i, j].get_yaxis().set_visible(False)

		# for k in range(size_figure_grid*size_figure_grid):
		# 	i = k // size_figure_grid
		# 	j = k % size_figure_grid
		# 	ax[i, j].cla()
		# 	im = images[k,:,:,:]
		# 	ax[i, j].imshow(im, cmap='gray')

		label = 'Epoch {0}'.format(num_epoch)
		plt.title(label, fontsize=8)
		# fig.text(0.5, 0.04, label, ha='center')
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

		# fig1, ax1 = plt.subplots(size_figure_grid, size_figure_grid, figsize=(5, 5))
		# for i in range(size_figure_grid):
		# 	for j in range(size_figure_grid):
		# 		ax1[i, j].get_xaxis().set_visible(False)
		# 		ax1[i, j].get_yaxis().set_visible(False)

		# for k in range(size_figure_grid*size_figure_grid):
		# 	i = k // size_figure_grid
		# 	j = k % size_figure_grid
		# 	ax1[i, j].cla()
		# 	gt = self.reals[k,:,:,:]
		# 	ax1[i, j].imshow(gt, cmap='gray')

		label = 'Epoch {0}'.format(num_epoch)
		plt.title(label, fontsize=8)
		# fig1.text(0.5, 0.04, label, ha='center')
		# fig1.tight_layout()
		if save:
			plt.tight_layout()
			plt.savefig(path.split('.')[0]+'gt.png')

		if show:
			plt.show()
		else:
			plt.close()


	def CIFAR10_Classifier(self):
		self.FID_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, pooling='avg', weights='imagenet', input_tensor=None, input_shape=(299,299,3), classes=1000)

	def FID_cifar10(self):

		def data_preprocess(image):
			with tf.device('/CPU'):
				image = tf.image.resize(image,[299,299])
				# This will convert to float values in [0, 1]

				# image = tf.divide(image,255.0)
				image = tf.scalar_mul(2.0,image)
				image = tf.subtract(image,1.0)

				# image = tf.image.convert_image_dtype(image, tf.float16)
			return image


		if self.FID_load_flag == 0:
			### First time FID call setup
			self.FID_load_flag = 1	
			random_points = tf.keras.backend.random_uniform([self.FID_num_samples], minval=0, maxval=int(self.fid_train_images.shape[0]), dtype='int32', seed=None)
			print(random_points)
			self.fid_train_images = self.fid_train_images[random_points]
			# self.fid_train_images = tf.image.resize(self.fid_train_images, [128,128])
			# self.fid_train_images = tf.subtract(self.fid_train_images,0.50)
			# self.fid_train_images = tf.scalar_mul(2.0,self.fid_train_images)
			# self.fid_images = self.fid_train_images.numpy()

			# self.fid_train_images_names = self.fid_train_images[random_points]

			## self.fid_train_images has the names to be read. Make a dataset with it
			self.fid_image_dataset = tf.data.Dataset.from_tensor_slices(self.fid_train_images)
			if self.FID_kind != 'latent':
				self.fid_image_dataset = self.fid_image_dataset.map(data_preprocess,num_parallel_calls=int(self.num_parallel_calls))
			self.fid_image_dataset = self.fid_image_dataset.batch(self.fid_batch_size)

			
			self.CIFAR10_Classifier()


		if self.mode == 'fid':
			print(self.checkpoint_dir)
			# print('logs/130919_ELeGANt_mnist_lsgan_base_01/130919_ELeGANt_mnist_lsgan_base_Results_checkpoints')
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
					preds = tf.subtract(preds,0.50)
					preds = tf.scalar_mul(2.0,preds)
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
		# 	preds = tf.image.resize(preds, [128,128])
		# 	preds = tf.scalar_mul(2.0,preds)
		# 	preds = tf.subtract(preds,1.0)
		# 	preds = preds.numpy()

		# 	self.act1 = self.FID_model.predict(self.fid_images)
		# 	self.act2 = self.FID_model.predict(preds)
		# 	self.eval_FID()
		# 	return