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


		out = tf.keras.layers.Conv2DTranspose(3, 4,strides=1,padding='same', kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(denc1) #64x64x3
		out =  tf.keras.layers.Activation( activation = 'tanh')(out)


		self.Decoder = tf.keras.Model(inputs=encoded, outputs=out)
		self.Encoder = tf.keras.Model(inputs=inputs, outputs=enc)

		return


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
		# return model

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

			if self.FID_kind != 'latent':
				self.SVHN_Classifier()


		if self.mode == 'fid':
			print(self.checkpoint_dir)

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
					preds = self.Decoder(self.get_noise(tf.constant(100)), training=False)
					preds = tf.image.resize(preds, [299,299])
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
