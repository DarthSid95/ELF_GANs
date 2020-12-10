from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions


class ARCH_g2():	
	def __init__(self):
		return

	def generator_model_g2_base(self):
		# init_fn = tf.keras.initializers.glorot_uniform()
		# init_fn = tf.function(init_fn, autograph=False)
		# bias_init_fn = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=42)
		# bias_init_fn = tf.function(bias_init_fn, autograph=False)
		# bias_init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None)
		# bias_init_fn = tf.function(bias_init_fn, autograph=False)
		# init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.1, seed=None)
		# init_fn = tf.function(init_fn, autograph=False)

		init_fn = tf.keras.initializers.Identity()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(2, use_bias=True, input_shape=(self.noise_dims,), kernel_initializer=init_fn, bias_initializer = bias_init_fn))
		# model.add(layers.ReLU())
		model.add(layers.Dense(2, use_bias=True, kernel_initializer=init_fn ,bias_initializer = bias_init_fn))
		# model.add(layers.ReLU())
		# model.add(layers.Dense(2, use_bias=True,kernel_initializer=init_fn))
		# model.add(layers.LeakyReLU())
		return model

	def discriminator_model_g2_base(self):
		inputs = tf.keras.Input(shape=(self.output_size,))

		w0_nt_x = tf.keras.layers.Dense(self.L, activation=None, use_bias = False)(inputs)
		w0_nt_x2 = tf.math.scalar_mul(2., w0_nt_x)
		cos_terms = tf.keras.layers.Activation( activation = tf.math.cos)(w0_nt_x)
		sin_terms = tf.keras.layers.Activation( activation = tf.math.sin)(w0_nt_x)
		cos2_terms  = tf.keras.layers.Activation( activation = tf.math.cos)(w0_nt_x2)

		cos_sum = tf.keras.layers.Dense(1, activation=None, use_bias = True)(cos_terms)
		sin_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(sin_terms)

		cos2_c_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(cos2_terms) #Tau_c weights
		cos2_s_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(cos2_terms) #Tau_s weights

		lambda_x_term = tf.keras.layers.Subtract()([cos2_s_sum, cos2_c_sum]) #(tau_s  - tau_r)
		#DS 1 - log, 2 - red_sum, 3 - abs(red_Sum), 4 - log/4pi
		# Out = tf.keras.layers.Add()([cos_sum, sin_sum, tf.divide(tf.abs(tf.reduce_sum(inputs,axis=-1,keepdims=True)),self.latent_dims)])
		if self.homo_flag:
			if self.latent_dims == 1:
				Out = tf.keras.layers.Add()([cos_sum, sin_sum, tf.divide(tf.abs(inputs),2.)])
			elif self.latent_dims >= 2:
				Out = tf.keras.layers.Add()([cos_sum, sin_sum, tf.divide(tf.reduce_sum(inputs,axis=-1,keepdims=True),self.latent_dims)])
				# Out = tf.keras.layers.Add()([cos_sum, sin_sum, tf.divide(tf.math.log(tf.norm(inputs,ord=2,keepdims=True)),-2.*np.pi)])
			else:
				const = tf.divide(tf.math.pow(np.pi,(self.latent_dims/2.)),self.latent_dims*(self.latent_dims-2)*tf.math.exp(tf.math.lgamma((self.latent_dims/2.) + 1)))
				Out = tf.keras.layers.Add()([cos_sum, sin_sum, tf.math.scalar_mul(tf.math.pow(tf.norm(inputs,ord=2,keepdims=True), (2 - self.latent_dims)), const)])
		else:
			Out = tf.keras.layers.Add()([cos_sum, sin_sum])
		model = tf.keras.Model(inputs=inputs, outputs=[Out,lambda_x_term])

		return model
		
		# cos_terms = tf.keras.layers.Dense(self.L, activation=tf.math.cos, use_bias = False)(inputs)
		# sin_terms = tf.keras.layers.Dense(self.L, activation=tf.math.sin, use_bias = False)(inputs)
		# cos2_terms = tf.keras.layers.Dense(self.L, activation=tf.math.cos, use_bias = False)(inputs) #The weights gave the *2 inside
		# cos_sum = tf.keras.layers.Dense(1, activation=None, use_bias = True)(cos_terms)
		# sin_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(sin_terms)

		# cos2_c_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(cos2_terms) #Tau_c weights
		# cos2_s_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(cos2_terms) #Tau_s weights

		# lambda_x_term = tf.keras.layers.Subtract()([cos2_s_sum, cos2_c_sum]) #(tau_s  - tau_r)
		# Out = tf.keras.layers.Add()([cos_sum, sin_sum])
		# model = tf.keras.Model(inputs=inputs, outputs=[Out,lambda_x_term])
		# return model
		# inputs = tf.keras.Input(shape=(2,))

		# cos_terms = tf.keras.layers.Dense(self.L, activation=tf.math.cos, use_bias = False)(inputs)
		# sin_terms = tf.keras.layers.Dense(self.L, activation=tf.math.sin, use_bias = False)(inputs)
		# cos_sum = tf.keras.layers.Dense(1, activation=None, use_bias = True)(cos_terms)
		# sin_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(sin_terms)
		# Out = tf.keras.layers.Add()([cos_sum, sin_sum])
		# model = tf.keras.Model(inputs=inputs, outputs=Out)
		# return model



	def show_result_g2(self, images = None, num_epoch = 0, path = 'result.png', show = False, save = False):
		# print("Gaussian Stats : True mean {} True Sigma {} \n Fake mean {} Fake Sigma {}".format(np.mean(self.reals,axis = 0), np.cov(self.reals,rowvar = False), np.mean(self.fakes, axis = 0), np.cov(self.fakes,rowvar = False) ))
		if self.res_flag:
			self.res_file.write("Gaussian Stats : True mean {} True Sigma {} \n Fake mean {} Fake Sigma {}".format(np.mean(self.reals), np.cov(self.reals,rowvar = False), np.mean(self.fakes), np.cov(self.fakes,rowvar = False) ))

		if self.total_count.numpy() == 1 or self.total_count.numpy()%1000 == 0:
			np.save(path+'_reals.npy',np.array(self.reals))
			np.save(path+'_fakes.npy',np.array(self.fakes))
		
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

		with PdfPages(path+'_Classifier.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(False)
			ax1.get_yaxis().set_visible(False)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim([self.MIN,self.MAX])
			ax1.scatter(self.reals[:,0], self.reals[:,1], c='r', linewidth = 1.5,  marker = '.', alpha = 0.8)
			ax1.scatter(self.fakes[:,0], self.fakes[:,1], c='g', linewidth = 1.5,  marker = '.', alpha = 0.8)
			# ax1.legend(loc = 'upper right')
			fig1.tight_layout()
			pdf.savefig(fig1)
			plt.close(fig1)

	