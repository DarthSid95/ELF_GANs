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



class ARCH_gmm8():

	def __init__(self):
		print("CREATING ARCH_deq_gmm8 CLASS")
		return

	def generator_model_gmm8_base(self):
		init_fn = tf.random_normal_initializer(mean=0.0, stddev=0.5, seed=None)#tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.noise_dims,))

		enc1 = tf.keras.layers.Dense(64, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(inputs)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)

		# enc11 = tf.keras.layers.Dense(64, kernel_initializer=init_fn, use_bias = True, activation = 'tanh')(enc1)
		# enc11 = tf.keras.layers.ReLU()(enc11)
		
		# enc12 = tf.keras.layers.Dense(int(self.latent_dims*10), kernel_initializer=init_fn, use_bias = True, activation = 'sigmoid')(enc11)
		# enc12 = tf.keras.layers.ReLU()(enc12)

		enc2 = tf.keras.layers.Dense(32, kernel_initializer=init_fn, use_bias = True, bias_initializer = bias_init_fn)(enc1)
		# enc2 = tf.keras.layers.Activation( activation = 'tanh')(enc2)
		enc2 = tf.keras.layers.Activation( activation = 'sigmoid')(enc2)
		# enc2 = tf.keras.layers.LeakyReLU()(enc2)

		enc3 = tf.keras.layers.Dense(16, kernel_initializer=init_fn, use_bias = False)(enc2)
		# enc3 = tf.keras.layers.Activation( activation = 'tanh')(enc3)
		enc3 = tf.keras.layers.Activation( activation = 'sigmoid')(enc3)
		# enc3 = tf.keras.layers.LeakyReLU()(enc3)

		enc4 = tf.keras.layers.Dense(self.output_size, kernel_initializer=init_fn, use_bias = False)(enc3)
		# enc4 =  tf.keras.layers.Activation( activation = 'tanh')(enc4)
		# enc4 = tf.math.scalar_mul(1.2, enc4)
		enc4 =  tf.keras.layers.Activation( activation = 'sigmoid')(enc4)
		# enc4 = tf.keras.layers.ReLU(max_value = 1.)(enc4)

		model = tf.keras.Model(inputs = inputs, outputs = enc4)

		return model

		# init_fn = tf.keras.initializers.glorot_uniform()
		# # init_fn = tf.random_normal_initializer(mean = 1.0, stddev = 0.01)
		# init_fn = tf.function(init_fn, autograph=False)

		# inputs = tf.keras.Input(shape=(self.noise_dims,))

		# enc1 = tf.keras.layers.Dense(128, kernel_initializer=init_fn, use_bias = True)(inputs)
		# enc1 = tf.keras.layers.ReLU()(enc1)

		# # enc11 = tf.keras.layers.Dense(int(self.latent_dims*256), kernel_initializer=init_fn, use_bias = True, activation = 'sigmoid')(enc1)
		# # enc11 = tf.keras.layers.ReLU()(enc11)
		
		# # enc12 = tf.keras.layers.Dense(int(self.latent_dims*64), kernel_initializer=init_fn, use_bias = True, activation = 'sigmoid')(enc11)
		# # enc12 = tf.keras.layers.ReLU()(enc12)

		# enc2 = tf.keras.layers.Dense(128, kernel_initializer=init_fn, use_bias = True)(enc1)
		# enc2 = tf.keras.layers.ReLU()(enc2)

		# # enc3 = tf.keras.layers.Dense(int(self.latent_dims*4), kernel_initializer=init_fn, use_bias = True)(enc2)
		# # # enc3 = tf.keras.layers.ReLU()(enc3)

		# enc4 = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = True)(enc2)
		# enc4 =  tf.keras.layers.Activation( activation = 'sigmoid')(enc4)
		# # enc4 = tf.keras.layers.ReLU(max_value = 1.)(enc4)

		# model = tf.keras.Model(inputs = inputs, outputs = enc4)

		# return model


	def generator_model_gmm8_AE(self):

		init_fn = tf.keras.initializers.glorot_uniform()
		# init_fn = tf.random_normal_initializer(mean = 1.0, stddev = 5.0)
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.noise_dims,))

		enc1 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn, use_bias = True)(inputs)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Dense(int(self.latent_dims*4), kernel_initializer=init_fn, use_bias = True)(enc1)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)

		enc3 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn, use_bias = True)(enc2)
		# enc3 = tf.keras.layers.LeakyReLU()(enc3)

		enc4 = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = True)(enc3)
		enc4 = tf.keras.layers.ReLU(max_value = 1.)(enc4)

		model = tf.keras.Model(inputs = inputs, outputs = enc4)

		return model

	def generator_model_gmm8_Cycle(self):

		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.noise_dims,))

		enc1 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn)(inputs)
		# enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Dense(int(self.latent_dims*4), kernel_initializer=init_fn)(enc1)
		# enc2 = tf.keras.layers.LeakyReLU()(enc2)

		enc3 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn)(enc2)
		# enc3 = tf.keras.layers.LeakyReLU()(enc3)

		enc4 = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn, use_bias = True)(enc3)
		# enc3 = tf.keras.layers.ReLU()(enc3)


		encoded = tf.keras.Input(shape=(self.latent_dims,))

		dec1 = tf.keras.layers.Dense(int((self.latent_dims)*2), kernel_initializer=init_fn)(encoded)
		# dec1 = tf.keras.layers.LeakyReLU()(dec1)

		dec2 = tf.keras.layers.Dense(int((self.latent_dims)*4), kernel_initializer=init_fn)(dec1)
		# dec2 = tf.keras.layers.LeakyReLU()(dec2)

		dec3 = tf.keras.layers.Dense(int((self.latent_dims)*2), kernel_initializer=init_fn)(dec2)
		# dec3 = tf.keras.layers.LeakyReLU()(dec3)

		out = tf.keras.layers.Dense(self.noise_dims, kernel_initializer=init_fn)(dec3)

		model = tf.keras.Model(inputs = inputs, outputs = enc4)
		self.generator_dec = tf.keras.Model(inputs = encoded, outputs = out)

		print("\n\n GENERATOR DECODER MODEL: \n\n")
		print(self.generator_dec.summary())

		return model


	def encoder_model_gmm8_AE(self):  # FOR BASE AE

		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.output_size,))

		enc0 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn, activation ='tanh')(inputs)
		# enc0 = tf.keras.layers.Dropout(0.5)(enc0)
		enc0 = tf.keras.layers.LeakyReLU()(enc0)

		enc1 = tf.keras.layers.Dense(int(self.latent_dims*4), kernel_initializer=init_fn, activation ='tanh')(enc0)
		# enc1 = tf.keras.layers.Dropout(0.5)(enc1)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn, activation ='tanh')(enc1)
		# enc2 = tf.keras.layers.Dropout(0.5)(enc2)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)

		enc3 = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn)(enc2)
		# enc3 = tf.keras.layers.LeakyReLU()(enc3)

		encoded = tf.keras.Input(shape=(self.latent_dims,))

		dec1 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn)(encoded)
		# dec1 = tf.keras.layers.Dropout(0.5)(dec1)
		dec1 = tf.keras.layers.LeakyReLU()(dec1)

		dec2 = tf.keras.layers.Dense(int(self.latent_dims*4), kernel_initializer=init_fn)(dec1)
		# dec2 = tf.keras.layers.Dropout(0.5)(dec2)
		dec2 = tf.keras.layers.LeakyReLU()(dec2)

		dec3 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn)(dec2)
		# dec3 = tf.keras.layers.Dropout(0.5)(dec3)
		dec3 = tf.keras.layers.LeakyReLU()(dec3)

		out = tf.keras.layers.Dense(int(self.output_size), kernel_initializer=init_fn)(dec3)
		# out_enc = tf.keras.layers.LeakyReLU()(out_enc)

		self.Encoder = tf.keras.Model(inputs = inputs, outputs = enc3)
		self.Decoder = tf.keras.Model(inputs = encoded, outputs = out)
		
		return self.Encoder

	def encoder_model_gmm8_Cycle(self):  # FOR BASE AE

		init_fn = tf.keras.initializers.glorot_uniform()
		init_fn = tf.function(init_fn, autograph=False)

		inputs = tf.keras.Input(shape=(self.output_size,))

		enc0 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn)(inputs)
		# enc0 = tf.keras.layers.Dropout(0.5)(enc0)
		enc0 = tf.keras.layers.LeakyReLU()(enc0)

		enc1 = tf.keras.layers.Dense(int(self.latent_dims*4), kernel_initializer=init_fn)(enc0)
		# enc1 = tf.keras.layers.Dropout(0.5)(enc1)
		enc1 = tf.keras.layers.LeakyReLU()(enc1)

		enc2 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn)(enc1)
		# enc2 = tf.keras.layers.Dropout(0.5)(enc2)
		enc2 = tf.keras.layers.LeakyReLU()(enc2)

		enc3 = tf.keras.layers.Dense(self.latent_dims, kernel_initializer=init_fn)(enc2)
		# enc3 = tf.keras.layers.LeakyReLU()(enc3)

		encoded = tf.keras.Input(shape=(self.latent_dims,))

		dec1 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn)(encoded)
		# dec1 = tf.keras.layers.Dropout(0.5)(dec1)
		dec1 = tf.keras.layers.LeakyReLU()(dec1)

		dec2 = tf.keras.layers.Dense(int(self.latent_dims*4), kernel_initializer=init_fn)(dec1)
		# dec2 = tf.keras.layers.Dropout(0.5)(dec2)
		dec2 = tf.keras.layers.LeakyReLU()(dec2)

		dec3 = tf.keras.layers.Dense(int(self.latent_dims*2), kernel_initializer=init_fn)(dec2)
		# dec3 = tf.keras.layers.Dropout(0.5)(dec3)
		dec3 = tf.keras.layers.LeakyReLU()(dec3)

		out = tf.keras.layers.Dense(int(self.output_size), kernel_initializer=init_fn, use_bias=False)(dec3)
		# out_enc = tf.keras.layers.LeakyReLU()(out_enc)

		self.Encoder = tf.keras.Model(inputs = inputs, outputs = enc3)
		self.Decoder = tf.keras.Model(inputs = encoded, outputs = out)
		
		return self.Encoder

	def discriminator_model_gmm8_base(self):
		inputs = tf.keras.Input(shape=(self.output_size,))
		# cos_terms = tf.keras.layers.Dense(self.L, activation=tf.math.cos, use_bias = False)(inputs)
		# sin_terms = tf.keras.layers.Dense(self.L, activation=tf.math.sin, use_bias = False)(inputs)
		# cos2_terms = tf.keras.layers.Dense(self.L, activation=tf.math.cos, use_bias = False)(inputs) #The weights gave the *2 inside

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
		# inputs = tf.keras.Input(shape=(self.latent_dims,)) #used to be self.N
		# cos_terms = tf.keras.layers.Dense(self.L, activation=tf.math.cos, use_bias = False)(inputs)
		# sin_terms = tf.keras.layers.Dense(self.L, activation=tf.math.sin, use_bias = False)(inputs)
		# cos_sum = tf.keras.layers.Dense(1, activation=None, use_bias = True)(cos_terms)
		# sin_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(sin_terms)
		# Out = tf.keras.layers.Add()([cos_sum, sin_sum])
		# # Out = tf.keras.layers.Activation(activation = 'sigmoid')(Out)
		# model = tf.keras.Model(inputs=inputs, outputs= Out)
		# return model

	def discriminator_model_gmm8_AE(self):
		inputs = tf.keras.Input(shape=(self.latent_dims,)) #used to be self.N
		cos_terms = tf.keras.layers.Dense(self.L, activation=tf.math.cos, use_bias = False)(inputs)
		sin_terms = tf.keras.layers.Dense(self.L, activation=tf.math.sin, use_bias = False)(inputs)
		cos_sum = tf.keras.layers.Dense(1, activation=None, use_bias = True)(cos_terms)
		sin_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(sin_terms)
		Out = tf.keras.layers.Add()([cos_sum, sin_sum])
		# Out = tf.keras.layers.Activation(activation = 'sigmoid')(Out)
		model = tf.keras.Model(inputs=inputs, outputs= Out)
		return model

	def discriminator_model_gmm8_Cycle(self):
		inputs = tf.keras.Input(shape=(self.latent_dims,)) #used to be self.N
		cos_terms = tf.keras.layers.Dense(self.L, activation=tf.math.cos, use_bias = False)(inputs)
		sin_terms = tf.keras.layers.Dense(self.L, activation=tf.math.sin, use_bias = False)(inputs)
		cos_sum = tf.keras.layers.Dense(1, activation=None, use_bias = True)(cos_terms)
		sin_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(sin_terms)
		Out = tf.keras.layers.Add()([cos_sum, sin_sum])
		model = tf.keras.Model(inputs=inputs, outputs= Out)
		return model


	def show_result_gmm8(self, images = None, num_epoch = 0, path = 'result.png', show = False, save = True):
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

		with PdfPages(path+'_Data.pdf') as pdf:
			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(False)
			ax1.get_yaxis().set_visible(False)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim([self.MIN,self.MAX])
			ax1.scatter(self.reals[:,0], self.reals[:,1], c='r', linewidth = 1, marker = '.', alpha = 0.5)
			ax1.scatter(self.fakes[:,0], self.fakes[:,1], c='g', linewidth = 1, marker = '.', alpha = 0.5)
			# ax1.legend(loc = 'upper right')
			fig1.tight_layout()
			pdf.savefig(fig1)
			plt.close(fig1)

			