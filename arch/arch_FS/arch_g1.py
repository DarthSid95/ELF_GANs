from __future__ import print_function
import os, sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import math

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pgf import FigureCanvasPgf
# matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
# from matplotlib.backends.backend_pgf import PdfPages
import tensorflow_probability as tfp
tfd = tfp.distributions


class ARCH_g1():	
	def __init__(self):
		return
		
	def generator_model_g1_base(self):
		# init_fn = tf.keras.initializers.glorot_uniform()
		# init_fn = tf.function(init_fn, autograph=False)
		iden_init_fn = tf.keras.initializers.Identity()
		iden_init_fn = tf.function(iden_init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(1, use_bias=True, input_shape=(self.noise_dims,),kernel_initializer=iden_init_fn, bias_initializer = bias_init_fn))
		# model.add(layers.ReLU())
		return model

	def discriminator_model_g1_base(self):
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
		# inputs = tf.keras.Input(shape=(1,))

		# cos_terms = tf.keras.layers.Dense(self.L, activation=tf.math.cos, use_bias = False)(inputs)
		# sin_terms = tf.keras.layers.Dense(self.L, activation=tf.math.sin, use_bias = False)(inputs)
		# cos_sum = tf.keras.layers.Dense(1, activation=None, use_bias = True)(cos_terms)
		# sin_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(sin_terms)
		# Out = tf.keras.layers.Add()([cos_sum, sin_sum])
		# model = tf.keras.Model(inputs=inputs, outputs=Out)
		# return model


	def show_result_g1(self, images = None, num_epoch = 0, path = 'result.png', show = False, save = False):
		# print("Gaussian Stats : True mean {} True Sigma {}, Fake mean {} Fake Sigma {}".format(np.mean(self.reals), np.std(self.reals), np.mean(self.fakes), np.std(self.fakes) ))
		# Define a single scalar Normal distribution.
		# pd_dist = tfd.Normal(loc=np.mean(self.reals), scale=np.std(self.reals))
		# pg_dist = tfd.Normal(loc=np.mean(self.fakes), scale=np.std(self.fakes))
		pd_dist = tfd.TruncatedNormal(loc=np.mean(self.reals), scale=np.std(self.reals), low = -10., high = 20.)
		pg_dist = tfd.TruncatedNormal(loc=np.mean(self.fakes), scale=np.std(self.fakes), low = -10., high = 20.)

		beta_c,beta_s = eval('self.Fourier_Series_Comp(self.fakes)')

		# self.pdf.set_weights([self.Coeffs, self.alpha_c,self.bias, self.alpha_s, self.Tau_c, self.Tau_s])
		# self.pgf.set_weights([self.Coeffs, self.beta_c,self.bias, self.beta_s, self.Tau_c, self.Tau_s])


		basis = np.expand_dims(np.linspace(self.MIN, self.MAX, int(1e4), dtype=np.float32), axis = 1)
		pd_vals = pd_dist.prob(basis)
		pg_vals = pg_dist.prob(basis)

		disc,_ = self.discriminator_B(self.discriminator_A(basis,training = False),training=False)
		disc = disc - min(disc)
		disc /= max(abs(disc))*1.0
		disc -= 0.50

		# pd_vals_FS,_ = self.pdf(basis, training = False)
		# pg_vals_FS,_ = self.pgf(basis, training = False)
		true_classifier = np.ones_like(basis)
		true_classifier[pd_vals > pg_vals] = 0

		if self.paper and (self.total_count.numpy() == 1 or self.total_count.numpy()%1000 == 0):
			np.save(path+'_disc.npy',np.array(disc))
			np.save(path+'_reals.npy',np.array(self.reals))
			np.save(path+'_fakes.npy',np.array(self.fakes))
			# np.save(path+'_pd_FS.npy',np.array(pd_vals_FS))
			# np.save(path+'_pg_FS.npy',np.array(pg_vals_FS))
		
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
				# "pgf.preamble": [
				# 	 r"\usepackage[utf8x]{inputenc}",
				# 	 r"\usepackage[T1]{fontenc}",
				# 	 r"\usepackage{cmbright}",
				# 	 ]
			})

		with PdfPages(path+'_Classifier.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim(bottom=-0.5,top=1.8)
			ax1.plot(basis,pd_vals, linewidth = 1.5, c='r')
			ax1.plot(basis,pg_vals, linewidth = 1.5, c='g')
			ax1.scatter(self.reals, np.zeros_like(self.reals), c='r', linewidth = 1.5, label='Real Data', marker = '.')
			ax1.scatter(self.fakes, np.zeros_like(self.fakes), c='g', linewidth = 1.5, label='Fake Data', marker = '.')
			# ax1.plot(basis,disc, c='b', linewidth = 1.5, label='Discriminator')
			# if self.total_count < 20:
			# 	ax1.plot(basis,true_classifier,'c--', linewidth = 1.5, label='True Classifier')
			ax1.legend(loc = 'upper right')
			fig1.tight_layout()
			pdf.savefig(fig1)
			plt.close(fig1)

			# fig2 = plt.figure(figsize=(3.5, 3.5))
			# ax2 = fig2.add_subplot(111)
			# ax2.cla()
			# ax2.get_xaxis().set_visible(True)
			# ax2.get_yaxis().set_visible(True)
			# ax2.set_xlim([self.MIN,self.MAX])
			# ax2.set_ylim(bottom=-0.5,top=1.8)
			# ax2.plot(basis,pd_vals, linewidth = 1.5, c='r', alpha = 0.1)
			# ax2.plot(basis,pg_vals, linewidth = 1.5, c='g', alpha = 0.1)
			# # ax2.plot(basis,pd_vals_FS, linewidth = 1, c='m')
			# # ax2.plot(basis,pg_vals_FS, linewidth = 1, c='b')
			# # ax2.legend(loc = 'upper right')
			# fig2.tight_layout()
			# pdf.savefig(fig2)
			# plt.close(fig2)



			