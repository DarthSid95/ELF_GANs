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

	def generator_model_g2(self):

		init_fn = tf.keras.initializers.Identity()
		init_fn = tf.function(init_fn, autograph=False)
		bias_init_fn = tf.keras.initializers.Zeros()
		bias_init_fn = tf.function(bias_init_fn, autograph=False)

		model = tf.keras.Sequential()
		model.add(layers.Dense(2, use_bias=True, input_shape=(self.noise_dims,), kernel_initializer=init_fn, bias_initializer = bias_init_fn))
		# model.add(layers.ReLU())
		model.add(layers.Dense(2, use_bias=True, kernel_initializer=init_fn ,bias_initializer = bias_init_fn))
		return model


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

	