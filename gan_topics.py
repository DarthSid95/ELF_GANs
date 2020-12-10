from __future__ import print_function
import os, sys, time, argparse
from datetime import date
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math
from absl import app
from absl import flags
import json

from gan_data import *
from gan_arch import *

import tensorflow_probability as tfp
tfd = tfp.distributions
from matplotlib.backends.backend_pgf import PdfPages

# FLAGS(sys.argv)
# tf.keras.backend.set_floatx('float64')

'''
GAN_topic is the Overarching class file, where corresponding parents are instantialized, along with setting up the calling functions for these and files and folders for resutls, etc. data reading is also done from here. Sometimes display functions, architectures, etc may be modified here if needed (overloading parent classes)
'''

'''***********************************************************************************
********** GAN Baseline setup ********************************************************
***********************************************************************************'''
class GAN_Base(GAN_ARCH, GAN_DATA_Base):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_ARCH class - defines all GAN architectures'''

		GAN_ARCH.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_Base.__init__(self)
		# eval('GAN_DATA_'+FLAGS.topic+'.__init__(self,data)')


	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'


		''' Define dataset and tf.data function. batch sizing done'''
		self.get_data()
		print(" Batch Size {}, Batch Size * Dloop {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, self.batch_size_big, self.num_batches,self.print_step, self.save_step))


	def get_data(self):
		# with tf.device('/CPU'):
		self.train_data = eval(self.gen_func)

		self.batch_size_big = tf.constant(self.batch_size*self.Dloop,dtype='int64')
		self.num_batches = int(np.floor((self.train_data.shape[0] * self.reps)/self.batch_size))
		''' Set PRINT and SAVE iters if 0'''
		self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
		self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

		self.train_dataset = eval(self.dataset_func)

		self.train_dataset_size = self.train_data.shape[0]


'''***********************************************************************************
********** GAN ELEGANT setup *********************************************************
***********************************************************************************'''
class GAN_ELeGANt(GAN_ARCH, GAN_DATA_Base):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_ARCH class - defines all GAN architectures'''

		GAN_ARCH.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_Base.__init__(self)
		# eval('GAN_DATA_'+FLAGS.topic+'.__init__(self,data)')


	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		if self.loss == 'FS':
			self.gen_model = 'self.generator_model_'+self.data+'_'+self.latent_kind+'()'
			self.disc_model = 'self.discriminator_model_'+self.data+'_'+self.latent_kind+'()' 
			self.EncDec_func = 'self.encoder_model_'+self.data+'_'+self.latent_kind+'()'
			self.DEQ_func = 'self.discriminator_ODE()'

		# ''' Create Run colation folder'''
		# if not os.path.exists(self.log_dir):
		# 	os.mkdir(self.log_dir)
		# 	print("Directory " , self.log_dir ,  " Created ")
		# else:    
		# 	print("Directory " , self.log_dir ,  " already exists")
		# if not os.path.exists(self.run_loc):
		# 	os.mkdir(self.run_loc)
		# 	print("Directory " , self.run_loc ,  " Created ")
		# else:    
		# 	print("Directory " , self.run_loc ,  " already exists")

		# self.cache_loc = self.run_loc+'/'+self.runID+'_Cache.txt'

		''' Define dataset and tf.data function. batch sizing done'''
		self.get_data()
		print(" Batch Size {}, Batch Size * Dloop {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, self.batch_size_big, self.num_batches,self.print_step, self.save_step))


	def get_data(self):
		# with tf.device('/CPU'):
		self.train_data = eval(self.gen_func)

		self.batch_size_big = tf.constant(self.batch_size*self.Dloop,dtype='int64')
		self.num_batches = int(np.floor((self.train_data.shape[0] * self.reps)/self.batch_size))
		''' Set PRINT and SAVE iters if 0'''
		self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
		self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

		self.train_dataset = eval(self.dataset_func)

		self.train_dataset_size = self.train_data.shape[0]


'''***********************************************************************************
********** GAN ACGAN setup ***********************************************************
***********************************************************************************'''
class GAN_ACGAN(GAN_ARCH, GAN_DATA_ACGAN):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_ARCH class - defines all GAN architectures'''

		GAN_ARCH.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_ACGAN.__init__(self)
		# eval('GAN_DATA_'+FLAGS.topic+'.__init__(self,data)')


	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.train_labels, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		if self.loss == 'FS':
			self.gen_model = 'self.generator_model_'+self.data+'_'+self.latent_kind+'()'
			self.disc_model = 'self.discriminator_model_'+self.data+'_'+self.latent_kind+'()' 
			self.EncDec_func = 'self.encoder_model_'+self.data+'_'+self.latent_kind+'()'
			self.DEQ_func = 'self.discriminator_ODE()'

		''' Define dataset and tf.data function. batch sizing done'''
		self.get_data()
		print(" Batch Size {}, Batch Size * Dloop {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, self.batch_size_big, self.num_batches,self.print_step, self.save_step))


	def get_data(self):
		# with tf.device('/CPU'):
		self.train_data, self.train_labels = eval(self.gen_func)

		self.batch_size_big = tf.constant(self.batch_size*self.Dloop,dtype='int64')
		self.num_batches = int(np.floor((self.train_data.shape[0])/self.batch_size))
		''' Set PRINT and SAVE iters if 0'''
		self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
		self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

		self.train_dataset = eval(self.dataset_func)
		print("Dataset created - this is it")
		print(self.train_dataset)

		self.train_dataset_size = self.train_data.shape[0]

	def get_noise(self,noise_case,batch_size):
		noise = tf.random.normal([batch_size, self.noise_dims], mean = self.noise_mean, stddev = self.noise_stddev)
		if noise_case == 'test':
			if self.data in ['mnist', 'cifar10']:
				if self.testcase in ['single', 'few']:
					noise_labels = self.number*np.ones((batch_size,1)).astype('int32')
				elif self.testcase in ['sharp']:
					noise_labels = np.expand_dims(np.random.choice([1,2,4,5,7,9], batch_size), axis = 1).astype('int32')
				elif self.testcase in ['even']:
					noise_labels = np.expand_dims(np.random.choice([0,2,4,6,8], batch_size), axis = 1).astype('int32')
				elif self.testcase in ['odd']:
					noise_labels = np.expand_dims(np.random.choice([1,3,5,7,9], batch_size), axis = 1).astype('int32')
				elif self.testcase in ['animals']:
					noise_labels = np.expand_dims(np.random.choice([2,3,4,5,6,7], batch_size), axis = 1).astype('int32')
			elif self.data in ['celeba']:
				if self.testcase in ['male', 'fewmale', 'bald', 'hat']:
					noise_labels = np.ones((batch_size,1)).astype('int32')
				elif self.testcase in ['female', 'fewfemale']:
					noise_labels = np.zeros((batch_size,1)).astype('int32')
		if noise_case == 'train':
			noise_labels = np.random.randint(0, self.num_classes, batch_size)
			if self.data == 'celeba':
				noise_labels = np.expand_dims(noise_labels, axis = 1)

		return noise, noise_labels




'''***********************************************************************************
********** GAN RumiGAN setup *********************************************************
***********************************************************************************'''
class GAN_RumiGAN(GAN_ARCH, GAN_DATA_RumiGAN):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_ARCH class - defines all GAN architectures'''
		GAN_ARCH.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_RumiGAN.__init__(self)


	# def show_result_g1(self, images = None, num_epoch = 0, show = False, save = False, path = 'result.png'):
		# fig = plt.figure()
		# ax = fig.add_subplot(111)
		# ax.get_xaxis().set_visible(True)
		# ax.get_yaxis().set_visible(True)
		# ax.set_xlim([self.MIN,self.MAX])

		# basis = np.expand_dims(np.array(np.arange(-10000*self.MIN,10000*self.MAX,1)/10000.0),axis=1)
		# print(basis)
		# disc = self.discriminator(basis,training= False)
		# print("Gaussian Stats : BG mean {} BG Sigma {}, Neg mean {} Neg Sigma {}, Fake mean {} Fake Sigma {}".format(np.mean(self.reals_pos), np.std(self.reals_pos), np.mean(self.reals_neg), np.std(self.reals_neg), np.mean(images), np.std(images) ))
		# if self.res_flag:
		# 	self.res_file.write("Gaussian Stats : BG mean {} BG Sigma {}, Neg mean {} Neg Sigma {}, Fake mean {} Fake Sigma {}".format(np.mean(self.reals_pos), np.std(self.reals_pos), np.mean(self.reals_neg), np.std(self.reals_neg), np.mean(images), np.std(images) ))



		# pd_dist = tfd.Normal(loc=np.mean(self.reals_pos), scale=np.std(self.reals_pos))
		# pdm_dist = tfd.Normal(loc=np.mean(self.reals_neg), scale=np.std(self.reals_neg))
		# pg_dist = tfd.Normal(loc=np.mean(images), scale=np.std(images))
		# # basis = np.expand_dims(np.array(np.arange(self.MIN*10,self.MAX*10,1)/10.0),axis=1)
		# disc = self.discriminator(basis,training = False)

		# pd_vals = pd_dist.prob(basis)
		# pdm_vals = pdm_dist.prob(basis)
		# pg_vals_data = pg_dist.prob(basis)

		# pg_vals = 1.5*np.array(pd_vals) - np.array(pdm_vals)
		# pg_vals[np.where(pg_vals< 0)[0]] = 0
		# # pg_vals /= np.sum(pg_vals,axis = 0)
		# list(pg_vals)
		# print(np.divide(np.array(pg_vals_data)+np.array(pdm_vals), np.array(pd_vals))[300:320])
		# # pg_vals = pg_vals/pg_vals.sum(axis=0,keepdims=1)

		# # pd_vals = pd_vals/max(pd_vals)
		# # pg_vals = pg_vals/max(pg_vals)
		# # pdm_vals = pdm_vals/max(pdm_vals)
		# # pg_vals_data = pg_vals_data/max(pg_vals_data)
		# ax.cla()
		# # ax.scatter(basis,pd_vals, linewidth = 3.5, c='r',label='Background Class', alpha = 0.5)	
		# ax.scatter(self.reals_pos[:10000], np.zeros_like(self.reals_pos[:10000]), c='r',label='Real Data')
		# ax.scatter(self.reals_neg[:10000], np.zeros_like(self.reals_neg[:10000]), alpha = 0.5, c='b',label='Negative Data')
		# ax.scatter(images[:10000], np.zeros_like(images[:10000]), c='g', alpha = 0.05, label='Fake Data')
		# # ax.plot(basis, disc, c='b',label='Discriminator')
		# ax.legend(loc = 'upper right')

		# label = 'Epoch {0}'.format(num_epoch)
		# fig.text(0.5, 0.04, label, ha='center')
		# if save:
		# 	plt.savefig(path)


		# plt.rcParams.update({
		# 			"pgf.texsystem": "pdflatex",
		# 			"font.family": "serif",  # use serif/main font for text elements
		# 			"font.size":18,
		# 			"font.serif": [], 
		# 			"text.usetex": True,     # use inline math for ticks
		# 			"pgf.rcfonts": False,    # don't setup fonts from rc parameters
		# 			"pgf.preamble": [
		# 				 r"\usepackage[utf8x]{inputenc}",
		# 				 r"\usepackage[T1]{fontenc}",
		# 				 r"\usepackage{cmbright}",
		# 				 ]
		# 		})
		# with PdfPages(path+'_Classifier.pdf', metadata={'author': 'Siddarth Asokan'}) as pdf:

		# 	fig1 = plt.figure(figsize=(7,5))
		# 	ax1 = fig1.add_subplot(111)
		# 	ax1.cla()
		# 	ax1.get_xaxis().set_visible(True)
		# 	ax1.get_yaxis().set_visible(True)
		# 	ax1.set_xlim([self.MIN,self.MAX])
		# 	# ax1.set_ylim(bottom=-0.25,top=1.8)
		# 	ax1.plot(basis,pd_vals, linewidth = 3.5, c='r',label='Background Class', alpha = 0.5)
		# 	ax1.plot(basis,pdm_vals, linewidth = 3.5, c='b',label='Negative Subset', alpha = 0.5)
		# 	ax1.plot(basis,pg_vals, linewidth = 3.5, c='g',label='True Generator', alpha = 0.5)
		# 	ax1.plot(basis,pg_vals_data, 'g--', linewidth = 2.5,label='Learnt Generator', alpha = 0.75)
		# 	ax1.scatter(self.reals_pos[:1000], np.zeros_like(self.reals_pos[:1000]), linewidth = 3.5, c='r', marker = '.', alpha = 0.25)
		# 	ax1.scatter(self.reals_neg[:1000], np.zeros_like(self.reals_neg[:1000]), linewidth = 3.5, c='b', marker = '.', alpha = 0.95)
		# 	ax1.scatter(images[:1000], np.zeros_like(images[:1000]), c='g', linewidth = 3.5, marker = '.', alpha = 0.95)
			
			
		# 	# ax1.plot(basis,disc, c='b', linewidth = 1.5, label='Discriminator')
		# 	ax1.legend(loc = 'upper right')
		# 	fig1.tight_layout()
		# 	pdf.savefig(fig1)
		# 	plt.close(fig1)

		# with PdfPages(path+'_Samples.pdf', metadata={'author': 'Siddarth Asokan'}) as pdf:

		# 	fig2 = plt.figure(figsize=(7,5))
		# 	ax2 = fig2.add_subplot(111)
		# 	ax2.cla()
		# 	ax2.get_xaxis().set_visible(True)
		# 	ax2.get_yaxis().set_visible(True)
		# 	ax2.set_xlim([self.MIN,self.MAX])
		# 	ax2.set_ylim(bottom=-0.2,top=0.4)
		# 	ax2.scatter(self.reals_pos[:10000], np.zeros_like(self.reals_pos[:10000]), linewidth = 1.5, c='r',label='Real Data', marker = '.')
		# 	ax2.scatter(self.reals_neg[:10000], np.zeros_like(self.reals_neg[:10000]), linewidth = 1.5, alpha = 1, c='b',label='Negative Data', marker = '.')
		# 	ax2.scatter(images[:10000], np.zeros_like(images[:10000]), c='g', linewidth = 1.5, alpha = 1, label='Fake Data', marker = '.')
		# 	# ax2.plot(basis,disc, alpha = 0.5,  c='b',label='Discriminator')
		# 	ax2.legend(loc = 'upper right')
		# 	fig2.tight_layout()
		# 	pdf.savefig(fig2)
		# 	plt.close(fig2)

		# if show:
		# 	plt.show()
		# else:
		# 	plt.close()

	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_model = 'self.generator_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data_pos, self.train_data_neg, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		''' Define dataset and tf.data function. batch sizing done'''
		self.get_data()
		print(" Batch Size {}, Batch Size * Dloop {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, self.batch_size_big, self.num_batches,self.print_step, self.save_step))

	def get_data(self):
		
		with tf.device('/CPU'):
			self.train_data_pos, self.train_data_neg = eval(self.gen_func)

			if self.data in [ 'wce', 'kid']:
				self.max_data_size = 4*max(self.train_data_pos.shape[0],self.train_data_neg.shape[0])
			else:
				self.max_data_size = max(self.train_data_pos.shape[0],self.train_data_neg.shape[0])

			self.batch_size_big = tf.constant(self.batch_size*self.Dloop,dtype='int64')
			self.num_batches = int(np.floor(self.max_data_size/self.batch_size))
			''' Set PRINT and SAVE iters if 0'''
			self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
			self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

			self.train_dataset_pos, self.train_dataset_neg = eval(self.dataset_func)

			self.train_dataset_size = self.max_data_size


'''***********************************************************************************
********** GAN AAE setup *************************************************************
***********************************************************************************'''
class GAN_WAE(GAN_ARCH, GAN_DATA_WAE):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_ARCH class - defines all GAN architectures'''

		# self.lr_AE_Enc = FLAGS_dict['lr_AE_Enc']
		# self.lr_AE_Dec = FLAGS_dict['lr_AE_Dec']
		# self.AE_count = FLAGS_dict['AE_count']

		GAN_ARCH.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_WAE.__init__(self)
		# eval('GAN_DATA_'+FLAGS.topic+'.__init__(self,data)')

		self.noise_setup()

	def noise_setup(self):
		self.num_of_components = 20

		probs = list((1/self.num_of_components)*np.ones([self.num_of_components]))
		stddev_scale = list(0.8*np.ones([self.num_of_components]))
		# locs = list(np.random.uniform(size = [10, self.latent_dims], low = 1., high = 8.))
		locs = np.random.uniform(size = [self.num_of_components, self.latent_dims], low = -3., high = 3.)
		self.locs = tf.Variable(locs)
		locs = [list(x) for x in list(locs)]
		
		# print(locs)       #[[7.5, 5], [5, 7.5], [2.5,5], [5,2.5], [7.5*0.7071, 7.5*0.7071], [2.5*0.7071, 7.5*0.7071], [7.5*0.7071, 2.5*0.7071], [2.5*0.7071, 2.5*0.7071] ]
		# stddev_scale = [.5, .5, .5, .5, .5, .5, .5, .5, .5, .5]

		# self.gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=stddev_scale))

		# probs = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
		# locs = [[0.75, 0.5], [0.5, 0.75], [0.25,0.5], [0.5,0.25], [0.5*1.7071, 0.5*1.7071], [0.5*0.2929, 0.5*1.7071], [0.5*1.7071, 0.5*0.2929], [0.5*0.2929, 0.5*0.2929] ]
		# stddev_scale = [.04, .04, .04, .04, .04, .04, .04, .04]
		self.gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=stddev_scale))

		self.gN = tfd.Normal(loc=1.25, scale=1.)

	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.encdec_model = 'self.encdec_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		if self.loss == 'FS':
			self.disc_model = 'self.discriminator_model_FS()' 
			self.DEQ_func = 'self.discriminator_ODE()'

		''' Define dataset and tf.data function. batch sizing done'''
		self.get_data()
		print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, self.num_batches,self.print_step, self.save_step))

	def find_sharpness(self,input_ims):
		def laplacian(input, ksize, mode=None, constant_values=None, name=None):
			"""
			Apply Laplacian filter to image.
			Args:
			  input: A 4-D (`[N, H, W, C]`) Tensor.
			  ksize: A scalar Tensor. Kernel size.
			  mode: A `string`. One of "CONSTANT", "REFLECT", or "SYMMETRIC"
			    (case-insensitive). Default "CONSTANT".
			  constant_values: A `scalar`, the pad value to use in "CONSTANT"
			    padding mode. Must be same type as input. Default 0.
			  name: A name for the operation (optional).
			Returns:
			  A 4-D (`[N, H, W, C]`) Tensor.
			"""

			input = tf.convert_to_tensor(input)
			ksize = tf.convert_to_tensor(ksize)

			tf.debugging.assert_none_equal(tf.math.mod(ksize, 2), 0)

			ksize = tf.broadcast_to(ksize, [2])

			total = ksize[0] * ksize[1]
			index = tf.reshape(tf.range(total), ksize)
			g = tf.where(
		    	tf.math.equal(index, tf.math.floordiv(total - 1, 2)),
		    	tf.cast(1 - total, input.dtype),
		    	tf.cast(1, input.dtype),
			)

			# print(g)

			# input = pad(input, ksize, mode, constant_values)

			channel = tf.shape(input)[-1]
			shape = tf.concat([ksize, tf.constant([1, 1], ksize.dtype)], axis=0)
			g = tf.reshape(g, shape)
			shape = tf.concat([ksize, [channel], tf.constant([1], ksize.dtype)], axis=0)
			g = tf.broadcast_to(g, shape)
			return tf.nn.depthwise_conv2d(input, g, [1, 1, 1, 1], padding="VALID")

		import tensorflow_io as tfio
		lap_img = laplacian(input_ims,3)
		if input_ims.shape[3] == 3:
			reduction_axis = [1,2,3]
		else:
			reduction_axis = [1,2]
		var = tf.square(tf.math.reduce_std(lap_img, axis = reduction_axis))
		var_out = np.mean(var)
		# print(var_out)
		return var_out

	def get_data(self):
		# with tf.device('/CPU'):
		self.train_data = eval(self.gen_func)

		# self.batch_size_big = tf.constant(self.batch_size*self.Dloop,dtype='int64')
		self.num_batches = int(np.floor((self.train_data.shape[0] * self.reps)/self.batch_size))
		''' Set PRINT and SAVE iters if 0'''
		self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
		self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

		self.train_dataset = eval(self.dataset_func)
		self.train_dataset_size = self.train_data.shape[0]


	def get_noise(self,batch_size):
		###Uncomment for the continues CelebaCode on Vega
		if self.noise_kind == 'gaussian_trunc':
			noise = tfp.distributions.TruncatedNormal(loc=0., scale=0.3, low=-1., high=1.).sample([batch_size, self.latent_dims])

		###Uncomment for the continues CIFAR10Code on Vayu
		if self.noise_kind == 'gmm':
			noise = self.gmm.sample(sample_shape=(int(batch_size.numpy())))

		if self.noise_kind == 'gN':
			noise = self.gN.sample(sample_shape=(int(batch_size.numpy()),self.latent_dims))

		# noise = tfp.distributions.TruncatedNormal(loc=1.25, scale=0.75, low=0., high=3.).sample([batch_size, self.latent_dims])

		# tf.random.normal([100, self.latent_dims], mean = self.locs.numpy()[i], stddev = 1.)
		if self.noise_kind == 'gaussian':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.0, stddev = 1.0)

		if self.noise_kind == 'gaussian_s2':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.0, stddev = np.sqrt(2))

		if self.noise_kind == 'gaussian_s4':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.0, stddev = 2)

		if self.noise_kind == 'gaussian_1m1':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.0, stddev = 0.25)

		if self.noise_kind == 'gaussian_05':
			noise = tfp.distributions.TruncatedNormal(loc=2.5, scale=1., low=0., high=5.).sample([batch_size, self.latent_dims])

		if self.noise_kind == 'gaussian_02':
			noise = tf.random.normal([batch_size, self.latent_dims], mean = 0.7*np.ones((1,self.latent_dims)), stddev = 0.2*np.ones((1,self.latent_dims)))

		if self.noise_kind == 'gaussian_01':
			noise = tfp.distributions.TruncatedNormal(loc=0.5, scale=0.2, low=0., high=1.).sample([batch_size, self.latent_dims])



		return noise



'''***********************************************************************************
********** GAN RumiGAN setup *********************************************************
***********************************************************************************'''
class GAN_ImNoise2Im(GAN_ARCH, GAN_DATA_ImNoise2Im):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_ARCH class - defines all GAN architectures'''
		GAN_ARCH.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_ImNoise2Im.__init__(self)

	def initial_setup(self):
		''' Initial Setup function. define function names '''
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.gen_func_noise = 'self.gen_func_'+self.noise_data+'()'
		self.gen_model = 'self.generator_model_'+self.data+'()'
		self.disc_model = 'self.discriminator_model_'+self.data+'()' 
		self.loss_func = 'self.loss_'+self.loss+'()'   
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data, self.batch_size, reps_train)'
		self.dataset_func_noise = 'self.dataset_'+self.noise_data+'(self.train_data_noise, self.batch_size, reps_noise)'
		self.show_result_func = 'self.show_result_'+self.data+'(images = predictions, num_epoch=epoch, show = False, save = True, path = path)'
		self.FID_func = 'self.FID_'+self.data+'()'

		''' Define dataset and tf.data function. batch sizing done'''
		self.get_data()
		print(" Batch Size {}, Final Num Batches {}, Print Step {}, Save Step {}".format(self.batch_size, self.num_batches,self.print_step, self.save_step))

	def get_data(self):
		
		with tf.device('/CPU'):
			self.train_data = eval(self.gen_func)
			self.train_data_noise = eval(self.gen_func_noise)
			self.ratio = self.train_data.shape[0]/self.train_data_noise.shape[0] # is the num of reps noise data needs, to match train data
			reps_train = np.ceil(1/float(self.ratio))
			reps_noise = np.ceil(self.ratio)
			print("reps_train",reps_train)
			print("reps_noise",reps_noise)

			# if ratio < 1 :
			# 	reps_train = np.ceil(1/float(self.ratio))
			# if ratio >= 1 else:
			# 	reps = np.ceil(self.ratio) 

			# self.batch_size_big = tf.constant(self.batch_size*self.Dloop,dtype='int64')
			self.max_data_size = max(self.train_data.shape[0],self.train_data_noise.shape[0])
			self.num_batches = int(np.floor(self.max_data_size/self.batch_size))
			''' Set PRINT and SAVE iters if 0'''
			self.print_step = tf.constant(max(int(self.num_batches/10),1),dtype='int64')
			self.save_step = tf.constant(max(int(self.num_batches/2),1),dtype='int64')

			self.train_dataset = eval(self.dataset_func)
			self.noise_dataset = eval(self.dataset_func_noise)

			self.train_dataset_size = self.max_data_size
