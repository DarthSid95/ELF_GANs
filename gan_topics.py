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
********** GAN AAE setup *************************************************************
***********************************************************************************'''
class GAN_WAE(GAN_ARCH, GAN_DATA_WAE):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_ARCH class - defines all GAN architectures'''

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

	def test(self):
		###### Random Saples
		for i in range(10):
			path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			noise = self.get_noise(self.batch_size)
			images = self.Decoder(noise)
			

			sharpness = self.find_sharpness(images)
			try:
				sharpness_vec.append(sharpness)
			except:
				shaprpness_vec = [sharpness]

			images = (images + 1.0)/2.0
			size_figure_grid = 10
			images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(7,7))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			# fig.text(0.5, 0.04, label, ha='center')
			plt.savefig(path)
			plt.close()

		###### Random Samples - Sharpness
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("Random Sharpness - " + str(overall_sharpness))
			if self.res_flag:
				self.res_file.write("Random Sharpness - "+str(overall_sharpness))
		else:
			if self.res_flag:
				self.res_file.write("Random Sharpness - "+str(overall_sharpness))

		i = 0
		for image_batch in self.train_dataset:
			i+=1
			sharpness = self.find_sharpness(image_batch)
			try:
				sharpness_vec.append(sharpness)
			except:
				shaprpness_vec = [sharpness]
			if i==100:
				break

		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("Dataset Sharpness 10k samples - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("Dataset Sharpness 10k samples - "+str(overall_sharpness))


		# ####### Recon - Output
		for image_batch in self.recon_dataset:				
			path = self.impath+'_TestingRecon_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			images = self.Decoder(self.Encoder(image_batch))
			images = (images + 1.0)/2.0
			size_figure_grid = 10
			images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(7,7))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()

		# ###### Recon - org
			path = self.impath+'_TestingReconOrg_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
			images = image_batch
			images = (images + 1.0)/2.0
			size_figure_grid = 10
			images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
			fig1 = plt.figure(figsize=(7,7))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.axis("off")
			if images_on_grid.shape[2] == 3:
				ax1.imshow(np.clip(images_on_grid,0.,1.))
			else:
				ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

			label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
			plt.title(label)
			plt.tight_layout()
			plt.savefig(path)
			plt.close()
			break

		####### Interpolation
		num_interps = 10
		if self.mode == 'test':
			num_figs = int(400/(2*num_interps))
		else:
			num_figs = 9
		# there are 400 samples in the batch. to make 10x10 images, 
		for image_batch in self.interp_dataset:
			for j in range(num_figs):
				path = self.impath+'_TestingInterpolationV2_'+str(self.total_count.numpy())+'_TestCase_'+str(j)+'.png'
				current_batch = image_batch[2*num_interps*j:2*num_interps*(j+1)]
				image_latents = self.Encoder(current_batch)
				for i in range(num_interps):
					start = image_latents[i:1+i].numpy()
					end = image_latents[num_interps+i:num_interps+1+i].numpy()
					stack = np.vstack([start, end])

					linfit = interp1d([1,num_interps+1], stack, axis=0)
					interp_latents = linfit(list(range(1,num_interps+1)))
					cur_interp_figs = self.Decoder(interp_latents)

					sharpness = self.find_sharpness(cur_interp_figs)

					try:
						sharpness_vec.append(sharpness)
					except:
						shaprpness_vec = [sharpness]
					cur_interp_figs_with_ref = np.concatenate((current_batch[i:1+i],cur_interp_figs.numpy(),current_batch[num_interps+i:num_interps+1+i]), axis = 0)
					# print(cur_interp_figs_with_ref.shape)
					try:
						batch_interp_figs = np.concatenate((batch_interp_figs,cur_interp_figs_with_ref),axis = 0)
					except:
						batch_interp_figs = cur_interp_figs_with_ref

				images = (batch_interp_figs + 1.0)/2.0
				# print(images.shape)
				size_figure_grid = num_interps
				images_on_grid = self.image_grid(input_tensor = images, grid_shape = (size_figure_grid,size_figure_grid+2),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
				fig1 = plt.figure(figsize=(num_interps,num_interps+2))
				ax1 = fig1.add_subplot(111)
				ax1.cla()
				ax1.axis("off")
				if images_on_grid.shape[2] == 3:
					ax1.imshow(np.clip(images_on_grid,0.,1.))
				else:
					ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

				label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
				plt.title(label)
				plt.tight_layout()
				plt.savefig(path)
				plt.close()
				del batch_interp_figs

		###### Interpol samples - Sharpness
		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("Interpolation Sharpness - " + str(overall_sharpness))
		if self.res_flag:
			self.res_file.write("Interpolation Sharpness - "+str(overall_sharpness))



