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
from gan_src import *

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
class GAN_Base(GAN_SRC, GAN_DATA_Base):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all GAN architectures'''

		GAN_SRC.__init__(self,FLAGS_dict)

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
class GAN_ELeGANt(GAN_SRC, GAN_DATA_Base, FourierSolver):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all GAN architectures'''

		GAN_SRC.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_Base.__init__(self)
		# eval('GAN_DATA_'+FLAGS.topic+'.__init__(self,data)')

		''' Set up the Fourier Series Solver common to WAEFR and WGAN-FS'''
		FourierSolver.__init__(self)


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
			# self.gen_model = 'self.generator_model_'+self.data+'_'+self.latent_kind+'()'
			# self.disc_model = 'self.discriminator_model_'+self.data+'_'+self.latent_kind+'()' 
			# self.EncDec_func = 'self.encoder_model_'+self.data+'_'+self.latent_kind+'()'
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
class GAN_WAE(GAN_SRC, GAN_DATA_WAE, FourierSolver):

	def __init__(self,FLAGS_dict):
		''' Set up the GAN_SRC class - defines all GAN architectures'''

		GAN_SRC.__init__(self,FLAGS_dict)

		''' Set up the GAN_DATA class'''
		GAN_DATA_WAE.__init__(self)
		# eval('GAN_DATA_'+FLAGS.topic+'.__init__(self,data)')

		''' Set up the Fourier Series Solver common to WAEFR and WGAN-FS'''
		FourierSolver.__init__(self)

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


class FourierSolver():

	def __init__(self):
		self.M = self.terms #Number of terms in FS
		self.T = self.sigma
		self.W = np.pi/self.T
		self.W0 = 1/self.T

		## For 1-D and 2-D Gaussians, no latent projections are needed. So latent dims are full dims itself.
		if self.data in ['g1']:
			self.latent_dims = 1
		if self.data in ['g2', 'gmm8']:
			self.latent_dims = 2
		self.N = self.latent_dims

		''' If M is small, take all terms in FS expanse, else, a sample few of them '''
		if self.N <= 3:
			num_terms = list(np.arange(1,self.M+1))
			self.L = ((self.M)**self.N)
			print(num_terms) # nvec = Latent x Num_terms^latent
			self.n_vec = tf.cast(np.array([p for p in cart_prod(num_terms,repeat = self.N)]).transpose(), dtype = 'float32') # self.N x self.L lengthmatrix, each column is a desired N_vec to use
		else:
			# self.L = L#50000# + self.N + 1
			with tf.device(self.device):
				'''need to do poisson disc sampling'''  #temp is self.M^self.N here
				temp = self.latent_dims
				vec1 = np.concatenate((np.ones([temp, 1]), np.concatenate(tuple([np.ones([temp,temp]) + k*np.eye(temp) for k in range(1,self.M+1)]),axis = 1)), axis = 1)
				print("VEC1",vec1)
				# vec2 = tf.cast(tf.random.uniform((temp,self.L),minval = 1, maxval = self.M, dtype = 'int32'),dtype='float32')
				vec2_basis = np.random.choice(self.M-1,self.L) + 1
				vec2 = np.concatenate(tuple([np.expand_dims(np.roll(vec2_basis,k),axis=0) for k in range(temp)]), axis = 0)
				print("VEC2",vec2)
				# self.n_vec = tf.cast(np.concatenate((vec1,vec2.numpy()), axis = 1),dtype='float32')
				self.n_vec = tf.cast(np.concatenate((vec1,vec2), axis = 1),dtype='float32')
				self.L += self.M*temp + 1
				print("NVEC",self.n_vec)


		with tf.device(self.device):
			print(self.n_vec, self.W)
			self.Coeffs = tf.multiply(self.n_vec, self.W)
			print(self.Coeffs)
			self.n_norm = tf.expand_dims(tf.square(tf.linalg.norm(tf.transpose(self.n_vec), axis = 1)), axis=1)
			self.bias = np.array([0.])


		## Target data is for evaluateion of alphas (Check Main Manuscript)
		## Generator data is for evaluateion of betas (Check Main Manuscript)
		if self.gan == 'WGAN':
			self.target_data = 'self.reals_enc'
			self.generator_data = 'self.fakes_enc'
		elif: self.gan = 'WAE':
			self.traget_data = 'self.fakes_enc'
			self.generator_data = 'self.reals_enc'
		return

	def discriminator_model_FS_A(self):
		inputs = tf.keras.Input(shape=(self.latent_dims,)) #used to be self.N

		w0_nt_x = tf.keras.layers.Dense(self.L, activation=None, use_bias = False)(inputs)
		w0_nt_x2 = tf.math.scalar_mul(2., w0_nt_x)

		cos_terms = tf.keras.layers.Activation( activation = tf.math.cos)(w0_nt_x)
		sin_terms = tf.keras.layers.Activation( activation = tf.math.sin)(w0_nt_x)
		cos2_terms  = tf.keras.layers.Activation( activation = tf.math.cos)(w0_nt_x2)

		model = tf.keras.Model(inputs=inputs, outputs= [inputs, cos_terms, sin_terms, cos2_terms])
		return model

	def discriminator_model_FS_B(self):
		inputs = tf.keras.Input(shape=(self.latent_dims,))
		cos_terms = tf.keras.Input(shape=(self.L,)) #used to be self.N
		sin_terms = tf.keras.Input(shape=(self.L,))
		cos2_terms = tf.keras.Input(shape=(self.L,))

		cos_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(cos_terms)
		sin_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(sin_terms)

		cos2_c_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(cos2_terms) #Tau_c weights
		cos2_s_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(cos2_terms) #Tau_s weights

		lambda_x_term = tf.keras.layers.Subtract()([cos2_s_sum, cos2_c_sum]) #(tau_s  - tau_r)

		if self.latent_dims == 1:
			phi0_x = inputs
		else:
			phi0_x = tf.divide(tf.reduce_sum(inputs,axis=-1,keepdims=True),self.latent_dims)

		if self.homo_flag:
			Out = tf.keras.layers.Add()([cos_sum, sin_sum, phi0_x])
		else:
			Out = tf.keras.layers.Add()([cos_sum, sin_sum])

		model = tf.keras.Model(inputs= [inputs, cos_terms, sin_terms, cos2_terms], outputs=[Out,lambda_x_term])
		return model

	def Fourier_Series_Comp(self,f):

		mu = tf.convert_to_tensor(np.expand_dims(np.mean(f,axis = 0),axis=1), dtype = 'float32')
		cov = tf.convert_to_tensor(np.cov(f,rowvar = False), dtype = 'float32')
		# print(self.reals.shape,self.fakes.shape)
		# self.T = tf.convert_to_tensor(2*max(np.mean(self.reals_enc), np.mean(self.fakes_enc)), dtype = 'float32')
		# # print("T",self.T)
		# self.W = 2*np.pi/self.T
		# self.freq = 1/self.T
		# self.Coeffs = tf.multiply(self.n_vec, self.W)
		# self.coefficients.set_weights([self.Coeffs, self.Coeffs])

		with tf.device(self.device):
			if self.distribution == 'generic':
				_, ar, ai, _ = self.discriminator_A(f, training = False)
				ar = tf.expand_dims(tf.reduce_mean(ar, axis = 0), axis = 1)#Lx1 vector
				ai = tf.expand_dims(tf.reduce_mean(ai, axis = 0), axis = 1)#Lx1 vector

				## Error calc between true and estimate values. Only for Sanity Check
				if self.data != 'g1':
					nt_mu = tf.linalg.matmul(tf.transpose(self.n_vec),mu)
					nt_cov_n =  tf.expand_dims(tf.linalg.tensor_diag_part(tf.linalg.matmul(tf.transpose(self.n_vec),tf.linalg.matmul(cov,self.n_vec))), axis=1)
				else:
					nt_mu = mu*self.n_vec
					nt_cov_n = cov * tf.expand_dims(tf.linalg.tensor_diag_part(tf.linalg.matmul(tf.transpose(self.n_vec),self.n_vec)), axis=1)
				#### FIX POWER OF T
				#tf.constant((1/(self.T))**1)
				ar_true =  tf.constant((1/(self.T))**0) * tf.math.exp(-0.5 * tf.multiply(nt_cov_n, self.W**2))*(tf.math.cos(tf.multiply(nt_mu, self.W)))
				ai_true =  tf.constant((1/(self.T))**0) * tf.math.exp(-0.5 * tf.multiply(nt_cov_n, self.W**2 ))*(tf.math.sin(tf.multiply(nt_mu, self.W)))

				error = tf.reduce_mean(tf.abs(ar-ar_true)) + tf.reduce_mean(tf.abs(ai-ai_true))
				# self.lambda_vec.append(np.log(error.numpy()))


			if self.distribution == 'gaussian':
				if self.data != 'g1':
					nt_mu = tf.linalg.matmul(tf.transpose(self.n_vec),mu)
					nt_cov_n =  tf.expand_dims(tf.linalg.tensor_diag_part(tf.linalg.matmul(tf.transpose(self.n_vec),tf.linalg.matmul(cov,self.n_vec))), axis=1)
				else:
					nt_mu = mu*tf.transpose(self.n_vec)
					nt_cov_n = cov * tf.expand_dims(tf.linalg.tensor_diag_part(tf.linalg.matmul(tf.transpose(self.n_vec,[1,0]),self.n_vec)), axis=1)

				ar =  tf.constant((1/(self.T))**0) * tf.math.exp(-0.5 * tf.multiply(nt_cov_n, self.W**2))*(tf.math.cos(tf.multiply(nt_mu, self.W)))
				ai =  tf.constant((1/(self.T))**0) * tf.math.exp(-0.5 * tf.multiply(nt_cov_n, self.W**2 ))*(tf.math.sin(tf.multiply(nt_mu, self.W)))
				# print(ar,ai)
			if self.distribution == 'uniform':
				#DEPRICATED - UNIFORM IS A BAD IDEA
				a_vec = tf.expand_dims(tf.reduce_min(f, axis = 0),0)
				b_vec = tf.expand_dims(tf.reduce_max(f, axis = 0),0)
				nt_a = tf.transpose(tf.linalg.matmul(a_vec, self.n_vec),[1,0])
				nt_b = tf.transpose(tf.linalg.matmul(b_vec, self.n_vec),[1,0])
				nt_bma = tf.transpose(tf.linalg.matmul(b_vec - a_vec, self.n_vec),[1,0])

				# tf.constant((1/(self.T))**1)
				ar =  1 * tf.divide(tf.math.sin(tf.multiply(nt_b ,self.W)) - tf.math.sin(tf.multiply(nt_a ,self.W)), tf.multiply(nt_bma,self.W))
				ai = - 1 * tf.divide(tf.math.cos(tf.multiply(nt_b ,self.W)) - tf.math.cos(tf.multiply(nt_a ,self.W)), tf.multiply(nt_bma,self.W))
			
		return  ar, ai

	def discriminator_ODE(self): 
		self.alpha_c, self.alpha_s = self.Fourier_Series_Comp(eval(self.traget_data))
		self.beta_c, self.beta_s = self.Fourier_Series_Comp(eval(self.generator_data))

		with tf.device(self.device):
			# Vec of len Lx1 , wach entry is ||n||
			self.Gamma_s = tf.math.divide(tf.constant(1/(self.W**2))*tf.subtract(self.alpha_s, self.beta_s), self.n_norm)
			self.Gamma_c = tf.math.divide(tf.constant(1/(self.W**2))*tf.subtract(self.alpha_c, self.beta_c), self.n_norm)
			self.Tau_s = tf.math.divide(tf.constant(1/(2.*(self.W**2)))*tf.square(tf.subtract(self.alpha_s, self.beta_s)), self.n_norm)
			self.Tau_c = tf.math.divide(tf.constant(1/(2.*(self.W**2)))*tf.square(tf.subtract(self.alpha_c, self.beta_c)), self.n_norm)
			self.sum_Tau = 1.*tf.reduce_sum(tf.add(self.Tau_s,self.Tau_c))

	def find_and_divide_lambda(self):
		self.lamb = tf.divide(tf.reduce_sum(self.lambda_x_terms_2) + tf.reduce_sum(self.lambda_x_terms_1),tf.cast(self.batch_size, dtype = 'float32')) + self.sum_Tau
		self.lamb = tf.cast(2*self.L, dtype = 'float32')*self.lamb
		self.lamb = tf.sqrt(self.lamb)
		self.real_output = tf.divide(self.real_output, self.lamb)
		self.fake_output = tf.divide(self.fake_output, self.lamb)

