from __future__ import print_function
import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)
import tensorflow_probability as tfp
from matplotlib.backends.backend_pgf import PdfPages
tfd = tfp.distributions

import matplotlib.pyplot as plt
import math
import tensorflow as tf
from absl import app
from absl import flags

from gan_topics import *
# tf.keras.backend.set_floatx('float64')


'''***********************************************************************************
********** Baseline WGANs ************************************************************
***********************************************************************************'''
class WGAN_Base(GAN_Base):

	def __init__(self,FLAGS_dict):

		# self.KLD_flag = KLD_flag
		# self.KLD = []
		GAN_Base.__init__(self,FLAGS_dict)

		self.lambda_GP = 0.1 #100 for normal data, 0.1 for synth
		self.lambda_ALP = 10.0 #100 for normal data, 0.1 for synth
		self.lambda_LP = 0.1 #10 for normal? 0.1 for synth

	#################################################################
	
	def main_func(self):
		with tf.device(self.device):
			self.total_count = tf.Variable(0,dtype='int64')
			self.generator = eval(self.gen_model)
			self.discriminator = eval(self.disc_model)
			##### Added for ICML rebuttal
			# x = self.EncDec_model_mnist()

			print("Model Successfully made")

			print(self.generator.summary())
			print(self.discriminator.summary())

			if self.res_flag == 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n GENERATOR MODEL: \n\n")
					self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
					self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))


			
			if self.loss == 'GP' :
				self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=200, decay_rate=0.9, staircase=True)
				self.G_optimizer = tf.keras.optimizers.Adam(self.lr_schedule, self.beta1, self.beta2)
			elif self.loss == 'ALP' :
				self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=100, decay_rate=0.9, staircase=True)
				self.G_optimizer = tf.keras.optimizers.Adam(self.lr_schedule, self.beta1, self.beta2)
			else:
				self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G, self.beta1, self.beta2)
			self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D, self.beta1, self.beta2)


			print("Optimizers Successfully made")		

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer,
								 Disc_optimizer = self.Disc_optimizer,
								 generator = self.generator,
								 discriminator = self.discriminator,
								 total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			# self.total_count = int(temp.split('-')[-1])
			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			# self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
			# self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size_big) / (self.train_data.shape[0])) + 1))

	#################################################################

	def train(self):    
		start = int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1 
		for epoch in range(start,self.num_epochs):
			if self.pbar_flag:
				bar = self.pbar(epoch)
			start = time.time()
			batch_count = tf.Variable(0,dtype='int64')
			start_1 = 0

			for image_batch in self.train_dataset:
				self.total_count.assign_add(1)
				# batch_count.assign_add(self.Dloop)
				batch_count.assign_add(1)
				start_1 = time.time()
				
				with tf.device(self.device):
					self.train_step(image_batch)
					self.eval_metrics()
				
				train_time = time.time()-start_1

					
				if self.pbar_flag:
					bar.postfix[0] = f'{batch_count.numpy():6.0f}'
					bar.postfix[1] = f'{self.D_loss.numpy():2.4e}'
					bar.postfix[2] = f'{self.G_loss.numpy():2.4e}'
					bar.update(self.batch_size.numpy())
				if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:
					if self.res_flag:
						self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f} \n".format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy()))
					

				self.print_batch_outputs(epoch)

				# Save the model every SAVE_ITERS iterations
				if (self.total_count.numpy() % self.save_step.numpy()) == 0:
					if self.save_all:
						self.checkpoint.save(file_prefix = self.checkpoint_prefix)
					else:
						self.manager.save()

			if self.pbar_flag:
				bar.close()
				del bar

			tf.print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
			self.generator.save(self.checkpoint_dir + '/model_generator.h5', overwrite = True)
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)

		# if self.KLD_flag:
		# 	self.printKLD()

	def print_batch_outputs(self,epoch):		
		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % 100) == 0 and self.data in ['g1', 'g2']:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)

	#################################################################

	def test(self):
		self.impath += '_Testing_'
		for img_batch in self.train_dataset:
			self.reals = img_batch
			self.generate_and_save_batch(0)
			return

	#################################################################

	def train_step(self,reals_all):
		for i in tf.range(self.Dloop):
			with tf.device(self.device):
				noise = tf.random.normal([self.batch_size, self.noise_dims], mean = self.noise_mean, stddev = self.noise_stddev)
			self.reals = reals_all

			with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
				
				self.fakes = self.generator(noise, training=True)

				self.real_output = self.discriminator(self.reals, training = True)
				self.fake_output = self.discriminator(self.fakes, training = True)
				eval(self.loss_func)

			self.D_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
			self.Disc_optimizer.apply_gradients(zip(self.D_grads, self.discriminator.trainable_variables))

			if self.loss == 'base':
				wt = []
				for w in self.discriminator.get_weights():
					w = tf.clip_by_value(w, -0.1,0.1) #0.01 for [0,1] data, 0.1 for [0,10]
					wt.append(w)
				self.discriminator.set_weights(wt)
			if i >= (self.Dloop - self.Gloop):
				self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
				self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


	#################################################################

	def loss_base(self):

		loss_fake = tf.reduce_mean(self.fake_output)

		loss_real = tf.reduce_mean(self.real_output) 

		self.D_loss = 1 * (-loss_real + loss_fake)

		self.G_loss = 1 * (loss_real - loss_fake)

	#################################################################

	def loss_GP(self):

		loss_fake = tf.reduce_mean(self.fake_output)

		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_GP * self.gp 

		self.G_loss = 1 * (loss_real - loss_fake)

	def gradient_penalty(self):
		if self.data in ['g1', 'g2', 'gmm2', 'gmm8', 'u1']:
			alpha = tf.random.uniform([self.batch_size, 1], 0., 1.)
		else:
			alpha = tf.random.uniform([self.batch_size, 1, 1, 1], 0., 1.)
		diff = tf.cast(self.fakes,dtype='float32') - tf.cast(self.reals,dtype='float32')
		inter = tf.cast(self.reals,dtype='float32') + (alpha * diff)
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])[0]
		if self.data in ['g1', 'g2', 'gmm2', 'gmm8', 'u1']:
			slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
		else:
			slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
		self.gp = tf.reduce_mean((slopes - 1.)**2)
		return 


	#################################################################

	def loss_LP(self):

		loss_fake = tf.reduce_mean(self.fake_output)

		loss_real = tf.reduce_mean(self.real_output)  

		self.lipschitz_penalty()

		self.D_loss = -loss_real + loss_fake + self.lambda_LP * self.lp 

		self.G_loss = loss_real - loss_fake

	def lipschitz_penalty(self):
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py

		self.K = 1
		self.p = 2

		if self.data in ['g1', 'g2', 'gmm2', 'gmm8', 'u1']:
			epsilon = tf.random.uniform([tf.shape(self.reals)[0], 1], 0.0, 1.0)
		else:
			epsilon = tf.random.uniform([tf.shape(self.reals)[0], 1, 1, 1], 0.0, 1.0)
		x_hat = epsilon * self.fakes + (1 - epsilon) * self.reals

		with tf.GradientTape() as t:
			t.watch(x_hat)
			D_vals = self.discriminator(x_hat, training = False)
		grad_vals = t.gradient(D_vals, [x_hat])[0]

		#### args.p taken from github as default p=2
		dual_p = 1 / (1 - 1 / self.p) if self.p != 1 else np.inf

		#gradient_norms = stable_norm(gradients, ord=dual_p)
		grad_norms = tf.norm(grad_vals, ord=dual_p, axis=1, keepdims=True)

		#### Default K = 1
		# lp = tf.maximum(gradient_norms - args.K, 0)
		self.lp = tf.reduce_mean(tf.maximum(grad_norms - self.K, 0)**2)
		# lp_loss = args.lambda_lp * reduce_fn(lp ** 2)

	#################################################################

	def loss_ALP(self):
		
		loss_fake = tf.reduce_mean(self.fake_output)

		loss_real = tf.reduce_mean(self.real_output)  

		self.adversarial_lipschitz_penalty()

		self.D_loss = 1 * (-loss_real + loss_fake) + self.lambda_ALP * self.alp 

		self.G_loss = 1 * (loss_real - loss_fake)


	def adversarial_lipschitz_penalty(self):
		def normalize(x, ord):
			return x / tf.maximum(tf.norm(x, ord=ord, axis=1, keepdims=True), 1e-10)
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py
		self.eps_min = 0.1
		self.eps_max = 10.0
		self.xi = 10.0
		self.ip = 1
		self.p = 2
		self.K = 5 #was 1. made 5 for G2 compares

		samples = tf.concat([self.reals, self.fakes], axis=0)
		if self.data in ['g1', 'g2', 'gmm2', 'gmm8', 'u1']:
			noise = tf.random.uniform([tf.shape(samples)[0], 1], 0, 1)
		else:
			noise = tf.random.uniform([tf.shape(samples)[0], 1, 1, 1], 0, 1)

		eps = self.eps_min + (self.eps_max - self.eps_min) * noise

		with tf.GradientTape(persistent = True) as t:
			t.watch(samples)
			validity = self.discriminator(samples, training = False)

			d = tf.random.uniform(tf.shape(samples), 0, 1) - 0.5
			d = normalize(d, ord=2)
			t.watch(d)
			for _ in range(self.ip):
				samples_hat = tf.clip_by_value(samples + self.xi * d, clip_value_min=-1, clip_value_max=1)
				validity_hat = self.discriminator(samples_hat, training = False)
				dist = tf.reduce_mean(tf.abs(validity - validity_hat))
				grad = t.gradient(dist, [d])[0]
				# print(grad)
				d = normalize(grad, ord=2)
			r_adv = d * eps

		samples_hat = tf.clip_by_value(samples + r_adv, clip_value_min=-1, clip_value_max=1)

		d_lp                   = lambda x, x_hat: tf.norm(x - x_hat, ord=self.p, axis=1, keepdims=True)
		d_x                    = d_lp

		samples_diff = d_x(samples, samples_hat)
		samples_diff = tf.maximum(samples_diff, 1e-10)

		validity      = self.discriminator(samples    , training = False)
		validity_hat  = self.discriminator(samples_hat, training = False)
		validity_diff = tf.abs(validity - validity_hat)

		alp = tf.maximum(validity_diff / samples_diff - self.K, 0)
		# alp = tf.abs(validity_diff / samples_diff - args.K)

		nonzeros = tf.greater(alp, 0)
		count = tf.reduce_sum(tf.cast(nonzeros, tf.float32))

		self.alp = tf.reduce_mean(alp**2)
		# alp_loss = args.lambda_lp * reduce_fn(alp ** 2)

	#####################################################################

	def loss_AE(self):
		mse = tf.keras.losses.MeanSquaredError()
		loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))#mse(self.reals, self.reals_dec)
		self.AE_loss =  loss_AE_reals 


'''***********************************************************************************
********** WGAN ELEGANT WITH LATENT **************************************************
***********************************************************************************'''
''' nEED to CLEAN Generate and Save '''
class WGAN_ELeGANt(GAN_ELeGANt):

	def __init__(self,FLAGS_dict):

		from itertools import product as cart_prod

		self.lambda_GP = 10.
		GAN_ELeGANt.__init__(self,FLAGS_dict)

		if self.data in ['g1']:
			self.latent_dims = 1
		if self.data in ['g2', 'gmm8']:
			self.latent_dims = 2

		self.N = self.latent_dims

		self.postfix = {0: f'{0:3.0f}', 1: f'{0:2.4e}', 2: f'{0:2.4e}'}
		self.bar_format = '{n_fmt}/{total_fmt} |{bar}| {rate_fmt}  Batch: {postfix[0]} ETA: {remaining}  Elapsed Time: {elapsed}  D_Loss: {postfix[1]}  G_Loss: {postfix[2]}'




		self.M = self.terms #FLAGS_dict['terms'] #Number of terms in FS
		# self.T = 2*np.pi*sigmul*self.sigma
		# self.T = 1
		self.T = self.sigmul*self.sigma
		self.W = 1*np.pi/self.T
		self.freq = 1/self.T


		''' If M is small, take all terms in FS expanse, else, a sample few of them '''
		if self.N <= 3:
			num_terms = list(np.arange(1,self.M+1))
			self.L = ((self.M)**self.N)
			print(num_terms) # nvec = Latent x Num_terms^latent
			self.n_vec = tf.cast(np.array([p for p in cart_prod(num_terms,repeat = self.N)]).transpose(), dtype = 'float32') # self.N x self.L lengthmatrix, each column is a desired N_vec to use
			# self.L = self.n_vec.numpy().shape[1]
		else:
			self.L = L#50000# + self.N + 1
			with tf.device(self.device):
				'''need to do poisson disc sampling'''  #temp is self.M^self.N here
				if self.latent_kind != 'DCT' and self.latent_kind != 'AE3':
					temp = self.latent_dims
				elif self.latent_kind == 'DCT' or self.latent_kind == 'AE3':
					temp = self.latent_dims*self.latent_dims

				#Deterministically selected harmonics
				# num_terms = list(np.arange(1,3))
				vec1 = np.concatenate((np.ones([temp, 1]), np.concatenate(tuple([np.ones([temp,temp]) + k*np.eye(temp) for k in range(1,self.M+1)]),axis = 1)), axis = 1)
				# vec1 = np.concatenate((np.ones([self.N, 1]), np.concatenate(tuple([np.ones([self.N,self.N]) + k*np.eye(self.N) for k in range(1,self.M+1)]),axis = 1)), axis = 1)#tf.cast(np.array([p for p in cart_prod(num_terms,repeat = self.N)]).transpose(),dtype='float32')
				print("VEC1",vec1)
				# vec2 = tf.cast(tf.random.uniform((temp,self.L),minval = 1, maxval = self.M, dtype = 'int32'),dtype='float32')
				# dist = tfd.Beta(2,5)
				# vec2 = tf.cast(tf.math.ceil((self.M-1)*dist.sample([temp,self.L])),dtype='float32')
				# import sympy
				vec2_basis = np.random.choice(self.M-1,self.L) + 1
				# vec2_basis = np.arange(self.L)
				vec2 = np.concatenate(tuple([np.expand_dims(np.roll(vec2_basis,k),axis=0) for k in range(temp)]), axis = 0)



				print("VEC2",vec2)
				# self.n_vec = tf.cast(np.concatenate((vec1,vec2.numpy()), axis = 1),dtype='float32')
				self.n_vec = tf.cast(np.concatenate((vec1,vec2), axis = 1),dtype='float32')
				# self.n_vec = tf.cast(vec2, dtype = 'float32')
				self.L += self.M*temp + 1
				print("NVEC",self.n_vec)
				# print(self.n.shape,xxx)


		with tf.device(self.device):
			print(self.n_vec, self.W)
			self.Coeffs = tf.multiply(self.n_vec, self.W)
			print(self.Coeffs)
			self.n_norm = tf.expand_dims(tf.square(tf.linalg.norm(tf.transpose(self.n_vec), axis = 1)), axis=1)
			# print(xxx)



	def main_func(self):

		with tf.device(self.device):

			self.total_count = tf.Variable(0,dtype='int64')
			self.generator = eval(self.gen_model)
			self.discriminator_A = self.discriminator_model_FS_A()
			self.discriminator_A.set_weights([self.Coeffs])
			self.discriminator_B = self.discriminator_model_FS_B()
			
			print("Model Successfully made")

			#### FIX POWER OF 0.5
			self.bias = np.array([0])
			self.pdf = eval(self.disc_model)
			self.pgf = eval(self.disc_model)

			print("Model Successfully made")
			print("\n\n GENERATOR MODEL: \n\n")
			print(self.generator.summary())
			print("\n\n DISCRIMINATOR PART A MODEL: \n\n")
			print(self.discriminator_A.summary())
			print("\n\n DISCRIMINATOR PART B MODEL: \n\n")
			print(self.discriminator_B.summary())


			if self.res_flag == 1 and self.resume != 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n GENERATOR MODEL: \n\n")
					self.generator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR PART A MODEL: \n\n")
					self.discriminator_A.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR PART B MODEL: \n\n")
					self.discriminator_B.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))

			self.lr_G_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=50, decay_rate=1.1, staircase=True)
			self.G_optimizer = tf.keras.optimizers.Nadam(self.lr_G) #Nadam?

			print("Optimizers Successfully made")			

		self.checkpoint = tf.train.Checkpoint(G_optimizer = self.G_optimizer, \
							generator = self.generator, \
							discriminator_A = self.discriminator_A, \
							discriminator_B = self.discriminator_B, \
							total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			# self.total_count = int(temp.split('-')[-1])
			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			self.generator = tf.keras.models.load_model(self.checkpoint_dir+'/model_generator.h5')
			# self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
			print("Model restored...")
			print("Starting at Iteration "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size_big) / (self.train_data.shape[0]*self.reps)) + 1))


	def train(self):    
		start = int((self.total_count.numpy() * self.batch_size_big) / (self.train_data.shape[0]*self.reps)) + 1
		for epoch in range(start,self.num_epochs): 

			if self.pbar_flag:
				bar = self.pbar(epoch)  
			start = time.time()
			batch_count = tf.Variable(0,dtype='int64')
			start_1 = 0
			for image_batch in self.train_dataset:

				self.total_count.assign_add(1)
				batch_count.assign_add(1)
				start_1 = time.time()
				with tf.device(self.device):
					# eval('self.train_step_'+self.latent_kind+'(image_batch)')
					self.train_step(image_batch)
					self.eval_metrics()


				train_time = time.time()-start_1

				if self.pbar_flag:
					bar.postfix[0] = f'{batch_count.numpy():3.0f}'
					bar.postfix[1] = f'{self.D_loss.numpy():2.4e}'
					bar.postfix[2] = f'{self.G_loss.numpy():2.4e}'
					bar.update(self.batch_size.numpy())
				if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:
					if self.res_flag:
						self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4e}; G_loss - {:>2.4e}\n".format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy()))


				#print AE resuts every 100 steps during AE training, and every 1000 steps after AE training block
				self.print_batch_outputs(epoch)

				# Save the model every SAVE_ITERS iterations
				if (self.total_count.numpy() % self.save_step.numpy()) == 0:
					if self.save_all:
						self.checkpoint.save(file_prefix = self.checkpoint_prefix)
					else:
						self.manager.save()

			if self.pbar_flag:
				bar.close()
				del bar
			
			tf.print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
			if self.res_flag:
				self.res_file.write('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
		# if self.KLD_flag:
		# 	self.printKLD()


	def print_batch_outputs(self,epoch):
		if (self.total_count.numpy() <= 5) and self.data != 'gmm8':
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)


	def test(self):
		self.impath += '_Testing_'
		for img_batch in self.train_dataset:
			self.reals = img_batch
			self.generate_and_save_batch(0)
			return

	def train_step(self,reals_all):

		with tf.device(self.device):
			noise = tf.random.normal([self.batch_size, self.noise_dims], mean = self.noise_mean, stddev = self.noise_stddev)
		self.reals = self.reals_enc = reals_all

		with tf.GradientTape() as gen_tape:

			self.fakes = self.fakes_enc = self.generator(noise, training=True)

			with gen_tape.stop_recording():
				if self.total_count.numpy()%FLAGS.ODE_step == 0 or self.total_count.numpy() <= 2:
					self.discriminator_ODE()
					self.discriminator_B.set_weights([self.Gamma_c, self.Gamma_s, self.Tau_c, self.Tau_s])
				
			self.real_output, self.lambda_x_terms_1 = self.discriminator_B(self.discriminator_A(self.reals_enc, training = True), training = True)
			self.fake_output, self.lambda_x_terms_2 = self.discriminator_B(self.discriminator_A(self.fakes_enc, training = True), training = True)

			self.find_and_divide_lambda()
			
			eval(self.loss_func)

			self.G_grads = gen_tape.gradient(self.G_loss, self.generator.trainable_variables)
			# print(self.G_grads)
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.generator.trainable_variables))


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
				ar = tf.expand_dims(tf.reduce_mean(ar, axis = 0), axis = 1) #Lx1 vector
				ai = tf.expand_dims(tf.reduce_mean(ai, axis = 0), axis = 1) #Lx1 vector


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
					# print(cov)
					nt_mu = mu*tf.transpose(self.n_vec)
					nt_cov_n = cov * tf.expand_dims(tf.linalg.tensor_diag_part(tf.linalg.matmul(tf.transpose(self.n_vec,[1,0]),self.n_vec)), axis=1)

				ar =  tf.constant((1/(self.T))**0) * tf.math.exp(-0.5 * tf.multiply(nt_cov_n, self.W**2))*(tf.math.cos(tf.multiply(nt_mu, self.W)))
				ai =  tf.constant((1/(self.T))**0) * tf.math.exp(-0.5 * tf.multiply(nt_cov_n, self.W**2 ))*(tf.math.sin(tf.multiply(nt_mu, self.W)))
				# print(ar,ai)
			if self.distribution == 'uniform':
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
		self.alpha_c, self.alpha_s = eval('self.Fourier_Series_Comp(self.reals_enc)') #alpha is reals=> target
		self.beta_c, self.beta_s = eval('self.Fourier_Series_Comp(self.fakes_enc)')

		with tf.device(self.device):
			# Vec of len Lx1 , wach entry is ||n||
			self.Gamma_s = tf.math.divide(tf.constant(1/(self.W**2))*tf.subtract(self.alpha_s, self.beta_s), self.n_norm)
			self.Gamma_c = tf.math.divide(tf.constant(1/(self.W**2))*tf.subtract(self.alpha_c, self.beta_c), self.n_norm)
			self.Tau_s = tf.math.divide(tf.constant(1/(2.*(self.W**2)))*tf.square(tf.subtract(self.alpha_s, self.beta_s)), self.n_norm)
			self.Tau_c = tf.math.divide(tf.constant(1/(2.*(self.W**2)))*tf.square(tf.subtract(self.alpha_c, self.beta_c)), self.n_norm)
			self.sum_Tau = 1.*tf.reduce_sum(tf.add(self.Tau_s,self.Tau_c))

	def find_and_divide_lambda(self):
		self.lamb = tf.divide(tf.reduce_sum(self.lambda_x_terms_2) + tf.reduce_sum(self.lambda_x_terms_1),tf.cast(self.batch_size, dtype = 'float32')) + self.sum_Tau
		self.lamb = tf.cast(2*self.L, dtype = 'float32')*self.lamb # Dont put the sqrt????
		self.lamb = tf.sqrt(self.lamb)
		self.real_output = tf.divide(self.real_output, self.lamb)
		self.real_output = tf.add(self.real_output,0.)
		self.fake_output = tf.divide(self.fake_output, self.lamb)
		self.fake_output = tf.add(self.fake_output,0.)

	def loss_FS(self):
		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)

		self.D_loss = 1 * (-loss_real + loss_fake)
		self.G_loss = 1 * (loss_real - loss_fake)



