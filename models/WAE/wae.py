from __future__ import print_function
import os, sys, time, argparse
import numpy as np
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import FigureCanvasPgf
import tensorflow_probability as tfp
from matplotlib.backends.backend_pgf import PdfPages
tfd = tfp.distributions
from itertools import product as cart_prod

import matplotlib.pyplot as plt
import math
import tensorflow as tf
from absl import app
from absl import flags
from scipy.interpolate import interp1d

from gan_topics import *
'''***********************************************************************************
********** WAEFeR ********************************************************************
***********************************************************************************'''
class WAE_ELeGANt(GAN_WAE, FourierSolver):

	def __init__(self,FLAGS_dict):

		GAN_WAE.__init__(self,FLAGS_dict)

		''' Set up the Fourier Series Solver common to WAEFR and WGAN-FS'''
		FourierSolver.__init__(self)
	

		if self.colab and (self.data in ['mnist', 'celeba', 'cifar10', 'svhn']):
			self.bar_flag = 0
		else:
			self.bar_flag = 1


	def create_models(self):
		with tf.device(self.device):
			self.total_count = tf.Variable(0,dtype='int64')
			eval(self.encdec_model)
			self.discriminator_A = self.discriminator_model_FS_A()
			self.discriminator_A.set_weights([self.Coeffs])
			self.discriminator_B = self.discriminator_model_FS_B()
			

			print("Model Successfully made")

			print("\n\n ENCODER MODEL: \n\n")
			print(self.Encoder.summary())
			print("\n\n DECODER MODEL: \n\n")
			print(self.Decoder.summary())
			print("\n\n DISCRIMINATOR PART A MODEL: \n\n")
			print(self.discriminator_A.summary())
			print("\n\n DISCRIMINATOR PART B MODEL: \n\n")
			print(self.discriminator_B.summary())


			if self.res_flag == 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n ENCODER MODEL: \n\n")
					self.Encoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DECODER MODEL: \n\n")
					self.Decoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR PART A MODEL: \n\n")
					self.discriminator_A.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR PART B MODEL: \n\n")
					self.discriminator_B.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return

	def create_optimizer(self):

		with tf.device(self.device):
			lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=100000, decay_rate=0.95, staircase=True)
			#### Added for IMCL rebuttal
			self.E_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Enc)
			self.D_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Dec)
			self.G_optimizer = tf.keras.optimizers.Nadam(self.lr_G)
			# self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D)
		print("Optimizers Successfully made")	
		return	

	def create_load_checkpoint(self):

		self.checkpoint = tf.train.Checkpoint(E_optimizer = self.E_optimizer,
								 D_optimizer = self.D_optimizer,
								 G_optimizer = self.G_optimizer,
								 Encoder = self.Encoder,
								 Decoder = self.Decoder,
								 discriminator_A = self.discriminator_A,
								 discriminator_B = self.discriminator_B,
								 locs = self.locs,
								 total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.Encoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Encoder.h5')
					self.Decoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Decoder.h5')
					self.discriminator_A = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator_A.h5')
					self.discriminator_B = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator_B.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))


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
				batch_count.assign_add(self.Dloop)
				start_1 = time.time()
				
				with tf.device(self.device):
					if epoch < self.GAN_pretrain_epochs:
						self.pretrain_step_GAN(image_batch)
					else:
						self.train_step(image_batch)
						self.eval_metrics()
				
				train_time = time.time()-start_1

				if self.pbar_flag:
					bar.postfix[0] = f'{batch_count.numpy():4.0f}'
					bar.postfix[1] = f'{self.D_loss.numpy():2.4e}'
					bar.postfix[2] = f'{self.G_loss.numpy():2.4e}'
					bar.postfix[3] = f'{self.AE_loss.numpy():2.4e}'
					bar.update(self.batch_size.numpy())

				if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:
					if self.res_flag:
						self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f}; AE_loss - {:>2.4f} \n".format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy(),self.AE_loss.numpy()))

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
			self.Encoder.save(self.checkpoint_dir + '/model_Encoder.h5', overwrite = True)
			self.Decoder.save(self.checkpoint_dir + '/model_Decoder.h5', overwrite = True)
			self.discriminator_A.save(self.checkpoint_dir + '/model_discriminator_A.h5', overwrite = True)
			self.discriminator_B.save(self.checkpoint_dir + '/model_discriminator_B.h5', overwrite = True)

		# if self.KLD_flag:
		# 	self.printKLD()



	def print_batch_outputs(self,epoch):		
		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % 100) == 0 and self.data in ['g1', 'g2']:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % 150) == 0:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % 1000) == 0:
			self.test()



	def pretrain_step_GAN(self,reals_all):

		## Actually Pretrain GAN. - Will make a sperate flag nd control if it does infact work out
		with tf.device(self.device):
			self.fakes_enc = target_noise = self.get_noise(self.batch_size)
			self.reals = reals_all
			self.AE_loss = tf.constant(0)

		with tf.GradientTape() as G_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			# print(self.reals_enc.numpy())
			
			with G_tape.stop_recording():
				if self.total_count.numpy()%1 == 0:
					eval(self.DEQ_func)
					self.discriminator_B.set_weights([self.Gamma_c,self.bias, self.Gamma_s, self.Tau_c, self.Tau_s])


			self.real_output, self.lambda_x_terms_1 = self.discriminator_B(self.discriminator_A(target_noise, training = True), training = True)
			self.fake_output, self.lambda_x_terms_2 = self.discriminator_B(self.discriminator_A(self.reals_enc, training = True), training = True)

			self.find_and_divide_lambda()
			eval(self.loss_func)

			self.G_grads = G_tape.gradient(self.G_loss, self.Encoder.trainable_variables)
			# print("FS Grads",self.E_grads,"=========================================================")
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.Encoder.trainable_variables))



	#### Enc-Dec version added for ICML rebuttal. Uncomment only if needed
	def train_step(self,reals_all):
		with tf.device(self.device):
			self.fakes_enc = target_noise = self.get_noise(self.batch_size)
			self.reals = reals_all

		with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as G_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			self.reals_dec = self.Decoder(self.reals_enc, training = True)

			self.loss_AE()
			self.E_grads = enc_tape.gradient(self.AE_loss, self.Encoder.trainable_variables)
			self.D_grads = dec_tape.gradient(self.AE_loss, self.Decoder.trainable_variables)
			# print("AE Grads",self.E_grads, "=================================================")
			self.E_optimizer.apply_gradients(zip(self.E_grads, self.Encoder.trainable_variables))
			self.D_optimizer.apply_gradients(zip(self.D_grads, self.Decoder.trainable_variables))


			with G_tape.stop_recording():
				if self.total_count.numpy()%1 == 0:
					eval(self.DEQ_func)
					self.discriminator_B.set_weights([self.Gamma_c,self.bias, self.Gamma_s, self.Tau_c, self.Tau_s])


			self.real_output, self.lambda_x_terms_1 = self.discriminator_B(self.discriminator_A(target_noise, training = True), training = True)
			self.fake_output, self.lambda_x_terms_2 = self.discriminator_B(self.discriminator_A(self.reals_enc, training = True), training = True)

			self.find_and_divide_lambda()
			eval(self.loss_func)

			self.G_grads = G_tape.gradient(self.G_loss, self.Encoder.trainable_variables)
			# print("FS Grads",self.E_grads,"=========================================================")
			self.G_optimizer.apply_gradients(zip(self.G_grads, self.Encoder.trainable_variables))


	def loss_FS(self):
		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output) 
		# used 0.1
		self.D_loss = (-loss_real + loss_fake) #+ self.AE_loss
		self.G_loss = (loss_real - loss_fake) #+ self.AE_loss


	def loss_AE(self):
		mse = tf.keras.losses.MeanSquaredError()
		if self.data in ['celeba','lsun']:
			loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))
		elif self.data in ['mnist', 'svhn', 'cifar10']:
			loss_AE_reals = 0.5*tf.reduce_mean(tf.abs(self.reals - self.reals_dec)) + 1.5*(mse(self.reals, self.reals_dec))
		else:
			loss_AE_reals = mse(self.reals, self.reals_dec)

		self.AE_loss =  loss_AE_reals 

'''***********************************************************************************
********** WAE ********************************************************************
***********************************************************************************'''
class WAE_Base(GAN_WAE):

	def __init__(self,FLAGS_dict):

		self.lambda_GP = 1.
		GAN_WAE.__init__(self,FLAGS_dict)


		if self.colab and self.data in ['mnist', 'svhn', 'celeba','cifar10']:
			self.bar_flag = 0
		else:
			self.bar_flag = 1

	
	def create_models(self):
		with tf.device(self.device):
			self.total_count = tf.Variable(0,dtype='int64')
			eval(self.encdec_model)
			self.discriminator = eval(self.disc_model)

			print("Model Successfully made")

			print("\n\n ENCODER MODEL: \n\n")
			print(self.Encoder.summary())
			print("\n\n DECODER MODEL: \n\n")
			print(self.Decoder.summary())
			print("\n\n DISCRIMINATOR MODEL: \n\n")
			print(self.discriminator.summary())

			if self.res_flag == 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n ENCODER MODEL: \n\n")
					self.Encoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DECODER MODEL: \n\n")
					self.Decoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
					self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
		return


	def create_optimizer(self):
		with tf.device(self.device):
			lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=1000000, decay_rate=0.95, staircase=True)

			self.E_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Enc)
			self.D_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Dec)
			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G)
			self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D)

			print("Optimizers Successfully made")
		return

	def create_load_checkpoint(self):
		self.checkpoint = tf.train.Checkpoint(E_optimizer = self.E_optimizer,
								 D_optimizer = self.D_optimizer,
								 Disc_optimizer = self.Disc_optimizer,
								 Encoder = self.Encoder,
								 Decoder = self.Decoder,
								 discriminator = self.discriminator,
								 total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			try:
				self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			except:
				print("Checkpoint loading Failed. It could be a model mismatch. H5 files will be loaded instead")
				try:
					self.Encoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Encoder.h5')
					self.Decoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Decoder.h5')
					self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
				except:
					print("H5 file loading also failed. Please Check the LOG_FOLDER and RUN_ID flags")

			print("Model restored...")
			print("Starting at Iteration - "+str(self.total_count.numpy()))
			print("Starting at Epoch - "+str(int((self.total_count.numpy() * self.batch_size) / (self.train_data.shape[0])) + 1))


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
				batch_count.assign_add(1)
				start_1 = time.time()
				
				with tf.device(self.device):
					if epoch < self.GAN_pretrain_epochs:
						self.pretrain_step_GAN(image_batch)
					else:
						self.train_step(image_batch)
						self.eval_metrics()
				
				train_time = time.time()-start_1

					
				if self.pbar_flag:
					bar.postfix[0] = f'{batch_count.numpy():4.0f}'
					bar.postfix[1] = f'{self.D_loss.numpy():2.4e}'
					bar.postfix[2] = f'{self.G_loss.numpy():2.4e}'
					bar.postfix[3] = f'{self.AE_loss.numpy():2.4e}'
					bar.update(self.batch_size.numpy())
				if (batch_count.numpy() % self.print_step.numpy()) == 0 or self.total_count <= 2:
					# tf.print ('Epoch {:>3d} batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f}'.format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy()))
					if self.res_flag:
						self.res_file.write("Epoch {:>3d} Batch {:>3d} in {:>2.4f} sec; D_loss - {:>2.4f}; G_loss - {:>2.4f}; AE_loss - {:>2.4f} \n".format(epoch,batch_count.numpy(),train_time,self.D_loss.numpy(),self.G_loss.numpy(),self.AE_loss.numpy()))
					

				self.print_batch_outputs(epoch)

				# Save the model every SAVE_ITERS iterations
				if (self.total_count.numpy() % self.save_step.numpy()) == 0:
					if self.save_all:
						self.checkpoint.save(file_prefix = self.checkpoint_prefix)
					else:
						self.manager.save()

				if (self.total_count.numpy() % 1000) == 0:
					self.test()

			if self.pbar_flag:
				bar.close()
				del bar


			tf.print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
			self.Encoder.save(self.checkpoint_dir + '/model_Encoder.h5', overwrite = True)
			self.Decoder.save(self.checkpoint_dir + '/model_Decoder.h5', overwrite = True)
			self.discriminator.save(self.checkpoint_dir + '/model_discriminator.h5', overwrite = True)


	def print_batch_outputs(self,epoch):		
		if self.total_count.numpy() <= 2:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % 100) == 0 and self.data in ['g1', 'g2']:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % self.save_step.numpy()) == 0:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % 250) == 0:
			self.generate_and_save_batch(epoch)
		if (self.total_count.numpy() % 1000) == 0:
			self.test()
	

	def pretrain_step_GAN(self,reals_all):
		with tf.device(self.device):
			self.fakes_enc = target_noise = self.get_noise(self.batch_size)
			self.reals = reals_all

		with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as G_tape, tf.GradientTape(persistent = True) as disc_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			self.AE_loss = tf.constant(0)

			for i in tf.range(self.Dloop):

				self.real_output = self.discriminator(target_noise, training = True)
				self.fake_output = self.discriminator(self.reals_enc, training = True)
				
				eval(self.loss_func)
				# self.D_loss = self.G_loss

				self.Disc_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
				self.Disc_optimizer.apply_gradients(zip(self.Disc_grads, self.discriminator.trainable_variables))

				if self.loss == 'base':
					wt = []
					for w in self.discriminator.get_weights():
						w = tf.clip_by_value(w, -0.01,0.01) #0.01 for [0,1] data
						wt.append(w)
					self.discriminator.set_weights(wt)

				if i >= (self.Dloop - self.Gloop):
					self.G_grads = G_tape.gradient(self.G_loss, self.Encoder.trainable_variables)
					self.G_optimizer.apply_gradients(zip(self.G_grads, self.Encoder.trainable_variables))



	def train_step(self,reals_all):
		with tf.device(self.device):
			self.fakes_enc = target_noise = self.get_noise(self.batch_size)
			self.reals = reals_all

		with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape, tf.GradientTape() as G_tape, tf.GradientTape(persistent = True) as disc_tape:

			self.reals_enc = self.Encoder(self.reals, training = True)
			self.reals_dec = self.Decoder(self.reals_enc, training = True)

			self.loss_AE()
			self.E_grads = enc_tape.gradient(self.AE_loss, self.Encoder.trainable_variables)
			self.D_grads = dec_tape.gradient(self.AE_loss, self.Decoder.trainable_variables)
			self.E_optimizer.apply_gradients(zip(self.E_grads, self.Encoder.trainable_variables))
			self.D_optimizer.apply_gradients(zip(self.D_grads, self.Decoder.trainable_variables))

			for i in tf.range(self.Dloop):

				self.real_output = self.discriminator(target_noise, training = True)
				self.fake_output = self.discriminator(self.reals_enc, training = True)
				
				eval(self.loss_func)
				# self.D_loss = self.G_loss

				self.Disc_grads = disc_tape.gradient(self.D_loss, self.discriminator.trainable_variables)
				self.Disc_optimizer.apply_gradients(zip(self.Disc_grads, self.discriminator.trainable_variables))

				if self.loss == 'base':
					wt = []
					for w in self.discriminator.get_weights():
						w = tf.clip_by_value(w, -0.1,0.1) #0.01 for [0,1] data
						wt.append(w)
					self.discriminator.set_weights(wt)

				if i >= (self.Dloop - self.Gloop):
					self.G_grads = G_tape.gradient(self.G_loss, self.Encoder.trainable_variables)
					self.G_optimizer.apply_gradients(zip(self.G_grads, self.Encoder.trainable_variables))


	def loss_base(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output) 

		self.D_loss = (-loss_real + loss_fake)
		self.G_loss = (loss_real - loss_fake)

	#################################################################

	def loss_KL(self):

		logloss_D_fake = tf.math.log(1 - self.fake_output)
		logloss_D_real = tf.math.log(self.real_output) 

		logloss_G_fake = tf.math.log(self.fake_output)

		self.D_loss = -tf.reduce_mean(logloss_D_fake + logloss_D_real)
		self.G_loss = -tf.reduce_mean(logloss_G_fake)

	#################################################################

	def loss_JS(self):

		cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

		D_real_loss = cross_entropy(tf.ones_like(self.real_output), self.real_output)
		D_fake_loss = cross_entropy(tf.zeros_like(self.fake_output), self.fake_output)

		G_fake_loss = cross_entropy(tf.ones_like(self.fake_output), self.fake_output)

		self.D_loss = D_real_loss + D_fake_loss
		self.G_loss = G_fake_loss


	#################################################################

	def loss_GP(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.gradient_penalty()

		self.D_loss = (-loss_real + loss_fake + self.lambda_GP * self.gp )
		self.G_loss = (loss_real - loss_fake)


	def gradient_penalty(self):
		alpha = tf.random.uniform([self.batch_size, 1], 0., 1.)
		diff = tf.cast(self.fakes_enc,dtype = 'float32') - tf.cast(self.reals_enc,dtype = 'float32')
		inter = tf.cast(self.reals_enc,dtype = 'float32') + (alpha * diff)
		with tf.GradientTape() as t:
			t.watch(inter)
			pred = self.discriminator(inter, training = True)
		grad = t.gradient(pred, [inter])[0]
		slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1]))
		self.gp = tf.reduce_mean((slopes - 1.)**2)
		return 


	#################################################################

	def loss_LP(self):

		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output)  

		self.lipschitz_penalty()

		self.D_loss = (-loss_real + loss_fake + self.lambda_GP * self.lp )
		self.G_loss = (loss_real - loss_fake)

	def lipschitz_penalty(self):
		# Code Courtesy of Paper Author dterjek's GitHub
		# URL : https://github.com/dterjek/adversarial_lipschitz_regularization/blob/master/wgan_alp.py

		self.K = 1
		self.p = 2

		epsilon = tf.random.uniform([self.batch_size, 1], 0.0, 1.0)
		x_hat = epsilon * tf.cast(self.fakes_enc,dtype = 'float32') + (1 - epsilon) * tf.cast(self.reals_enc,dtype = 'float32')

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

		self.D_loss = (-loss_real + loss_fake + self.lambda_GP * self.alp)
		self.G_loss = (loss_real - loss_fake)


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
		self.K = 1

		samples = tf.concat([tf.cast(self.reals_enc,dtype = 'float32'), tf.cast(self.fakes_enc,dtype = 'float32')], axis=0)
		noise = tf.random.uniform([tf.shape(samples)[0], 1], 0, 1)
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

		if self.data in ['celeba','lsun']:
			loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))
		elif self.data in ['mnist', 'svhn', 'cifar10']:
			loss_AE_reals = 0.5*tf.reduce_mean(tf.abs(self.reals - self.reals_dec)) + 1.5*(mse(self.reals, self.reals_dec))
		else:
			loss_AE_reals = mse(self.reals, self.reals_dec)

		self.AE_loss =  loss_AE_reals 

