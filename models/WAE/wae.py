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
class WAE_ELeGANt(GAN_WAE):

	def __init__(self,FLAGS_dict):

		GAN_WAE.__init__(self,FLAGS_dict)

		# self.lambda_d = FLAGS_dict['lambda_d']

		# self.latent_dims = FLAGS_dict['latent_dims']
		# self.N = FLAGS_dict['latent_dims']
		# self.sigma = FLAGS_dict['sigma']
		# self.sigmul = FLAGS_dict['sigmul']

		# self.noise_kind = FLAGS_dict['noise_kind']
		# self.homo_flag = FLAGS_dict['homo_flag']


		self.M = self.terms#FLAGS_dict['terms'] #Number of terms in FS

		self.T = self.sigmul*self.sigma
		self.W = np.pi/self.T
		self.W0 = 1/self.T
		# self.distribution = FLAGS_dict['distribution'] #'generic'
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
				# print(self.n.shape,xxx)


		with tf.device(self.device):
			print(self.n_vec, self.W)
			self.Coeffs = tf.multiply(self.n_vec, self.W)
			print(self.Coeffs)
			self.n_norm = tf.expand_dims(tf.square(tf.linalg.norm(tf.transpose(self.n_vec), axis = 1)), axis=1)
			self.bias = np.array([0.])
	

		if self.colab and (self.data in ['mnist', 'celeba', 'cifar10', 'svhn']):
			self.bar_flag = 0
		else:
			self.bar_flag = 1


	def main_func(self):
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



			# from flopco_keras import FlopCoKeras
			# stats = FlopCoKeras(self.discriminator_B)

			# print(f"FLOPs: {stats.total_flops}")
			# print(f"MACs: {stats.total_macs}")
			# print(f"Relative FLOPs: {stats.relative_flops}")

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

			lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=100000, decay_rate=0.95, staircase=True)

			#### Added for IMCL rebuttal
			self.E_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Enc)
			self.D_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Dec)
			self.G_optimizer = tf.keras.optimizers.Nadam(self.lr_G)
			self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D)

			print("Optimizers Successfully made")		

		self.checkpoint = tf.train.Checkpoint(E_optimizer = self.E_optimizer,
								 D_optimizer = self.D_optimizer,
								 G_optimizer = self.G_optimizer,
								 Disc_optimizer = self.Disc_optimizer,
								 Encoder = self.Encoder,
								 Decoder = self.Decoder,
								 discriminator_A = self.discriminator_A,
								 discriminator_B = self.discriminator_B,
								 locs = self.locs,
								 total_count = self.total_count)
		self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=10)
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

		if self.resume:
			# self.total_count = int(temp.split('-')[-1])
			print("Checking for checkpoint file in "+self.checkpoint_dir)
			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			# self.Encoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Encoder.h5')
			# self.Decoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Decoder.h5')
			# self.discriminator_A = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator_A.h5')
			# self.discriminator_B = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator_B.h5')
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
					if epoch < self.AE_count:
						self.pretrain_step_AE(image_batch)
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


	def test(self):
		###### Random Saples
		# for i in range(10):
		# 	path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
		# 	noise = self.get_noise(self.batch_size)
		# 	images = self.Decoder(noise)
			

		# 	sharpness = self.find_sharpness(images)
		# 	try:
		# 		sharpness_vec.append(sharpness)
		# 	except:
		# 		shaprpness_vec = [sharpness]

		# 	images = (images + 1.0)/2.0
		# 	size_figure_grid = 10
		# 	images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
		# 	fig1 = plt.figure(figsize=(7,7))
		# 	ax1 = fig1.add_subplot(111)
		# 	ax1.cla()
		# 	ax1.axis("off")
		# 	if images_on_grid.shape[2] == 3:
		# 		ax1.imshow(np.clip(images_on_grid,0.,1.))
		# 	else:
		# 		ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

		# 	label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
		# 	plt.title(label)
		# 	plt.tight_layout()
		# 	# fig.text(0.5, 0.04, label, ha='center')
		# 	plt.savefig(path)
		# 	plt.close()

		# overall_sharpness = np.mean(np.array(shaprpness_vec))
		# if self.mode == 'test':
		# 	print("Random Sharpness - " + str(overall_sharpness))
		# 	if self.res_flag:
		# 		self.res_file.write("Random Sharpness - "+str(overall_sharpness))
		# else:
		# 	if self.res_flag:
		# 		self.res_file.write("Random Sharpness - "+str(overall_sharpness))

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
		else:
			if self.res_flag:
				self.res_file.write("Dataset Sharpness 10k samples - "+str(overall_sharpness))


		# ####### Recon - Output
		# for image_batch in self.recon_dataset:				
		# 	path = self.impath+'_TestingRecon_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
		# 	images = self.Decoder(self.Encoder(image_batch))
		# 	images = (images + 1.0)/2.0
		# 	size_figure_grid = 10
		# 	images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
		# 	fig1 = plt.figure(figsize=(7,7))
		# 	ax1 = fig1.add_subplot(111)
		# 	ax1.cla()
		# 	ax1.axis("off")
		# 	if images_on_grid.shape[2] == 3:
		# 		ax1.imshow(np.clip(images_on_grid,0.,1.))
		# 	else:
		# 		ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

		# 	label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
		# 	plt.title(label)
		# 	plt.tight_layout()
		# 	plt.savefig(path)
		# 	plt.close()

		# ###### Recon - org
		# 	path = self.impath+'_TestingReconOrg_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
		# 	images = image_batch
		# 	images = (images + 1.0)/2.0
		# 	size_figure_grid = 10
		# 	images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
		# 	fig1 = plt.figure(figsize=(7,7))
		# 	ax1 = fig1.add_subplot(111)
		# 	ax1.cla()
		# 	ax1.axis("off")
		# 	if images_on_grid.shape[2] == 3:
		# 		ax1.imshow(np.clip(images_on_grid,0.,1.))
		# 	else:
		# 		ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

		# 	label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
		# 	plt.title(label)
		# 	plt.tight_layout()
		# 	plt.savefig(path)
		# 	plt.close()
		# 	break

		####### Interpolation
		# num_interps = 10
		# if self.mode == 'test':
		# 	num_figs = int(400/(2*num_interps))
		# else:
		# 	num_figs = 9
		# # there are 400 samples in the batch. to make 10x10 images, 
		# for image_batch in self.interp_dataset:
		# 	for j in range(num_figs):
		# 		path = self.impath+'_TestingInterpolationV2_'+str(self.total_count.numpy())+'_TestCase_'+str(j)+'.png'
		# 		current_batch = image_batch[2*num_interps*j:2*num_interps*(j+1)]
		# 		image_latents = self.Encoder(current_batch)
		# 		for i in range(num_interps):
		# 			start = image_latents[i:1+i].numpy()
		# 			end = image_latents[num_interps+i:num_interps+1+i].numpy()
		# 			stack = np.vstack([start, end])
		# 			# print(stack.shape)
		# 			linfit = interp1d([1,num_interps+1], stack, axis=0)
		# 			# try:
		# 			# 	interp_latents=np.concatenate((interp_latents,linfit(list(range(1,num_interps+1)))),axis = 0)
		# 			# except:
		# 			# 	interp_latents = linfit(list(range(1,num_interps+1)))
		# 			interp_latents = linfit(list(range(1,num_interps+1)))
		# 			cur_interp_figs = self.Decoder(interp_latents)

		# 			sharpness = self.find_sharpness(cur_interp_figs)

		# 			try:
		# 				sharpness_vec.append(sharpness)
		# 			except:
		# 				shaprpness_vec = [sharpness]
		# 			cur_interp_figs_with_ref = np.concatenate((current_batch[i:1+i],cur_interp_figs.numpy(),current_batch[num_interps+i:num_interps+1+i]), axis = 0)
		# 			# print(cur_interp_figs_with_ref.shape)
		# 			try:
		# 				batch_interp_figs = np.concatenate((batch_interp_figs,cur_interp_figs_with_ref),axis = 0)
		# 			except:
		# 				batch_interp_figs = cur_interp_figs_with_ref
		# 			# print(batch_interp_figs.shape)


		# 		# interpolation_figs = self.Decoder(interp_latents)

		# 		images = (batch_interp_figs + 1.0)/2.0
		# 		# print(images.shape)
		# 		size_figure_grid = num_interps
		# 		images_on_grid = self.image_grid(input_tensor = images, grid_shape = (size_figure_grid,size_figure_grid+2),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
		# 		fig1 = plt.figure(figsize=(num_interps,num_interps+2))
		# 		ax1 = fig1.add_subplot(111)
		# 		ax1.cla()
		# 		ax1.axis("off")
		# 		if images_on_grid.shape[2] == 3:
		# 			ax1.imshow(np.clip(images_on_grid,0.,1.))
		# 		else:
		# 			ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

		# 		label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
		# 		plt.title(label)
		# 		plt.tight_layout()
		# 		plt.savefig(path)
		# 		plt.close()
		# 		del batch_interp_figs

		# overall_sharpness = np.mean(np.array(shaprpness_vec))
		# if self.mode == 'test':
		# 	print("Interpolation Sharpness - " + str(overall_sharpness))
		# 	if self.res_flag:
		# 		self.res_file.write("Interpolation Sharpness - "+str(overall_sharpness))
		# else:
		# 	if self.res_flag:
		# 		self.res_file.write("nterpolation Sharpness - "+str(overall_sharpness))




	# def pretrain_step_AE(self,reals_all):
	# 	with tf.device(self.device):
	# 		self.reals = reals_all

	# 	with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:

	# 		self.reals_enc = self.Encoder(self.reals, training = True)
	# 		self.reals_dec = self.Decoder(self.reals_enc, training = True)

	# 		self.loss_AE()
	# 		self.D_loss = self.G_loss = tf.constant(0)
	# 		self.E_grads = enc_tape.gradient(self.AE_loss, self.Encoder.trainable_variables)
	# 		self.D_grads = dec_tape.gradient(self.AE_loss, self.Decoder.trainable_variables)
	# 		# print("AE Grads",self.E_grads, "=================================================")
	# 		self.E_optimizer.apply_gradients(zip(self.E_grads, self.Encoder.trainable_variables))
	# 		self.D_optimizer.apply_gradients(zip(self.D_grads, self.Decoder.trainable_variables))

	def pretrain_step_AE(self,reals_all):

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

		cos_sum = tf.keras.layers.Dense(1, activation=None, use_bias = True)(cos_terms)
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

		with tf.device(self.device):
			if self.distribution == 'generic':
				# ar, ai = self.coefficients(f, training = False)
				_, ar, ai, _ = self.discriminator_A(f, training = False)
				ar = tf.expand_dims(tf.reduce_mean( ar, axis = 0), axis = 1)
				ai = tf.expand_dims(tf.reduce_mean( ai, axis = 0), axis = 1)
			if self.distribution == 'gaussian':
				if self.data != 'g1':
					with tf.device('/CPU'):
						nt_mu = tf.linalg.matmul(tf.transpose(self.n_vec),mu)
						nt_cov_n =  tf.expand_dims(tf.linalg.tensor_diag_part(tf.linalg.matmul(tf.transpose(self.n_vec),tf.linalg.matmul(cov,self.n_vec))), axis=1)
				else:
						nt_mu = mu*self.n_vec
						nt_cov_n = cov * tf.expand_dims(tf.linalg.tensor_diag_part(tf.linalg.matmul(tf.transpose(self.n_vec),self.n_vec)), axis=1)
				#### FIX POWER OF T
				#tf.constant((1/(self.T))**1)
				ar =  tf.constant((1/(self.T))**0) * tf.math.exp(-0.5 * tf.multiply(nt_cov_n, self.W**2))*(tf.math.cos(tf.multiply(nt_mu, self.W)))
				ai =  tf.constant((1/(self.T))**0) * tf.math.exp(-0.5 * tf.multiply(nt_cov_n, self.W**2 ))*(tf.math.sin(tf.multiply(nt_mu, self.W)))
		return  ar, ai

	def discriminator_ODE(self): ###### CURRENT WORKING PROPER VERSION

		# with tf.device('/CPU'):
		with tf.device(self.device):
			self.alpha_c, self.alpha_s = eval('self.Fourier_Series_Comp(self.fakes_enc)') #alpha is targets => fakes
			self.beta_c, self.beta_s = eval('self.Fourier_Series_Comp(self.reals_enc)')
			# self.Coeffs = tf.multiply(self.n_vec, self.W)

			# Vec of len Lx1 , wach entry is ||n||
			# temp = tf.constant(-0.5/(self.W**2))*tf.subtract(self.alpha_s, self.beta_s) #1./200 gave 29th's good images
			self.Gamma_s = tf.math.divide(tf.constant(1/(self.W**2))*tf.subtract(self.alpha_s, self.beta_s), self.n_norm)
			self.Gamma_c = tf.math.divide(tf.constant(1/(self.W**2))*tf.subtract(self.alpha_c, self.beta_c), self.n_norm)
			self.Tau_s = tf.math.divide(tf.constant(1/(2.*(self.W**2)))*tf.square(tf.subtract(self.alpha_s, self.beta_s)), self.n_norm)
			self.Tau_c = tf.math.divide(tf.constant(1/(2.*(self.W**2)))*tf.square(tf.subtract(self.alpha_c, self.beta_c)), self.n_norm)
			self.sum_Tau = 1.*tf.reduce_sum(tf.add(self.Tau_s,self.Tau_c))

	def find_and_divide_lambda(self):
		# print(self.lambda_x_terms_1, self.lambda_x_terms_2, self.sum_Tau, "====================")
		self.lamb = tf.divide(tf.reduce_sum(self.lambda_x_terms_2) + tf.reduce_sum(self.lambda_x_terms_1),tf.cast(self.batch_size, dtype = 'float32')) + self.sum_Tau
		# self.lamb = tf.reduce_sum(self.lambda_x_terms_2) + tf.reduce_sum(self.lambda_x_terms_1) + self.sum_Tau
		self.lamb = tf.cast(2*self.L, dtype = 'float32')*self.lamb # Dont put the sqrt????
		self.lamb = tf.sqrt(self.lamb)
		# self.lambda_vec.append(self.lamb.numpy())
		# print(self.lamb,"=======================")
		# print(self.real_output, self.fake_output,"====================")
		self.real_output = tf.divide(self.real_output, self.lamb)
		self.fake_output = tf.divide(self.fake_output, self.lamb)


	def loss_FS(self):
		loss_fake = tf.reduce_mean(self.fake_output)
		loss_real = tf.reduce_mean(self.real_output) 
		# used 0.1
		self.D_loss = (-loss_real + loss_fake) #+ self.AE_loss
		self.G_loss = (loss_real - loss_fake) #+ self.AE_loss


	def loss_AE(self):
		mse = tf.keras.losses.MeanSquaredError()
		# loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))
		# loss_AE_reals += mse(self.reals, self.reals_dec)
		# if self.data == 'mnist':
			# loss_AE_reals += mse(self.reals, self.reals_dec)
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

		# self.latent_dims = FLAGS_dict['latent_dims']
		# self.noise_kind = FLAGS_dict['noise_kind']
		
		GAN_WAE.__init__(self,FLAGS_dict)


		if self.colab and self.data in ['mnist', 'svhn', 'celeba','cifar10']:
			self.bar_flag = 0
		else:
			self.bar_flag = 1

	
	def main_func(self):
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

			# from flopco_keras import FlopCoKeras
			# stats = FlopCoKeras(self.discriminator)

			# print(f"FLOPs: {stats.total_flops}")
			# print(f"MACs: {stats.total_macs}")
			# print(f"Relative FLOPs: {stats.relative_flops}")

			if self.res_flag == 1:
				with open(self.run_loc+'/'+self.run_id+'_Models.txt','a') as fh:
					# Pass the file handle in as a lambda function to make it callable
					fh.write("\n\n ENCODER MODEL: \n\n")
					self.Encoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DECODER MODEL: \n\n")
					self.Decoder.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))
					fh.write("\n\n DISCRIMINATOR MODEL: \n\n")
					self.discriminator.summary(line_length=80, print_fn=lambda x: fh.write(x + '\n'))


			lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(self.lr_G, decay_steps=1000000, decay_rate=0.95, staircase=True)

			self.E_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Enc)
			self.D_optimizer = tf.keras.optimizers.Adam(self.lr_AE_Dec)
			self.G_optimizer = tf.keras.optimizers.Adam(self.lr_G)
			self.Disc_optimizer = tf.keras.optimizers.Adam(self.lr_D)

			print("Optimizers Successfully made")		

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
			# self.total_count = int(temp.split('-')[-1])
			self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
			# self.Encoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Encoder.h5')
			# self.Decoder = tf.keras.models.load_model(self.checkpoint_dir+'/model_Decoder.h5')
			# self.discriminator = tf.keras.models.load_model(self.checkpoint_dir+'/model_discriminator.h5')
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
					if epoch < self.AE_count:
						self.pretrain_step_AE(image_batch)
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
	
	# def pretrain_step_AE(self,reals_all):
	# 	with tf.device(self.device):
	# 		self.reals = reals_all

	# 	with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:

	# 		self.reals_enc = self.Encoder(self.reals, training = True)
	# 		self.reals_dec = self.Decoder(self.reals_enc, training = True)

	# 		self.loss_AE()
	# 		self.D_loss = self.G_loss = tf.constant(0)
	# 		self.E_grads = enc_tape.gradient(self.AE_loss, self.Encoder.trainable_variables)
	# 		self.D_grads = dec_tape.gradient(self.AE_loss, self.Decoder.trainable_variables)
	# 		# print("AE Grads",self.E_grads, "=================================================")
	# 		self.E_optimizer.apply_gradients(zip(self.E_grads, self.Encoder.trainable_variables))
	# 		self.D_optimizer.apply_gradients(zip(self.D_grads, self.Decoder.trainable_variables))

	def pretrain_step_AE(self,reals_all):
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


	def test(self):

		##### Random
		# for i in range(10):
		# 	path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
		# 	noise = self.get_noise(self.batch_size)
		# 	images = self.Decoder(noise)

		# 	sharpness = self.find_sharpness(images)

		# 	try:
		# 		sharpness_vec.append(sharpness)
		# 	except:
		# 		shaprpness_vec = [sharpness]

		# 	images = (images + 1.0)/2.0

		# 	size_figure_grid = 10
		# 	images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
		# 	fig1 = plt.figure(figsize=(7,7))
		# 	ax1 = fig1.add_subplot(111)
		# 	ax1.cla()
		# 	ax1.axis("off")
		# 	if images_on_grid.shape[2] == 3:
		# 		ax1.imshow(np.clip(images_on_grid,0.,1.))
		# 	else:
		# 		ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

		# 	# images = tf.reshape(images, [images.shape[0],self.output_size,self.output_size,1])
		# 	# fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(10, 10))
		# 	# for i in range(size_figure_grid):
		# 	# 	for j in range(size_figure_grid):
		# 	# 		ax[i, j].get_xaxis().set_visible(False)
		# 	# 		ax[i, j].get_yaxis().set_visible(False)

		# 	# for k in range(size_figure_grid*size_figure_grid):
		# 	# 	i = k // size_figure_grid
		# 	# 	j = k % size_figure_grid
		# 	# 	ax[i, j].cla()
		# 	# 	if images.shape[3] == 1:
		# 	# 		im = images[k,:,:,0]
		# 	# 		ax[i, j].imshow(im, cmap='gray')
		# 	# 	else:
		# 	# 		im = images[k,:,:,:]
		# 	# 		ax[i, j].imshow(im)

		# 	label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
		# 	plt.title(label)
		# 	plt.tight_layout()
		# 	# fig.text(0.5, 0.04, label, ha='center')
		# 	plt.savefig(path)
		# 	plt.close()
		
		# overall_sharpness = np.mean(np.array(shaprpness_vec))
		# if self.mode == 'test':
		# 	print("Random Sharpness - " + str(overall_sharpness))
		# 	if self.res_flag:
		# 		self.res_file.write("Random Sharpness - "+str(overall_sharpness))
		# else:
		# 	if self.res_flag:
		# 		self.res_file.write("Random Sharpness - "+str(overall_sharpness))


		i = 0
		for image_batch in self.train_dataset:
			i+=1
			sharpness = self.find_sharpness(image_batch)
			try:
				sharpness_vec.append(sharpness)
			except:
				shaprpness_vec = [sharpness]
			if i==10:
				break

		overall_sharpness = np.mean(np.array(shaprpness_vec))
		if self.mode == 'test':
			print("Dataset Sharpness - " + str(overall_sharpness))
			if self.res_flag:
				self.res_file.write("Dataset Sharpness - "+str(overall_sharpness))
		else:
			if self.res_flag:
				self.res_file.write("Dataset Sharpness - "+str(overall_sharpness))


		##### Recon - Output
		# for image_batch in self.recon_dataset:
		# 	path = self.impath+'_TestingRecon_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
		# 	images = self.Decoder(self.Encoder(image_batch))
		# 	images = (images + 1.0)/2.0
		# 	size_figure_grid = 10
		# 	images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
		# 	fig1 = plt.figure(figsize=(7,7))
		# 	ax1 = fig1.add_subplot(111)
		# 	ax1.cla()
		# 	ax1.axis("off")
		# 	if images_on_grid.shape[2] == 3:
		# 		ax1.imshow(np.clip(images_on_grid,0.,1.))
		# 	else:
		# 		ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

		# 	label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
		# 	plt.title(label)
		# 	plt.tight_layout()
		# 	# fig.text(0.5, 0.04, label, ha='center')
		# 	plt.savefig(path)
		# 	plt.close()

		##### Recon - ORG 
		# 	path = self.impath+'_TestingReconOrg_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
		# 	images = image_batch
		# 	images = (images + 1.0)/2.0
		# 	size_figure_grid = 10
		# 	images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
		# 	fig1 = plt.figure(figsize=(7,7))
		# 	ax1 = fig1.add_subplot(111)
		# 	ax1.cla()
		# 	ax1.axis("off")
		# 	if images_on_grid.shape[2] == 3:
		# 		ax1.imshow(np.clip(images_on_grid,0.,1.))
		# 	else:
		# 		ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

		# 	label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
		# 	plt.title(label)
		# 	plt.tight_layout()
		# 	plt.savefig(path)
		# 	plt.close()
		# 	break

		##### Interpolation
		# num_interps = 10
		# if self.mode == 'test':
		# 	num_figs = int(400/(2*num_interps))
		# else:
		# 	num_figs = 9
		# # there are 400 samples in the batch. to make 10x10 images, 
		# for image_batch in self.interp_dataset:
		# 	for j in range(num_figs):
		# 		path = self.impath+'_TestingInterpolationV2_'+str(self.total_count.numpy())+'_TestCase_'+str(j)+'.png'
		# 		current_batch = image_batch[2*num_interps*j:2*num_interps*(j+1)]
		# 		image_latents = self.Encoder(current_batch)
		# 		for i in range(num_interps):
		# 			start = image_latents[i:1+i].numpy()
		# 			end = image_latents[num_interps+i:num_interps+1+i].numpy()
		# 			stack = np.vstack([start, end])
		# 			# print(stack.shape)
		# 			linfit = interp1d([1,num_interps+1], stack, axis=0)
		# 			# try:
		# 			# 	interp_latents=np.concatenate((interp_latents,linfit(list(range(1,num_interps+1)))),axis = 0)
		# 			# except:
		# 			# 	interp_latents = linfit(list(range(1,num_interps+1)))
		# 			interp_latents = linfit(list(range(1,num_interps+1)))
		# 			cur_interp_figs = self.Decoder(interp_latents)

		# 			sharpness = self.find_sharpness(cur_interp_figs)

		# 			try:
		# 				sharpness_vec.append(sharpness)
		# 			except:
		# 				shaprpness_vec = [sharpness]

		# 			cur_interp_figs_with_ref = np.concatenate((current_batch[i:1+i],cur_interp_figs.numpy(),current_batch[num_interps+i:num_interps+1+i]), axis = 0)

		# 			try:
		# 				batch_interp_figs = np.concatenate((batch_interp_figs,cur_interp_figs_with_ref),axis = 0)
		# 			except:
		# 				batch_interp_figs = cur_interp_figs_with_ref
		# 			# print(interp_latents.shape)

		# 		# interpolation_figs = self.Decoder(interp_latents)

		# 		images = (batch_interp_figs + 1.0)/2.0
		# 		# print(images.shape)
		# 		size_figure_grid = num_interps
		# 		images_on_grid = self.image_grid(input_tensor = images, grid_shape = (size_figure_grid,size_figure_grid+2),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
		# 		fig1 = plt.figure(figsize=(num_interps,num_interps))
		# 		ax1 = fig1.add_subplot(111)
		# 		ax1.cla()
		# 		ax1.axis("off")
		# 		if images_on_grid.shape[2] == 3:
		# 			ax1.imshow(np.clip(images_on_grid,0.,1.))
		# 		else:
		# 			ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

		# 		label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
		# 		plt.title(label)
		# 		plt.tight_layout()
		# 		plt.savefig(path)
		# 		plt.close()
		# 		del batch_interp_figs
			
		# overall_sharpness = np.mean(np.array(shaprpness_vec))
		# if self.mode == 'test':
		# 	print("Interpolation Sharpness - " + str(overall_sharpness))
		# 	if self.res_flag:
		# 		self.res_file.write("Interpolation Sharpness - "+str(overall_sharpness))
		# else:
		# 	if self.res_flag:
		# 		self.res_file.write("nterpolation Sharpness - "+str(overall_sharpness))





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
		#used 0.01
		self.D_loss = (-loss_real + loss_fake)

		self.G_loss = (loss_real - loss_fake)

	#################################################################

	def loss_KL(self):


		logloss_D_fake = tf.math.log(1 - self.fake_output)
		logloss_D_real = tf.math.log(self.real_output) 

		logloss_G_fake = tf.math.log(self.fake_output)
		#used 0.01

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
		#used 0.001
		self.D_loss = (-loss_real + loss_fake + self.lambda_GP * self.gp )

		self.G_loss = (loss_real - loss_fake)
		#0th on 30th is 0.0001, 1st is 0.001, 2nd is 0.1, 3rd is 0.01


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
		#used 0.01
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
		#used 1

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
		# loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))
		# if self.data == 'mnist':
		# 	loss_AE_reals += mse(self.reals, self.reals_dec)

		if self.data in ['celeba','lsun']:
			loss_AE_reals = tf.reduce_mean(tf.abs(self.reals - self.reals_dec))
		elif self.data in ['mnist', 'svhn', 'cifar10']:
			loss_AE_reals = 0.5*tf.reduce_mean(tf.abs(self.reals - self.reals_dec)) + 1.5*(mse(self.reals, self.reals_dec))
		else:
			loss_AE_reals = mse(self.reals, self.reals_dec)

		self.AE_loss =  loss_AE_reals 




###########OLD TEST DUMPS - FS

		# for image_batch in self.interp_dataset:
		# 	for i in range(4):
		# 		path = self.impath+'_TestingInterpReconOrg_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
		# 		images = image_batch
		# 		images = (images + 1.0)/2.0
		# 		size_figure_grid = 10
		# 		images_on_grid = self.image_grid(input_tensor = images[i*(size_figure_grid*size_figure_grid):(i+1)*(size_figure_grid*size_figure_grid)], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
		# 		fig1 = plt.figure(figsize=(10,10))
		# 		ax1 = fig1.add_subplot(111)
		# 		ax1.cla()
		# 		ax1.axis("off")
		# 		if images_on_grid.shape[2] == 3:
		# 			ax1.imshow(np.clip(images_on_grid,0.,1.))
		# 		else:
		# 			ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

		# 		label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
		# 		plt.title(label)
		# 		plt.tight_layout()
		# 		plt.savefig(path)
		# 		plt.close()
		# 	break
		############################################
		# for image_batch in self.recon_dataset:
		# 	for j in range(7):
		# 		path = self.impath+'_TestingInterpolation_'+str(self.total_count.numpy())+'_TestCase_'+str(j)+'.png'
		# 		image_latents = self.Encoder(image_batch[12*j:12*(j+1)])
		# 		for i in range(6):
		# 			start = image_latents[i:i+1].numpy()
		# 			end = image_latents[6+i:7+i].numpy()
		# 			stack = np.vstack([start, end])
		# 			# print(stack.shape)
		# 			linfit = interp1d([1,6], stack, axis=0)
		# 			try:
		# 				interp_latents=np.concatenate((interp_latents,linfit([1,2,3,4,5,6])),axis = 0)
		# 			except:
		# 				interp_latents = linfit([1,2,3,4,5,6])
		# 			# print(interp_latents.shape)

		# 		interpolation_figs = self.Decoder(interp_latents)

		# 		images = (interpolation_figs + 1.0)/2.0
		# 		# print(images.shape)
		# 		size_figure_grid = 6
		# 		images_on_grid = self.image_grid(input_tensor = images[0:size_figure_grid*size_figure_grid], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
		# 		fig1 = plt.figure(figsize=(7,7))
		# 		ax1 = fig1.add_subplot(111)
		# 		ax1.cla()
		# 		ax1.axis("off")
		# 		if images_on_grid.shape[2] == 3:
		# 			ax1.imshow(np.clip(images_on_grid,0.,1.))
		# 		else:
		# 			ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

		# 		label = 'INTERPOLATED IMAGES AT ITERATION '+str(self.total_count.numpy())
		# 		plt.title(label)
		# 		plt.tight_layout()
		# 		plt.savefig(path)
		# 		plt.close()
		# 		del interp_latents


		# for image_batch in self.interp_dataset:
		# 	for i in range(4):
		# 		path = self.impath+'_TestingInterpReconOrg_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
		# 		images = image_batch
		# 		images = (images + 1.0)/2.0
		# 		size_figure_grid = 10
		# 		images_on_grid = self.image_grid(input_tensor = images[i*(size_figure_grid*size_figure_grid):(i+1)*(size_figure_grid*size_figure_grid)], grid_shape = (size_figure_grid,size_figure_grid),image_shape=(self.output_size,self.output_size),num_channels=images.shape[3])
		# 		fig1 = plt.figure(figsize=(10,10))
		# 		ax1 = fig1.add_subplot(111)
		# 		ax1.cla()
		# 		ax1.axis("off")
		# 		if images_on_grid.shape[2] == 3:
		# 			ax1.imshow(np.clip(images_on_grid,0.,1.))
		# 		else:
		# 			ax1.imshow(np.clip(images_on_grid[:,:,0],0.,1.), cmap='gray')

		# 		label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
		# 		plt.title(label)
		# 		plt.tight_layout()
		# 		plt.savefig(path)
		# 		plt.close()
		# 	break
		############################################

		# for i in range(10):
		# 	path = self.impath+'_Testing_'+str(self.total_count.numpy())+'_TestCase_'+str(i)+'.png'
		# 	noise = self.get_noise(self.batch_size)
		# 	images = self.Decoder(noise)
		# 	size_figure_grid = 10
		# 	# images = tf.reshape(images, [images.shape[0],self.output_size,self.output_size,1])
		# 	fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(10, 10))
		# 	for i in range(size_figure_grid):
		# 		for j in range(size_figure_grid):
		# 			ax[i, j].get_xaxis().set_visible(False)
		# 			ax[i, j].get_yaxis().set_visible(False)

		# 	for k in range(size_figure_grid*size_figure_grid):
		# 		i = k // size_figure_grid
		# 		j = k % size_figure_grid
		# 		ax[i, j].cla()
		# 		if images.shape[3] == 1:
		# 			im = images[k,:,:,0]
		# 			ax[i, j].imshow(im, cmap='gray')
		# 		else:
		# 			im = images[k,:,:,:]
		# 			ax[i, j].imshow(im)

		# 	label = 'TEST SAMPLES AT ITERATION '+str(self.total_count.numpy())
		# 	fig.text(0.5, 0.04, label, ha='center')
		# 	plt.savefig(path)
		# 	plt.close()


######## OLD TEST DUMPS _ WAE
