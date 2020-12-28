from __future__ import print_function
import os, sys, time, argparse
from datetime import date
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math
from absl import app
from absl import flags
import json
import glob
from tqdm.autonotebook import tqdm
import shutil

# import tensorflow_probability as tfp
# tfd = tfp.distributions

##FOR FID
from numpy import cov
from numpy import trace
from scipy.linalg import sqrtm
import scipy as sp
from numpy import iscomplexobj

from ext_resources import *

class GAN_Metrics():

	def __init__(self):

		self.KLD_flag = 0
		self.FID_flag = 0
		self.PR_flag = 0
		self.lambda_flag = 0
		self.recon_flag = 0
		self.GradGrid_flag = 0
		self.class_prob_flag = 0
		self.metric_counter_vec = []


		if self.loss == 'FS' and self.mode != 'metrics':
			self.lambda_flag = 1
			self.lambda_vec = []

		if 'KLD' in self.metrics:				
			self.KLD_flag = 1
			self.KLD_vec = []

			if self.data in ['g1', 'g2', 'gmm8']:
				self.KLD_steps = 10
				if self.data in [ 'gmm8',]:
					self.KLD_func = self.KLD_sample_estimate
				else:
					self.KLD_func = self.KLD_Gaussian
			else:
				self.KLD_flag = 1
				self.KLD_steps = 100
				if self.loss == 'FS' and self.gan != 'WAE':
					if self.distribution == 'gaussian' or self.data in ['g1','g2']:
						self.KLD_func = self.KLD_Gaussian
					else:
						self.KLD_func = self.KLD_sample_estimate
				if self.gan == 'WAE':
					if 'gaussian' in self.noise_kind:
						self.KLD_func = self.KLD_Gaussian
					else:
						self.KLD_func = self.KLD_sample_estimate
				print('KLD is not an accurate metric on this datatype')
				

		if 'FID' in self.metrics:
			self.FID_flag = 1
			self.FID_load_flag = 0
			self.FID_vec = []
			self.FID_vec_new = []

			if self.data in ['mnist', 'svhn']:
				self.FID_steps = 500
				if self.mode == 'metrics':
					self.FID_num_samples = 1000
				else:
					self.FID_num_samples = 15000
			elif self.data in ['cifar10']:
				self.FID_steps = 500
				if self.mode == 'metrics':
					self.FID_num_samples = 1000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['celeba', 'ukiyoe']:
				self.FID_steps = 1500 #2500 for Rumi
				if self.mode == 'metrics':
					self.FID_num_samples = 1000
				else:
					self.FID_num_samples = 5000
			elif self.data in ['gN']:
				self.FID_steps = 100
			else:
				self.FID_flag = 0
				print('FID cannot be evaluated on this dataset')

		if 'recon' in self.metrics:
			self.recon_flag = 1
			self.recon_vec = []
			self.FID_vec_new = []

			if self.data in ['mnist', 'svhn']:
				self.recon_steps = 500
			elif self.data in ['cifar10']:
				self.recon_steps = 1500
			elif self.data in ['celeba']:
				self.recon_steps = 1500 
			elif self.data in ['ukiyoe']:
				self.recon_steps = 1500# 1500 of 
			elif self.data in ['gN']:
				self.recon_steps = 100
			else:
				self.recon_flag = 0
				print('Reconstruction cannot be evaluated on this dataset')

		if 'GradGrid' in self.metrics:
			if self.data in ['g2', 'gmm8']:
				self.GradGrid_flag = 1
				self.GradGrid_steps = 100
			else:
				print("Cannot plot Gradient grid. Not a 2D dataset")

	def eval_metrics(self):
		update_flag = 0

		if self.FID_flag and (self.total_count.numpy()%self.FID_steps == 0 or self.mode == 'metrics'):
			update_flag = 1
			self.update_FID()
			if self.mode != 'metrics':
				np.save(self.metricpath+'FID.npy',np.array(self.FID_vec))
				self.print_FID()


		if self.KLD_flag and ((self.total_count.numpy()%self.KLD_steps == 0 or self.total_count.numpy() == 1)  or self.mode == 'metrics'):
			update_flag = 1
			self.update_KLD()
			if self.mode != 'metrics':
				np.save(self.metricpath+'KLD.npy',np.array(self.KLD_vec))
				self.print_KLD()

		if self.recon_flag and ((self.total_count.numpy()%self.recon_steps == 0 or self.total_count.numpy() == 1)  or self.mode == 'metrics'):
			update_flag = 1
			self.eval_recon()
			if self.mode != 'metrics':
				np.save(self.metricpath+'recon.npy',np.array(self.recon_vec))
				self.print_recon()

		if self.lambda_flag and (self.loss == 'FS' or self.mode == 'metrics'):
			update_flag = 1
			self.update_Lambda()
			if self.mode != 'metrics':
				np.save(self.metricpath+'Lambda.npy',np.array(self.lambda_vec))
				self.print_Lambda()

		if self.GradGrid_flag and ((self.total_count.numpy()%self.GradGrid_steps == 0 or self.total_count.numpy() == 1) or self.mode == 'metrics'):
			update_flag = 1
			self.print_GradGrid()

		if self.res_flag and update_flag:
			self.res_file.write("Metrics avaluated at Iteration " + str(self.total_count.numpy()) + '\n')



	def update_FID(self):
		eval(self.FID_func)

	def eval_FID(self):
		mu1, sigma1 = self.act1.mean(axis=0), cov(self.act1, rowvar=False)
		mu2, sigma2 = self.act2.mean(axis=0), cov(self.act2, rowvar=False)
		# calculate sum squared difference between means
		ssdiff = np.sum((mu1 - mu2)**2.0)
		# calculate sqrt of product between cov
		covmean = sqrtm(sigma1.dot(sigma2))
		# check and correct imaginary numbers from sqrt
		if iscomplexobj(covmean):
			covmean = covmean.real
		# calculate score
		self.fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
		self.FID_vec.append([self.fid, self.total_count.numpy()])
		if self.mode == 'metrics':
			print("Final FID score - "+str(self.fid))
			if self.res_flag:
				self.res_file.write("Final FID score - "+str(self.fid))

		if self.res_flag:
			self.res_file.write("FID score - "+str(self.fid))
		# self.eval_FID_new()
		# np.save(self.impath+'_FID.npy',np.array(self.FID_vec))
		return

	def print_FID(self):
		path = self.metricpath
		#Colab has issues with latex. Better not waste time printing on Colab. Use the NPY later, offline,...
		if self.colab:
			from matplotlib.backends.backend_pdf import PdfPages
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		vals = list(np.array(self.FID_vec)[:,0])
		locs = list(np.array(self.FID_vec)[:,1])

		with PdfPages(path+'FID_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'FID vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)


	def eval_recon(self):
		# print('Evaluating Recon Loss\n')
		mse = tf.keras.losses.MeanSquaredError()
		for image_batch in self.recon_dataset:
			# print("batch 1\n")
			recon_images = self.Decoder(self.Encoder(image_batch, training= False) , training = False)
			try:
				recon_loss = 0.5*(recon_loss) + 0.25*tf.reduce_mean(tf.abs(image_batch - recon_images)) + 0.75*(mse(image_batch,recon_images))
			except:
				recon_loss = 0.5*tf.reduce_mean(tf.abs(image_batch - recon_images)) + 1.5*(mse(image_batch,recon_images))
		self.recon_vec.append([recon_loss, self.total_count.numpy()])
		if self.mode == 'metrics':
			print("Final Reconstruction error - "+str(recon_loss))
			if self.res_flag:
				self.res_file.write("Final Reconstruction error - "+str(recon_loss))

		if self.res_flag:
			self.res_file.write("Reconstruction error - "+str(recon_loss))

	def print_recon(self):
		path = self.metricpath
		#Colab has issues with latex. Better not waste time printing on Colab. Use the NPY later, offline,...
		if self.colab:
			from matplotlib.backends.backend_pdf import PdfPages
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		vals = list(np.array(self.recon_vec)[:,0])
		locs = list(np.array(self.recon_vec)[:,1])

		with PdfPages(path+'recon_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'Reconstruction Error')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)


	def KLD_sample_estimate(self,P,Q):
		def skl_estimator(s1, s2, k=1):
			from sklearn.neighbors import NearestNeighbors
			### Code Courtesy nheartland 
			### URL : https://github.com/nhartland/KL-divergence-estimators/blob/master/src/knn_divergence.py
			""" KL-Divergence estimator using scikit-learn's NearestNeighbours
			s1: (N_1,D) Sample drawn from distribution P
			s2: (N_2,D) Sample drawn from distribution Q
			k: Number of neighbours considered (default 1)
			return: estimated D(P|Q)
			"""
			n, m = len(s1), len(s2)
			d = float(s1.shape[1])
			D = np.log(m / (n - 1))

			s1_neighbourhood = NearestNeighbors(k+1, 10).fit(s1)
			s2_neighbourhood = NearestNeighbors(k, 10).fit(s2)

			for p1 in s1:
				s1_distances, indices = s1_neighbourhood.kneighbors([p1], k+1)
				s2_distances, indices = s2_neighbourhood.kneighbors([p1], k)
				rho = s1_distances[0][-1]
				nu = s2_distances[0][-1]
				D += (d/n)*np.log(nu/rho)
			return D
		KLD = skl_estimator(P,Q)
		self.KLD_vec.append([KLD, self.total_count.numpy()])
		return

	def KLD_Gaussian(self,P,Q):

		def get_mean(f):
			return np.mean(f,axis = 0).astype(np.float64)
		def get_cov(f):
			return np.cov(f,rowvar = False).astype(np.float64)
		def get_std(f):
			return np.std(f).astype(np.float64)
		try:
			if self.data == 'g1':
				Distribution = tfd.Normal
				P_dist = Distribution(loc=get_mean(P), scale=get_std(P))
				Q_dist = Distribution(loc=get_mean(Q), scale=get_std(Q))
			else:
				Distribution = tfd.MultivariateNormalFullCovariance
				P_dist = Distribution(loc=get_mean(P), covariance_matrix=get_cov(P))
				Q_dist = Distribution(loc=get_mean(Q), covariance_matrix=get_cov(Q))
		
			self.KLD_vec.append([P_dist.kl_divergence(Q_dist).numpy(), self.total_count.numpy()])
		except:
			print("KLD error - Falling back to prev value")
			try:
				self.KLD_vec.append([self.KLD_vec[-1]*0.9, self.total_count.numpy()])
			except:
				self.KLD_vec.append([0, self.total_count.numpy()])
		# print('KLD: ',self.KLD_vec[-1])
		return


	def update_KLD(self):
		
		if self.topic == 'ELeGANt':
			if self.loss == 'FS' and (self.latent_kind == 'AE' or self.latent_kind == 'AAE'):
				self.KLD_func(self.reals_enc,self.fakes_enc)
			else:
				self.KLD_func(self.reals,self.fakes)
		elif self.topic == 'AAE':
			self.KLD_func(self.fakes_enc,self.reals_enc)
		else:
			self.KLD_func(self.reals,self.fakes)

			

	def print_KLD(self):
		path = self.metricpath
		if self.colab:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size":12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		vals = list(np.array(self.KLD_vec)[:,0])
		locs = list(np.array(self.KLD_vec)[:,1])
		if self.topic == 'ELeGANt':
			if self.loss == 'FS' and self.latent_kind == 'AE':
				locs = list(np.array(self.KLD_vec)[:,1] - self.AE_steps)
		

		with PdfPages(path+'KLD_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'KL Divergence Vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

	def update_Lambda(self):
		self.lambda_vec.append([self.lamb.numpy(),self.total_count.numpy()])

	def print_Lambda(self):
		path = self.metricpath
		if self.colab:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size": 12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		# lbasis  = np.expand_dims(np.array(np.arange(0,len(self.lambda_vec))),axis=1)
		vals = list(np.array(self.lambda_vec)[:,0])
		locs = list(np.array(self.lambda_vec)[:,1])

		with PdfPages(path+'Lambda_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.plot(locs,vals, c='r',label = 'Lambda Vs. Iterations')
			ax1.legend(loc = 'upper right')
			pdf.savefig(fig1)
			plt.close(fig1)

	def print_GradGrid(self):

		path = self.metricpath + str(self.total_count.numpy()) + '_'

		if self.colab:
			from matplotlib.backends.backend_pdf import PdfPages
			plt.rc('text', usetex=False)
		else:
			from matplotlib.backends.backend_pgf import PdfPages
			plt.rcParams.update({
				"pgf.texsystem": "pdflatex",
				"font.family": "helvetica",  # use serif/main font for text elements
				"font.size": 12,
				"text.usetex": True,     # use inline math for ticks
				"pgf.rcfonts": False,    # don't setup fonts from rc parameters
			})

		
		from itertools import product as cart_prod

		x = np.arange(self.MIN,self.MAX+0.1,0.1)
		y = np.arange(self.MIN,self.MAX+0.1,0.1)

		# X, Y = np.meshgrid(x, y)
		prod = np.array([p for p in cart_prod(x,repeat = 2)])
		# print(x,prod)

		X = prod[:,0]
		Y = prod[:,1]

		# print(prod,X,Y)
		# print(XXX)

		with tf.GradientTape() as disc_tape:
			prod = tf.cast(prod, dtype = 'float32')
			disc_tape.watch(prod)
			d_vals = self.discriminator(prod,training = False)
		grad_vals = disc_tape.gradient(d_vals, [prod])[0]

		#Flag to control normalization of D(x) values for printing on the contour plot
		Normalize_Flag = False
		try:
			# print(d_vals[0])
			
			if Normalize_Flag and ((min(d_vals[0]) <= -2) or (max(d_vals[0]) >= 2)):
				### IF NORMALIZATION IS NEEDED
				d_vals_sub = d_vals[0] - min(d_vals[0])
				d_vals_norm = d_vals_sub/max(d_vals_sub)
				d_vals_norm -= 0.5
				d_vals_norm *= 3
				# d_vals_new = np.expand_dims(np.array(d_vals_norm),axis = 1)
				d_vals_new = np.reshape(d_vals_norm,(x.shape[0],y.shape[0])).transpose()
				# d_vals_norm = np.expand_dims(np.array(d_vals_sub/max(d_vals_sub)),axis = 1)
				# d_vals_new = np.subtract(d_vals_norm,0.5)
				# d_vals_new = np.multiply(d_vals_new,3.)
				# print(d_vals_new)
			else:
				### IF NORMALIZATION IS NOT NEEDED
				d_vals_norm = d_vals[0]
				d_vals_new = np.reshape(d_vals_norm,(x.shape[0],y.shape[0])).transpose()
		except:
			d_vals_new = np.reshape(d_vals,(x.shape[0],y.shape[0])).transpose()
		# print(d_vals_new)
		dx = grad_vals[:,1]
		dy = grad_vals[:,0]
		# print(XXX)
		n = -1
		color_array = np.sqrt(((dx-n)/2)**2 + ((dy-n)/2)**2)

		with PdfPages(path+'GradGrid_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(True)
			ax1.get_yaxis().set_visible(True)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim(bottom=self.MIN,top=self.MAX)
			ax1.quiver(X,Y,dx,dy,color_array)
			ax1.scatter(self.reals[:1000,0], self.reals[:1000,1], c='r', linewidth = 1, label='Real Data', marker = '.', alpha = 0.1)
			ax1.scatter(self.fakes[:1000,0], self.fakes[:1000,1], c='g', linewidth = 1, label='Fake Data', marker = '.', alpha = 0.1)
			pdf.savefig(fig1)
			plt.close(fig1)

		with PdfPages(path+'Contourf_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(False)
			ax1.get_yaxis().set_visible(False)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim([self.MIN,self.MAX])
			cs = ax1.contourf(x,y,d_vals_new,alpha = 0.5, levels = list(np.arange(-1.5,1.5,0.1)), extend = 'both' )
			ax1.scatter(self.reals[:,0], self.reals[:,1], c='r', linewidth = 1, marker = '.', alpha = 0.75)
			ax1.scatter(self.fakes[:,0], self.fakes[:,1], c='g', linewidth = 1, marker = '.', alpha = 0.75)
			# cbar = fig1.colorbar(cs, shrink=1., orientation = 'horizontal')
			# Can be used with figure size (2,10) to generate a colorbar with diff colors as plotted
			# Good for a table wit \multicol{5}
			# cbar = fig1.colorbar(cs, aspect = 40, shrink=1., ticks = [0, 1.0], orientation = 'horizontal')
			# cbar.ax.set_xticklabels(['Min', 'Max'])
			# # cbar.set_ticks_position(['bottom', 'top'])
			pdf.savefig(fig1)
			plt.close(fig1)

		with PdfPages(path+'Contourf_plot_cBar.pdf') as pdf:

			fig1 = plt.figure(figsize=(8, 8))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(False)
			ax1.get_yaxis().set_visible(False)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim([self.MIN,self.MAX])
			cs = ax1.contourf(x,y,d_vals_new,alpha = 0.5, levels = list(np.arange(-1.5,1.6,0.1)), extend = 'both' )
			ax1.scatter(self.reals[:,0], self.reals[:,1], c='r', linewidth = 1, marker = '.', alpha = 0.75)
			ax1.scatter(self.fakes[:,0], self.fakes[:,1], c='g', linewidth = 1, marker = '.', alpha = 0.75)
			# cbar = fig1.colorbar(cs, shrink=1., orientation = 'horizontal')
			# Can be used with figure size (10,2) to generate a colorbar with diff colors as plotted
			# Good for a table wit \multicol{5}
			cbar = fig1.colorbar(cs, aspect = 40, shrink=1., ticks = [-1.5, -1, -0.5, 0, 0.5, 1, 1.5], orientation = 'horizontal')
			cbar.ax.set_xticklabels(['$-1.5$', '$-1$', '$-0.5$', '$0$', '$0.5$', '$1$', '$1.5$'])
			# # cbar.set_ticks_position(['bottom', 'top'])
			pdf.savefig(fig1)
			plt.close(fig1)

		with PdfPages(path+'Contour_plot.pdf') as pdf:

			fig1 = plt.figure(figsize=(3.5, 3.5))
			ax1 = fig1.add_subplot(111)
			ax1.cla()
			ax1.get_xaxis().set_visible(False)
			ax1.get_yaxis().set_visible(False)
			ax1.set_xlim([self.MIN,self.MAX])
			ax1.set_ylim([self.MIN,self.MAX])
			ax1.contour(x,y,d_vals_new,10,linewidths = 0.5,alpha = 0.4 )
			ax1.scatter(self.reals[:,0], self.reals[:,1], c='r', linewidth = 1, marker = '.', alpha = 0.75)
			ax1.scatter(self.fakes[:,0], self.fakes[:,1], c='g', linewidth = 1, marker = '.', alpha = 0.75)
			# cbar = fig1.colorbar(cs, shrink=1., orientation = 'horizontal')
			pdf.savefig(fig1)
			plt.close(fig1)
