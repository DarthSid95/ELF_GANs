from __future__ import print_function
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import glob
from absl import flags
import csv

from scipy import io as sio

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pgf import PdfPages

import tensorflow_datasets as tfds
# import tensorflow_datasets as tfds


### Need to prevent tfds downloads bugging out? check
import urllib3
urllib3.disable_warnings()


FLAGS = flags.FLAGS

'''***********************************************************************************
********** Base Data Loading Ops *****************************************************
***********************************************************************************'''
class GAN_DATA_ops:

	def __init__(self):
		self.gen_func = 'self.gen_func_'+self.data+'()'
		self.dataset_func = 'self.dataset_'+self.data+'(self.train_data)'
		self.noise_mean = 0.0
		self.noise_stddev = 1.00
		#Default Number of repetitions of a dataset in tf.dataset mapping
		self.reps = 1

		if self.data == 'g1':
			self.MIN = 0
			self.MAX = 1
			self.noise_dims = 1
			self.output_size = 1
			self.noise_mean = 0.0
			self.noise_stddev = 1.0
		elif self.data == 'g2':
			self.MIN = -1
			self.MAX = 1.2
			self.noise_dims = 2
			self.output_size = 2
			self.noise_mean = 0.0
			self.noise_stddev = 1.0
		elif self.data == 'gmm8':
			self.noise_dims = 100
			self.output_size = 2
			self.noise_mean = 0.0
			self.noise_stddev = 1.0
		else: 
			self.noise_dims = 100
			if self.data in ['celeba', 'ukiyoe']:
				self.output_size = self.out_size
			elif self.data == 'mnist':
				self.output_size = 28
			elif self.data in ['cifar10', 'svhn']:
				self.output_size = 32


	def mnist_loader(self):
		(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
		train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], 1).astype('float32')
		train_labels = train_labels.reshape(train_images.shape[0], 1).astype('float32')
		train_images = (train_images - 127.5) / 127.5
		# train_images = (train_images - 0.) / 255.0
		test_images = test_images.reshape(test_images.shape[0],test_images.shape[1], test_images.shape[2], 1).astype('float32')
		self.test_images = (test_images - 127.5) / 127.5
		# self.test_images = (test_images - 0.) / 255.0

		return train_images, train_labels, test_images, test_labels


	def svhn_loader(self):
		SVHN_train_data = sio.loadmat('data/SVHN/train_32x32.mat')

		train_images = tf.transpose(tf.cast(SVHN_train_data['X'],dtype='float32'),[3,0,1,2]).numpy()
		train_images= (train_images - 127.5) / 127.5
		train_labels = SVHN_train_data['y']

		return train_images, train_labels


	def celeba_loader(self):
		if self.colab:
			try:
				with open("data/CelebA/Colab_CelebA_Names.txt","r") as names:
					true_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")
			except:
				true_files = sorted(glob.glob('/content/colab_data_faces/img_align_celeba/*.jpg'))
				print("Data File Created. Saving filenames")
				with open("data/CelebA/Colab_CelebA_Names.txt","w") as names:
					for name in true_files:
						names.write(str(name)+'\n')
		else:
			try:
				with open("data/CelebA/CelebA_Names.txt","r") as names:
					true_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")
			except:
				true_files = sorted(glob.glob('data/CelebA/img_align_celeba/*.jpg'))
				print("Data File Created. Saving filenames")
				with open("data/CelebA/CelebA_Names.txt","w") as names:
					for name in true_files:
						names.write(str(name)+'\n')
		train_images = np.expand_dims(np.array(true_files),axis=1)

		attr_file = 'data/CelebA/list_attr_celeba.csv'

		with open(attr_file,'r') as a_f:
			data_iter = csv.reader(a_f,delimiter = ',',quotechar = '"')
			data = [data for data in data_iter]
		# print(data,len(data))
		label_array = np.asarray(data)

		return train_images, label_array


	def ukiyoe_loader(self):
		if self.colab:
			try:
				with open("data/local_data/Colab_UkiyoE_Names.txt","r") as names:
					true_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")
			except:
				true_files = sorted(glob.glob('/content/local_data/ukiyoe-1024/*.jpg'))
				print("Data File Created. Saving filenames")
				with open("data/UkiyoE/Colab_UkiyoE_Names.txt","w") as names:
					for name in true_files:
						names.write(str(name)+'\n')
		else:
			try:
				with open("data/UkiyoE/CelebA_Names.txt","r") as names:
					true_files = np.array([line.rstrip() for line in names])
					print("Data File Found. Reading filenames")
			except:
				true_files = sorted(glob.glob('data/UkiyoE/ukiyoe-1024/*.jpg'))
				print("Data File Created. Saving filenames")
				with open("data/UkiyoE/CelebA_Names.txt","w") as names:
					for name in true_files:
						names.write(str(name)+'\n')
		train_images = np.expand_dims(np.array(true_files),axis=1)

		return train_images


	def cifar10_loader(self):

		(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
		train_images = train_images.reshape(train_images.shape[0],self.output_size, self.output_size, 3).astype('float32')
		train_labels = train_labels.reshape(train_images.shape[0], 1).astype('float32')
		train_images = (train_images - 127.5) / 127.5
		test_images = test_images.reshape(test_images.shape[0],self.output_size, self.output_size, 3).astype('float32')
		test_labels = test_labels.reshape(test_images.shape[0], 1).astype('float32')
		test_images = (test_images - 127.5) / 127.5

		return train_images, train_labels, test_images, test_labels


'''
GAN_DATA functions are specific to the topic, ELeGANt, RumiGAN, PRDeep or DCS. Data reading and dataset making functions per data, with init having some specifics generic to all, such as printing instructions, noise params. etc.
'''
'''***********************************************************************************
********** GAN_DATA_Baseline *********************************************************
***********************************************************************************'''
class GAN_DATA_Base(GAN_DATA_ops):

	def __init__(self):#,data,testcase,number,out_size):
		self.noise_mean = 0.0
		self.noise_stddev = 1.00
		GAN_DATA_ops.__init__(self)

	def gen_func_g1(self):
		self.MIN = -3.5
		self.MAX = 10.5
		g1 = tfp.distributions.TruncatedNormal(loc=self.data_mean, scale=self.data_var, low=-20., high=20.)
		# g1 = tfp.distributions.TruncatedNormal(loc=1.0, scale=0.50, low=-20., high=20.)
		return g1.sample([800*self.batch_size, 1])
		# return tf.random.normal([1000*self.batch_size, 1], mean = 8.0, stddev = 1.)


	def dataset_g1(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)
		return train_dataset

	def gen_func_g2(self):
		self.MIN = -2.2
		self.MAX = 2.
		return tf.random.normal([500*self.batch_size.numpy(), 2], mean =np.array([1.0,1.0]), stddev = np.array([0.20,0.20]))

	def dataset_g2(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)
		return train_dataset

	def gen_func_gmm8(self):
		tfd = tfp.distributions
		probs = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125]
		## Cirlce
		# locs = [[1., 0.], [0., 1.], [-1.,0.], [0.,-1.], [1*0.7071, 1*0.7071], [-1*0.7071, 1*0.7071], [1*0.7071, -1*0.7071], [-1*0.7071, -1*0.7071] ]
		# self.MIN = -1.3 #-1.3 for circle, 0 for pattern
		# self.MAX = 1.3 # +1.3 for cicle , 1 for pattern

		## ?
		# locs = [[0.25, 0.], [0., 0.25], [-0.25,0.], [0.,-0.25], [0.25*0.7071, 0.5*0.7071], [-0.25*0.7071, 0.25*0.7071], [0.25*0.7071, -0.25*0.7071], [-0.25*0.7071, -0.25*0.7071] ]

		## random
		# locs = [[0.75, 0.5], [0.5, 0.75], [0.25,0.5], [0.5,0.25], [0.75*0.7071, 0.75*0.7071], [0.25*0.7071, 0.75*0.7071], [0.75*0.7071, 0.25*0.7071], [0.25*0.7071, 0.25*0.7071] ]

		## Pattern
		locs = [[0.75, 0.5], [0.5, 0.75], [0.25,0.5], [0.5,0.25], [0.5*1.7071, 0.5*1.7071], [0.5*0.2929, 0.5*1.7071], [0.5*1.7071, 0.5*0.2929], [0.5*0.2929, 0.5*0.2929] ]
		self.MIN = -0. #-1.3 for circle, 0 for pattern
		self.MAX = 1.0 # +1.3 for cicle , 1 for pattern

		# stddev_scale = [.04, .04, .04, .04, .04, .04, .04, .04]
		stddev_scale = [.03, .03, .03, .03, .03, .03, .03, .03]
		# stddev_scale = [1., 1., 1., 1., 1., 1., 1., 1. ]
		# stddev_scale = [1., 1., 1., 1., 1., 1., 1., 1. ]
		# covs = [ [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]], [[0.001, 0.],[0., 0.001]]   ]

		gmm = tfd.MixtureSameFamily(mixture_distribution=tfd.Categorical(probs=probs),components_distribution=tfd.MultivariateNormalDiag(loc=locs,scale_identity_multiplier=stddev_scale))

		return gmm.sample(sample_shape=(int(500*self.batch_size.numpy())))

	def dataset_gmm8(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		train_dataset = train_dataset.shuffle(4)
		train_dataset = train_dataset.batch(batch_size)
		train_dataset = train_dataset.prefetch(5)
		return train_dataset



'''***********************************************************************************
********** GAN_DATA_WAE **************************************************************
***********************************************************************************'''
class GAN_DATA_WAE(GAN_DATA_ops):

	def __init__(self):#,data,testcase,number,out_size):
		GAN_DATA_ops.__init__(self)#,data,testcase,number,out_size)


	def gen_func_mnist(self):
		train_images, train_labels, test_images, test_labels = self.mnist_loader()

		self.fid_train_images = train_images

		if self.testcase == 'single':
			train_images = train_images[np.where(train_labels == self.number)[0]][0:50]
			self.reps = int(60000/train_images.shape[0])+1
		if self.testcase == 'even':
			train_images = train_images[np.where(train_labels%2 == 0)[0]]
		if self.testcase == 'odd':
			train_images = train_images[np.where(train_labels%2 != 0)[0]]

		return train_images

	def dataset_mnist(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		if self.testcase == 'single':
			train_dataset = train_dataset.repeat(self.reps-1)
		train_dataset = train_dataset.shuffle(40)
		train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
		train_dataset = train_dataset.prefetch(10)

		recon_dataset = tf.data.Dataset.from_tensor_slices((self.test_images[0:10000:100]))
		self.recon_dataset = recon_dataset.batch(100,drop_remainder=True)

		interp_dataset = tf.data.Dataset.from_tensor_slices((self.test_images[0:10000:25]))
		interp_dataset = interp_dataset.shuffle(10)
		self.interp_dataset = interp_dataset.batch(400,drop_remainder=True)


		return train_dataset


	def gen_func_svhn(self):
		train_images, train_labels = self.svhn_loader()

		self.fid_train_images = train_images

		if self.testcase == 'single':
			train_images = train_images[np.where(train_labels == self.number)[0]][0:50]
			self.reps = int(60000/train_images.shape[0])+1
		if self.testcase == 'even':
			train_images = train_images[np.where(train_labels%2 == 0)[0]]
		if self.testcase == 'odd':
			train_images = train_images[np.where(train_labels%2 != 0)[0]]

		return train_images

	def dataset_svhn(self,train_data,batch_size):
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		if self.testcase == 'single':
			train_dataset = train_dataset.repeat(self.reps-1)
		train_dataset = train_dataset.shuffle(40)
		train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
		train_dataset = train_dataset.prefetch(10)

		recon_dataset = tf.data.Dataset.from_tensor_slices((train_data[0:10000:100]))
		self.recon_dataset = recon_dataset.batch(100,drop_remainder=True)

		interp_dataset = tf.data.Dataset.from_tensor_slices((train_data[0:10000:25]))
		self.interp_dataset = interp_dataset.batch(400,drop_remainder=True)

		return train_dataset

	def gen_func_celeba(self):

		train_images, data_array = self.celeba_loader()
		# print(data_array,data_array.shape)
		tags = data_array[0,:] # print to find which col to pull for what
		gender = data_array[1:,21]
		bald_labels = data_array[1:,5]
		hat_labels = data_array[1:,-5]
		# print(gender,gender.shape)
		male = gender == '1'
		male = male.astype('uint8')

		bald = bald_labels == '1'
		bald = bald.astype('uint8')

		hat = hat_labels == '1'
		hat = hat.astype('uint8')

		self.fid_train_images = train_images

		if self.testcase == 'single':
			self.fid_train_images = train_images[np.where(male == 0)]
			train_images = np.repeat(train_images[np.where(male == 0)][0:5000],20,axis = 0)
		if self.testcase == 'female':
			train_images = train_images[np.where(male == 0)]
			self.fid_train_images = train_images
		if self.testcase == 'male':
			train_images = train_images[np.where(male == 1)]
			self.fid_train_images = train_images
		if self.testcase == 'bald':
			self.fid_train_images = train_images[np.where(bald == 1)]
			train_images = np.repeat(train_images[np.where(bald == 1)],20,axis = 0)
		if self.testcase == 'hat':
			self.fid_train_images = train_images[np.where(hat == 1)]
			train_images = np.repeat(train_images[np.where(hat == 1)],20,axis = 0)


		return train_images

	def dataset_celeba(self,train_data,batch_size):	
		def data_reader_faces(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([218,178,3])
				image  = tf.image.crop_to_bounding_box(image, 38, 18, 140,140)
				image = tf.image.resize(image,[self.output_size,self.output_size])
				# This will convert to float values in [0, 1]
				# image = tf.divide(image,255.0)
				image = tf.subtract(image,127.0)
				image = tf.divide(image,127.0)
				# image = tf.image.convert_image_dtype(image, tf.float16)
			return image

		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		# if not self.colab:
		train_dataset = train_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
		train_dataset = train_dataset.shuffle(500)
		train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
		train_dataset = train_dataset.prefetch(15)

		recon_dataset = tf.data.Dataset.from_tensor_slices((train_data[0:100]))
		recon_dataset=recon_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
		self.recon_dataset = recon_dataset.batch(100,drop_remainder=True)

		interp_dataset = tf.data.Dataset.from_tensor_slices((train_data[0:400]))
		interp_dataset=interp_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
		self.interp_dataset = interp_dataset.batch(400,drop_remainder=True)

		return train_dataset

	def gen_func_ukiyoe(self):
		train_images = self.ukiyoe_loader()
		# print(data_array,data_array.shape)
		self.fid_train_images = train_images
		self.reps = 20
		return train_images

	def dataset_ukiyoe(self,train_data,batch_size):	
		def data_reader_faces(filename):
			with tf.device('/CPU'):
				print(tf.cast(filename[0],dtype=tf.string))
				image_string = tf.io.read_file(tf.cast(filename[0],dtype=tf.string))
				# Don't use tf.image.decode_image, or the output shape will be undefined
				image = tf.image.decode_jpeg(image_string, channels=3)
				image.set_shape([1024,1024,3])
				image = tf.image.resize(image,[self.output_size,self.output_size])
				# This will convert to float values in [0, 1]
				# image = tf.divide(image,255.0)
				image = tf.subtract(image,127.0)
				image = tf.divide(image,127.0)
				# image = tf.image.convert_image_dtype(image, tf.float16)
			return image

		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		# if not self.colab:
		train_dataset = train_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
		train_dataset = train_dataset.repeat(self.reps-1)
		train_dataset = train_dataset.shuffle(5000)
		train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
		train_dataset = train_dataset.prefetch(5)

		recon_dataset = tf.data.Dataset.from_tensor_slices((train_data[0:100]))
		recon_dataset=recon_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
		self.recon_dataset = recon_dataset.batch(100,drop_remainder=True)

		interp_dataset = tf.data.Dataset.from_tensor_slices((train_data[0:400]))
		interp_dataset=interp_dataset.map(data_reader_faces,num_parallel_calls=int(self.num_parallel_calls))
		self.interp_dataset = interp_dataset.batch(400,drop_remainder=True)

		return train_dataset


	def gen_func_cifar10(self):
		# self.output_size = int(28)

		train_images, train_labels, test_images, test_labels = self.cifar10_loader()

		self.test_images = test_images
		self.fid_train_images = train_images

		if self.testcase == 'single':
			train_images = train_images[np.where(train_labels == self.number)[0]]
			self.fid_train_images = train_images
			self.test_images = self.test_images[np.where(test_labels == self.number)[0]]
			self.reps = int(50000/train_images.shape[0])+1

		return train_images

	def dataset_cifar10(self,train_data,batch_size):

		def data_to_grey(image):
			image = tf.image.rgb_to_grayscale(image)
			return image
		
		train_dataset = tf.data.Dataset.from_tensor_slices((train_data))
		# train_dataset = train_dataset.map(data_to_grey, num_parallel_calls=int(self.num_parallel_calls))
		if self.testcase == 'single':
			train_dataset = train_dataset.repeat(self.reps-1)
		train_dataset = train_dataset.shuffle(10)
		train_dataset = train_dataset.batch(batch_size,drop_remainder=True)
		train_dataset = train_dataset.prefetch(5)

		recon_dataset = tf.data.Dataset.from_tensor_slices((self.test_images))	
		self.recon_dataset = recon_dataset.batch(100,drop_remainder=True)

		interp_dataset = tf.data.Dataset.from_tensor_slices((self.test_images[0:400]))	
		self.interp_dataset = interp_dataset.batch(400,drop_remainder=True)

		return train_dataset










