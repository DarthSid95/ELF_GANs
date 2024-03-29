from __future__ import print_function
import os, sys, time, argparse, signal, json, struct
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# import tensorflow_addons as tfa
from tensorflow.python import debug as tf_debug
import traceback

print(tf.__version__)
from absl import app
from absl import flags



# from mnist_cnn_icp_eval import *
# tf.keras.backend.set_floatx('float64')

def signal_handler(sig, frame):
	print('\n\n\nYou pressed Ctrl+C! \n\n\n')
	sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

'''Generic set of FLAGS. learning_rate and batch_size are redefined in GAN_ARCH if g1/g2'''
FLAGS = flags.FLAGS
flags.DEFINE_float('lr_G', 0.0001, """learning rate for generator""")
flags.DEFINE_float('lr_D', 0.0001, """learning rate for discriminator""")
flags.DEFINE_float('beta1', 0.5, """beta1 for Adam""")
flags.DEFINE_float('beta2', 0.9, """beta2 for Adam""")
flags.DEFINE_integer('colab', 0, """ set 1 to run code in a colab friendy way """)
flags.DEFINE_integer('homo_flag', 1, """ set 1 to read data in a colab friendy way """)
flags.DEFINE_integer('batch_size', 100, """Batch size.""")
flags.DEFINE_integer('paper', 1, """1 for saving images for a paper""")
flags.DEFINE_integer('resume', 1, """1 vs 0 for Yes Vs. No""")
flags.DEFINE_integer('saver', 1, """1-Save events for Tensorboard. 0 O.W.""")
flags.DEFINE_integer('res_flag', 1, """1-Write results to a file. 0 O.W.""")
flags.DEFINE_integer('pbar_flag', 1, """1-Display Progress Bar, 0 O.W.""")
flags.DEFINE_integer('out_size', 32, """CelebA output reshape size""")
flags.DEFINE_list('metrics', '', 'CSV for the metrics to evaluate. KLD, FID, PR')
flags.DEFINE_integer('save_all', 0, """1-Save all the models. 0 for latest 10""") #currently functions as save_all internally
flags.DEFINE_integer('seed', 42, """Initialize the random seed of the run (for reproducibility).""")
flags.DEFINE_integer('num_epochs', 200, """Number of epochs to train for.""")
flags.DEFINE_integer('Dloop', 1, """Number of loops to run for D.""")
flags.DEFINE_integer('Gloop', 1, """Number of loops to run for G.""")
flags.DEFINE_integer('ODE_step', 1, """Number of loops to run for G. DEPRICATED UNTIL FIX""")
flags.DEFINE_integer('num_parallel_calls', 5, """Number of parallel calls for dataset map function""")
flags.DEFINE_string('run_id', 'default', """ID of the run, used in saving.""")
flags.DEFINE_string('log_folder', 'default', """ID of the run, used in saving.""")
flags.DEFINE_string('mode', 'train', """Operation mode: train, test, fid """)
flags.DEFINE_string('topic', 'ELeGANt', """Base or ELeGANt""")
flags.DEFINE_string('data', 'mnist', """Type of Data to run for""")
flags.DEFINE_string('gan', 'WAE', """WGAN or WAE""")
flags.DEFINE_string('loss', 'base', """Type of Loss function to use""")
flags.DEFINE_string('GPU', '0,1', """GPU's made visible '0', '1', or '0,1' """)
flags.DEFINE_string('device', '0', """Which GPU device to run on: 0,1 or -1(CPU)""")
flags.DEFINE_string('noise_kind', 'gaussian', """Type of Noise for WAE latent prior""")

flags.DEFINE_float('data_mean', 0.0, """Mean of taget Gaussian data""")
flags.DEFINE_float('data_var', 1.0, """Variance of taget Gaussian data""")

'''Flags just for WGAN-FS forms'''
flags.DEFINE_integer('terms', 50, """N for 0-M for FS.""") #Matters only if g
flags.DEFINE_float('sigma',75, """approximation sigma of data distribution""") 
flags.DEFINE_integer('lambda_d', 20000, """Period as a multiple of sigmul*sigma""") ##NeedToKill
flags.DEFINE_string('latent_kind', 'base', """AE/DCT/W/AE2/AE3/Cycle - need to make W""") ##NeedToKill
flags.DEFINE_string('distribution', 'generic', """generic/gaussian""")
flags.DEFINE_integer('latent_dims', 10, """Dimension of latent representation""") #20 on GMM 8 worked #Matters only if not g  ;AE3 takes lxl 14 or 7; DCt lxl
flags.DEFINE_integer('L', 25000, """Number of terms in summation""")
# flags.DEFINE_integer('AE_steps', 20000, """Dimension of latent representation""") #1000 for GMM8
flags.DEFINE_integer('GAN_pretrain_epochs', 5, """Num of AE pre-training Epochs""")
flags.DEFINE_integer('train_D', 0, """Set 1 to backprop and update Disc FS weights with backprop""")
flags.DEFINE_string('FID_kind', 'none', """if FID is latent, calculates on WAE latent space""")

flags.DEFINE_float('lr_AE_Enc', 0.01, """learning rate""")
flags.DEFINE_float('lr_AE_Dec', 0.01, """learning rate""")



FLAGS(sys.argv)
from models import *


if __name__ == '__main__':
	'''Enable Flags and various tf declarables on GPU processing '''
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.GPU
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	print('Visible Physical Devices: ',physical_devices)
	for gpu in physical_devices:
		print(gpu)
		tf.config.experimental.set_memory_growth(gpu, True)
	tf.config.threading.set_inter_op_parallelism_threads(6)
	tf.config.threading.set_intra_op_parallelism_threads(6)

	
	# Level | Level for Humans | Level Description                  
	# ------|------------------|------------------------------------ 
	# 0     | DEBUG            | [Default] Print all messages       
	# 1     | INFO             | Filter out INFO messages           
	# 2     | WARNING          | Filter out INFO & WARNING messages 
	# 3     | ERROR            | Filter out all messages
	tf.get_logger().setLevel('ERROR')
	# tf.debugging.set_log_device_placement(True)
	# if FLAGS.colab and FLAGS.data == 'celeba':
	os.environ["TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD"] = "500G"
	if FLAGS.colab:
		import warnings
		warnings.filterwarnings("ignore")



	''' Set random seed '''
	np.random.seed(FLAGS.seed)
	tf.random.set_seed(FLAGS.seed)

	FLAGS_dict = FLAGS.flag_values_dict()

	###	EXISTING Variants:
	##
	##
	##	(3) WGAN - 
	##		(A) Base
	##		(B) ELF_GAN
	##
	##	(4) WAE - 
	##		(A) Base
	##		(B) FS
	##
	##
	### -----------------

	gan_call = FLAGS.gan + '_' + FLAGS.topic + '(FLAGS_dict)'

	# with tf.device('/GPU:'+FLAGS.device):
	# try:
	print('GAN setup')
	gan = eval(gan_call)
	gan.initial_setup()
	gan.get_data()
	gan.create_models()
	gan.create_optimizer()
	gan.create_load_checkpoint()
	print('Worked')

	if gan.mode == 'train':
		print(gan.mode)
		gan.train()
		gan.test()

	if gan.mode == 'h5_from_checkpoint':
		gan.h5_from_checkpoint()

	if gan.mode == 'test':
		gan.test()

	if gan.mode == 'metrics':
		gan.eval_metrics()

	# except Exception as e:
	# 	print("Exiting Execution due to error.")
	# 	print(e)
	# 	exit(0)

###############################################################################  
	
	
	print('Completed.')
