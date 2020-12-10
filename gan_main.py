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
flags.DEFINE_string('topic', 'ELeGANt', """ELeGANt or RumiGAN""")
flags.DEFINE_string('data', 'mnist', """Type of Data to run for""")
flags.DEFINE_string('gan', 'sgan', """Type of GAN for""")
flags.DEFINE_string('loss', 'base', """Type of Loss function to use""")
flags.DEFINE_string('GPU', '0,1', """GPU's made visible '0', '1', or '0,1' """)
flags.DEFINE_string('device', '0', """Which GPU device to run on: 0,1 or -1(CPU)""")
flags.DEFINE_string('noise_kind', 'gaussian', """Type of Noise for WAE latent prior""")
flags.DEFINE_string('noise_data', 'mnist', """Type of Data to feed as noise""")


'''Flags just for RumiGAN'''
flags.DEFINE_integer('number', 3, """ Class selector in Multi-class data""")
flags.DEFINE_integer('num_few', 200, """ 200 for MNIST, 1k for C10 and 10k for CelebA""")
flags.DEFINE_integer('GaussN', 3, """ N for Gaussian""")
flags.DEFINE_string('testcase', 'female', """Test cases for RumiGAN""")
flags.DEFINE_string('label_style', 'base', """base vs. embed for how labels are fed to the net""")
flags.DEFINE_float('label_a', -0.5, """Class label - a """)
flags.DEFINE_float('label_bp', 2.0, """Class label - bp for +ve data """)
flags.DEFINE_float('label_bn', -2.0, """Class label - bn for -ve data """)
flags.DEFINE_float('label_c', 2.0, """Class label - c for generator """)
flags.DEFINE_float('alphap', 0.9, """alpha weight for +ve class """)
flags.DEFINE_float('alphan', 0.1, """alpha weight for -ve class""")

'''
Defined Testcases:
1. even - learn only the even numbers 
2. odd - learn only the odd mnist numbers
3. male - learn males in CelebA
4. female - learn females in CelebA
5. single - learn a single digit in MNIST - uses number flag to deice number
'''

'''Flags just for WGAN-FS forms'''
flags.DEFINE_integer('terms', 15, """N for 0-M for FS.""") #Matters only if g
flags.DEFINE_float('sigma',10, """approximation sigma of data distribution""") 
flags.DEFINE_integer('lambda_d', 20000, """Period as a multiple of sigmul*sigma""")
flags.DEFINE_integer('sigmul', 1, """Period as a multiple of sigmul*sigma""") #10e5 is mnist, 10e3 if g ### NEW 100 works for MNIST, 10^5 is celeba ### Set to 1. make 100 for save for paper in 1D
flags.DEFINE_string('latent_kind', 'AE', """AE/DCT/W/AE2/AE3/Cycle - need to make W""")
flags.DEFINE_string('distribution', 'generic', """generic/gaussian""")
flags.DEFINE_integer('latent_dims', 10, """Dimension of latent representation""") #20 on GMM 8 worked #Matters only if not g  ;AE3 takes lxl 14 or 7; DCt lxl
flags.DEFINE_integer('L', 25000, """Number of terms in summation""")
flags.DEFINE_integer('AE_steps', 20000, """Dimension of latent representation""") #1000 for GMM8
flags.DEFINE_integer('AE_count', 5, """Num of AE pre-training Epochs""")
flags.DEFINE_integer('train_D', 0, """Dimension of latent representation""")
flags.DEFINE_string('FID_kind', 'none', """if FID is latent, calculates on WAE latent space""")

flags.DEFINE_float('lr_AE_Enc', 0.01, """learning rate""")
flags.DEFINE_float('lr_AE_Dec', 0.01, """learning rate""")
flags.DEFINE_float('lr_GenEnc', 0.0001, """learning rate""") #0.1 here works on base g
flags.DEFINE_float('lr_GenDec', 0.01, """learning rate""")
flags.DEFINE_float('lr_Disc', 0.0, """learning rate""")
flags.DEFINE_float('lr2_AE_Enc', 0.0000001, """learning rate""")
flags.DEFINE_float('lr2_AE_Dec', 0.0000001, """learning rate""")
flags.DEFINE_float('lr2_GenEnc', 0.000001, """learning rate""") #0.001 is good? 0.00001 too high for g2
flags.DEFINE_float('lr2_GenDec', 0, """learning rate""")
flags.DEFINE_float('lr2_Disc', 0.0, """learning rate""")



def email_success():
	import smtplib, ssl

	port = 587  # For starttls
	smtp_server = "smtp.gmail.com"
	sender_email = "darthsidcodes@gmail.com"
	receiver_email = "darthsidcodes@gmail.com"
	password = "kamehamehaX100#"
	SUBJECT = "Execution Completed"
	TEXT = "Execution of code: "+str(gan.run_id)+" completed Successfully."
	message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT) 
	context = ssl.create_default_context()
	with smtplib.SMTP(smtp_server, port) as server:
		server.ehlo()  # Can be omitted
		server.starttls(context=context)
		server.ehlo()  # Can be omitted
		server.login(sender_email, password)
		server.sendmail(sender_email, receiver_email, message)


def email_error(error):
	traceback.print_exc()
	error = traceback.format_exc()
	import smtplib, ssl

	port = 587  # For starttls
	smtp_server = "smtp.gmail.com"
	sender_email = "darthsidcodes@gmail.com"
	receiver_email = "darthsidcodes@gmail.com"
	password = "kamehamehaX100#"
	SUBJECT = "An Error Occured"
	TEXT = "Error: "+error
	message = 'Subject: {}\n\n{}'.format(SUBJECT, TEXT) 
	context = ssl.create_default_context()
	with smtplib.SMTP(smtp_server, port) as server:
		server.ehlo()  # Can be omitted
		server.starttls(context=context)
		server.ehlo()  # Can be omitted
		server.login(sender_email, password)
		server.sendmail(sender_email, receiver_email, message)



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
	##	(1) SGAN - 
	##		(A) Base
	##		(B) RumiGAN
	##		(C) ACGAN
	##
	##	(2) LSGAN - 
	##		(A) Base
	##		(B) RumiGAN
	##
	##	(3) WGAN - 
	##		(A) Base
	##		(B) ELeGANt
	##		(C) Rumi
	##
	##	(4) WAE - 
	##		(A) Base
	##		(B) ELeGANt
	##
	##
	### -----------------
	### Have to add CycleGAN for future work. Potentially a cGAN to separate out ACGAN style stuff from cGAN. Currently, no plans.

	gan_call = FLAGS.gan + '_' + FLAGS.topic + '(FLAGS_dict)'

	# with tf.device('/GPU:'+FLAGS.device):
	try:
		print('trying')
		gan = eval(gan_call)
		gan.initial_setup()
		gan.main_func()
		print('Worked')

		if gan.mode == 'train':
			print(gan.mode)
			gan.train()
			email_success()
			gan.test()

		if gan.mode == 'save_model':
			gan.model_saver()

		if gan.mode == 'test':
			gan.test()

		# if gan.mode == 'fid':
		# 	gan.update_FID()

		if gan.mode == 'metrics':
			gan.eval_metrics()

	except Exception as e:
		email_error(str(e))
		print("Exiting Execution due to error.")
		exit(0)

###############################################################################  
	
	
	print('Completed.')
