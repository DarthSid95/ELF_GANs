import os
import tarfile
import warnings

pwd = os.popen('pwd').read().strip('\n')
working_folder = pwd.split('/')[-1]
if working_folder != 'ELF_GAN':
	warnings.warn("Not in the ELF_GANs working direcorty. The images will get stored in the wrong path. This could be intentionally, but will not facilitate running of the Code without modifications.",ImportWarning)
if not os.path.exists(pwd+'/data'):
	os.mkdir(pwd+'/data')
if not os.path.exists(pwd+'/data/UkiyoE'):
	os.mkdir(pwd+'/data/UkiyoE')
os.system('gdown -O '+pwd+'/data/UkiyoE/ https://drive.google.com/uc?id=1zEgVLrKVp8oCZuX0NENcAeh-kdaKJzNG')


with tarfile.open(pwd+'/data/UkiyoE/ukiyoe-1024-v2.tar') as tar_ref:
	tar_ref.extractall(pwd+'/data/UkiyoE/')