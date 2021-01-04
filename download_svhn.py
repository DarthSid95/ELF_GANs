import os
import zipfile
import warnings

pwd = os.popen('pwd').read().strip('\n')
working_folder = pwd.split('/')[-1]
if working_folder != 'ELF_GAN':
	warnings.warn("Not in the ELF_GANs working direcorty. The images will get stored in the wrong path. This could be intentionally, but will not facilitate running of the Code without modifications.",ImportWarning)
if not os.path.exists(pwd+'/data'):
	os.mkdir(pwd+'/data')
if not os.path.exists(pwd+'/data/SVHN'):
	os.mkdir(pwd+'/data/SVHN')
os.system('wget -P '+pwd+'/data/SVHN/ http://ufldl.stanford.edu/housenumbers/train_32x32.mat')
os.system('wget -P '+pwd+'/data/SVHN/ http://ufldl.stanford.edu/housenumbers/test_32x32.mat')
