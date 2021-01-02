import os
import zipfile
import warnings

pwd = os.popen('pwd').read().strip('\n')
working_folder = pwd.split('/')[-1]
if working_folder != 'ELF_GAN':
	warnings.warn("Not in the ELF_GANs working direcorty. The images will get stored in the wrong path. This could be intentionally, but will not facilitate running of the Code without modifications.",ImportWarning)
if not os.path.exists(pwd+'/data'):
	os.mkdir(pwd+'/data')
if not os.path.exists(pwd+'/data/CelebA'):
	os.mkdir(pwd+'/data/CelebA')
os.system('wget -P '+pwd+'/data/CelebA/ https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/celeba.zip')


with zipfile.ZipFile(pwd+'/data/CelebA/celeba.zip',"r") as zip_ref:
  zip_ref.extractall(pwd+'/data/CelebA/')