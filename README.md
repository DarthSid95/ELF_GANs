Euler Lagrange Analysis of Generative Adversarial Networks
====================

## Introduction

This is the Code submission accompanying the JMLR Submission "Euler Lagrange Analysis of GANs"

This codebase consists of the TensorFlow2.0 implementation WGAN-FS and WAEFR, along with the baseline comparisons. Additionally, high-resolution counterparts of the figures presented in the paper are also included.  

Dependencies can be installed via anaconda. The ``ELFGAN_GPU.yml`` file list the dependencies to setup the GPU system based environment: 

```
GPU accelerated TensorFlow2.0 Environment:
- pip=20.0.2
- python=3.6.10
- cudatoolkit=10.1.243
- cudnn=7.6.5
- pip:
    - absl-py==0.9.0
    - h5py==2.10.0
    - ipython==7.15.0
    - ipython-genutils==0.2.0
    - matplotlib==3.1.3
    - numpy==1.18.1
    - scikit-learn==0.22.1
    - scipy==1.4.1
    - tensorflow-gpu==2.2.0
    - tensorboard==2.2.0
    - tensorflow-addons
    - tensorflow-estimator==2.2.0
    - tqdm==4.42.1
    - gdown==3.12
```
If a GPU is unavailable, the CPU only environment can be built  with ``ELFGAN_CPU.yml``. This setting is meant to run evaluation code. While the WGAN-FS codes could potentially be trained on the CPU, WAEFR training is not advisable.
```
CPU based TensorFlow2.0 Environment:
- pip=20.0.2
- python=3.6.10
- pip:
    - absl-py==0.9.0
    - h5py==2.10.0
    - ipython==7.15.0
    - ipython-genutils==0.2.0
    - matplotlib==3.1.3
    - numpy==1.18.1
    - scikit-learn==0.22.1
    - scipy==1.4.1
    - tensorboard==2.0.2
    - tensorflow-addons==0.6.0
    - tensorflow-datasets==3.0.1
    - tensorflow-estimator==2.0.1
    - tensorflow==2.0.0
    - tensorflow-probability==0.8.0
    - tqdm==4.42.1
    - gdown==3.12
```

Codes were tested locally on the following system configurations:

```
*SYSTEM 1: Ubuntu 18.04.4LTS
- GPU:			'NVIDIA GeForce GTX 1080'
- RAM:			'32GB'
- CPU:			'Intel Core i7-7820HK @2.9GHz x 8'
- CUDA:			'10.2'
- NVIDIA_drivers:	'440.82' 

*SYSTEM 2: macOS Catalina, Version 10.15.6
- GPU:			-
- RAM:			'16GB'
- CPU:			'8-Core Intel Core i9 @2.3GHz'
- CUDA:			-
- NVIDIA_drivers:	-
```

## Fourier Series Implementation

The main difference between existing WGAN variants and out WGAN-FS, is the inclusion of a Fourier series solver in place of the gradient descent based optimization of the discriminator network. To highlight this fact, the snippets of code associated with the Fourier Solver (FS) are described below:

The Fourier series network, as described in Figure 6 of the main manuscript is implemented as follows. The first layer, consisting of static weights, is implemented as below:
```
inputs = tf.keras.Input(shape=(latent_dims,))

w0_nt_x = tf.keras.layers.Dense(L, activation=None, use_bias = False)(inputs)
w0_nt_x2 = tf.math.scalar_mul(2., w0_nt_x)

cos_terms = tf.keras.layers.Activation( activation = tf.math.cos)(w0_nt_x)
sin_terms = tf.keras.layers.Activation( activation = tf.math.sin)(w0_nt_x)
cos2_terms  = tf.keras.layers.Activation( activation = tf.math.cos)(w0_nt_x2)

discriminator_model_A = tf.keras.Model(inputs=inputs, outputs= [inputs, cos_terms, sin_terms, cos2_terms])
```
The weight layers ``w0_nt_x`` is the predetermined ``w_o M`` matrix. Subsequently, the Fourier coefficients of  ``p_g`` and ``p_d`` are evaluated by the sample estimates of the characteristic function:
```
_, a_real, a_imag, _ = discriminator_model_A(input_data, training = False)
a_real = tf.expand_dims(tf.reduce_mean( ar, axis = 0), axis = 1)
a_imag = tf.expand_dims(tf.reduce_mean( ai, axis = 0), axis = 1)
```
Next, the Fourier coefficients ``D(x)`` are evaluated based on the solution to the Poisson's PDE, and are given by:

```
Gamma_imag = tf.math.divide(tf.constant(1/(w_o**2))*tf.subtract(alpha_imag, beta_imag), n_norm)
Gamma_real = tf.math.divide(tf.constant(1/(w_o**2))*tf.subtract(alpha_real, beta_real), n_norm)
Tau_imag = tf.math.divide(tf.constant(1/(2*(w_o**2)))*tf.square(tf.subtract(alpha_imag, beta_imag)), n_norm)
Tau_real = tf.math.divide(tf.constant(1/(2*(w_o**2)))*tf.square(tf.subtract(alpha_real, beta_real)), n_norm)
Tau_sum = tf.reduce_sum(tf.add(Tau_imag,Tau_real))
```
Finally, with these terms, the second layer of the discriminator can be defined:
```
inputs = tf.keras.Input(shape=(self.latent_dims,))
cos_terms = tf.keras.Input(shape=(self.L,))
sin_terms = tf.keras.Input(shape=(self.L,))
cos2_terms = tf.keras.Input(shape=(self.L,))

cos_sum = tf.keras.layers.Dense(1, activation=None, use_bias = True)(cos_terms) # Gamma_real weights
sin_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(sin_terms) # Gamma_imag weights

cos2_c_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(cos2_terms) # Tau_real weights
cos2_s_sum = tf.keras.layers.Dense(1, activation=None, use_bias = False)(cos2_terms) # Tau_imag weights

lambda_x_term = tf.keras.layers.Subtract()([cos2_s_sum, cos2_c_sum]) # (tau_imag  - tau_real) term

if self.latent_dims == 1:
	phi0_x = inputs # 1-D Homogeneous component 
else:
	phi0_x = tf.divide(tf.reduce_sum(inputs, axis=-1, keepdims=True), latent_dims) # n-D Homogeneous component 

if homogeneous_component:
	Out = tf.keras.layers.Add()([cos_sum, sin_sum, phi0_x])
else:
	Out = tf.keras.layers.Add()([cos_sum, sin_sum])

discriminator_model_B = tf.keras.Model(inputs= [inputs, cos_terms, sin_terms, cos2_terms], outputs=[Out,lambda_x_term])

```

## Training Data

MNIST and CIFAR-10 are loaded from TensorFlow-Datasets. The CelebA dataset (**1.2GB**) can be downloaded by running the following code (requires ``wget`` dependency):

```
python download_celeba.py
```
Alternatively you can manually download [the ``img_align_celeba`` folder](https://drive.google.com/file/d/0B7EVK8r0v71pZjFTYXZWM3FlRnM/view?usp=sharing) from the official [CelebA website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html). Additionally, [the ``list_attr_celeba.csv`` file](https://drive.google.com/file/d/0B7EVK8r0v71pblRyaVFSWGxPY0U/view?usp=sharing) can also be downloaded directly. Both must be saved at ``ELF_GANs/data/CelebA/``.

Similarly, the Ukiyo-E dataset (**1.23 GB**) can be downloaded by running the following script (requires the ``gdown`` dependency. It can be installed manually, or is a part of the ELFGAN environment setup):
```
python download_ukiyoe.py
```
Alternatively, you can download the [official tarball]() from the [Ukiyo-E dataset website](https://www.justinpinkney.com/ukiyoe-dataset/) and extract the ``ukiyoe-1024`` folder to ``ELF_GAN/data/UkiyoE/``.

Finally, the SVHN dataset's aligned and cropped ``32x32`` mat file is used for training. This can be downloaded by running the following script:

```
python download_svhn.py
```
Alternatively, you can directly download the [train_32x32.mat](http://ufldl.stanford.edu/housenumbers/train_32x32.mat) file (**182 MB**), and if needed for evaluation, [test_32x32.mat](http://ufldl.stanford.edu/housenumbers/test_32x32.mat) file (**61.3 MB**), from the [SVHN website](http://ufldl.stanford.edu/housenumbers/), and placed in ``ELF_GAN/data/SVHN/``.


## Training  

The code provides training procedure on synthetic Gaussians for WGAN-FS, along with the baseline WGAN with weight clipping, WGAN-GP, WGAN-LP and WGAN-ALP are included. For image data, the proposed WAEFR, and the WAE-GAN baseline with JS, KL, WGAN-LP and WGAN-ALP cost penalties are provided


1) **Running ``train_*.sh`` bash files**: The fastest was to train a model is by running the bash files in ``ELF_GANs/bash_files/train/``. To train the Model for a given test case: Code to train each ``Figure X Subfig (y)`` is present in these files. Uncomment the desired command to train for the associated testcase. For example, to generate the discriminator function plot from Figure 1(a), the associated code can be uncommented in ``train_WGANFS.sh``, and the file can be run from the ``ELF_GAN`` folder as
```
bash bash_files/train/train_WGANFS.sh
```
2) **Manually running ``gam_main.py``**: Aternatively, you can train any model of your choice by running ``gan_main.py`` with custom flags and modifiers. The list of flags and their defauly values are are defined in  ``gan_main.py``.    

3) **Training on Colab**: This code is capable of training models on Google Colab (although it is *not* optimized for this). For those familiar with the approach, this repository could be cloned to your google drive and steps (1) or (2) could be used for training. CelebA must be downloaded to you current instance on Colab as reading data from GoogleDrive currently causes a Timeout error.  Setting the flags ``--colab_data 1``,  ``--latex_plot_flag 0``, and ``--pbar_flag 0`` is advisable. The ``colab_data`` flag modifies CelebA, Ukiyo-E and SVHN data-handling code to read data from the local folder, rather than ``ELF_GANs/data/CelebA/``. Additionally, you may need to manually download the SVHN mat file, or the Ukiyo-E tarball to the target GoogleDrive folder.  The ``latex_plot_flag`` flag removes code dependency on latex for plot labels, since the Colab isntance does not native include this. (Alternatively, you could install texlive_full in your colab instance). Lastly, turning off the ``pbar_flag`` was found to prevent the browser from eating too much RAM when training the model. **The .ipynb file for training on Colab will be included with the public release of the paper**. 



----------------------------------
----------------------------------

### License
The license is committed to the repository in the project folder as `LICENSE.txt`.  
Please see the `LICENSE.txt` file for full informations.

----------------------------------

**Siddarth Asokan**  
**Robert Bosch Centre for Cyber Physical Systems**  
**Indian Institute of Science**  
**Bangalore, India**  
**Email:** *siddartha@iisc.ac.in*

----------------------------------
----------------------------------