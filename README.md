Euler Lagrange Analysis of Generative Adversarial Networks
====================

## Introduction

This is the Code submission accompanying the JMLR Submission "Euler Lagrange Analysis of GANs"

This codebase consists of the TensorFlow2.0 implementation WGAN-FS and WAEFR, along with the baseline comparisons. Additionally, high-resolution counterparts of the figures presented in the paper are also included.  

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

MNIST, Fashion MNIST and CIFAR-10 are loaded from TensorFlow-Datasets. The CelebA dataset (**1.2GB**) can be downloaded by running the following code (requires ``wget`` dependency):

```
python download_celeba.py
```
Alternatively you can manually download the ``img_align_celeba`` folder and the ``list_attr_celeba.csv`` file, and save them at ``RumiGANs/data/CelebA/``.


## Training  

The code provides training procedure for baseline Standard GAN^1, LSGAN^2, WGAN^3, WGAN-GP^4, and each of their corresponding *Rumi* variants. Additionally, ported implementations of auxiliary classifier GAN (ACGAN^5), Twin ACGAN^6 and conditional GAN with projection discriminator (CGAN-PD^7) are included.   


1) The fastest was to train a model is by running the bash files in ``RumiGANs/bash_files/train/``. The train the Model for a given test case: Code to train each ``Figure X Subfig (y)`` is present in these files. Uncomment the desired command to train for the associated testcase. For example, to generate images from Rumi-LSGAN on CelebA with class imbalance, Figure 4(i), uncomment ``Code for Figure 4.i`` in the ``train_RumiGAN.sh`` file in the above folder and run  
```
bash bash_files/train/train_RumiGAN.sh
```
2) Aternatively, you can train any model of your choice by running ``gan_main.py`` with custom flags and modifiers. The list of flags and their defauly values are are defined in  ``gan_main.py``.    

3) **Training on Colab**: This code is capable of training models on Google Colab (although it is *not* optimized for this). For those familiar with the approach, this repository could be cloned to your google drive and steps (1) or (2) could be used for training. CelebA must be downloaded to you current instance on Colab as reading data from GoogleDrive currently causes a Timeout error.  Setting the flags ``--colab_data 1``,  ``--latex_plot_flag 0``, and ``--pbar_flag 0`` is advisable. The ``colab_data`` flag modifies CelebA data-handling code to read data from the local folder, rather than ``RumiGANs/data/CelebA/``.  The ``latex_plot_flag`` flag removes code dependency on latex for plot labels, since the Colab isntance does not native include this. (Alternatively, you could install texlive_full in your colab instance). Lastly, turning off the ``pbar_flag`` was found to prevent the browser from eating too much RAM when training the model. **The .ipynb file for training on Colab will be included shortly**. 

----------------------------------
----------------------------------

### License
The license is committed to the repository in the project folder as `LICENSE.txt`.  
Please see the `LICENSE.txt` file for full informations.

----------------------------------

**Siddarth Asokan**  
**Robert Bosch Centre for Cyber Physical Systems **  
**Indian Institute of Science**  
**Bangalore, India **  
**Email:** *siddartha@iisc.ac.in*

----------------------------------
----------------------------------