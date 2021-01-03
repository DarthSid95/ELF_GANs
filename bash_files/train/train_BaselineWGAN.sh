set -e
if [ "${CONDA_DEFAULT_ENV}" != "ELFGAN" ]; then
	echo 'You are not in the <ELFGAN> environment. Attempting to activate the ELFGAN environment via conda. Please run "conda activate ELFGAN" and try again if this fails. If the ELFGAN environment has not been installed, please refer the README.md file for further instructions.'
	condaActivatePath=$(which activate)
	source ${condaActivatePath} ELFGAN
fi


# ---- 1-D Gaussian learning. Fig 1.a ---- #

# python ./gan_main.py   --run_id 'new' --resume 0 --GPU '' --device '-1' --topic 'ELeGANt' --mode 'train' --data 'g1' --gan 'WGAN' --loss 'FS' --saver 1 --num_epochs 4 --paper 1 --res_flag 1 --latent_kind 'base' --latent_dims '2' --lr_G '0.005' --lr_D '0.005'  --distribution 'gaussian' --batch_size '500' --sigma '5' --sigmul '1' --terms '100' --metrics 'KLD,GradGrid' --colab 1 --homo_flag 0


## Data set to noise N(0,1) and target N(7,1). MIN = -3.5, MAX = 10.5. Num batches cut to 5 nd epochs cut to 2. Initializers set to Identity for weights and zero for bias.

# python ./gan_main.py   --run_id 'new' --resume 0 --GPU '' --device '-1' --topic 'ELeGANt' --mode 'train' --data 'g1' --gan 'WGAN' --loss 'FS' --saver 1 --num_epochs 2 --paper 1 --res_flag 1 --latent_kind 'base' --latent_dims '2' --lr_G '0.001' --lr_D '0.001'  --distribution 'gaussian' --batch_size '500' --sigma '10' --sigmul '1' --terms '100' --metrics 'KLD,GradGrid' --colab 1

# python ./gan_main.py   --run_id 'new' --resume 0 --GPU '' --device '-1' --topic 'ELeGANt' --mode 'train' --data 'g1' --gan 'WGAN' --loss 'LP' --saver 1 --num_epochs 2 --paper 1 --res_flag 1 --lr_G '0.005' --lr_D '0.005' --batch_size '500' --metrics 'KLD,GradGrid' --Dloop 30 --colab 1

# python ./gan_main.py   --run_id 'new' --resume 0 --GPU '' --device '-1' --topic 'ELeGANt' --mode 'train' --data 'g1' --gan 'WGAN' --loss 'ALP' --saver 1 --num_epochs 2 --paper 1 --res_flag 1 --lr_G '0.005' --lr_D '0.005' --batch_size '500' --metrics 'KLD,GradGrid' --Dloop 30 --colab 1

# python ./gan_main.py   --run_id 'new' --resume 0 --GPU '' --device '-1' --topic 'ELeGANt' --mode 'train' --data 'g1' --gan 'WGAN' --loss 'GP' --saver 1 --num_epochs 2 --paper 1 --res_flag 1 --lr_G '0.005' --lr_D '0.005' --batch_size '500' --metrics 'KLD,GradGrid' --Dloop 30 --colab 1

# python ./gan_main.py   --run_id 'new' --resume 0 --GPU '' --device '-1' --topic 'ELeGANt' --mode 'train' --data 'g1' --gan 'WGAN' --loss 'base' --saver 1 --num_epochs 2 --paper 1 --res_flag 1 --lr_G '0.005' --lr_D '0.005' --batch_size '500' --metrics 'KLD,GradGrid' --Dloop 30 --colab 1