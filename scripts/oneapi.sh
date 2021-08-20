#!/bin/bash

FMSG="Acute Lymphoblastic Leukemia Jetson Nano Classifier oneAPI installation terminated!"

printf -- 'This script will install oneAPI on your machine.\n';
printf -- '\033[33m WARNING: This script assumes Ubuntu 20.04. \033[0m\n';
printf -- '\033[33m WARNING: This script assumes you have not already installed the oneAPI Basekit. \033[0m\n';
printf -- '\033[33m WARNING: This script assumes you have not already installed the oneAPI AI Analytics Toolkit. \033[0m\n';
printf -- '\033[33m WARNING: If any of the above are not relevant to you, please comment out the relevant sections below before running this installation script. \033[0m\n';
printf -- '\033[33m WARNING: This is an inteteractive installation, please follow instructions provided. \033[0m\n';

read -p "Proceed (y/n)? " proceed
if [ "$proceed" = "Y" -o "$proceed" = "y" ]; then
	printf -- 'Installing oneAPI on your machine.\n';
	# Comment out the following if you have already installed oneAPI Basekit
	wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB -O - | sudo apt-key add -
	echo "deb https://apt.repos.intel.com/oneapi all main" | sudo tee /etc/apt/sources.list.d/oneAPI.list
	sudo apt update
	sudo apt install intel-basekit
	sudo apt -y install cmake pkg-config build-essential
	echo 'source /opt/intel/oneapi/setvars.sh' >> ~/.bashrc
	source ~/.bashrc
	# Comment out the following if you have already installed oneAPI AI Analytics
	sudo apt install intel-aikit
	# The following will create a Conda env with Intel Optimized TensorFlow
	conda create -n all-jetson-nano -c intel intel-aikit-tensorflow
	# Comment out the following if you have already installed the Intel GPU drivers
	# or do not have an Intel GPU on your training device
	sudo apt-get install -y gpg-agent wget
	wget -qO - https://repositories.intel.com/graphics/intel-graphics.key |
	sudo apt-key add -
	sudo apt-add-repository \
	'deb [arch=amd64] https://repositories.intel.com/graphics/ubuntu focal main'
	sudo apt-get update
	sudo apt-get install \
	intel-opencl-icd \
	intel-level-zero-gpu level-zero \
	intel-media-va-driver-non-free libmfx1
	stat -c "%G" /dev/dri/render*
	groups ${USER}
	sudo gpasswd -a ${USER} render
	newgrp render
	sudo usermod -a -G video ${USER}
else
	echo $FMSG;
	exit 1
fi