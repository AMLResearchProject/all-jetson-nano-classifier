#!/bin/bash

FMSG="Acute Lymphoblastic Leukemia Jetson Nano Classifier trainer installation terminated!"

printf -- 'This script will install the Acute Lymphoblastic Leukemia Jetson Nano Classifier trainer on your machine.\n';
printf -- '\033[33m WARNING: This script assumes Ubuntu 20.04. \033[0m\n';
printf -- '\033[33m WARNING: This script assumes you have already run scripts/oenapi.sh. \033[0m\n';
printf -- '\033[33m WARNING: This script assumes you have already activated the all-jetson-nano environment. \033[0m\n';
printf -- '\033[33m WARNING: This is an inteteractive installation, please follow instructions provided. \033[0m\n';

read -p "Proceed (y/n)? " proceed
if [ "$proceed" = "Y" -o "$proceed" = "y" ]; then
	printf -- 'Installing the Acute Lymphoblastic Leukemia Jetson Nano Classifier trainer on your machine.\n';
	conda activate all-jetson-nano
	conda install jupyter
	conda install nb_conda
	conda install -c conda-forge mlxtend
	conda install matplotlib
	conda install Pillow
	conda install opencv
	conda install scipy
	conda install scikit-learn
	conda install scikit-image
	conda install -c conda-forge onnx
	conda install protobuf=3.9 -c conda-forge
	conda install -c conda-forge tf2onnx
	printf -- '\033[32m SUCCESS: Acute Lymphoblastic Leukemia Jetson Nano Classifier trainer installed! \033[0m\n';
else
	echo $FMSG;
	exit 1
fi