#!/bin/bash
# Based on AWS DeepLearning AMI with Ubuntu 18.04

rm /usr/local/cuda
ln -s /usr/local/cuda-10.1 /usr/local/cuda

# Upgrade pip3 and install tensorflow, tensorboard, dataset packages
pip3 install --upgrade pip
pip3 install tf-nightly
pip3 install tensorboard==2.2
pip3 install tensorboard_plugin_profile
pip3 install tensorflow_datasets
