#  Entity Aspect Linking System (Repo for my master thesis)

## This repository is used to put my source code of experiments for my master thesis. 

## Installation
1. install requirements.txt via pip
2. if you want to used our pretrained model directly, you can download model files in 'model' folder. Or you can run from the beginning.
3. before you would like to use our embedding ranking model, please be sure to download GloVe model from here: https://nlp.stanford.edu/projects/glove/ 
We currently use glove.6b.300d
4. before you want to use our bert ranking model, be sure to download and install bert pretained model.
Check this repo for the installation and detailed usage:
https://github.com/hanxiao/bert-as-service 

## Data Source
Raw data is under dir /data

Other training datasets are under dir /trained

## Usage

before run any models, set the env var customer in ['sub', 'obj', 'both_subj', 'both_obj']
There is no main.py. The scripts run separately. 
The detailed usage please see each file.



