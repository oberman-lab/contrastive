#!/bin/bash

# An script to train LeNet on MNIST with SGD

LOGDIR='./logs/'
DATADIR='./data/'
mkdir -p $LOGDIR
mkdir -p $DATADIR
GPUID=0 # Select a GPU. If you want say two GPUs, set GPUID="0,1"

CUDA_VISIBLE_DEVICES=$GPUID \

for VARIABLE in 0.01 0.05 0.1 0.2 0.3 0.4 0.5
do

python -u ./main.py\
  --data-dir $DATADIR\
  --dropout 0.25\
  --momentum 0.9\
  --compare True \
  --dataset MNIST \
  --frac-labeled $VARIABLE \
  --epochs 10 >> $LOGDIR/log.out

done
