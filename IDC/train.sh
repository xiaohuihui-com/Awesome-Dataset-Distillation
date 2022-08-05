#!/usr/bin/env bash


python train.py --dataset='cifar10' --model='convnet' &
python train.py --dataset='cifar100' --model='convnet' &
python train.py --dataset='mnist' --model='convnet' &
python train.py --dataset='fashion' --model='convnet' &
python train.py --dataset='svhn' --model='convnet' &
wait
