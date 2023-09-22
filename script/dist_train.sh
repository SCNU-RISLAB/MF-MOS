#!/bin/bash

DatasetPath=DATAROOT
ArchConfig=./train_yaml/ddp_mos_coarse_stage.yml
DataConfig=./config/labels/semantic-kitti-mos.raw.yaml
LogPath=./log/Train

export CUDA_VISIBLE_DEVICES=0,1 && python3 -m torch.distributed.launch --nproc_per_node=2 \
                                           ./train.py -d $DatasetPath \
                                                      -ac $ArchConfig \
                                                      -dc $DataConfig \
                                                      -l $LogPath