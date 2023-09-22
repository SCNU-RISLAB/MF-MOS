#!/bin/bash

DatasetPath=DATAROOT
ArchConfig=./train_yaml/mos_pointrefine_stage.yml
DataConfig=./config/labels/semantic-kitti-mos.raw.yaml
LogPath=./log/TrainWithSIEM
FirstStageModelPath=FirstStageModelPath

export CUDA_VISIBLE_DEVICES=0 && python train_2stage.py -d $DatasetPath \
                                                        -ac $ArchConfig \
                                                        -dc $DataConfig \
                                                        -l $LogPath \
                                                        -p $FirstStageModelPath