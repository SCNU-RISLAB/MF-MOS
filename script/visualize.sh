#!/bin/bash

DatasetPath=DATAROOT
Seq=08
DataConfig=./config/labels/semantic-kitti-mos.raw.yaml
Version=fuse # Version in ["moving", "movable", "fuse"] for predictions
#PredictionPath=./log/valid/predictions

python3 utils/visualize_mos.py -d $DatasetPath \
                               -s $Seq \
                               -c $DataConfig \
                               -v $Version \
                               # -p $PredictionPath
