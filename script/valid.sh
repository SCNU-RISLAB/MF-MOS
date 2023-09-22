#!/bin/bash

DatasetPath=DATAROOT
ModelPath=MODELPATH
SavePath=./log/Valid/predictions/
SPLIT=valid # valid or test

# If you want to use SIEM, set pointrefine on
export CUDA_VISIBLE_DEVICES=0 && python3 infer.py -d $DatasetPath \
                                                  -m $ModelPath \
                                                  -l $SavePath \
                                                  -s $SPLIT \
                                                  --pointrefine \
                                                  --movable # Whether to save the label of movable objects
