#!/bin/bash

dataset=/data1/MotionSeg3d_Dataset/data_odometry_velodyne/dataset

# -v in ["moving", "movable", "fuse"] for predictions
python3 utils/visualize_mos.py -d $dataset \
                               -s 08 \
                               -c config/labels/semantic-kitti-mos.raw.yaml \
                               -v fuse \
                               # -p ./log/valid/predictions
