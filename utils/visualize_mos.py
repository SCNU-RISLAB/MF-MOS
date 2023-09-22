#!/usr/bin/env python3
# Developed by Xieyuanli Chen
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
from auxiliary.laserscan import LaserScan, SemLaserScan
from auxiliary.laserscanvis import LaserScanVis
import copy

def get_args():
    parser = argparse.ArgumentParser("./visualize.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to visualize. No Default',
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=False,
        default="config/labels/semantic-kitti-mos.yaml",
        help='Dataset config file. Defaults to %(default)s',
    )
    parser.add_argument(
        '--sequence', '-s',
        type=str,
        default="00",
        required=False,
        help='Sequence to visualize. Defaults to %(default)s',
    )
    parser.add_argument(
        '--predictions', '-p',
        type=str,
        default=None,
        required=False,
        help='Alternate location for labels, to use predictions folder. '
        'Must point to directory containing the predictions in the proper format '
        ' (see readme)'
        'Defaults to %(default)s',
    )
    parser.add_argument(
        '--ignore_semantics', '-i',
        dest='ignore_semantics',
        default=False,
        action='store_true',
        help='Ignore semantics. Visualizes uncolored pointclouds.'
        'Defaults to %(default)s',
    )
    parser.add_argument(
        '--do_instances', '-di',
        dest='do_instances',
        default=False,
        action='store_true',
        help='Visualize instances too. Defaults to %(default)s',
    )
    parser.add_argument(
        '--offset',
        type=int,
        default=0,
        required=False,
        help='Sequence to start. Defaults to %(default)s',
    )
    parser.add_argument(
        '--ignore_safety',
        dest='ignore_safety',
        default=False,
        action='store_true',
        help='Normally you want the number of labels and ptcls to be the same,'
        ', but if you are not done inferring this is not the case, so this disables'
        ' that safety.'
        'Defaults to %(default)s',
    )
    parser.add_argument(
        '--version', '-v',
        type=str,
        default="moving",
        choices=["moving", "movable", "fuse"],
        required=False,
        help="which version is selected to visualize",
    )
    parser.add_argument(
        '--gt_vis',
        default=False,
        action='store_true',
        help='Whether to visualize the ground truth',
    )
    return parser


if __name__ == '__main__':
    
    parser = get_args()
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("*" * 80)
    print("  INTERFACE:")
    print("  Dataset:", FLAGS.dataset)
    print("  Config:", FLAGS.config)
    print("  Sequence:", FLAGS.sequence)
    print("  Predictions:", FLAGS.predictions)
    print("  ignore_semantics:", FLAGS.ignore_semantics)
    print("  do_instances:", FLAGS.do_instances)
    print("  ignore_safety:", FLAGS.ignore_safety)
    print("  offset:", FLAGS.offset)
    print("  visulization_version:", FLAGS.version)
    print("*" * 80)

    H, W = 64, 2048

    # open config file
    try:
        print("Opening config file %s" % FLAGS.config)
        CFG = yaml.safe_load(open(FLAGS.config, 'r'))
    except Exception as e:
        print(e)
        print("Error opening yaml file.")
        quit()

    # fix sequence name
    FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

    # does sequence folder exist?
    scan_paths = os.path.join(FLAGS.dataset, "sequences", FLAGS.sequence, "velodyne")
    if os.path.isdir(scan_paths):
        print("Sequence folder exists! Using sequence from %s" % scan_paths)
    else:
        print("Sequence folder doesn't exist! Exiting...")
        quit()

    # populate the pointclouds
    scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(scan_paths)) for f in fn]
    scan_names.sort()

    pred_label_names = None

    # does sequence folder exist?
    if not FLAGS.ignore_semantics:
        gt_label_paths = os.path.join(FLAGS.dataset, "sequences", FLAGS.sequence, "labels")

        if os.path.isdir(gt_label_paths):
            print("Labels folder exists! Using labels from %s" % gt_label_paths)
        else:
            print(gt_label_paths)
            print("Labels folder doesn't exist! Exiting...")
            quit()

        # populate the pointclouds
        gt_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(gt_label_paths)) for f in fn]
        gt_label_names.sort()

        # check that there are same amount of labels and scans
        if not FLAGS.ignore_safety:
            print(f"len(gt_label_names):{len(gt_label_names)}, len(scan_names)={len(scan_names)}")
            assert(len(gt_label_names) == len(scan_names))

        if FLAGS.predictions is not None:
            if FLAGS.version == "moving":
                pred_label_paths = os.path.join(FLAGS.predictions, "sequences", FLAGS.sequence, "predictions")
            elif FLAGS.version == "movable":
                pred_label_paths = os.path.join(FLAGS.predictions, "sequences", FLAGS.sequence, "predictions_movable")
            else:
                pred_label_paths = os.path.join(FLAGS.predictions, "sequences", FLAGS.sequence, "predictions_fuse")

            if os.path.isdir(pred_label_paths):
                print("Predictions labels folder exists! Using labels from %s" % pred_label_paths)
            else:
                raise FileNotFoundError("Predictions labels doesn't exist! Exiting...")
            pred_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(pred_label_paths)) for f in fn]
            pred_label_names.sort()

            # check that there are same amount of labels and scans
            if not FLAGS.ignore_safety:
                print(f"len(pred_label_names):{len(pred_label_names)}, len(scan_names)={len(scan_names)}")
                assert (len(pred_label_names) == len(scan_names))

        color_dict = CFG["color_map"]
        gt_color_dict = copy.deepcopy(color_dict)
        moving_learning_map = CFG["moving_learning_map"]
        movable_learning_map = CFG["movable_learning_map"]
        moving_learning_map_inv = CFG["moving_learning_map_inv"]
        movable_learning_map_inv = CFG["movable_learning_map_inv"]
        # # import pdb; pdb.set_trace()

        for key in color_dict.keys():
            if (key == 250) or (key in movable_learning_map.keys() and movable_learning_map[key] == 2):
                color_dict[key] = [255, 0, 0]
                if key != 250 and moving_learning_map[key] == 2:
                    color_dict[key] = [0, 0, 255]
            else:
                color_dict[key] = [255, 255, 255]
        nclasses = len(color_dict)
        print(color_dict)
        scan = SemLaserScan(nclasses, color_dict, H=H, W=W, project=True)
    else:
        gt_label_names = None
        scan = LaserScan(H=H, W=W, project=True)  # project all opened scans to spheric proj

    # create a visualizer
    vis = LaserScanVis(H=H, W=W,
                       scan=scan,
                       scan_names=scan_names,
                       gt_label_names=gt_label_names,
                       pred_label_names=pred_label_names,
                       offset=FLAGS.offset,
                       semantics=not FLAGS.ignore_semantics, instances=FLAGS.do_instances and not FLAGS.ignore_semantics)

    # print instructions
    print("To navigate:")
    print("\tb: back (previous scan)")
    print("\tn: next (next scan)")
    print("\tq: quit (exit program)")
    print("Describe:\n"
          "image:     \t-------------\n"
          "           \tRange image\n"
          "           \tGround Truth\n"
          "           \t{Predictions}\n"
          "           \t-------------\n"
          "PointCloud:\t[Ground Truth  ||  {Predictions}]")

    # run the visualizer
    vis.run()
