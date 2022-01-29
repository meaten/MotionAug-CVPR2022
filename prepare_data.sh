#!/bin/bash

cd data/bvh
unzip hdm05.zip
python align.py --dirpath hdm05 --outpath hdm05_aligned_split --split_config split_config.txt
cd ../../
python data/bvh/bvhToNp.py --dirpath data/bvh/hdm05_aligned_split/ --outpath motionAE/dataset/dataset_ortho6d_rel_rp_rr_mirror_split.npz --rep ortho6d
