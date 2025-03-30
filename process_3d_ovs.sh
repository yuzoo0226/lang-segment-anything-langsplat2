#!/bin/bash

category_names=("bench" "blue_sofa" "covered_desk" "lawn" "office_desk" "room" "snacks" "sofa")

for category_name in "${category_names[@]}"; do
    python evaluate.py --label /mnt/home/yuga-y/usr/splat_ws/datasets/3d_ovs/$category_name/labels --output_dir outputs/3d_ovs/$category_name
done