#!/bin/bash
WKSPACE_DIR=$(dirname $(dirname $(pwd)))
# docker run -it --rm --gpus all -v $WKSPACE_DIR:$WKSPACE_DIR gdiff-img \
#     "cd $WKSPACE_DIR/grasp_diffusion/ && pip install -e . &&
#     python scripts/train/train_pointcloud_6d_grasp_diffusion.py"
apptainer run --nv -B $WKSPACE_DIR:$WKSPACE_DIR gdiff.sif \
    "cd $WKSPACE_DIR/grasp_diffusion/ && ls > tmp.txt && \
    pip install -e . && python scripts/train/train_pointcloud_6d_grasp_diffusion.py"