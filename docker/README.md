## Docker

```shell
# build image
bash build-image.sh
# run test
bash run-container.sh
```

## Apptainer

```shell
# spin-up a local docker image registry (if not already up)
docker run -d -p 5000:5000 --restart=always --name registry registry:2
# push docker image to registry, then pull as apptainer image
bash apptainer-build-from-docker.sh
# run test
bash run-apptainer.sh
```

## Execution

```shell
WKSPACE_DIR=$(dirname $(dirname $(pwd)))

# docker
docker run --rm --gpus all -v $WKSPACE_DIR:$WKSPACE_DIR gdiff-img \
    "cd $WKSPACE_DIR/grasp_diffusion/ && pip install -e . && python scripts/train/train_pointcloud_6d_grasp_diffusion.py"

# apptainer
apptainer run --nv -B $WKSPACE_DIR:$WKSPACE_DIR gdiff.sif \
    "cd $WKSPACE_DIR/grasp_diffusion/ && pip install -e . && python scripts/train/train_pointcloud_6d_grasp_diffusion.py"
```