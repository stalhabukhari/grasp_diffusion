#!/bin/bash
# start a local docker registry
# docker run -d -p 5000:5000 --restart=always --name registry registry:2

# tag and push local container to it
docker tag gdiff-img:latest localhost:5000/gdiff-img:latest
docker push localhost:5000/gdiff-img:latest

# build apptainer container
# apptainer build gdiff.sif Apptainer
# apptainer build gdiff.sif docker://localhost:5000/gdiff-img:latest
apptainer build gdiff.sif docker-daemon://localhost:5000/gdiff-img:latest