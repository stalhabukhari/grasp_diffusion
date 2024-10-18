#!/bin/bash
# --rm is added for testing and debugging
docker run -it --rm --gpus all -v "$(pwd)":"/code-dir" \
  gdiff-img "python test_script.py"