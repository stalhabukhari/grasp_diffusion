#!/bin/bash
apptainer run --nv gdiff.sif "cd $(pwd) && echo $(python --version) &&
        python test_script.py"