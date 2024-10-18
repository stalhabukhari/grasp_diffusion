# This Dockerfile sets up a GraspDiffusion environment with PyTorch (GPU)
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && \
    apt install -y \
        wget git net-tools vim curl build-essential x11vnc zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
        libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev freeglut3-dev && \
        # following is for opencv-python (uncomment if not needed)
        apt install -y --no-install-recommends libglib2.0-0 libxrender1 libxext6 libsm6 libgl1-mesa-glx && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# install miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mkdir /root/.conda && \
    bash Miniconda3-latest-Linux-x86_64.sh -b && \
    rm -f Miniconda3-latest-Linux-x86_64.sh && \
    echo "Running $(conda --version)"

# install python packages
COPY requirements.txt /env-setup/requirements.txt
RUN conda init bash && . /root/.bashrc && conda create -n gdiff python=3.8 && \
    source activate gdiff && pip install --upgrade pip && \
    conda install -y pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia && \
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    
RUN source activate gdiff && \
    # scikit-sparse fix: https://stackoverflow.com/a/76322833/6597334
    conda install -c conda-forge suitesparse==5.10.1 scikit-sparse==0.4.8 && \
    pip install -r /env-setup/requirements.txt

RUN source activate gdiff && cd /env-setup/ && \
    git clone https://github.com/robotgradient/mesh_to_sdf.git && \
    pip install -e mesh_to_sdf

WORKDIR /code-dir
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "gdiff", "/bin/bash", "-c"]