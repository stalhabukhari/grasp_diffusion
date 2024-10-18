# Pytorch implementation of Diffusion models in SE(3) for grasp and motion generation

This library provides the tools for training and sampling diffusion models in SE(3),
implemented in PyTorch. 
We apply them to learn 6D grasp distributions. We use the learned distribution as cost function
for grasp and motion optimization problems.
See reference [1] for additional details.

[[Website]](https://sites.google.com/view/se3dif/home)      [[Preprint]](https://arxiv.org/pdf/2209.03855.pdf)

<img src="assets/grasp_dif.gif" alt="diffusion" style="width:800px;"/>

## Installation

Create a conda environment
```python
conda env create -f environment.yml
```
Activate the environment and install the library
```python
conda activate se3dif_env && pip install -e .
```
Clone https://github.com/TheCamusean/mesh_to_sdf and install
```python
pip install -e .
```
#### Installation Issues
1. ```pip install theseus-ai``` not working.
I suggest trying to install Theseus from source https://github.com/AI-App/Theseus
2. ```pip install pytorch3d``` not working.
Try: ```conda install pytorch3d -c pytorch3d```.
or ```pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"```

3. Custom pytorch:
Try: ```conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia```


## Download Data and Trained Models

We define the source of the dataset and trained models in ```se3dif/utils/directory_utils.py```
Originally, the data root folder is set in the folder in which the repository is (one folder before the repository). 
Nevertheless, you can change it by changing ```root_directory``` in ```se3dif/utils/directory_utils.py```.

```
root
└─── data
│   │   grasps
│   │   meshes
│   │   sdf
│   └─── models
│   │   │ graspdif_model_0
│   │   │ graspdif_model_1
│ 
└─── grasp_diffusion (repository)
```

### Processed Data
#### (Based on Acronym [2] and Shapenet dataset [3])
We provide indications on how to prepare the training 
dataset in ```scripts/create_data```.


The already prepared data can be downloaded by
```cd ..```
and download [data](https://drive.google.com/drive/folders/1ULWuYZYyFncIBqBhRMNrVOrosGGRITZU?usp=sharing).

### Trained Models

In the base folder of the repository

```python
cd .. && mkdir data
cd data
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/camusean/grasp_diffusion models
```

## Sample Grasps

Sample given the whole object pointcloud
```azure
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 10 --obj_id 0 --obj_class 'ScrewDriver'
```

Sample given a mug-specialized model
```azure
python scripts/sample/generate_pointcloud_6d_grasp_poses.py --n_grasps 10 --obj_id 10 --obj_class 'Mug' --model 'grasp_dif_mugs'
```

Sample given a partial pointcloud
```azure
python scripts/sample/generate_partial_pointcloud_6d_grasp_poses.py --n_grasps 10 --obj_id 12 --obj_class 'Mug'
```

## Train a new model


Train pointcloud conditioned model
```azure
python scripts/train/train_pointcloud_6d_grasp_diffusion.py
```

Train partial pointcloud conditioned model
```azure
python scripts/train/train_partial_pointcloud_6d_grasp_diffusion.py
```

## Evaluate Generated Grasps in Simulation (Isaac Gym) and compute Earth Moving Distance

To evaluate a trained model in Isaac Gym, you first have to install the simulator and install it into your conda environment.
Note: for our experiments, we used Isaac Gym preview3.

In the file *scripts/evaluate/evaluate_pointcloud_6d_grasp_poses.py* we showcase how we evaluate the quality of the trained model

```azure
python scripts/evaluate/evaluate_pointcloud_6d_grasp_poses.py --n_grasps 100 --obj_id 0 --obj_class 'Mug' --model 'grasp_dif_mugs' --device "cuda:0"
```



## References

[1] Julen Urain*, Niklas Funk*, Jan Peters, Georgia Chalvatzaki. 
"SE(3)-DiffusionFields: Learning smooth cost functions for joint grasp and motion optimization through diffusion" 
ICRA 2023.
[[arxiv]](https://arxiv.org/pdf/2209.03855.pdf)

```
@article{urain2022se3dif,
  title={SE(3)-DiffusionFields: Learning smooth cost functions for joint grasp and motion optimization through diffusion},
  author={Urain, Julen and Funk, Niklas and Peters, Jan and Chalvatzaki, Georgia},
  journal={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
```

[2] Eppner Clemens, Arsalan Mousavian, Dieter Fox. 
"Acronym: A large-scale grasp dataset based on simulation." 
*IEEE International Conference on Robotics and Automation (ICRA)*. 
2021 [[arxiv]](https://arxiv.org/abs/2011.09584)


[3] Chang Angel X., et al. 
"Shapenet: An information-rich 3d model repository." 
*arXiv preprint arXiv:1512.03012*. 2015 [[arxiv]](https://arxiv.org/abs/1512.03012)

## Contributions
This code repository is the joint effort of [Julen Urain](http://robotgradient.com/) and [Niklas Funk](https://niklasfunk.com/).

### Contact

If you have any questions or find any bugs, please let me know: [Julen Urain](http://robotgradient.com/) julen[at]robot-learning[dot]de
