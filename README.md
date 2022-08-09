# **Point Cloud Instance Segmentation with Semi-supervised Bounding-Box Mining**



## Introduction

This repository is inference code release for our T-PAMI 2021 paper (arXiv report [here](https://arxiv.org/pdf/2111.15210.pdf)).



## Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{YongbinLiao2022PointCI,
  title={Point Cloud Instance Segmentation with Semi-supervised Bounding-Box Mining},
  author={Yongbin Liao and Hongyuan Zhu and Yanggang Zhang and Chuangguan Ye and Tao Chen and Jiayuan Fan},
  year={2022}
}
```



## Installation

Install [Pytorch](https://pytorch.org/get-started/locally/) and [Tensorflow](https://github.com/tensorflow/tensorflow) (for TensorBoard). It is required that you have access to GPUs.The code is tested with Ubuntu 18.04, Pytorch v1.8, TensorFlow v1.15.

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used in the backbone network:

```shell
cd pointnet2
python setup.py install
```

To see if the compilation is successful, try to run `python models/votenet.py` to see if a forward pass works.

Install the following Python dependencies (with `pip install`):

```
matplotlib
opencv-python
plyfile
'trimesh>=2.35.39,<2.35.40'
'networkx>=2.2,<2.3'
scipy
```

Install the following Python dependencies (with `conda install`):

```shell
conda install -c conda-forge point_cloud_utils
```



## Dataset preparation

We follow the VoteNet codebase for preprocessing our data. The instructions for preprocessing SUN RGB-D are [here](https://github.com/facebookresearch/votenet/tree/main/sunrgbd) and ScanNet are [here](https://github.com/facebookresearch/votenet/tree/main/scannet).



## Run eval

You can download pre-trained models and sample point clouds [HERE](https://drive.google.com/file/d/1mVA9R-KzwWdFRnir0_ZUI7PiCcMlTKKg/view?usp=sharing). Unzip the file under the project root path (`/path/to/project/demo_files`) and then run:

```shell
python eval.py
```

