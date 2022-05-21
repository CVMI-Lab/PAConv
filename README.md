# PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds
<img src="./figure/paconv.jpg" width="900"/>

by [Mutian Xu*](https://mutianxu.github.io/), [Runyu Ding*](), [Hengshuang Zhao](https://hszhao.github.io/), and [Xiaojuan Qi](https://xjqi.github.io/).


## Introduction
This repository is built for the official implementation of:

__PAConv__: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds ___(CVPR2021)___ [[arXiv](https://arxiv.org/abs/2103.14635)]
<br>

If you find our work useful in your research, please consider citing:

```
@inproceedings{xu2021paconv,
  title={PAConv: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds},
  author={Xu, Mutian and Ding, Runyu and Zhao, Hengshuang and Qi, Xiaojuan},
  booktitle={CVPR},
  year={2021}
}
```

## Highlight

* All initialization models and trained models are available.
* Provide fast multiprocessing training ([nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/_modules/torch/nn/parallel/distributed.html)) with official [nn.SyncBatchNorm](https://pytorch.org/docs/master/nn.html#torch.nn.SyncBatchNorm).
* Incorporated with [tensorboardX](https://github.com/lanpa/tensorboardX) for better visualization of the whole training process.
* Support recent versions of PyTorch.
* Well designed code structures for easy reading and using.

## Usage

We provide scripts for different point cloud processing tasks:

* [Object Classification](./obj_cls) task on Modelnet40.
 
* [Shape Part Segmentation](./part_seg) task on ShapeNetPart.
 
* [Indoor Scene Segmentation](./scene_seg) task on S3DIS.

You can find the instructions for running these tasks in the above corresponding folders.

## Performance
The following tables report the current performances on different tasks and datasets. ( __*__ denotes the backbone architectures)

### Object Classification on ModelNet40

| Method | OA |
| :--- | :---: |
| PAConv _(*PointNet)_   | 93.2%|
| PAConv _(*DGCNN)_      | **93.9%** |

### Object Classification under Corruptions on ModelNet-C.
| Method |  mCE | Clean OA |
| :--- | :---: | :---: |
| PAConv _(*DGCNN)_    | **1.104** | **0.936** |


### Shape Part Segmentation on ShapeNet Part
| Method |  Class mIoU | Instance mIoU |
| :--- | :---: | :---: |
| PAConv _(*DGCNN)_    | **84.6%** | **86.1%** |



### Indoor Scene Segmentation on S3DIS Area-5

| Method |  S3DIS mIoU  |
| :--- | :---: |
| PAConv _(*PointNet++)_| **66.58%** |


## Contact

You are welcome to send pull requests or share some ideas with us. Contact information: Mutian Xu (mino1018@outlook.com) or Runyu Ding (ryding@eee.hku.hk).

## Acknowledgement

Our code base is partially borrowed from [PointWeb](https://github.com/hszhao/PointWeb), [DGCNN](https://github.com/WangYueFt/dgcnn) and [PointNet++](https://github.com/charlesq34/pointnet2).

## Update

20/05/2022:

Our method is officially supported by [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) for indoor scene segmentation.

We got competitive performance on [ModelNet-C](https://github.com/jiawei-ren/ModelNet-C) dataset for object classification under corruptions.
