# 3D Semantic Segmentation


<img src="./figure/semseg_vis.jpg" width="900"/>


## Installation

### Requirements
   - Hardware: 1 GPU
   - Software: 
      PyTorch>=1.5.0, Python>=3, CUDA>=10.2, tensorboardX
      tqdm, h5py, pyYaml

### Dataset
- Download S3DIS [dataset](https://drive.google.com/drive/folders/12wLblskNVBUeryt1xaJTQlIoJac2WehV) and symlink the paths to them as follows (you can alternatively modify the relevant paths specified in folder `config`):
    ```
     mkdir -p dataset
     ln -s /path_to_s3dis_dataset dataset/s3dis
     ```

## Usage

1. Requirement:

   - Hardware: 1 GPU to hold 6000MB for CUDA version, 2 GPUs to hold 10000MB for non-CUDA version.
   - Software: 
      PyTorch>=1.5.0, Python3.7, CUDA>=10.2, tensorboardX, tqdm, h5py, pyYaml

2. Train:

   - Specify the gpu used in config and then do training:

     ```shell
     sh tool/train.sh s3dis pointnet2_paconv                   # non-cuda version
     sh tool/train.sh s3dis pointnet2_paconv_cuda              # cuda version
     ```
   
   We also provide pretrained models. One is implemented by CUDA mIoU=66.01(w/o voting) and the other achieves  66.33 mIoU (w/o voting) in s3dis Area-5 validation set.

4. Test:

   - Download trained segmentation models and put them under folder specified in config or modify the specified paths.

   - For full testing (get listed performance):

     ```shell
     CUDA_VISIBLE_DEVICES=0 sh tool/test.sh s3dis pointnet2_paconv        # non-cuda version
     CUDA_VISIBLE_DEVICES=0 sh tool/test.sh s3dis pointnet2_paconv_cuda   # cuda version
     ```

[comment]: <> (5. Visualization: [tensorboardX]&#40;https://github.com/lanpa/tensorboardX&#41; incorporated for better visualization.)

[comment]: <> (   ```shell)

[comment]: <> (   tensorboard --logdir=run1:$EXP1,run2:$EXP2 --port=6789)

[comment]: <> (   ```)


[comment]: <> (6. Other:)

[comment]: <> (   - Video predictions: Youtube [LINK]&#40;&#41;.)


## Citation

If you find our work helpful in your research, please consider citing:

```
@inproceedings{xu2020paconv,
  title={{PAConv}: Position Adaptive Convolution with Dynamic Kernel Assembling on Point Clouds},
  author={Xu, Mutian and Ding, Runyu and Zhao, Hengshuang and Qi, Xiaojuan},
  booktitle={CVPR},
  year={2021}
}
```

## Contact
 
You are welcome to send pull requests or share some ideas with us. Contact information: Mutian Xu (mino1018@outlook.com) or Runyu Ding (ryding@eee.hku.hk).

## Acknowledgement
The code is partially borrowed from [PointWeb](https://github.com/hszhao/PointWeb).
