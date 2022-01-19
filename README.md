# idn-solver
[**Paper**](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhao_A_Confidence-Based_Iterative_Solver_of_Depths_and_Surface_Normals_for_ICCV_2021_paper.pdf) | [**Project Page**](http://b1ueber2y.me/projects/IDN-Solver//) <br>


This repository contains the code release of our ICCV 2021 paper:

A Confidence-based Iterative Solver of Depths and Surface Normals for Deep Multi-view Stereo 

[Wang Zhao*](https://github.com/thuzhaowang), [Shaohui Liu*](http://b1ueber2y.me/), [Yi Wei](https://weiyithu.github.io/), [Hengkai Guo](https://github.com/guohengkai), [Yong-Jin Liu](https://cg.cs.tsinghua.edu.cn/people/~Yongjin/Yongjin.htm)
  
## Installation
We recommend to use conda to setup a specified environment. Run
```bash
conda env create -f environment.yml
```

## Test on a sequence
First download the pretrained model from <a href="https://drive.google.com/file/d/1ddKYtn9_9pPsXi0l9rY5atgrzfl5_yj6/view?usp=sharing">here</a> and put it under ./pretrain/ folder.

Prepare the sequence data with color images, camera poses (4x4 cam2world transformation) and intrinsics. The sequence data structure should be like:
```
sequence_name
  | color
      | 00000.jpg
  | pose
      | 00000.txt
  | K.txt
```

Run the following command to get the outputs:
```bash
python infer_folder.py --seq_dir /path/to/the/sequence/data --output_dir /path/to/save/outputs --config ./configs/test_folder.yaml
```
Tune the "reference gap" parameter to make sure there are sufficient overlaps and camera translations within an image pair. For ScanNet-like sequence, we recommend to use reference_gap of 20.

## Test on ScanNet
### Prepare ScanNet test split data
Download the ScanNet test split data from the <a href="https://github.com/ScanNet/ScanNet">official site</a> and pre-process the data using:
```bash
python ./data/preprocess.py --data_dir /path/to/scannet/test/split/ --output_dir /path/to/save/pre-processed/scannet/test/data
```
This includes 1. resize the color images to 480x640 resolution 2. sample the data with interval of 20


### Run evaluation
```bash
python eval_scannet.py --data_dir /path/to/processed/scannet/test/split/ --config ./configs/test_scannet.yaml
```

## Train
### Prepare ScanNet training data
We use the pre-processed ScanNet data from <a href="https://github.com/udaykusupati/Normal-Assisted-Stereo">NAS</a>, 
you could download the data using <a href="https://drive.google.com/drive/folders/1PTi37xlPxqhHNyxs_4xiGGj1OsnTQhWD?usp=sharing">this link</a>. The data structure is like:
```
scannet
  | scannet_nas
    | train
      | scene0000_00
          | color
            | 0000.jpg
          | pose
            | 0000.txt
          | depth
            | 0000.npy
          | intrinsic
          | normal
            | 0000_normal.npy
    | val
  | scans_test_sample (preprocessed ScanNet test split)
```

### Run training
Modify the "dataset_path" variable with yours in the config yaml. 

The network is trained with a two-stage strategy. The whole training process takes ~6 days with 4 Nvidia V100 GPUs. 
```bash
python train.py ./configs/scannet_stage1.yaml
python train.py ./configs/scannet_stage2.yaml
```

### Citation
If you find our work useful in your research, please consider citing:
```
@InProceedings{Zhao_2021_ICCV,
    author    = {Zhao, Wang and Liu, Shaohui and Wei, Yi and Guo, Hengkai and Liu, Yong-Jin},
    title     = {A Confidence-Based Iterative Solver of Depths and Surface Normals for Deep Multi-View Stereo},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {6168-6177}
}
```

### Acknowledgement
This project heavily relies codes from <a href="https://github.com/udaykusupati/Normal-Assisted-Stereo">NAS</a> and we thank the authors for releasing their code.

We also thank <a href="https://www.xxlong.site/">Xiaoxiao Long</a> for kindly helping with ScanNet evaluations.


