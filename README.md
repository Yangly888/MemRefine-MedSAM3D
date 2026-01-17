[![DOI](https://zenodo.org/badge/1126782636.svg)](https://doi.org/10.5281/zenodo.18137967)
<h1 align="center">● Enhancing 3D Medical Image Segmentation: Memory-Enhanced and Feature-Optimized Model</h1>

MemRefine-MedSAM3D is an advanced 3D medical image segmentation model that utilizes the [SAM 2](https://github.com/facebookresearch/segment-anything-2) framework to address 3D medical image segmentation tasks. The model introduces two novel modules: **CPGF (Cross-slice Progressive Gated Fusion)** for enhancing cross-slice contextual consistency, and **LGFF (Local-Global Feature Fusion)** for strengthening the representation of boundaries and fine-grained structures.

## 🧐 Requirement

 Install the environment:

 ``conda env create -f environment.yml``

 ``conda activate memrefine-medsam3d``

 You can download SAM2 checkpoint from checkpoints folder:

 ``bash download_ckpts.sh``

 Further Note: We tested on the following system environment and you may have to handle some issue due to system difference.
```
Operating System: Ubuntu 22.04
Conda Version: 23.7.4
Python Version: 3.12.4
```

 ## 🎯 Example Cases
 #### Download BTCV or FLARE or your own dataset and put in the ``data`` folder, create the folder if it does not exist ⚒️

 ### 3D case - BTCV Abdominal Multiple Organs Segmentation

 **Step1:** Download pre-processed [BTCV](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752) dataset manually from [here](https://huggingface.co/datasets/jiayuanz3/btcv/tree/main), or using command lines:

 ``wget https://huggingface.co/datasets/jiayuanz3/btcv/resolve/main/btcv.zip``

 ``unzip btcv.zip``

**Step2:** Run the training and validation by:

``python train_3d.py -net sam2 -exp_name BTCV_MemRefine -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -val_freq 5 -prompt bbox -prompt_freq 2 -dataset btcv -data_path ./data/btcv -cpgf 1  -use_lgff 1 ``

 ### 3D case - FLARE22 Abdominal Multiple Organs Segmentation

 **Step1:** Download [FLARE22](https://flare22.grand-challenge.org/) dataset manually from [here](https://flare22.grand-challenge.org/).

**Step2:** Run the training and validation by:

``python train_3d.py -net sam2 -exp_name FLARE_MemRefine -sam_ckpt ./checkpoints/sam2_hiera_small.pt -sam_config sam2_hiera_s -image_size 1024 -val_freq 5 -prompt bbox -prompt_freq 2 -dataset flare -data_path ./data/FLARE22 -cpgf 1 -use_lgff 1 ``

## 📝 Cite
 ~~~
@article{memrefine2025,
    title={Enhancing 3D Medical Image Segmentation: Memory-Enhanced and Feature-Optimized Model},
    author={Liying Yang, Shengwei Tian, et al},
    note={Under review at The Visual Computer},
    year={2025}
}
 ~~~
