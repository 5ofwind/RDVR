## Video Rescaling With Recurrent Diffusion
D. Li, Y. Liu, Z. Wang and J. Yang, "Video rescaling with recurrent diffusion," IEEE Transactions on Circuits and Systems for Video Technology., vol. 34, no. 10, pp. 9386–9399, 2024.

https://ieeexplore.ieee.org/abstract/document/10521831

## Abstract
Video rescaling helps to fit different display devices. In video rescaling systems, videos are downsampled for easier storage, transmission and preview. The downsampled videos can be upsampled with a neural network to restore the details when needed. Previous group-based video rescaling algorithms benefit from the joint downsampling and joint upsampling of multiple frames, but are restricted by the fully joint operation. In this paper, we propose a recurrent diffusion-based framework for video rescaling. We employ biased joint operation and recurrent diffusion, to make a better use of the temporal relation within different frames in each image group. We explicitly control the direction of information propagation by arranging the processing order of all frames. In biased joint operation, we concentrate on restoring one frame, i.e., the middle frame. The other frames in the group are coarsely reconstructed. Our recurrent diffusion compensates the coarse frames by gradually propagating information from the middle to borders backwardly and forwardly. The recurrent diffusion module is performed by fusing the information of adjacent frames. Biased joint operation and recurrent diffusion are jointly trained. We design several propagation variants and find that our recurrent diffusion is the best among them. It is also shown that recurrent diffusion is better than non-recurrent diffusion in terms of reconstruction quality and model size. We also adopt a high-resolution fine-tuning strategy to further improve the quality of high-resolution frames. Experimental results demonstrate the effectiveness of the proposed method in terms of visual quality, quantitative evaluations, and computational efficiency.

## Prerequisite
- Python 3
- PyTorch >= 1.4.0
- NVIDIA GPU + CUDA
- Python Package: "pip install numpy opencv-python lmdb pyyaml"
- FFmpeg with H.265 codec activated, for RDVR-H265

## Training Data Preparation
We adopt the LMDB format for training. The script is provided in `codes/data_scripts`.

## The trained models can be found at:

https://pan.baidu.com/s/1YOC3zna5mSAU9touCU-Wfw

Password：w3pg

## The definitions of the folders

RDVR: Our basic RDVR.

RDVRplus: RDVR+, our basic RDVR + high-resolution fine-tuning.

RDVRplusplus: RDVR++, our basic RDVR + high-resolution fine-tuning + low-resolution enhancer (please see our supplymentary material for the details of low-resolution enhancer).

RDVR-H265: Combing our basic RDVR with H.265 video compression. There are two stages in trainning. For stage 1 we train a basic RDVR for scale factor of 2 with BD downsampling for 250000 iterations. The second stage has 50000 iterations with H.265 video compression. At the second stage (with 50000 iterations in total), BD downsampling is also utilized. We apply the BD downsampled frames for the inputs of H.265 encoding in the first 25000 iterations. We use the outputs of the downsampling network for the inputs of H.265 encoding in the next 25000 iterations at Stage 2. We find that the two-step approach at Stage 2 leads to slight improvement in terms of MS-SSIM (about 0.0002 on average), compared with the situation that Step 1 is removed and Step 2 has 50000 iterations. Note that our final model is "RDVR-H265-Stage2-Step2".

## For training one model, enter the "codes" folder of the model, and run

sh train.sh

## For testing one model, enter the "codes" folder of the model, and run

sh test.sh

## You can edit test.sh" to select the testing datasets.

## If you find that our work is useful, please cite:

@article{li2024video,

  title={Video rescaling with recurrent diffusion},
  
  author={Li, Dingyi and Liu, Yu and Wang, Zengfu and Yang, Jian},
  
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  
  volume={34},
  
  number={10},
  
  pages={9386--9399},
  
  year={2024}
  
}
