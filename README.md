## This is the code for our paper "Video Rescaling with Recurrent Diffusion".
Li, D., Liu, Y., Wang, Z., Yang, J.: Video rescaling with recurrent diffusion. IEEE Transactions on Circuits and Systems for Video Technology. pp. 1–14 (2024)

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

## The definitions of the folders:

RDVR: Our basic RDVR.

RDVRplus: Our basic RDVR + high-resolution fine-tuning.

RDVRplusplus: Our basic RDVR + high-resolution fine-tuning + low-resolution enhancer (please see our supplymentary material for the details of low-resolution enhancer).

RDVR-H265: Combing our basic RDVR with H.265 video compression. There are two stages in trainning. For stage 1 we train a basic RDVR for scale factor of 2 with BD downsampling for 250000 iterations. The second stage has 50000 iterations with H.265 video compression. At the second stage we apply the bicubically downsampled frames for the inputs of H.265 encoding in the first 25000 iterations. We use the outputs of the downsampling network for the inputs of H.265 encoding in the next 25000 iterations at stage 2.

## For training one model, enter the "codes" folder of the model, and run

sh train.sh

## For testing one model, enter the "codes" folder of the model, and run

sh test.sh

You can edit test.sh" to select the testing datasets.

## If you find that our work is useful, please cite:

@article{li2024video,
  title={Video Rescaling with Recurrent Diffusion},
  author={Li, Dingyi and Liu, Yu and Wang, Zengfu and Yang, Jian},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2024}
}
