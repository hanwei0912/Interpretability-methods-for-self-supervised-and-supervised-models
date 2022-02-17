# Interpretability-methods-for-self-supervised-and-supervised-models
In recent years, the rapid development of Deep Neural Networks (DNN) has led to a remarkable performance in many complex tasks in the field of computer vision at the cost of the models’ complexity. The more complex the models get, the higher the need is for understanding them. The primary objective of this repo is to give visual explanations on what both supervised and self-supervised methods really learn during training. Self-supervised and supervised state-of-the-art pre-trained models will be investigated. As backbone networks, for both categories convnets and Transformers based architectures will be used. Variation of visualization techniques will be used. 

This repo is a reproduction of https://github.com/jacobgil/pytorch-grad-cam to better understand self-supervised and supervised methods.
So any further information about the requirements of the environment and the way each script is executed can be found in the original code

In this set of experiments I try to interpret the following models.

Cait source --> https://github.com/facebookresearch/deit model:cait_s24_224 

Deit source --> https://github.com/facebookresearch/deit model:deit_base_patch16_224

Dino_res source --> https://github.com/facebookresearch/dino backbone:dino_resnet50   /classifier weights:dino_resnet50_linearweights.pth 

Dino_xcit source --> https://github.com/facebookresearch/dino backbone:dino_xcit_small_12_p16    /classifier weights:dino_xcit_small_12_p16_linearweights.pth

Res50 source --> torchvision.models.resnet.resnet50

SwinT source --> https://fastai.github.io/timmdocs/ model:swin_base_patch4_window7_224

Xcit source --> https://github.com/facebookresearch/xcit model weights:xcit_small_12_p16_224.pth

moco-v3_res source --> https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md backbone:torchvision.models.resnet.resnet50 /model weights:linear-1000ep.pth.tar

moco-v3_deit source --> https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md backbone:deit_base_patch16_224 /model weights:linear-vit-b-300ep.pth.tar

In order to execute the Xcit_main.py one has to download first the xcit_small_12_p16_224.pth from https://github.com/facebookresearch/xcit and simply put the .pth file to the pretrained_weights folder!

Then for the moco-v3_deit_main.py, download first the linear-vit-b-300ep.pth.tar from https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md and simply put the .pth file to the pretrained_weights folder. 

With the same thinking, for the moco-v3_res_main.py, download first the linear-1000ep.pth.tar from https://github.com/facebookresearch/moco-v3/blob/main/CONFIG.md and simply put the .pth file to the pretrained_weights folder.

More samples for the experiments can be found https://github.com/EliSchwartz/imagenet-sample-images

Each model script contains the quantitative evaluation of the activation map. If one wants only the qualitative results, he must comment some code lines!